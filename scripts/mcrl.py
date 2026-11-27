"""命令行入口。

  python scripts/mcrl.py tables                          生成任务得分表（Markdown + HTML）
  python scripts/mcrl.py demo   --out runs/demo ...      采集特权专家示范
  python scripts/mcrl.py bc     --demos a.npz b.npz ...  行为克隆
  python scripts/mcrl.py dagger --run runs/dagger ...    DAgger
  python scripts/mcrl.py ppo    --run runs/ppo_a ...     PPO（可从 BC 初始化，带 BC 辅助损失）
  python scripts/mcrl.py eval   --checkpoint best.pt     10 个评测种子（未改动的 make_env）
  python scripts/mcrl.py render --run runs/ppo_a         把回放 npz 渲染成 HTML
  python scripts/mcrl.py dashboard --runs runs/* --out runs/dashboard.html

--env mock 用替身环境（不启动 Minecraft），用于冒烟测试。
"""

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _display_env(args):
    """Minecraft 需要 X 显示。没有 DISPLAY 时自己起一个 Xvfb。"""
    if args.env == "mock":
        return None
    from mc_rl.display import ensure_display

    return ensure_display()


def _vec(args, expert=True):
    from mc_rl.vec_env import SubprocVecEnv

    spec = {
        "kind": args.env,
        "privileged": True,
        "expert": expert,
        "action_repeat": args.action_repeat,
        "display": os.environ.get("DISPLAY"),
        "launch_stagger": 0.0 if args.env == "mock" else args.start_delay,
        "env_vars": {k: os.environ[k] for k in ("JAVA_HOME", "PATH", "MINERL_GRADLE_USER_HOME") if k in os.environ},
    }
    return SubprocVecEnv(args.num_envs, spec)


def _device(args):
    import torch

    if args.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return args.device


def _log_to(run_dir):
    os.makedirs(run_dir, exist_ok=True)
    handle = open(os.path.join(run_dir, "log.txt"), "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        handle.write(msg + "\n")
        handle.flush()

    return log


def cmd_tables(args):
    from mc_rl.render import write_score_tables_html
    from mc_rl.tasks import score_series, score_tables_markdown

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "score_tables.md"), "w", encoding="utf-8") as handle:
        handle.write(score_tables_markdown())
    with open(os.path.join(args.out, "score_tables.json"), "w", encoding="utf-8") as handle:
        json.dump(score_series(), handle, ensure_ascii=False, indent=2)
    write_score_tables_html(os.path.join(args.out, "score_tables.html"))
    print("得分表已写入", args.out)


def cmd_demo(args):
    from mc_rl.imitation import collect, summarize_episodes
    from mc_rl.ppo import parse_seeds
    from mc_rl.render import render_all_replays

    _display_env(args)
    log = _log_to(args.run)
    vec = _vec(args)
    try:
        weights = json.loads(args.task_weights) if args.task_weights else None
        _, episodes = collect(vec, None, args.steps, args.tasks.split(","), parse_seeds(args.train_seeds),
                              beta=1.0, seed=args.seed, out_path=os.path.join(args.run, f"demo_{args.seed}.npz"),
                              run_dir=args.run, log_fn=log, task_weights=weights)
    finally:
        vec.close()
    log(json.dumps(summarize_episodes(episodes), ensure_ascii=False, indent=1))
    render_all_replays(args.run)


def cmd_bc(args):
    import torch

    from mc_rl.imitation import load_demos, train_bc

    log = _log_to(args.run)
    drop = [t for t in (args.drop_tasks or "").split(",") if t]
    data = load_demos(args.demos + list(args.drop_from or []), drop_tasks=drop, drop_from=args.drop_from or [])
    log(f"示范样本 {len(data['labels'])}")
    init = None
    if args.init:
        from mc_rl.models import load_checkpoint

        init, _ = load_checkpoint(args.init, _device(args), upgrade=True)
    model, history = train_bc(data, model=init, arch=args.arch, epochs=args.epochs, device=_device(args), log_fn=log,
                              balance=args.balance, lr=args.lr)
    best = next((h for h in reversed(history) if h.get("best")), history[-1])
    torch.save({"config": model.config(), "state_dict": model.state_dict(), "extra": {"bc": best}},
               os.path.join(args.run, "bc.pt"))
    with open(os.path.join(args.run, "bc_history.json"), "w") as handle:
        json.dump(history, handle, indent=1, ensure_ascii=False)


def cmd_dagger(args):
    from mc_rl.imitation import run_dagger
    from mc_rl.models import ActorCritic, load_checkpoint
    from mc_rl.features import vector_size
    from mc_rl.macros import NUM_MACROS
    from mc_rl.ppo import parse_seeds
    from mc_rl.render import render_all_replays
    from mc_rl.tasks import TASKS

    _display_env(args)
    log = _log_to(args.run)
    device = _device(args)
    if args.init:
        model, _ = load_checkpoint(args.init, device, upgrade=True)
    else:
        model = ActorCritic(vector_size(len(TASKS)), NUM_MACROS, arch=args.arch).to(device)
    vec = _vec(args)
    try:
        run_dagger(vec, model, args.rounds, args.steps, args.tasks.split(","), parse_seeds(args.train_seeds),
                   args.run, device=device, epochs=args.epochs, beta0=args.beta0, beta_decay=args.beta_decay,
                   demo_paths=args.demos or (), log_fn=log, lr=args.lr,
                   task_weights=json.loads(args.task_weights) if args.task_weights else None)
    finally:
        vec.close()
    render_all_replays(args.run)


def cmd_ppo(args):
    from mc_rl.models import load_checkpoint
    from mc_rl.ppo import train_ppo
    from mc_rl.render import render_all_replays, render_dashboard

    _display_env(args)
    log = _log_to(args.run)
    device = _device(args)
    init = None
    if args.init:
        init, _ = load_checkpoint(args.init, device, upgrade=True)
        log(f"从 {args.init} 初始化")
    cfg = {k: getattr(args, k) for k in ("num_envs", "rollout_steps", "epochs", "minibatch", "gamma", "lam", "clip",
                                         "lr", "ent_coef", "vf_coef", "bc_coef", "bc_decay", "bc_min", "total_ticks",
                                         "tasks", "arch", "seed", "train_seeds", "replay_every", "action_repeat",
                                         "curriculum_floor", "staged", "start_stage")}
    cfg["staged"] = bool(cfg["staged"])
    if args.task_weights:
        cfg["task_weights"] = json.loads(args.task_weights)
    vec = _vec(args, expert=True)
    try:
        train_ppo(vec, cfg, args.run, init=init, device=device, log_fn=log)
    finally:
        vec.close()
    render_all_replays(args.run)
    render_dashboard([args.run], os.path.join(args.run, "dashboard.html"))


def cmd_eval(args):
    from mc_rl.evaluate import evaluate_parallel, write_report
    from mc_rl.render import render_all_replays
    from survive_day.constants import EVAL_SEEDS

    _display_env(args)
    seeds = EVAL_SEEDS if not args.seeds else tuple(int(s) for s in args.seeds.split(","))
    env_vars = {k: os.environ[k] for k in ("DISPLAY", "JAVA_HOME", "PATH") if k in os.environ}
    report = evaluate_parallel(args.agent, args.checkpoint, seeds=seeds, task=args.task,
                               action_repeat=args.action_repeat, workers=args.workers, record=args.record,
                               out_dir=args.run, env_kind=args.env, env_vars=env_vars)
    out = args.out or os.path.join(args.run, f"eval_{args.agent}.json")
    write_report(report, out)
    print(f"{args.agent}: 中位存活 {report['median_survival_steps']}，活过一天 {report['survived_day_count']}/"
          f"{len(report['episodes'])}，死因 {report['death_causes']} -> {out}")
    if args.record:
        render_all_replays(args.run)


def cmd_render(args):
    from mc_rl.render import render_all_replays, render_replay

    if args.npz:
        print(render_replay(args.npz))
    else:
        for path in render_all_replays(args.run):
            print(path)


def cmd_dashboard(args):
    from mc_rl.render import render_all_replays, render_dashboard

    runs = []
    for pattern in args.runs:
        runs += [p for p in sorted(glob.glob(pattern)) if os.path.exists(os.path.join(p, "metrics.jsonl"))]
    for run in runs:
        render_all_replays(run)
    evals = []
    for pattern in args.evals or []:
        evals += sorted(glob.glob(pattern))
    print(render_dashboard(runs, args.out, eval_files=evals))


def main(argv=None):
    from mc_rl.ppo import DEFAULTS
    from mc_rl.tasks import TASK_NAMES

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p, envs=True):
        p.add_argument("--env", choices=("minerl", "mock"), default="minerl")
        p.add_argument("--action-repeat", type=int, default=8)
        p.add_argument("--device", default="auto")
        p.add_argument("--seed", type=int, default=0)
        if envs:
            p.add_argument("--num-envs", type=int, default=8)
            p.add_argument("--start-delay", type=float, default=3.0)
            p.add_argument("--tasks", default=",".join(TASK_NAMES))
            p.add_argument("--train-seeds", default=DEFAULTS["train_seeds"])

    p = sub.add_parser("tables")
    p.add_argument("--out", default=os.path.join(ROOT, "docs"))
    p.set_defaults(fn=cmd_tables)

    p = sub.add_parser("demo")
    common(p)
    p.add_argument("--run", required=True)
    p.add_argument("--steps", type=int, default=500, help="每个环境的决策步")
    p.add_argument("--task-weights", help="任务采样权重 JSON")
    p.set_defaults(fn=cmd_demo)

    p = sub.add_parser("bc")
    common(p, envs=False)
    p.add_argument("--run", required=True)
    p.add_argument("--demos", nargs="+", required=True)
    p.add_argument("--arch", default="cnn")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--balance", type=float, default=0.5)
    p.add_argument("--init", help="从已有检查点继续训练")
    p.add_argument("--drop-from", nargs="*", help="这些示范文件里去掉 --drop-tasks 的样本（也会被加载）")
    p.add_argument("--drop-tasks", help="逗号分隔的任务名")
    p.set_defaults(fn=cmd_bc)

    p = sub.add_parser("dagger")
    common(p)
    p.add_argument("--run", required=True)
    p.add_argument("--init")
    p.add_argument("--demos", nargs="*")
    p.add_argument("--arch", default="cnn")
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--beta0", type=float, default=1.0)
    p.add_argument("--beta-decay", type=float, default=0.5)
    p.add_argument("--task-weights", help="任务采样权重 JSON")
    p.set_defaults(fn=cmd_dagger)

    p = sub.add_parser("ppo")
    common(p)
    p.add_argument("--run", required=True)
    p.add_argument("--init")
    p.add_argument("--task-weights", help='固定任务权重 JSON，例如 {"survive_day": 1}')
    for key in ("rollout_steps", "epochs", "minibatch", "total_ticks", "replay_every"):
        p.add_argument("--" + key.replace("_", "-"), type=int, default=DEFAULTS[key])
    for key in ("gamma", "lam", "clip", "lr", "ent_coef", "vf_coef", "bc_coef", "bc_decay", "bc_min",
                "curriculum_floor"):
        p.add_argument("--" + key.replace("_", "-"), type=float, default=DEFAULTS[key])
    p.add_argument("--arch", default=DEFAULTS["arch"], choices=("cnn", "mlp"))
    p.add_argument("--staged", type=int, default=1, help="1=按阶段解锁的课程，0=所有任务一起自适应采样")
    p.add_argument("--start-stage", type=int, default=0)
    p.set_defaults(fn=cmd_ppo)

    p = sub.add_parser("eval")
    common(p, envs=False)
    p.add_argument("--agent", choices=("policy", "random_macro", "daytime", "noop"), default="policy")
    p.add_argument("--checkpoint")
    p.add_argument("--task", default="survive_day")
    p.add_argument("--run", default=os.path.join(ROOT, "runs", "eval"))
    p.add_argument("--out")
    p.add_argument("--seeds")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--record", action="store_true")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("render")
    p.add_argument("--run")
    p.add_argument("--npz")
    p.set_defaults(fn=cmd_render)

    p = sub.add_parser("dashboard")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--evals", nargs="*")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_dashboard)

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
