"""多组参数并行训练。每组一个进程，自带 Xvfb 和若干个 Minecraft，GPU 轮流分配。

  python scripts/sweep.py --preset main --root runs/sweep1 --gpus 0,1,2,3
  python scripts/sweep.py --config my_grid.json --root runs/sweep2

配置是 {名字: {ppo 参数...}} 的字典，键名同 `mcrl.py ppo` 的命令行参数（下划线写法）。
看板：python scripts/mcrl.py dashboard --runs 'runs/sweep1/*' --out runs/sweep1/dashboard.html
"""

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ALL = "explore,wood,craft,tools,hunt,eat,shelter,nightfall,survive_day,torch,homing,combat,cook"

PRESETS = {
    # BC 初始化 + PPO + 衰减的模仿辅助损失（主方案）
    "main": {
        "bc_ppo": {"init": "BC", "bc_coef": 0.5, "bc_decay": 0.98, "lr": 1e-4, "ent_coef": 0.01},
        "bc_ppo_lowent": {"init": "BC", "bc_coef": 0.5, "bc_decay": 0.98, "lr": 1e-4, "ent_coef": 0.003},
        "bc_ppo_pure": {"init": "BC", "bc_coef": 0.0, "lr": 1e-4, "ent_coef": 0.005},
        "scratch_ppo_aux": {"bc_coef": 1.0, "bc_decay": 0.99, "lr": 3e-4, "ent_coef": 0.01},
        "scratch_ppo": {"bc_coef": 0.0, "lr": 3e-4, "ent_coef": 0.01},
        "vector_ppo": {"arch": "mlp", "bc_coef": 0.0, "lr": 3e-4, "ent_coef": 0.01, "rollout_steps": 256,
                       "minibatch": 64, "tasks": "survive_day"},
    },
    # 第二轮：安全生存专家 + 分阶段课程，7 组对比，每组一块 GPU（0–6）
    "v2": {
        "dapg_staged": {"init": "BC", "lr": 3e-5, "ent_coef": 0.001, "bc_coef": 1.0, "bc_decay": 0.995,
                        "bc_min": 0.3},
        "dapg_strong_staged": {"init": "BC", "lr": 3e-5, "ent_coef": 0.001, "bc_coef": 2.0, "bc_decay": 1.0,
                               "bc_min": 2.0},
        "dapg_flat": {"init": "BC", "lr": 3e-5, "ent_coef": 0.001, "bc_coef": 1.0, "bc_decay": 0.995,
                      "bc_min": 0.3, "staged": 0},
        "ppo_bc_staged": {"init": "BC", "lr": 1e-4, "ent_coef": 0.003, "bc_coef": 0.5, "bc_decay": 0.98,
                          "bc_min": 0.1},
        "scratch_aux_staged": {"lr": 3e-4, "ent_coef": 0.01, "bc_coef": 1.0, "bc_decay": 0.99, "bc_min": 0.1},
        "vector_ppo_staged": {"arch": "mlp", "lr": 3e-4, "ent_coef": 0.01, "bc_coef": 0.0},
        "vector_ppo_survive": {"arch": "mlp", "lr": 3e-4, "ent_coef": 0.01, "bc_coef": 0.0, "rollout_steps": 256,
                               "minibatch": 64, "tasks": "survive_day"},
    },
    # 第三轮：13 个任务（含扩展）分阶段课程；DAPG 为主，对照：无课程、只用核心任务、更大学习率、向量 PPO
    "v3": {
        "dapg_staged": {"init": "BC", "lr": 3e-5, "ent_coef": 0.001, "bc_coef": 1.0, "bc_decay": 0.995,
                        "bc_min": 0.5, "tasks": ALL},
        "dapg_strong": {"init": "BC", "lr": 3e-5, "ent_coef": 0.001, "bc_coef": 2.0, "bc_decay": 1.0,
                        "bc_min": 2.0, "tasks": ALL},
        "dapg_flat": {"init": "BC", "lr": 3e-5, "ent_coef": 0.001, "bc_coef": 1.0, "bc_decay": 0.995,
                      "bc_min": 0.5, "staged": 0, "tasks": ALL},
        "dapg_core": {"init": "BC", "lr": 3e-5, "ent_coef": 0.001, "bc_coef": 1.0, "bc_decay": 0.995,
                      "bc_min": 0.5},
        "dapg_lr1e4": {"init": "BC", "lr": 1e-4, "ent_coef": 0.001, "bc_coef": 1.0, "bc_decay": 0.995,
                       "bc_min": 0.5, "tasks": ALL},
        "vector_ppo_staged": {"arch": "mlp", "lr": 3e-4, "ent_coef": 0.01, "bc_coef": 0.0, "tasks": ALL},
        "vector_ppo_survive": {"arch": "mlp", "lr": 3e-4, "ent_coef": 0.01, "bc_coef": 0.0, "rollout_steps": 256,
                               "minibatch": 64, "tasks": "survive_day"},
    },
    "smoke": {
        "smoke_a": {"bc_coef": 0.5, "total_ticks": 40000, "rollout_steps": 64, "num_envs": 2},
        "smoke_b": {"bc_coef": 0.0, "total_ticks": 40000, "rollout_steps": 64, "num_envs": 2},
    },
}


def build_command(name, cfg, root, bc_path, env_kind):
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "mcrl.py"), "ppo", "--run", os.path.join(root, name),
           "--env", env_kind]
    for key, value in cfg.items():
        if key == "init":
            if value == "BC":
                if not bc_path:
                    raise SystemExit(f"{name} 需要 --bc 检查点")
                value = bc_path
            cmd += ["--init", value]
            continue
        if key == "task_weights":
            value = json.dumps(value)
        cmd += ["--" + key.replace("_", "-"), str(value)]
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="main", choices=sorted(PRESETS))
    parser.add_argument("--config", help="JSON 文件，覆盖 preset")
    parser.add_argument("--only", help="逗号分隔，只跑这些名字")
    parser.add_argument("--root", required=True)
    parser.add_argument("--bc", help="BC 检查点（init=BC 的组用）")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--total-ticks", type=int, default=3_000_000)
    parser.add_argument("--env", default="minerl", choices=("minerl", "mock"))
    parser.add_argument("--stagger", type=float, default=20.0, help="组与组之间的启动间隔（秒）")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    grid = dict(PRESETS[args.preset])
    if args.config:
        with open(args.config) as handle:
            grid = json.load(handle)
    # 种子按完整网格里的位置定：用 --only 补跑某几组时种子和第一次一样
    seeds = {name: index for index, name in enumerate(grid)}
    if args.only:
        keep = set(args.only.split(","))
        grid = {k: v for k, v in grid.items() if k in keep}
    gpus = args.gpus.split(",")
    os.makedirs(args.root, exist_ok=True)
    manifest_path = os.path.join(args.root, "sweep.json")
    manifest = {}
    if args.only and os.path.exists(manifest_path):  # 补跑：保留其余组的记录
        with open(manifest_path) as handle:
            manifest = json.load(handle)
    procs = []
    for index, (name, cfg) in enumerate(grid.items()):
        cfg = dict(cfg)
        cfg.setdefault("num_envs", args.num_envs)
        cfg.setdefault("total_ticks", args.total_ticks)
        cfg.setdefault("seed", seeds[name])
        cmd = build_command(name, cfg, args.root, args.bc, args.env)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpus[index % len(gpus)]
        env.pop("DISPLAY", None)  # 每组自己起 Xvfb
        manifest[name] = {"cmd": cmd, "gpu": env["CUDA_VISIBLE_DEVICES"], "cfg": cfg}
        print(name, "GPU", env["CUDA_VISIBLE_DEVICES"], " ".join(cmd), flush=True)
        if args.dry_run:
            continue
        os.makedirs(os.path.join(args.root, name), exist_ok=True)
        out = open(os.path.join(args.root, name, "stdout.txt"), "a")
        procs.append((name, subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env, cwd=ROOT,
                                             start_new_session=True)))
        manifest[name]["pid"] = procs[-1][1].pid
        time.sleep(args.stagger)
    if args.dry_run:
        return
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(f"已启动 {len(procs)} 组；进程号写在 {args.root}/sweep.json")


if __name__ == "__main__":
    main()
