"""多种子评测。会启动 Minecraft，因此不要在导入时创建环境。"""

import argparse
import json

from survive_day.constants import EVAL_SEEDS
from survive_day.scoring import check_seeds, summarize
from survive_day.survive_specs import make_env


def _plain(value):
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, (str, bytes)):
        try:
            return item()
        except (ValueError, AttributeError):
            return value
    return value


def run_episode(agent, seed, action_repeat):
    env = make_env(seed, action_repeat=action_repeat, strip_reward=True)
    try:
        observation = env.reset()
        done = False
        info = {
            "survival_steps": 0,
            "survived_day": False,
            "life": None,
            "food": None,
        }
        while not done:
            action = agent(observation, env)
            observation, _reward, done, info = env.step(action)
    finally:
        env.close()
    return {
        "seed": int(seed),
        "survival_steps": int(info["survival_steps"]),
        "survived_day": bool(info["survived_day"]),
        "life": _plain(info["life"]),
        "food": _plain(info["food"]),
    }


def evaluate(agent, seeds=EVAL_SEEDS, action_repeat=1):
    """agent(observation, env) -> action。奖励已被评测包装器丢掉。"""
    selected = tuple(seeds)
    check_seeds(selected)
    episodes = [
        run_episode(agent, seed, action_repeat=action_repeat) for seed in selected
    ]
    return summarize(episodes)


def write_report(report, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main(argv=None):
    from baselines.random_agent import act as random_act

    parser = argparse.ArgumentParser(description="评测 MineRLSurviveDay-v0")
    parser.add_argument("--out", required=True, help="结果 JSON 路径")
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--agent", choices=("random",), default="random")
    args = parser.parse_args(argv)
    if args.agent != "random":
        raise SystemExit("仓库只附带随机空操作。学习策略由学生自己接入 evaluate()。")
    report = evaluate(random_act, action_repeat=args.action_repeat)
    write_report(report, args.out)
    print(
        f"中位存活步数 {report['median_survival_steps']} / "
        f"{report['max_episode_steps']}，"
        f"活过一天 {report['survived_day_count']} / {len(report['episodes'])}"
    )


if __name__ == "__main__":
    main()
