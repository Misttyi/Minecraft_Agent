"""评测汇总。不导入 MineRL。"""

from survive_day.constants import EVAL_SEEDS, MAX_EPISODE_STEPS


def median(values):
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        raise ValueError("没有可汇总的数值")
    middle = count // 2
    if count % 2 == 1:
        return float(ordered[middle])
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def summarize(episodes):
    if not episodes:
        raise ValueError("没有评测局")
    return {
        "episodes": list(episodes),
        "median_survival_steps": median(
            episode["survival_steps"] for episode in episodes
        ),
        "survived_day_count": sum(
            1 for episode in episodes if episode["survived_day"]
        ),
        "max_episode_steps": MAX_EPISODE_STEPS,
        "seeds": [episode["seed"] for episode in episodes],
    }


def check_seeds(seeds):
    unknown = [seed for seed in seeds if seed not in EVAL_SEEDS]
    if unknown:
        raise ValueError(
            f"种子 {unknown} 不在固定评测列表 {list(EVAL_SEEDS)} 里"
        )
