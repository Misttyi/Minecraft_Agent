"""公开常数与包装器。MineRL 环境通过 make_env 延迟导入。"""

from survive_day.constants import (
    ENV_ID,
    EVAL_SEEDS,
    MAX_EPISODE_STEPS,
    REWARD_SCHEDULE,
)
from survive_day.scoring import summarize
from survive_day.wrappers import SurviveDayWrapper


def make_env(*args, **kwargs):
    from survive_day.survive_specs import make_env as build_env

    return build_env(*args, **kwargs)


__all__ = [
    "ENV_ID",
    "EVAL_SEEDS",
    "MAX_EPISODE_STEPS",
    "REWARD_SCHEDULE",
    "SurviveDayWrapper",
    "make_env",
    "summarize",
]
