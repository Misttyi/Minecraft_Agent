"""MineRL 生存一天：任务得分表、PPO / 模仿学习接口、并行训练和 HTML 回放。

不修改 survive_day 下的评测文件。训练时可以用 mc_rl.privileged 里带特权观测的
子类规格给专家看方块网格和实体；学生策略只读标准观测，评测仍走 survive_day.make_env。
"""

from mc_rl.macros import MACROS, MACRO_INDEX
from mc_rl.tasks import TASKS, TASK_NAMES, get_task

__all__ = ["MACROS", "MACRO_INDEX", "TASKS", "TASK_NAMES", "get_task"]
