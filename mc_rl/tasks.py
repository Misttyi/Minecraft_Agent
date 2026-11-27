"""任务系列与得分表。

每个任务都有：
- 一张得分表：一次性里程碑，points 合计 100，用来给任务打分（0–100）；
  reward 是训练用的奖励，可以和 points 不同。里程碑带前置条件，必须按顺序拿，
  避免策略绕过关键步骤刷分（例如没砍树就“捡到”木板）。
- 稠密塑形项：存活、受伤、死亡、夜间露天、饥饿等，每个决策步结算。
- 可选的特权塑形：训练环境带方块网格和实体观测时，奖励靠近最近的树或动物。
  这些量只进奖励，不进策略输入，评测时用不到。

任务从易到难，前面任务里的技能是后面任务的子步骤，课程采样按成功率自动调整。
最终评分口径见作业说明；本文件只提供训练得分表、奖励塑形和回放展示。
"""

import math
from collections import defaultdict

from survive_day.constants import MAX_EPISODE_STEPS, REWARD_SCHEDULE

SUNSET_TICK = 12000
NIGHT_TICK = 13000


class Milestone:
    def __init__(self, key, text, points, reward, check, requires=(), at_end=False):
        self.key = key
        self.text = text
        self.points = float(points)
        self.reward = float(reward)
        self.check = check
        self.requires = tuple(requires)
        self.at_end = at_end


class Achievement:
    """得分表之外的可选成就（速通、无伤、点亮庇护所……）：不计入 100 分，回放里显示徽章，训练给一点小奖励。"""

    def __init__(self, key, text, check, reward=2.0, at_end=False):
        self.key = key
        self.text = text
        self.check = check
        self.reward = float(reward)
        self.at_end = at_end


class Task:
    def __init__(
        self,
        name,
        title,
        goal,
        horizon,
        milestones,
        success,
        dense,
        weight=1.0,
        stage=0,
        stop_on_success=True,
        approach=None,
        skills=(),
        achievements=(),
        extension=False,
    ):
        self.name = name
        self.title = title
        self.goal = goal
        self.horizon = int(horizon)
        self.milestones = list(milestones)
        self.success_key = success
        self.dense = dict(dense)
        self.weight = float(weight)
        self.stage = int(stage)
        self.stop_on_success = stop_on_success
        self.approach = approach
        self.skills = tuple(skills)
        self.achievements = list(achievements)
        self.extension = extension
        total = sum(m.points for m in self.milestones)
        if abs(total - 100.0) > 1e-6:
            raise ValueError(f"任务 {name} 的得分表合计 {total}，应为 100")
        keys = [m.key for m in self.milestones]
        if success not in keys:
            raise ValueError(f"任务 {name} 的成功里程碑 {success} 不在得分表里")
        for m in self.milestones:
            for need in m.requires:
                if need not in keys:
                    raise ValueError(f"{name}.{m.key} 的前置 {need} 不存在")


# ---------------------------------------------------------------- 判定小工具


def got(item, count=1):
    return lambda t: t.collected[item] >= count


def depth(blocks):
    return lambda t: t.shaft_depth >= blocks


def in_shaft(t):
    """人在自己挖的竖井里：离第一次 dig_down 的位置水平 ≤ 1.5 格，且至少低于井口 1 格。"""
    a = t.dig_anchor
    if a is None:
        return False
    return math.hypot(t.state["x"] - a[0], t.state["z"] - a[2]) <= 1.5 and t.state["y"] <= a[1] - 1


def sealed(t):
    # 只算在自己的竖井里头顶被盖住；走出竖井钻进洞穴、树下、悬崖下都不算
    return t.state["sky"] < 0.5 and t.shaft_depth >= 2 and t.mem.sealed >= 1 and in_shaft(t)


def alive_until(tick):
    return lambda t: t.state["alive"] > 0.5 and t.mem.ticks >= tick


def sealed_before_sunset(t):
    return sealed(t) and t.mem.ticks <= SUNSET_TICK + 600


def sealed_for(ticks):
    return lambda t: t.sealed_ticks >= ticks


def event(name, count=1):
    return lambda t: t.events[name] >= count


def far(blocks):
    return lambda t: t.max_dist >= blocks


def end_life_at_least(value):
    return lambda t: t.state["alive"] > 0.5 and t.state["life"] >= value


def milestone_by(key, tick):
    """某个里程碑在 tick 之前拿到。"""
    return lambda t: key in t.done and t.done[key] <= tick


def unhurt(t):
    return t.state["alive"] > 0.5 and t.min_life >= 20


def back_home(radius=4.0, out=24):
    return lambda t: t.max_dist >= out and t.state["dist"] <= radius


def at_night_holding(item):
    return lambda t: t.mem.ticks >= NIGHT_TICK - 500 and t.state["hand"] == item


def end_food_at_least(value):
    return lambda t: t.state["alive"] > 0.5 and t.state["food"] >= value


M = Milestone

DENSE_BASE = {
    "alive": 0.0,  # 每 8 刻
    "damage": 0.5,  # 每掉 1 点血
    "death": 20.0,
    "hunger": 0.0,  # food < 6 时每 8 刻
    "night_exposed": 0.0,  # 夜里露天每 8 刻
    "night_sheltered": 0.0,  # 夜里在坑中每 8 刻
    "approach": 0.0,  # 特权塑形：靠近目标每格
    "invalid": 0.0,  # 选到被掩码屏蔽的动作（理论上不会发生）
}


def dense(**overrides):
    table = dict(DENSE_BASE)
    table.update(overrides)
    return table


TASKS = (
    Task(
        "explore",
        "探索移动",
        "离开出生点，跳上台阶，走出 32 格且不受伤。学会前进、转向、跳跃。",
        horizon=1600,
        milestones=[
            M("dist8", "离出生点 8 格", 25, 3, far(8)),
            M("dist16", "离出生点 16 格", 25, 3, far(16), ["dist8"]),
            M("dist32", "离出生点 32 格", 30, 5, far(32), ["dist16"]),
            M("unhurt", "结束时生命 ≥ 18", 20, 3, end_life_at_least(18), ["dist8"], at_end=True),
        ],
        success="dist32",
        dense=dense(damage=0.3),
        weight=0.5,
        stage=0,
        skills=("forward", "forward_jump", "turn_*"),
    ),
    Task(
        "wood",
        "采集原木",
        "找到树，对准树干连续挥击，收集 5 根原木。",
        horizon=2400,
        milestones=[
            M("log1", "收集 1 根原木", 40, 10, got("log", 1)),
            M("log3", "收集 3 根原木", 35, 8, got("log", 3), ["log1"]),
            M("log5", "收集 5 根原木", 25, 6, got("log", 5), ["log3"]),
        ],
        success="log3",
        dense=dense(approach=0.05),
        approach="log",
        stage=0,
        skills=("chop", "look_up", "forward"),
    ),
    Task(
        "craft",
        "基础合成",
        "原木 -> 木板 -> 木棍 -> 工作台，并把工作台放到地上。",
        horizon=3600,
        milestones=[
            M("log1", "收集 1 根原木", 10, 3, got("log", 1)),
            M("log3", "收集 3 根原木", 10, 3, got("log", 3), ["log1"]),
            M("planks", "合成 ≥4 块木板", 15, 4, got("planks", 4), ["log1"]),
            M("stick", "合成木棍", 15, 4, got("stick", 1), ["planks"]),
            M("table", "合成工作台", 20, 6, got("crafting_table", 1), ["planks"]),
            M("placed", "放下工作台", 30, 8, event("table_placed"), ["table"]),
        ],
        success="placed",
        dense=dense(approach=0.03),
        approach="log",
        stage=1,
        skills=("craft_planks", "craft_stick", "craft_table", "place_table"),
    ),
    Task(
        "tools",
        "木制工具",
        "在放下的工作台旁合成木剑和木镐，并拿起木剑。",
        horizon=6000,
        milestones=[
            M("log3", "收集 3 根原木", 10, 3, got("log", 3)),
            M("table", "合成工作台", 10, 3, got("crafting_table", 1), ["log3"]),
            M("placed", "放下工作台", 10, 4, event("table_placed"), ["table"]),
            M("sword", "合成木剑", 25, 10, got("wooden_sword", 1), ["placed"]),
            M("pickaxe", "合成木镐", 25, 10, got("wooden_pickaxe", 1), ["placed"]),
            M("wield", "手持木剑", 20, 4, event("hold_sword"), ["sword"]),
        ],
        success="pickaxe",
        dense=dense(approach=0.03),
        approach="log",
        stage=2,
        skills=("craft_sword", "craft_pickaxe", "equip_sword"),
    ),
    Task(
        "hunt",
        "狩猎",
        "找到牛或猪，靠近后挥击，捡起掉落的生肉。",
        horizon=6000,
        milestones=[
            M("meat1", "获得 1 块肉", 50, 20, got("meat", 1)),
            M("meat3", "获得 3 块肉", 25, 10, got("meat", 3), ["meat1"]),
            M("healthy", "结束时生命 ≥ 14", 25, 3, end_life_at_least(14), ["meat1"], at_end=True),
        ],
        success="meat1",
        dense=dense(approach=0.05),
        approach="animal",
        stage=2,
        skills=("attack", "forward", "turn_*"),
    ),
    Task(
        "eat",
        "饥饿时进食",
        "拿到生肉后装备到主手，按住右键吃下去。饥饿值没满就可以吃。",
        horizon=7200,
        milestones=[
            M("meat1", "获得 1 块肉", 30, 10, got("meat", 1)),
            M("hold", "主手拿肉", 15, 4, event("hold_meat"), ["meat1"]),
            M("ate", "吃下一块肉", 35, 15, event("ate"), ["hold"]),
            M("fed", "结束时饥饿 ≥ 18", 20, 3, end_food_at_least(18), ["ate"], at_end=True),
        ],
        success="ate",
        dense=dense(approach=0.04, hunger=0.05),
        approach="animal",
        stage=2,
        skills=("equip_meat", "eat"),
    ),
    Task(
        "shelter",
        "挖坑封顶",
        "原地向下挖 3 格，再抬头用泥土封住头顶，在坑里待满 400 刻。",
        horizon=2400,
        milestones=[
            M("dig1", "垂直下挖 1 格", 15, 4, depth(1)),
            M("dig2", "垂直下挖 2 格", 15, 4, depth(2), ["dig1"]),
            M("dig3", "垂直下挖 3 格", 20, 5, depth(3), ["dig2"]),
            M("sealed", "头顶封住、看不见天空", 30, 10, sealed, ["dig2"]),
            M("stay", "封顶后待满 400 刻", 20, 6, sealed_for(400), ["sealed"]),
        ],
        success="sealed",
        stop_on_success=False,
        dense=dense(damage=1.0),
        stage=1,
        skills=("dig_down", "seal_above", "sneak"),
    ),
    Task(
        "nightfall",
        "日落前进庇护所",
        "白天砍树、做工作台，日落前挖坑封顶，安全进入夜晚。",
        horizon=16000,
        milestones=[
            M("log3", "收集 3 根原木", 5, 3, got("log", 3)),
            M("table", "合成工作台", 5, 3, got("crafting_table", 1), ["log3"]),
            M("shelter", "日落前封顶（≤12600 刻）", 35, 20, sealed_before_sunset),
            M("dusk", "活到 13000 刻（入夜）", 20, 10, alive_until(NIGHT_TICK)),
            M("night", "活到 16000 刻", 35, 20, alive_until(16000), ["dusk"]),
        ],
        success="night",
        stop_on_success=False,
        dense=dense(damage=1.0, death=40.0, night_exposed=0.2, night_sheltered=0.02),
        approach="log",
        weight=1.5,
        stage=3,
        skills=("chop", "craft_*", "dig_down", "seal_above"),
    ),
    Task(
        "survive_day",
        "生存一天（总评任务）",
        "从清晨活到 24000 刻：采集、合成、日落前藏身、饿了进食。",
        horizon=MAX_EPISODE_STEPS,
        milestones=[
            M("log1", "收集原木", 5, 2, got("log", 1)),
            M("table", "合成工作台", 5, 3, got("crafting_table", 1), ["log1"]),
            M("shelter", "日落前封顶", 10, 15, sealed_before_sunset),
            M("t6000", "活到 6000 刻（正午）", 10, 5, alive_until(6000)),
            M("t12000", "活到 12000 刻（日落）", 15, 10, alive_until(12000), ["t6000"]),
            M("t18000", "活到 18000 刻（午夜）", 20, 20, alive_until(18000), ["t12000"]),
            M("t24000", "活到任务结束（≥23800 刻；Malmo 会在 24000 前几十刻按时间结束）", 25, 50,
              alive_until(MAX_EPISODE_STEPS - 200), ["t18000"]),
            M("fed", "结束时饥饿 ≥ 10", 10, 5, end_food_at_least(10), ["t12000"], at_end=True),
        ],
        success="t24000",
        stop_on_success=False,
        dense=dense(alive=0.01, damage=1.0, death=40.0, hunger=0.05,
                    night_exposed=0.2, night_sheltered=0.02),
        approach="log",
        weight=2.0,
        stage=3,
        skills=("全部",),
    ),
    Task(
        "torch",
        "煤炭与火把（进阶）",
        "做木镐，挖到石头和煤，合成火把并放下照明。",
        horizon=12000,
        milestones=[
            M("pickaxe", "合成木镐", 20, 8, got("wooden_pickaxe", 1)),
            M("cobble", "挖到圆石", 20, 8, got("cobblestone", 1), ["pickaxe"]),
            M("coal", "挖到煤炭", 25, 16, got("coal", 1), ["pickaxe"]),
            M("torch", "合成火把", 20, 16, got("torch", 1), ["coal"]),
            M("light", "放下火把", 15, 6, event("torch_placed"), ["torch"]),
        ],
        success="torch",
        dense=dense(approach=0.02),
        approach="log",
        weight=0.3,
        stage=4,
        skills=("equip_pickaxe", "chop", "craft_torch", "place_torch"),
    ),
)

A = Achievement

# 每个任务的成就：不改变 100 分得分表，只在回放和看板上显示徽章，训练时给一点小奖励。
ACHIEVEMENTS = {
    "explore": [A("sprint", "800 刻内走出 32 格", milestone_by("dist32", 800))],
    "wood": [A("speed", "800 刻内收集 5 根原木", milestone_by("log5", 800)),
             A("unhurt", "全程无伤", unhurt, at_end=True)],
    "craft": [A("speed", "1500 刻内放下工作台", milestone_by("placed", 1500))],
    "tools": [A("speed", "3000 刻内做出木镐", milestone_by("pickaxe", 3000))],
    "hunt": [A("hunter", "收集 3 块肉且全程无伤", lambda t: "meat3" in t.done and t.min_life >= 20, at_end=True)],
    "eat": [A("full", "吃过肉且结束时饥饿值满 20", lambda t: "ate" in t.done and t.state["food"] >= 20, at_end=True)],
    "shelter": [A("quick", "400 刻内封顶", milestone_by("sealed", 400)),
                A("unhurt", "全程无伤", unhurt, at_end=True)],
    "nightfall": [A("early", "6000 刻前就封顶", lambda t: sealed(t) and t.mem.ticks <= 6000),
                  A("lit", "在封顶的庇护所里放下火把", event("torch_sealed"))],
    "survive_day": [A("flawless", "全天无伤", unhurt, at_end=True),
                    A("lit", "在封顶的庇护所里放下火把", event("torch_sealed")),
                    A("well_fed", "结束时饥饿 ≥ 18", end_food_at_least(18), at_end=True)],
    "torch": [A("lit2", "放下 2 支火把", event("torch_placed", 2))],
}
for _task in TASKS:
    _task.achievements = ACHIEVEMENTS.get(_task.name, [])

# ---------------------------------------------------------------- 扩展任务
# 追加在核心 10 个任务之后：任务编号、观测向量和动作都向后兼容（旧检查点不受影响）。
# 默认课程只用核心任务；用 --tasks 显式加入扩展任务。

EXTENSION_TASKS = (
    Task(
        "homing",
        "外出回家",
        "走到离出生点 24 格以外，再回到出生点 4 格以内。练习用“出生点方位”特征导航和记忆。",
        horizon=3200,
        milestones=[
            M("out24", "离出生点 24 格", 30, 5, far(24)),
            M("home", "回到出生点 4 格内", 50, 15, back_home(), ["out24"]),
            M("unhurt", "结束时生命 ≥ 18", 20, 3, end_life_at_least(18), ["out24"], at_end=True),
        ],
        success="home",
        dense=dense(damage=0.5),
        stage=1,
        weight=0.8,
        skills=("forward", "turn_*", "出生点方位"),
        achievements=[A("speed", "2000 刻内回家", milestone_by("home", 2000))],
        extension=True,
    ),
    Task(
        "combat",
        "夜战",
        "白天做木剑，入夜后手持木剑击杀怪物（夜里经验值增加记为一次击杀），并活到 20000 刻。",
        horizon=20000,
        milestones=[
            M("sword", "手持木剑", 15, 5, event("hold_sword")),
            M("armed", "入夜时手持木剑", 15, 5, at_night_holding("wooden_sword"), ["sword"]),
            M("kill1", "夜里手持木剑击杀 1 只怪物", 35, 20, event("night_kill_sword"), ["sword"]),
            M("kill3", "夜里手持木剑击杀 3 只怪物", 15, 10, event("night_kill_sword", 3), ["kill1"]),
            M("survive", "活到 20000 刻", 20, 10, alive_until(20000)),
        ],
        success="kill1",
        stop_on_success=False,
        dense=dense(damage=0.5, death=20.0, approach=0.02),
        approach="hostile",
        stage=4,
        weight=0.4,
        skills=("craft_sword", "equip_sword", "attack"),
        achievements=[A("triple_unhurt", "三连杀且生命 ≥ 10", lambda t: "kill3" in t.done and t.state["life"] >= 10,
                        at_end=True)],
        extension=True,
    ),
    Task(
        "cook",
        "烤肉",
        "木镐挖 8 块圆石，在工作台旁合成熔炉并放下，再把生肉烤熟（MineRL 物品奖励 64）。",
        horizon=12000,
        milestones=[
            M("pickaxe", "合成木镐", 15, 6, got("wooden_pickaxe", 1)),
            M("cobble8", "收集 8 块圆石", 20, 8, got("cobblestone", 8), ["pickaxe"]),
            M("furnace", "合成熔炉", 15, 8, got("furnace", 1), ["cobble8"]),
            M("placed", "放下熔炉", 15, 6, event("furnace_placed"), ["furnace"]),
            M("meat", "获得生肉", 15, 6, got("raw_meat", 1)),
            M("cooked", "烤出熟肉", 20, 20, got("cooked", 1), ["placed", "meat"]),
        ],
        success="cooked",
        dense=dense(approach=0.02),
        approach="animal",
        stage=4,
        weight=0.4,
        skills=("equip_pickaxe", "craft_furnace", "place_furnace", "smelt_meat"),
        achievements=[A("chef", "烤出 2 块熟肉", got("cooked", 2))],
        extension=True,
    ),
)

TASK_NAMES = tuple(task.name for task in TASKS)  # 核心任务（默认课程）
ALL_TASKS = TASKS + EXTENSION_TASKS
ALL_TASK_NAMES = tuple(task.name for task in ALL_TASKS)

STAGES = (
    (0, "基础", "移动探索、砍树"),
    (1, "合成与庇护", "原木到工作台的合成链、挖坑封顶"),
    (2, "工具与食物", "工作台旁合成工具、打猎、进食"),
    (3, "生存", "日落前藏身、完整活过一天"),
    (4, "进阶", "木镐挖石头和煤、火把；扩展：夜战、烤肉"),
)
# 课程解锁：前沿阶段各任务（滑动平均）成功率均值 ≥ STAGE_UNLOCK_SUCCESS，或得分表分数均值 ≥ STAGE_UNLOCK_SCORE，
# 或者该阶段每个任务都已跑够 STAGE_UNLOCK_EPISODES 局，就解锁下一阶段。分数条件让“部分会了”的阶段（例如打猎偶尔成功、
# 但前几个里程碑都拿到了）也能往前走；局数上限保证不会被某个特别难的任务卡死。
STAGE_UNLOCK_SUCCESS = 0.5
STAGE_UNLOCK_SCORE = 50.0
STAGE_UNLOCK_EPISODES = 12  # 前沿阶段每个任务都跑满这么多局
TASK_INDEX = {name: index for index, name in enumerate(ALL_TASK_NAMES)}


def get_task(name):
    return ALL_TASKS[TASK_INDEX[name]]


# ---------------------------------------------------------------- 结算


class TaskTracker:
    """跟踪一局里的里程碑、累计收集量和事件，按得分表算奖励和分数。"""

    def __init__(self, task):
        self.task = task
        self.done = {}
        self.collected = defaultdict(float)
        self.events = defaultdict(int)
        self.sealed_ticks = 0
        self.shaft_depth = 0.0
        self.dig_anchor = None
        self.max_dist = 0.0
        self.state = None
        self.prev = None
        self.mem = None
        self.reward_parts = defaultdict(float)
        self.approach_best = None
        self.min_life = 20.0
        self.achieved = {}

    @property
    def score(self):
        return sum(m.points for m in self.task.milestones if m.key in self.done)

    @property
    def success(self):
        return self.task.success_key in self.done

    @property
    def complete(self):
        return all(m.key in self.done for m in self.task.milestones if not m.at_end)

    def start(self, state, mem):
        self.state = state
        self.prev = state
        self.mem = mem

    def _track(self, prev, state, macro):
        for item in ("log", "planks", "stick", "crafting_table", "wooden_pickaxe", "wooden_sword", "dirt",
                     "cobblestone", "coal", "torch", "meat", "raw_meat", "cooked", "furnace"):
            gain = state[item] - prev[item]
            if gain > 0:
                self.collected[item] += gain
        if macro == "place_table" and state["crafting_table"] < prev["crafting_table"]:
            self.events["table_placed"] += 1
        if macro == "place_torch" and state["torch"] < prev["torch"]:
            self.events["torch_placed"] += 1
            if sealed(self):  # self.state 已经是这一步的状态
                self.events["torch_sealed"] += 1
        if macro == "place_furnace" and state["furnace"] < prev["furnace"]:
            self.events["furnace_placed"] += 1
        # 夜里经验值上涨：只有击杀生物才会（熔炼也给经验，但夜战任务里不熔炼），记为一次击杀
        if self.mem is not None and self.mem.ticks >= NIGHT_TICK - 500 and state["xp"] > prev["xp"] \
                and state["alive"] > 0.5:
            self.events["night_kill"] += 1
            if state["hand"] == "wooden_sword" or prev["hand"] == "wooden_sword":
                self.events["night_kill_sword"] += 1
        self.min_life = min(self.min_life, state["life"])
        if state["hand"] == "wooden_sword":
            self.events["hold_sword"] += 1
        if state["hand"] in ("beef", "porkchop", "cooked_beef", "cooked_porkchop"):
            self.events["hold_meat"] += 1
        if macro == "eat" and (state["meat"] < prev["meat"] or state["food"] > prev["food"]):
            self.events["ate"] += 1
        # 只算自己向下挖的竖井：从第一次 dig_down 的位置算水平漂移，超过 1.5 格就不算。
        if macro == "dig_down" and self.dig_anchor is None:
            self.dig_anchor = (prev["x"], prev["y"], prev["z"])
        if self.dig_anchor is not None:
            drift = math.hypot(state["x"] - self.dig_anchor[0], state["z"] - self.dig_anchor[2])
            if drift <= 1.5:
                self.shaft_depth = max(self.shaft_depth, self.dig_anchor[1] - state["y"])
        self.max_dist = max(self.max_dist, state["dist"])

    def update(self, state, macro, ticks, privileged=None, final=False):
        """结算一步。返回 (奖励, 新拿到的里程碑 key 列表)。"""
        prev = self.state
        self.prev = prev
        self.state = state
        self._track(prev, state, macro)
        if sealed(self):
            self.sealed_ticks += ticks
        dense_table = self.task.dense
        units = ticks / 8.0
        parts = {}
        alive = state["alive"] > 0.5
        parts["alive"] = dense_table["alive"] * units if alive else 0.0
        parts["damage"] = -dense_table["damage"] * max(0.0, prev["life"] - state["life"])
        parts["death"] = -dense_table["death"] if not alive else 0.0
        parts["hunger"] = -dense_table["hunger"] * units if state["food"] < 6 else 0.0
        night = self.mem.ticks >= NIGHT_TICK and self.mem.ticks < MAX_EPISODE_STEPS - 1000
        exposed = state["sky"] > 0.5 and state["light"] < 12
        parts["night_exposed"] = (
            -dense_table["night_exposed"] * units if (night and exposed and alive) else 0.0
        )
        parts["night_sheltered"] = (
            dense_table["night_sheltered"] * units if (night and sealed(self) and alive) else 0.0
        )
        parts["approach"] = 0.0
        if privileged is not None and self.task.approach and dense_table["approach"] > 0:
            dist = privileged.get("nearest_" + self.task.approach)
            if dist is not None:
                # 只奖励刷新最近距离，来回走动拿不到分。
                if self.approach_best is None:
                    self.approach_best = dist
                elif dist < self.approach_best:
                    parts["approach"] = dense_table["approach"] * min(self.approach_best - dist, 2.0)
                    self.approach_best = dist
        newly = []
        for milestone in self.task.milestones:
            if milestone.key in self.done:
                continue
            if milestone.at_end and not final:
                continue
            if any(need not in self.done for need in milestone.requires):
                continue
            if milestone.check(self):
                self.done[milestone.key] = int(self.mem.ticks)
                newly.append(milestone.key)
        parts["milestone"] = sum(m.reward for m in self.task.milestones if m.key in newly)
        parts["achievement"] = 0.0
        for ach in self.task.achievements:
            if ach.key in self.achieved or (ach.at_end and not final):
                continue
            if ach.check(self):
                self.achieved[ach.key] = int(self.mem.ticks)
                parts["achievement"] += ach.reward
                newly.append("★" + ach.key)
        for key, value in parts.items():
            self.reward_parts[key] += value
        return float(sum(parts.values())), newly

    def summary(self):
        return {
            "task": self.task.name,
            "score": self.score,
            "success": self.success,
            "milestones": dict(self.done),
            "collected": {k: float(v) for k, v in self.collected.items()},
            "events": dict(self.events),
            "reward_parts": {k: round(v, 3) for k, v in self.reward_parts.items()},
            "shaft_depth": self.shaft_depth,
            "max_dist": self.max_dist,
            "achievements": dict(self.achieved),
        }


# ---------------------------------------------------------------- 得分表系列

DEATH_CAUSES = ("starvation", "fall", "night_monster", "cave", "drown", "other")
DEATH_CAUSE_TEXT = {
    "starvation": "饥饿",
    "fall": "摔落",
    "night_monster": "夜间怪物",
    "cave": "掉进洞穴",
    "drown": "溺水",
    "other": "其他",
}


def classify_death(history):
    """按最后几步的标准观测归类死因。history 是 snapshot 字典列表，最后一个是死亡帧。"""
    if not history:
        return "other"
    last = history[-1]
    window = history[-6:]
    before = window[0]
    if any(s["food"] <= 0.5 for s in window):
        return "starvation"
    if any(s["air"] < 30 for s in window):
        return "drown"
    drop_y = max(s["y"] for s in window) - last["y"]
    if drop_y >= 4 and before["life"] - last["life"] >= 3:
        return "fall"
    if last["dy"] <= -6 and last["sky"] < 0.5:
        return "cave"
    if last["ticks"] >= NIGHT_TICK - 500 or last["sun"] < 0.4:
        return "night_monster"
    return "other"


def item_reward_table():
    return [(entry["type"], entry["reward"]) for entry in REWARD_SCHEDULE]


def score_series():
    """全部得分表，供文档和 HTML 使用。"""
    return {
        "item_rewards": item_reward_table(),
        "stages": [list(st) for st in STAGES],
        "unlock": {"success": STAGE_UNLOCK_SUCCESS, "score": STAGE_UNLOCK_SCORE, "episodes": STAGE_UNLOCK_EPISODES},
        "tasks": [
            {
                "name": task.name,
                "title": task.title,
                "goal": task.goal,
                "horizon": task.horizon,
                "success": task.success_key,
                "weight": task.weight,
                "stage": task.stage,
                "extension": task.extension,
                "achievements": [{"key": a.key, "text": a.text, "reward": a.reward} for a in task.achievements],
                "skills": list(task.skills),
                "approach": task.approach,
                "dense": {k: v for k, v in task.dense.items() if v},
                "milestones": [
                    {
                        "key": m.key,
                        "text": m.text,
                        "points": m.points,
                        "reward": m.reward,
                        "requires": list(m.requires),
                        "at_end": m.at_end,
                    }
                    for m in task.milestones
                ],
            }
            for task in ALL_TASKS
        ],
    }


def score_tables_markdown():
    lines = ["# 任务系列得分表", ""]
    lines.append("由 `mc_rl/tasks.py` 生成：`python scripts/mcrl.py tables`。")
    lines.append("")
    lines.append("每个任务的里程碑分（points）合计 100，用来给任务打 0–100 分；")
    lines.append("训练奖励（reward）单独设定，只在训练里用。里程碑必须按前置条件依次达成。")
    lines.append("前 10 个是核心任务（默认课程）；标“扩展”的 3 个任务需要用 `--tasks` 显式加入，编号追加在后面，")
    lines.append("不影响已有检查点。每个任务还有若干**成就**（速通、无伤……），不计入 100 分，回放里显示徽章。")
    lines.append("")
    lines.append("## 课程总览")
    lines.append("")
    lines.append("课程按阶段由易到难解锁：只采样已解锁阶段的任务，当前前沿阶段权重加倍，早期阶段保留采样防止遗忘。")
    lines.append(f"前沿阶段各任务成功率均值 ≥ {STAGE_UNLOCK_SUCCESS}、或得分均值 ≥ {STAGE_UNLOCK_SCORE:g}、"
                 f"或该阶段每个任务都跑满 {STAGE_UNLOCK_EPISODES} 局，就解锁下一阶段。")
    lines.append("")
    lines.append("| 阶段 | 名称 | 内容 | 任务 |")
    lines.append("|---|---|---|---|")
    for stage, title, text in STAGES:
        names = ", ".join(f"`{t.name}`" + ("（扩展）" if t.extension else "") for t in ALL_TASKS if t.stage == stage)
        lines.append(f"| {stage} | {title} | {text} | {names} |")
    lines.append("")
    lines.append("| # | 阶段 | 任务 | 时长（刻） | 成功条件 | 采样权重 | 核心技能 |")
    lines.append("|---|---|---|---|---|---|---|")
    for index, task in sorted(enumerate(ALL_TASKS), key=lambda x: (x[1].stage, x[0])):
        success = next(m.text for m in task.milestones if m.key == task.success_key)
        lines.append(
            f"| {index} | {task.stage} | {task.title}{'（扩展）' if task.extension else ''} (`{task.name}`) | {task.horizon} | {success} | "
            f"{task.weight} | {', '.join(task.skills)} |"
        )
    for index, task in enumerate(ALL_TASKS):
        lines += ["", f"## {index}. {task.title}（`{task.name}`）", "", task.goal, ""]
        lines.append("| 里程碑 | 说明 | 得分 | 训练奖励 | 前置 | 结算时机 |")
        lines.append("|---|---|---|---|---|---|")
        for m in task.milestones:
            lines.append(
                f"| `{m.key}` | {m.text} | {m.points:g} | {m.reward:g} | "
                f"{', '.join(m.requires) or '—'} | {'局末' if m.at_end else '达成即得'} |"
            )
        if task.achievements:
            lines.append("")
            lines.append("成就（不计入 100 分）：" + "；".join(f"`{a.key}` {a.text}" for a in task.achievements))
        terms = {k: v for k, v in task.dense.items() if v}
        if terms:
            lines.append("")
            lines.append(
                "稠密塑形：" + "，".join(f"`{k}` {v:g}" for k, v in terms.items())
                + (f"；特权塑形目标：最近的 `{task.approach}`" if task.approach and terms.get("approach") else "")
            )
    lines += ["", "## MineRL 物品奖励（RewardForCollectingItemsOnce，每种一次）", ""]
    lines.append("| 物品 | 奖励 |")
    lines.append("|---|---|")
    for item, reward in item_reward_table():
        lines.append(f"| `{item}` | {reward} |")
    lines += ["", "## 死因分类", ""]
    lines.append("| 代码 | 含义 | 判定（最后 6 个决策步） |")
    lines.append("|---|---|---|")
    rules = {
        "starvation": "饥饿值降到 0",
        "drown": "氧气 < 30",
        "fall": "高度下降 ≥ 4 且掉血 ≥ 3",
        "cave": "低于出生点 6 格以上且看不见天空",
        "night_monster": "入夜（≥12500 刻）或太阳亮度 < 0.4",
        "other": "以上都不是",
    }
    for cause in DEATH_CAUSES:
        lines.append(f"| `{cause}` | {DEATH_CAUSE_TEXT[cause]} | {rules[cause]} |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- 可选扩展接口（不要求、不计分）


def make_my_task():
    """可选接口：返回一个自定义 Task；本作业不要求实现，也不计入评分。

    保留此函数只是为了演示如何在课程之外扩展任务。若自行实现并启用，
    建议让里程碑 points 合计 100、用 requires 串起关键步骤以防跳步刷分。
    """
    raise NotImplementedError("可选扩展接口，未实现；本作业不要求，也不计分")
