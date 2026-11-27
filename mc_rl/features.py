"""把 MineRL 观测收成标量状态和策略输入。

只读标准观测（survive_day 规格挂的 handler）。特权观测由 mc_rl.privileged 单独解析，
不进入学生策略的输入。
"""

import math

import numpy as np

from survive_day.constants import INVENTORY_ITEMS, MAX_EPISODE_STEPS

HAND_ITEMS = (
    "air",
    "wooden_pickaxe",
    "wooden_sword",
    "torch",
    "beef",
    "porkchop",
    "cooked_beef",
    "cooked_porkchop",
    "other",
)

MEAT_ITEMS = ("cooked_beef", "cooked_porkchop", "beef", "porkchop")


def _scalar(value, default=0.0):
    if value is None:
        return float(default)
    array = np.asarray(value)
    if array.size == 0:
        return float(default)
    try:
        return float(array.reshape(-1)[0])
    except (TypeError, ValueError):
        return float(default)


def sky_flag(value):
    """can_see_sky 可能是长度 2 的 one-hot，也可能是布尔标量。"""
    array = np.asarray(value).reshape(-1)
    if array.size >= 2:
        return float(array[1] > array[0])
    if array.size == 1:
        return float(bool(array[0]))
    return 0.0


def hand_type(observation):
    hand = ((observation.get("equipped_items") or {}).get("mainhand") or {}).get(
        "type", "air"
    )
    if not isinstance(hand, str):
        array = np.asarray(hand).reshape(-1)
        hand = str(array[0]) if array.size else "air"
    return hand if hand in HAND_ITEMS else ("air" if hand in ("none", "") else "other")


class EpisodeMemory:
    """一局里由智能体自己记下的量：出生点、步数、工作台位置、封顶次数、伤害历史。

    都来自过去的标准观测和自己发出的动作，评测时同样可以维护。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.ticks = 0
        self.decisions = 0
        self.spawn = None
        self.table_pos = None
        self.furnace_pos = None
        self.sealed = 0
        self.last_life = 20.0
        self.damage_recent = 0.0
        self.last_pos = None
        self.stuck = 0
        self.dig_anchor = None
        self.meals = 0
        self.last_macro = "noop"

    def observe(self, state):
        pos = (state["x"], state["y"], state["z"])
        if self.spawn is None:
            self.spawn = pos
        drop = max(0.0, self.last_life - state["life"])
        self.damage_recent = 0.8 * self.damage_recent + drop
        self.last_life = state["life"]
        if self.last_pos is not None:
            moved = math.hypot(pos[0] - self.last_pos[0], pos[2] - self.last_pos[2])
            self.stuck = self.stuck + 1 if moved < 0.15 else 0
        self.last_pos = pos

    @staticmethod
    def _front_block(state):
        """place_* 宏动作先把视线压到 50 度再放：方块落在身前约 1.4 格、和脚同高。返回方块中心。"""
        yaw = math.radians(state["yaw"])
        return (state["x"] - math.sin(yaw) * 1.4, state["y"] + 0.5, state["z"] + math.cos(yaw) * 1.4)

    def place_furnace(self, state):
        self.furnace_pos = self._front_block(state)

    @staticmethod
    def _offset(pos, state):
        if pos is None:
            return None
        return (pos[0] - state["x"], pos[1] - state["y"], pos[2] - state["z"])

    @staticmethod
    def _near(offset, radius=4.3):
        if offset is None:
            return False
        dx, dy, dz = offset
        dy -= 1.62
        return math.sqrt(dx * dx + dy * dy + dz * dz) <= radius

    def place_table(self, state):
        """place_table 先把视线压到 50 度再放：方块落在身前约 1.4 格、和脚同高。记下方块中心。"""
        yaw = math.radians(state["yaw"])
        self.table_pos = (state["x"] - math.sin(yaw) * 1.4, state["y"] + 0.5, state["z"] + math.cos(yaw) * 1.4)

    def table_offset(self, state):
        if self.table_pos is None:
            return None
        return (self.table_pos[0] - state["x"], self.table_pos[1] - state["y"], self.table_pos[2] - state["z"])

    def table_near(self, state, radius=4.3):
        """Malmo 的 nearbyCraft 要求工作台离头部 5 格以内并且在视野里；视野由宏动作先对准。"""
        offset = self.table_offset(state)
        if offset is None:
            return False
        dx, dy, dz = offset
        dy -= 1.62
        return math.sqrt(dx * dx + dy * dy + dz * dz) <= radius


def snapshot(observation, memory=None):
    """标准观测 -> 平铺字典。memory 提供出生点和时钟。"""
    life = observation.get("life_stats") or {}
    loc = observation.get("location_stats") or {}
    inventory = observation.get("inventory") or {}
    state = {
        "alive": float(bool(_scalar(life.get("is_alive"), 1.0))),
        "life": _scalar(life.get("life"), 20.0),
        "food": _scalar(life.get("food"), 20.0),
        "saturation": _scalar(life.get("saturation"), 5.0),
        "air": _scalar(life.get("air"), 300.0),
        "xp": _scalar(life.get("xp"), 0.0),
        "sun": _scalar(loc.get("sun_brightness"), 1.0),
        "sky_light": _scalar(loc.get("sky_light_level"), 1.0),
        "light": _scalar(loc.get("light_level"), 15.0),
        "sky": sky_flag(loc.get("can_see_sky", 1)),
        "x": _scalar(loc.get("xpos"), 0.0),
        "y": _scalar(loc.get("ypos"), 64.0),
        "z": _scalar(loc.get("zpos"), 0.0),
        "pitch": _scalar(loc.get("pitch"), 0.0),
        "yaw": _scalar(loc.get("yaw"), 0.0),
        "rain": float(bool(_scalar(loc.get("is_raining"), 0.0))),
        "hand": hand_type(observation),
    }
    for item in INVENTORY_ITEMS:
        state[item] = _scalar(inventory.get(item), 0.0)
    state["meat"] = sum(state[item] for item in MEAT_ITEMS)
    state["raw_meat"] = state["beef"] + state["porkchop"]
    state["cooked"] = state["cooked_beef"] + state["cooked_porkchop"]
    if memory is not None and memory.spawn is not None:
        state["dy"] = state["y"] - memory.spawn[1]
        state["dist"] = math.hypot(state["x"] - memory.spawn[0], state["z"] - memory.spawn[2])
        state["ticks"] = float(memory.ticks)
        state["sealed"] = float(memory.sealed)
        state["table_near"] = float(memory.table_near(state))
        offset = memory.table_offset(state)
        state["table_offset"] = offset
        state["damage_recent"] = memory.damage_recent
        state["stuck"] = float(memory.stuck)
        furnace = memory._offset(memory.furnace_pos, state)
        state["furnace_offset"] = furnace
        state["furnace_near"] = float(memory._near(furnace))
        # 出生点相对朝向的方位角（0 = 正前方），给“回家”用
        dx, dz = memory.spawn[0] - state["x"], memory.spawn[2] - state["z"]
        home_yaw = math.degrees(math.atan2(-dx, dz))
        state["home_rel"] = (home_yaw - state["yaw"] + 180.0) % 360.0 - 180.0
    else:
        state.update(dy=0.0, dist=0.0, ticks=0.0, sealed=0.0, table_near=0.0,
                     damage_recent=0.0, stuck=0.0, table_offset=None, furnace_offset=None,
                     furnace_near=0.0, home_rel=0.0)
    return state


VECTOR_NAMES = (
    ("life", 20.0),
    ("food", 20.0),
    ("saturation", 20.0),
    ("air", 300.0),
    ("sun", 1.0),
    ("sky_light", 1.0),
    ("light", 15.0),
    ("sky", 1.0),
    ("pitch", 90.0),
    ("dy", 8.0),
    ("dist", 32.0),
    ("ticks", float(MAX_EPISODE_STEPS)),
    ("sealed", 4.0),
    ("table_near", 1.0),
    ("damage_recent", 10.0),
    ("stuck", 8.0),
)


def _task_counts():
    from mc_rl.tasks import EXTENSION_TASKS, TASKS

    return len(TASKS), len(EXTENSION_TASKS)


# 第一版向量（核心 10 个任务）的长度。之后加的特征和扩展任务 one-hot 都接在它后面，
# 旧检查点只读前 CORE 维（models.ActorCritic 会自动截取），所以向后兼容。
EXTRA_NAMES = ("home_sin", "home_cos", "home_dist", "xp", "night", "furnace_near")


def core_vector_size():
    return len(VECTOR_NAMES) + 2 + len(INVENTORY_ITEMS) + len(HAND_ITEMS) + _task_counts()[0]


def vector_size(num_tasks=None):
    """完整向量长度（num_tasks 参数只为兼容旧调用，忽略）。"""
    return core_vector_size() + len(EXTRA_NAMES) + _task_counts()[1]


def task_of_vectors(vec):
    """从一批向量里读出任务编号（TASK_INDEX），兼容旧的核心长度向量和完整长度向量。
    核心任务 one-hot 在 [core-n_core, core)，扩展任务 one-hot 在完整向量的最后 n_ext 维。"""
    vec = np.asarray(vec)
    n_core, n_ext = _task_counts()
    core = core_vector_size()
    task = np.argmax(vec[:, core - n_core:core], axis=1)
    if vec.shape[1] >= core + len(EXTRA_NAMES) + n_ext and n_ext:
        ext = vec[:, core + len(EXTRA_NAMES):core + len(EXTRA_NAMES) + n_ext]
        is_ext = ext.max(axis=1) > 0.5
        task = np.where(is_ext, n_core + np.argmax(ext, axis=1), task)
    return task


def vectorize(state, task_index, num_tasks=None):
    """标量观测 -> 约 [-1, 1] 的向量。背包计数用 log1p，避免 64 块泥土压过一根木棍。

    布局：[核心标量][朝向][背包][主手][核心任务 one-hot] | [扩展特征][扩展任务 one-hot]
    """
    n_core, n_ext = _task_counts()
    values = [np.clip(state[name] / scale, -4.0, 4.0) for name, scale in VECTOR_NAMES]
    yaw = math.radians(state["yaw"])
    values += [math.sin(yaw), math.cos(yaw)]
    values += [math.log1p(max(state[item], 0.0)) / 3.0 for item in INVENTORY_ITEMS]
    values += [1.0 if state["hand"] == item else 0.0 for item in HAND_ITEMS]
    core = [0.0] * n_core
    if task_index < n_core:
        core[task_index] = 1.0
    values += core
    rel = math.radians(state.get("home_rel", 0.0))
    ticks = state.get("ticks", 0.0)
    values += [
        math.sin(rel), math.cos(rel),
        min(state.get("dist", 0.0), 64.0) / 32.0,
        math.log1p(max(state.get("xp", 0.0), 0.0)) / 4.0,
        1.0 if 12500 <= ticks <= 23000 else 0.0,
        float(state.get("furnace_near", 0.0)),
    ]
    ext = [0.0] * n_ext
    if task_index >= n_core:
        ext[task_index - n_core] = 1.0
    values += ext
    return np.asarray(values, dtype=np.float32)


def pov_array(observation, size=64):
    pov = observation.get("pov")
    if pov is None:
        return np.zeros((size, size, 3), dtype=np.uint8)
    pov = np.asarray(pov, dtype=np.uint8)
    if pov.shape[0] != size or pov.shape[1] != size:
        rows = np.linspace(0, pov.shape[0] - 1, size).astype(np.int64)
        cols = np.linspace(0, pov.shape[1] - 1, size).astype(np.int64)
        pov = pov[rows][:, cols]
    # MineRL 的 pov 可能是负步长视图（上下翻转），torch.as_tensor 不接受。
    return np.ascontiguousarray(pov)
