"""训练专用的特权观测。

在 SurviveDayEnvSpec 的子类上多挂三个 Malmo 观测：
- ObservationFromGrid：身边 21×9×21 的方块类型，编码成小整数；
- ObservationFromNearbyEntities：附近的动物、怪物和掉落物，转成相对坐标表；
- ObservationFromRay：准星指着的方块或实体。

只有专家（模仿学习的老师）和训练奖励会读这些量。学生策略的输入只来自标准观测，
评测仍然调用未改动的 survive_day.make_env。不修改 survive_specs.py 和 wrappers.py。
"""

import math

import numpy as np

GRID_MIN = (-10, -3, -10)
GRID_MAX = (10, 5, 10)
GRID_SHAPE = tuple(hi - lo + 1 for lo, hi in zip(GRID_MIN, GRID_MAX))  # x, y, z
GRID_CELLS = GRID_SHAPE[0] * GRID_SHAPE[1] * GRID_SHAPE[2]

BLOCK_CODES = {
    "air": 1,
    "log": 2,
    "log2": 2,
    "leaves": 3,
    "leaves2": 3,
    "dirt": 4,
    "grass": 4,
    "stone": 5,
    "coal_ore": 6,
    "water": 7,
    "flowing_water": 7,
    "lava": 8,
    "flowing_lava": 8,
    "sand": 9,
    "gravel": 9,
    "tallgrass": 1,
    "double_plant": 1,
    "red_flower": 1,
    "yellow_flower": 1,
}
CODE_NAMES = {0: "other", 1: "air", 2: "log", 3: "leaves", 4: "dirt", 5: "stone",
              6: "coal_ore", 7: "water", 8: "lava", 9: "sand"}

ENTITY_CODES = {
    "Cow": 1, "Pig": 2, "Sheep": 3, "Chicken": 4,
    "Zombie": 5, "Skeleton": 6, "Creeper": 7, "Spider": 8,
    "beef": 9, "porkchop": 10,
}
ENTITY_NAMES = {v: k for k, v in ENTITY_CODES.items()}
ENTITY_NAMES.update({11: "item", 12: "mob"})
ANIMALS = ("Cow", "Pig")
MEAT_DROPS = ("beef", "porkchop")
HOSTILES = ("Zombie", "Skeleton", "Creeper", "Spider", "mob")
MAX_ENTITIES = 24
ENTITY_RANGE = (32, 8, 32)

RAY_HIT = {"none": 0, "block": 1, "entity": 2}


def grid_index(dx, dy, dz):
    """Malmo 的网格顺序：x 最快，其次 z，最后 y。"""
    x = dx - GRID_MIN[0]
    y = dy - GRID_MIN[1]
    z = dz - GRID_MIN[2]
    return (y * GRID_SHAPE[2] + z) * GRID_SHAPE[0] + x


def encode_block(name):
    return BLOCK_CODES.get(str(name), 0)


def summarize_privileged(raw):
    """raw = {"logs": [(dx, dy, dz)], "entities": [{name, dx, dy, dz}], "ray": dict|None, ...}"""
    out = dict(raw)

    def nearest(points):
        best = None
        for point in points:
            dist = math.sqrt(point[0] ** 2 + point[2] ** 2 + 0.25 * point[1] ** 2)
            if best is None or dist < best:
                best = dist
        return best

    entities = raw.get("entities") or []
    animals = [(e["dx"], e["dy"], e["dz"]) for e in entities if e["name"] in ANIMALS]
    drops = [(e["dx"], e["dy"], e["dz"]) for e in entities if e["name"] in MEAT_DROPS]
    hostiles = [(e["dx"], e["dy"], e["dz"]) for e in entities if e["name"] in HOSTILES]
    out["nearest_log"] = nearest(raw.get("logs") or [])
    out["nearest_animal"] = nearest(animals + drops)
    out["nearest_drop"] = nearest(drops)
    out["nearest_hostile"] = nearest(hostiles)
    return out


def extract(observation):
    """真实环境观测里的编码数组 -> 和 mock_env 相同格式的特权字典。没有特权观测时返回 None。"""
    if "privileged" in observation:
        return observation["privileged"]
    grid = observation.get("priv_grid")
    if grid is None:
        return None
    grid = np.asarray(grid).reshape(-1)
    loc = observation.get("location_stats") or {}
    # 网格偏移是相对脚下方块的；换算到相对玩家精确位置要加上方块原点和玩家位置的差。
    offset = []
    for key in ("xpos", "ypos", "zpos"):
        value = float(np.asarray(loc.get(key, 0.0)).reshape(-1)[0])
        offset.append(math.floor(value) - value)
    logs = []
    if grid.size == GRID_CELLS:
        for flat in np.nonzero(grid == 2)[0]:
            flat = int(flat)
            x = flat % GRID_SHAPE[0]
            z = (flat // GRID_SHAPE[0]) % GRID_SHAPE[2]
            y = flat // (GRID_SHAPE[0] * GRID_SHAPE[2])
            # 网格坐标是相对脚下方块的；方块中心再加 0.5。
            logs.append((x + GRID_MIN[0] + 0.5 + offset[0], y + GRID_MIN[1] + 0.5 + offset[1],
                         z + GRID_MIN[2] + 0.5 + offset[2]))
    entities = []
    table = np.asarray(observation.get("priv_entities", np.zeros((0, 5)))).reshape(-1, 5)
    for code, dx, dy, dz, valid in table:
        if valid > 0.5:
            entities.append({"name": ENTITY_NAMES.get(int(code), "mob"), "dx": float(dx),
                             "dy": float(dy), "dz": float(dz)})
    ray = None
    ray_arr = np.asarray(observation.get("priv_ray", np.zeros(3))).reshape(-1)
    if ray_arr.size >= 3 and int(ray_arr[0]) != 0:
        hit = "block" if int(ray_arr[0]) == 1 else "entity"
        name = CODE_NAMES.get(int(ray_arr[1]), "other") if hit == "block" else ENTITY_NAMES.get(int(ray_arr[1]), "mob")
        ray = {"hit": hit, "type": name, "distance": float(ray_arr[2])}
    return summarize_privileged({"logs": logs, "entities": entities, "ray": ray, "grid": grid,
                                 "grid_offset": tuple(offset)})


# ---------------------------------------------------------------- MineRL handler（需要 minerl）


def _build_handlers():
    from minerl.herobraine.hero import spaces
    from minerl.herobraine.hero.handlers.translation import TranslationHandler

    class _CodedObservation(TranslationHandler):
        name = ""

        def to_string(self):
            return self.name

        def from_universal(self, x):
            return self.space.no_op()

    class GridCodes(_CodedObservation):
        name = "priv_grid"

        def __init__(self):
            super().__init__(spaces.Box(low=0, high=16, shape=(GRID_CELLS,), dtype=np.int8))

        def xml_template(self):
            lo, hi = GRID_MIN, GRID_MAX
            return (
                "<ObservationFromGrid><Grid name=\"priv_grid\">"
                f"<min x=\"{lo[0]}\" y=\"{lo[1]}\" z=\"{lo[2]}\"/>"
                f"<max x=\"{hi[0]}\" y=\"{hi[1]}\" z=\"{hi[2]}\"/>"
                "</Grid></ObservationFromGrid>"
            )

        def from_hero(self, info):
            cells = info.get("priv_grid")
            out = np.zeros(GRID_CELLS, dtype=np.int8)
            if cells:
                out[: len(cells)] = [encode_block(c) for c in cells[:GRID_CELLS]]
            return out

    class EntityTable(_CodedObservation):
        name = "priv_entities"

        def __init__(self):
            super().__init__(spaces.Box(low=-100, high=100, shape=(MAX_ENTITIES, 5), dtype=np.float32))

        def xml_template(self):
            rx, ry, rz = ENTITY_RANGE
            return (
                "<ObservationFromNearbyEntities>"
                f"<Range name=\"priv_entities\" xrange=\"{rx}\" yrange=\"{ry}\" zrange=\"{rz}\"/>"
                "</ObservationFromNearbyEntities>"
            )

        def from_hero(self, info):
            out = np.zeros((MAX_ENTITIES, 5), dtype=np.float32)
            # MineRL 0.4.4 的 FullStats 用小写 xpos/ypos/zpos；原版 Malmo 是 XPos。
            px = info.get("XPos", info.get("xpos"))
            py = info.get("YPos", info.get("ypos"))
            pz = info.get("ZPos", info.get("zpos"))
            rows = []
            for entity in info.get("priv_entities") or []:
                name = str(entity.get("name", ""))
                if px is None or name in ("",) or name.startswith("MineRLAgent"):
                    continue
                dx = float(entity.get("x", 0)) - px
                dy = float(entity.get("y", 0)) - py
                dz = float(entity.get("z", 0)) - pz
                if abs(dx) < 0.3 and abs(dz) < 0.3 and abs(dy) < 0.3:
                    continue  # 自己
                code = ENTITY_CODES.get(name)
                if code is None:
                    code = 11 if "quantity" in entity else 12
                rows.append((dx * dx + dz * dz, code, dx, dy, dz))
            rows.sort()
            for index, (_d, code, dx, dy, dz) in enumerate(rows[:MAX_ENTITIES]):
                out[index] = (code, dx, dy, dz, 1.0)
            return out

    class RayObservation(_CodedObservation):
        name = "priv_ray"

        def __init__(self):
            super().__init__(spaces.Box(low=0, high=100, shape=(3,), dtype=np.float32))

        def xml_template(self):
            return "<ObservationFromRay/>"

        def from_hero(self, info):
            sight = info.get("LineOfSight") or {}
            if not sight or not sight.get("inRange", True):
                return np.zeros(3, dtype=np.float32)
            hit = RAY_HIT.get(str(sight.get("hitType", "none")), 0)
            kind = str(sight.get("type", ""))
            code = encode_block(kind) if hit == 1 else ENTITY_CODES.get(kind, 12)
            return np.array([hit, code, float(sight.get("distance", 0.0))], dtype=np.float32)

    return GridCodes, EntityTable, RayObservation


def make_privileged_env(world_seed, action_repeat=8, strip_reward=False):
    """训练环境：标准 SurviveDay 规格 + 三个特权观测。包装器原样复用。"""
    from survive_day.constants import MAX_EPISODE_STEPS
    from survive_day.survive_specs import SurviveDayEnvSpec
    from survive_day.wrappers import SurviveDayWrapper

    grid_cls, entity_cls, ray_cls = _build_handlers()

    class PrivilegedSurviveDaySpec(SurviveDayEnvSpec):
        def create_observables(self):
            return super().create_observables() + [grid_cls(), entity_cls(), ray_cls()]

    spec = PrivilegedSurviveDaySpec(world_seed=world_seed)
    return SurviveDayWrapper(
        spec.make(),
        action_repeat=action_repeat,
        strip_reward=strip_reward,
        max_episode_steps=MAX_EPISODE_STEPS,
    )


def set_world_seed(env, seed):
    """不重启 Minecraft 换种子。MineRL 每次 reset 都调用 task.reset()，按规格重建全部 handler，
    SeededWorldGenerator 读的是 spec.world_seed，所以改这个属性即可，下次 reset 生效。"""
    base = env
    for _ in range(10):
        spec = getattr(base, "task", None)
        if spec is not None and hasattr(spec, "world_seed"):
            spec.world_seed = str(seed)
            return True
        base = getattr(base, "env", None)
        if base is None:
            break
    return False
