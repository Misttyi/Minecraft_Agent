"""离散宏动作。

一个宏动作展开成若干个底层 MineRL 动作，每个底层动作再被 SurviveDayWrapper
按 action_repeat（默认 8）重复。镜头增量按 repeat 均分，所以 turn_left 总共转 15 度。

掩码只看标准观测和智能体自己的记忆（例如自己把工作台放在了哪里），评测时同样可用。
"""

import numpy as np

from mc_rl.features import MEAT_ITEMS

# (名字, 说明)
MACRO_TABLE = (
    ("noop", "原地不动一拍"),
    ("forward", "向前走一拍"),
    ("forward_jump", "边走边跳，越过一格台阶"),
    ("back", "后退一拍"),
    ("turn_left", "左转 15°"),
    ("turn_right", "右转 15°"),
    ("turn_left_big", "左转 60°"),
    ("turn_right_big", "右转 60°"),
    ("look_up", "抬头 15°"),
    ("look_down", "低头 15°"),
    ("look_level", "视线回到水平"),
    ("attack", "按下再松开一次左键（打怪、打动物）"),
    ("chop", "连续挥击 6 拍（砍树、挖土）"),
    ("use", "右键一拍"),
    ("sneak", "蹲下一拍"),
    ("craft_planks", "原木 -> 木板"),
    ("craft_stick", "木板 -> 木棍"),
    ("craft_table", "木板 -> 工作台"),
    ("craft_torch", "煤炭 + 木棍 -> 火把"),
    ("place_table", "低头放下工作台"),
    ("craft_pickaxe", "工作台旁合成木镐"),
    ("craft_sword", "工作台旁合成木剑"),
    ("equip_sword", "手持木剑"),
    ("equip_pickaxe", "手持木镐"),
    ("equip_meat", "手持肉（熟肉优先）"),
    ("eat", "按住右键 5 拍进食"),
    ("dig_down", "看脚下并连续挖 6 拍"),
    ("seal_above", "抬头放泥土封顶"),
    ("place_torch", "低头放火把"),
    # ---- 以下是扩展动作（追加在末尾，旧检查点不受影响）
    ("craft_furnace", "工作台旁用 8 块圆石合成熔炉"),
    ("place_furnace", "低头放下熔炉"),
    ("smelt_meat", "熔炉旁把生肉烤熟"),
    ("sprint_jump", "疾跑跳跃（快速消耗饥饿值、赶路）"),
)

MACROS = tuple(name for name, _ in MACRO_TABLE)
MACRO_INDEX = {name: index for index, name in enumerate(MACROS)}
MACRO_DESCRIPTIONS = dict(MACRO_TABLE)
NUM_MACROS = len(MACROS)


def _cam(pitch, yaw, repeat):
    repeat = max(int(repeat), 1)
    return np.array([pitch / repeat, yaw / repeat], dtype=np.float32)


def _aim_pitch(current, target, limit=45.0):
    return float(np.clip(target - current, -limit, limit))


def _face_table(state, repeat, key="table_offset"):
    """nearbyCraft / nearbySmelt 要求工作台 / 熔炉在视野里：先把镜头转向记下的位置。"""
    import math

    offset = state.get(key)
    if not offset:
        return {}
    dx, dy, dz = offset
    yaw = math.degrees(math.atan2(-dx, dz))
    pitch = math.degrees(math.atan2(1.62 - dy, max(math.hypot(dx, dz), 1e-3)))
    dyaw = (yaw - state["yaw"] + 180.0) % 360.0 - 180.0
    return {"camera": _cam(_aim_pitch(state["pitch"], pitch, 120.0), dyaw, repeat)}


def expand(name, state, repeat):
    """宏动作 -> 底层动作覆盖列表。每个元素会合进 no_op 再发给环境。"""
    pitch = state["pitch"]
    if name == "noop":
        return [{}]
    if name == "forward":
        return [{"forward": 1}]
    if name == "forward_jump":
        return [{"forward": 1, "jump": 1}]
    if name == "sprint_jump":
        return [{"forward": 1, "jump": 1, "sprint": 1}] * 2
    if name == "back":
        return [{"back": 1}]
    if name == "turn_left":
        return [{"camera": _cam(0.0, -15.0, repeat)}]
    if name == "turn_right":
        return [{"camera": _cam(0.0, 15.0, repeat)}]
    if name == "turn_left_big":
        return [{"camera": _cam(0.0, -60.0, repeat)}]
    if name == "turn_right_big":
        return [{"camera": _cam(0.0, 60.0, repeat)}]
    if name == "look_up":
        return [{"camera": _cam(_aim_pitch(pitch, pitch - 15.0), 0.0, repeat)}]
    if name == "look_down":
        return [{"camera": _cam(_aim_pitch(pitch, pitch + 15.0), 0.0, repeat)}]
    if name == "look_level":
        return [{"camera": _cam(_aim_pitch(pitch, 0.0, 60.0), 0.0, repeat)}]
    if name == "attack":
        # 打生物要“按下-松开”：一直按住只算一次点击（方块才会持续挖）。按 8 刻、松 8 刻 = 一次满蓄力攻击。
        return [{"attack": 1}, {}]
    if name == "chop":
        return [{"attack": 1}] * 6
    if name == "use":
        return [{"use": 1}]
    if name == "sneak":
        return [{"sneak": 1}]
    if name == "craft_planks":
        return [{"craft": "planks"}]
    if name == "craft_stick":
        return [{"craft": "stick"}]
    if name == "craft_table":
        return [{"craft": "crafting_table"}]
    if name == "craft_torch":
        return [{"craft": "torch"}]
    if name == "place_table":
        return [
            {"camera": _cam(_aim_pitch(pitch, 50.0, 90.0), 0.0, repeat)},
            {"place": "crafting_table"},
        ]
    if name in ("craft_pickaxe", "craft_sword"):
        item = "wooden_pickaxe" if name == "craft_pickaxe" else "wooden_sword"
        return [_face_table(state, repeat), {"nearbyCraft": item}]
    if name == "equip_sword":
        return [{"equip": "wooden_sword"}, {}]
    if name == "equip_pickaxe":
        return [{"equip": "wooden_pickaxe"}, {}]
    if name == "equip_meat":
        item = next((m for m in MEAT_ITEMS if state.get(m, 0) > 0), "beef")
        return [{"equip": item}, {}]
    if name == "eat":
        return [{"use": 1, "sneak": 1}] * 5
    if name == "dig_down":
        # 90 度时射线常打不中脚下，85 度稳定命中。
        first = {"attack": 1, "camera": _cam(_aim_pitch(pitch, 85.0, 120.0), 0.0, repeat)}
        return [first] + [{"attack": 1}] * 5
    if name == "seal_above":
        # 抬头 70 度对着竖井井壁放泥土；每次再转 90 度，换一面井壁，失败几次也总能封上。
        return [
            {"sneak": 1, "camera": _cam(_aim_pitch(pitch, -70.0, 180.0), 90.0, repeat)},
            {"sneak": 1, "place": "dirt" if state.get("dirt", 0) >= 1 else "cobblestone"},
        ]
    if name == "craft_furnace":
        return [_face_table(state, repeat), {"nearbyCraft": "furnace"}]
    if name == "place_furnace":
        return [
            {"camera": _cam(_aim_pitch(pitch, 50.0, 90.0), 0.0, repeat)},
            {"place": "furnace"},
        ]
    if name == "smelt_meat":
        item = "cooked_beef" if state.get("beef", 0) >= 1 else "cooked_porkchop"
        return [_face_table(state, repeat, "furnace_offset"), {"nearbySmelt": item}]
    if name == "place_torch":
        return [
            {"camera": _cam(_aim_pitch(pitch, 60.0, 120.0), 0.0, repeat)},
            {"place": "torch"},
        ]
    raise KeyError(name)


def action_mask(state):
    """bool[NUM_MACROS]。只屏蔽一定无效的合成、装备、放置和进食。移动与挥击永远可选。"""
    mask = np.ones(NUM_MACROS, dtype=bool)
    hand = state.get("hand", "air")

    def off(name, condition):
        if not condition:
            mask[MACRO_INDEX[name]] = False

    off("craft_planks", state["log"] >= 1)
    off("craft_stick", state["planks"] >= 2)
    off("craft_table", state["planks"] >= 4)
    off("craft_torch", state["coal"] >= 1 and state["stick"] >= 1)
    off("place_table", state["crafting_table"] >= 1)
    near = state.get("table_near", 0.0) > 0.5
    off("craft_pickaxe", near and state["planks"] >= 3 and state["stick"] >= 2)
    off("craft_sword", near and state["planks"] >= 2 and state["stick"] >= 1)
    off("equip_sword", state["wooden_sword"] >= 1 and hand != "wooden_sword")
    off("equip_pickaxe", state["wooden_pickaxe"] >= 1 and hand != "wooden_pickaxe")
    off("equip_meat", state["meat"] >= 1 and hand not in MEAT_ITEMS)
    off("eat", hand in MEAT_ITEMS and state[hand] >= 1 and state["food"] < 20)
    off("seal_above", state["dirt"] >= 1 or state["cobblestone"] >= 1)
    off("place_torch", state["torch"] >= 1)
    off("craft_furnace", near and state["cobblestone"] >= 8)
    off("place_furnace", state["furnace"] >= 1)
    fuel = state["planks"] + state["coal"] + state["log"]
    off("smelt_meat", state.get("furnace_near", 0.0) > 0.5 and state["beef"] + state["porkchop"] >= 1 and fuel >= 1)
    return mask
