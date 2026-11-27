"""模仿学习的老师：看特权观测（方块网格、实体、准星）的脚本专家。

输出宏动作名字。学生策略只拿标准观测去模仿它，所以评测时不需要特权信息。
没有特权观测时退化成盲走 + 时钟循环。
"""

import math
from collections import defaultdict

import numpy as np

from mc_rl.features import MEAT_ITEMS

EYE = 1.62
REACH = 4.0
NIGHT_START = 13000


def wrap(angle):
    return (angle + 180.0) % 360.0 - 180.0


def target_angles(dx, dy, dz):
    """MC 约定：yaw 0 朝 +z，yaw 90 朝 -x；pitch 正值向下看。"""
    yaw = math.degrees(math.atan2(-dx, dz))
    horiz = math.hypot(dx, dz)
    pitch = math.degrees(math.atan2(EYE - dy, max(horiz, 1e-3)))
    return yaw, pitch, horiz


class Expert:
    def __init__(self, seed=0, eat_below=15, shelter_tick=10800, hunt_until=9000, safe_shelter_tick=2400):
        self.rng = np.random.default_rng(seed)
        self.safe_shelter_tick = safe_shelter_tick
        self.eat_below = eat_below
        self.shelter_tick = shelter_tick
        self.hunt_until = hunt_until

    def reset(self, task):
        self.task = task
        self.wander_turns = 0
        self.wander_steps = 0
        self.seal_tries = 0
        self.target = None
        self.best_dist = 0.0
        self.no_progress = 0
        self.chops = 0
        self.blacklist = set()
        self.attempts = defaultdict(int)
        self.last_gain = 0.0
        self.turn_dir = 1
        self.hazard_turns = 0
        self.hazard_cooldown = 0
        self.place_tries = 0
        self.mem = None
        self.shaft_anchor = None
        self.committed = False
        self.ground_search = 0

    # ------------------------------------------------------------ 基础动作
    def _turn_towards(self, state, yaw_target):
        diff = wrap(yaw_target - state["yaw"])
        if abs(diff) >= 40:
            return "turn_right_big" if diff > 0 else "turn_left_big"
        if abs(diff) >= 9:
            return "turn_right" if diff > 0 else "turn_left"
        return None

    def _pitch_towards(self, state, pitch_target, tolerance=10.0):
        diff = pitch_target - state["pitch"]
        if diff > tolerance:
            return "look_down"
        if diff < -tolerance:
            return "look_up"
        return None

    def _go(self, state, priv=None):
        if state["stuck"] >= 2:
            ray = (priv or {}).get("ray")
            # 被树叶、泥土、沙子挡住：直接挖开（徒手挖这些很快），石头挖不动就绕。
            if ray and ray["hit"] == "block" and ray["type"] in ("leaves", "dirt", "sand") and ray["distance"] <= 2.0:
                return "chop"
            if state["stuck"] >= 3 and state["stuck"] % 4 == 3:
                return "look_up" if state["pitch"] > -10 else "look_level"
            # 转身不会移动，stuck 会一直涨；所以每 4 拍转一次，其余时间跳着往前顶。
            if state["stuck"] % 4 == 0:
                if state["stuck"] >= 16:
                    self.turn_dir = -self.turn_dir
                return "turn_right_big" if self.turn_dir > 0 else "turn_left_big"
            return "forward_jump"
        return "forward"

    def _hazard_ahead(self, state, priv):
        """正前方是水、岩浆，或者紧挨着一个至少 4 格深的坑。连续转了一圈都有危险就不再管（避免原地打转）。"""
        from mc_rl.privileged import GRID_MIN, GRID_SHAPE, grid_index

        grid = (priv or {}).get("grid")
        if grid is None or len(grid) != GRID_SHAPE[0] * GRID_SHAPE[1] * GRID_SHAPE[2]:
            return False
        if self.hazard_turns >= 6:
            self.hazard_cooldown = 12
            self.hazard_turns = 0
        if self.hazard_cooldown > 0:
            self.hazard_cooldown -= 1
            return False
        yaw = math.radians(state["yaw"])
        hazard = False
        for dist in (1.2, 2.2):
            gx = int(math.floor(-math.sin(yaw) * dist + state["x"])) - int(math.floor(state["x"]))
            gz = int(math.floor(math.cos(yaw) * dist + state["z"])) - int(math.floor(state["z"]))
            column = [int(grid[grid_index(gx, gy, gz)]) for gy in (-3, -2, -1, 0)]
            if column[2] in (7, 8) or column[3] in (7, 8):
                hazard = True
            if dist < 2 and all(code == 1 for code in column):
                hazard = True
        self.hazard_turns = self.hazard_turns + 1 if hazard else 0
        return hazard

    def go_home(self, state, mem):
        """在水里或掉到出生点以下太深：朝出生点方向边跳边走（跳可以游泳、爬坡）。"""
        if mem is None or mem.spawn is None:
            return "forward_jump"
        dx, dz = mem.spawn[0] - state["x"], mem.spawn[2] - state["z"]
        turn = self._turn_towards(state, math.degrees(math.atan2(-dx, dz)))
        if turn and math.hypot(dx, dz) > 2:
            return turn
        if abs(state["pitch"]) > 20:
            return "look_level"
        return "forward_jump"

    def sprint_jump(self, state, priv):
        if self.in_danger(state) or self._hazard_ahead(state, priv):
            return self.go_home(state, self.mem) if self.in_danger(state) else "turn_right_big"
        if abs(state["pitch"]) > 20:
            return "look_level"
        return "sprint_jump"

    def in_danger(self, state):
        return state["air"] < 290 or state["dy"] < -8

    def wander(self, state, priv=None):
        """没有目标时的探索：沿一个方向走，躲开水和悬崖，偶尔换方向，撞墙就跳或绕开。"""
        if self.in_danger(state):
            return self.go_home(state, self.mem)
        if abs(state["pitch"]) > 20:
            return "look_level"
        self.wander_steps += 1
        if self._hazard_ahead(state, priv):
            return "turn_right_big" if self.turn_dir > 0 else "turn_left_big"
        if self.wander_steps % 150 == 0:
            self.turn_dir = 1 if self.rng.random() < 0.5 else -1
            return "turn_right_big" if self.turn_dir > 0 else "turn_left_big"
        if self.wander_steps % 30 == 0:
            return "turn_right" if self.rng.random() < 0.5 else "turn_left"
        return self._go(state, priv)

    # ------------------------------------------------------------ 技能
    def _key(self, state, point):
        return (round(state["x"] + point[0]), round(state["y"] + point[1]), round(state["z"] + point[2]))

    def chop_toward(self, state, priv, kinds=("log",)):
        """走到最近的可达目标方块前，对准并砍。没有进展的目标拉黑，被挡住就砍掉挡路的方块。"""
        ray = (priv or {}).get("ray")
        if ray and ray["hit"] == "block" and ray["type"] in kinds and ray["distance"] <= REACH:
            self.chops += 1
            if self.chops > 10 and self.target is not None:
                self.blacklist.add(self.target)
                self.target = None
                self.chops = 0
                return "turn_right_big"
            return "chop"
        gained = state["log"] + state["cobblestone"] + state["coal"]
        if gained > self.last_gain:
            self.attempts.clear()  # 拿到东西了：之前的尝试计数清零
        self.last_gain = gained
        points = (priv or {}).get("logs") if kinds == ("log",) else self._blocks(priv, kinds)
        reachable = [p for p in (points or []) if -0.5 <= p[1] <= 3.6 and self._key(state, p) not in self.blacklist]
        if not reachable:
            return self.wander(state, priv)
        best = min(reachable, key=lambda p: math.hypot(p[0], p[2]) + 0.6 * abs(p[1] - 1.0))
        key = self._key(state, best)
        # 每个目标累计尝试次数（跨目标切换也不清零），太多次还没拿到就拉黑——防止在两根原木之间来回切换卡死
        self.attempts[key] += 1
        if self.attempts[key] > 30:
            self.blacklist.add(key)
            self.target = None
            return "turn_left_big" if self.rng.random() < 0.5 else "turn_right_big"
        yaw, pitch, horiz = target_angles(*best)
        if key != self.target:
            self.target, self.best_dist, self.no_progress, self.chops = key, horiz, 0, 0
        elif horiz < self.best_dist - 0.25:
            self.best_dist, self.no_progress = horiz, 0
        else:
            self.no_progress += 1
        if self.no_progress > 24:
            self.blacklist.add(key)
            self.target = None
            return "turn_left_big" if self.rng.random() < 0.5 else "turn_right_big"
        turn = self._turn_towards(state, yaw)
        if turn:
            return turn
        if horiz > 2.4:
            if abs(state["pitch"]) > 25:
                return "look_level"
            if state["stuck"] >= 2:
                if ray and ray["hit"] == "block" and ray["type"] not in ("air", "water", "lava") and ray["distance"] <= 2.0:
                    return "chop"  # 树叶、泥土挡路：砍开
                return "forward_jump"
            if self._hazard_ahead(state, priv):
                self.blacklist.add(key)
                self.target = None
                return "turn_right_big" if self.turn_dir > 0 else "turn_left_big"
            return "forward"
        look = self._pitch_towards(state, pitch)
        if look:
            return look
        self.chops += 1
        if ray and ray["hit"] == "block" and ray["type"] == "leaves":
            return "chop"
        return "forward" if self.chops % 5 == 4 else "chop"

    def _blocks(self, priv, kinds):
        from mc_rl.privileged import CODE_NAMES, GRID_MIN, GRID_SHAPE

        grid = (priv or {}).get("grid")
        if grid is None:
            return []
        codes = [code for code, name in CODE_NAMES.items() if name in kinds]
        out = []
        for flat in np.nonzero(np.isin(grid, codes))[0]:
            flat = int(flat)
            x = flat % GRID_SHAPE[0]
            z = (flat // GRID_SHAPE[0]) % GRID_SHAPE[2]
            y = flat // (GRID_SHAPE[0] * GRID_SHAPE[2])
            off = (priv or {}).get("grid_offset", (0.0, 0.0, 0.0))
            out.append((x + GRID_MIN[0] + 0.5 + off[0], y + GRID_MIN[1] + 0.5 + off[1], z + GRID_MIN[2] + 0.5 + off[2]))
        return out

    def get_sword(self, state, priv, tracker):
        """打猎前先做木剑：徒手一下只掉 1 点血，被打的动物还会被击退、逃跑；木剑约 4 点，3 下就够。"""
        if state["wooden_sword"] >= 1:
            return None
        if tracker.collected["log"] < 3 and state["planks"] < 6:
            return self.chop_toward(state, priv)
        return self.craft_chain(state, want_tools=True, tracker=tracker) or self.chop_toward(state, priv)

    def hunt(self, state, priv):
        if state["wooden_sword"] >= 1 and state["hand"] != "wooden_sword":
            return "equip_sword"
        entities = (priv or {}).get("entities") or []
        drops = [e for e in entities if e["name"] in ("beef", "porkchop")]
        animals = [e for e in entities if e["name"] in ("Cow", "Pig")]
        ray = (priv or {}).get("ray")
        if ray and ray["hit"] == "entity" and ray["type"] in ("Cow", "Pig") and ray["distance"] <= 3.2:
            return "attack"
        targets = drops or animals
        if not targets:
            return self.wander(state, priv)
        target = min(targets, key=lambda e: math.hypot(e["dx"], e["dz"]))
        yaw, pitch, horiz = target_angles(target["dx"], target["dy"] + 0.6, target["dz"])
        turn = self._turn_towards(state, yaw)
        if turn:
            return turn
        if target in drops or horiz > 2.2:
            if abs(state["pitch"]) > 30:
                return "look_level"
            if self._hazard_ahead(state, priv) and horiz > 3:
                return "turn_right_big" if self.turn_dir > 0 else "turn_left_big"
            return self._go(state, priv)
        look = self._pitch_towards(state, pitch, tolerance=15)
        return look or "attack"

    def eat(self, state):
        if state["hand"] not in MEAT_ITEMS or state.get(state["hand"], 0) < 1:
            return "equip_meat"
        return "eat"

    def craft_chain(self, state, want_tools=False, want_table_placed=True, tracker=None):
        placed = tracker is not None and tracker.events.get("table_placed", 0) > 0
        need_planks = 4 + (5 if want_tools else 0) + (2 if state["stick"] < (3 if want_tools else 1) else 0)
        if state["log"] >= 1 and state["planks"] < need_planks:
            return "craft_planks"
        if state["crafting_table"] < 1 and not placed and state["planks"] >= 4 and \
                (not want_tools or state["planks"] >= 6 or state["log"] < 1):
            if state["stick"] < 1 and state["planks"] >= 6:
                return "craft_stick"
            return "craft_table"
        if state["stick"] < (3 if want_tools else 1) and state["planks"] >= 2:
            return "craft_stick"
        if want_table_placed and state["crafting_table"] >= 1 and not placed:
            self.place_tries += 1
            if self.place_tries % 3 == 0:
                return "turn_right_big" if self.place_tries % 6 == 0 else "back"
            return "place_table"
        if want_tools and state["table_near"] > 0.5:
            if state["wooden_sword"] < 1 and state["planks"] >= 2 and state["stick"] >= 1:
                return "craft_sword"
            if state["wooden_pickaxe"] < 1 and state["planks"] >= 3 and state["stick"] >= 2:
                return "craft_pickaxe"
        return None

    def shaft_depth(self, state):
        """当前所在竖井的深度：离自己开挖点水平 1.2 格以内才算在井里。"""
        a = self.shaft_anchor
        if a is None or math.hypot(state["x"] - a[0], state["z"] - a[2]) > 1.2:
            return None
        return a[1] - state["y"]

    def _ground(self, priv):
        """脚下方块的编码（特权网格），没有网格时返回 None。"""
        from mc_rl.privileged import GRID_SHAPE, grid_index

        grid = (priv or {}).get("grid")
        if grid is None or len(grid) != GRID_SHAPE[0] * GRID_SHAPE[1] * GRID_SHAPE[2]:
            return None
        return int(grid[grid_index(0, -1, 0)])

    def shelter(self, state, tracker, priv=None):
        """原地挖 3 格竖井并封顶。脚下是沙子、石头、水时先走到泥土地上（徒手挖不出能封顶的方块）。"""
        depth = self.shaft_depth(state)
        seal_blocks = state["dirt"] + state["cobblestone"]
        if depth is None:
            ground = self._ground(priv)
            if ground is not None and ground != 4 and seal_blocks < 2 and self.ground_search < 40:
                self.ground_search += 1
                return self.wander(state, priv)
            self.shaft_anchor = (state["x"], state["y"], state["z"])
            self.seal_tries = 0
            return "dig_down"
        if depth >= 2:
            self.committed = True
        if state["sky"] < 0.5 and depth >= 2 and state["sealed"] >= 1:
            if state["food"] < self.eat_below and state["meat"] >= 1:
                return self.eat(state)
            return "sneak"
        if depth < 3 or seal_blocks < 1:
            return "dig_down"
        self.seal_tries += 1
        if self.seal_tries > 8:
            self.seal_tries = 0
            return "dig_down"
        return "seal_above"

    # ------------------------------------------------------------ 扩展任务的技能
    def homing(self, state, priv, tracker):
        """先背对出生点走出 24 格，再按“出生点方位”走回去。"""
        home_yaw = state["yaw"] + state.get("home_rel", 0.0)
        if tracker.max_dist < 25:
            if state["dist"] > 3:
                turn = self._turn_towards(state, home_yaw + 180.0)
                if turn:
                    return turn
            return self.wander(state, priv) if state["dist"] <= 3 else self._go(state, priv)
        turn = self._turn_towards(state, home_yaw)
        if turn:
            return turn
        if abs(state["pitch"]) > 20:
            return "look_level"
        return self._go(state, priv)

    def fight(self, state, priv):
        """夜里对付怪物：躲开苦力怕，其余靠近后挥剑。"""
        entities = (priv or {}).get("entities") or []
        creepers = [e for e in entities if e["name"] == "Creeper" and math.hypot(e["dx"], e["dz"]) < 5]
        if creepers:
            c = creepers[0]
            turn = self._turn_towards(state, math.degrees(math.atan2(c["dx"], -c["dz"])))
            return turn or self._go(state, priv)
        ray = (priv or {}).get("ray")
        if ray and ray["hit"] == "entity" and ray["type"] in ("Zombie", "Skeleton", "Spider") and ray["distance"] <= 3.2:
            return "attack"
        targets = [e for e in entities if e["name"] in ("Zombie", "Skeleton", "Spider")]
        if not targets:
            return "sneak"
        target = min(targets, key=lambda e: math.hypot(e["dx"], e["dz"]))
        yaw, pitch, horiz = target_angles(target["dx"], target["dy"] + 1.0, target["dz"])
        turn = self._turn_towards(state, yaw)
        if turn:
            return turn
        if horiz > 2.5:
            return self._go(state, priv)
        return self._pitch_towards(state, pitch, tolerance=15) or "attack"

    def combat(self, state, priv, mem, tracker):
        if state["wooden_sword"] < 1:
            if tracker.collected["log"] < 3:
                return self.chop_toward(state, priv)
            return self.craft_chain(state, want_tools=True, tracker=tracker) or self.chop_toward(state, priv)
        if state["hand"] != "wooden_sword":
            return "equip_sword"
        if self.committed or state["life"] < 8:
            return self.shelter(state, tracker, priv)
        if mem.ticks < NIGHT_START - 500:
            if state["dist"] > 12:
                return self.go_home(state, mem)
            return "sneak"
        return self.fight(state, priv)

    def cook(self, state, priv, tracker):
        """木头 -> 木镐 -> 生肉 -> 8 块圆石 -> 工作台旁合成熔炉 -> 放下 -> 烤肉。木板留一块当燃料。"""
        if state["wooden_pickaxe"] < 1:
            if tracker.collected["log"] < 4:
                return self.chop_toward(state, priv)
            return self.craft_chain(state, want_tools=True, tracker=tracker) or self.chop_toward(state, priv)
        if state["planks"] + state["log"] < 2 and state["cooked"] < 1:
            return self.chop_toward(state, priv)  # 留燃料
        # 往下挖石头之后人在竖井里出不来，所以先在地面上找肉；但最多找到 4000 刻，动物少的种子不至于整条链卡住
        if state["raw_meat"] < 1 and state["cooked"] < 1 and self.mem is not None and self.mem.ticks < 4000:
            return self.get_sword(state, priv, tracker) or self.hunt(state, priv)
        if state["furnace"] < 1 and not tracker.events.get("furnace_placed"):
            if state["cobblestone"] < 8:
                if state["hand"] != "wooden_pickaxe":
                    return "equip_pickaxe"
                reachable = [p for p in self._blocks(priv, ("stone",))
                             if -0.5 <= p[1] <= 3.6 and math.hypot(p[0], p[2]) < 4
                             and self._key(state, p) not in self.blacklist]
                if reachable:
                    return self.chop_toward(state, priv, kinds=("stone",))
                return "dig_down"  # 石头都在脚下：带着木镐往下挖，挖到石层后井壁就是可达的石头
            if state["table_near"] < 0.5:
                if state["crafting_table"] >= 1:
                    return "place_table"
                if state["planks"] >= 5:
                    return "craft_table"
                return "craft_planks" if state["log"] >= 1 else self.chop_toward(state, priv)
            return "craft_furnace"
        if state["furnace"] >= 1:
            return "place_furnace"
        if state["raw_meat"] < 1 and state["cooked"] < 1:
            return self.get_sword(state, priv, tracker) or self.hunt(state, priv)
        if state["furnace_near"] < 0.5:
            return self.go_to_furnace(state, priv)
        if state["planks"] < 1 and state["log"] >= 1:
            return "craft_planks"
        return "smelt_meat"

    def go_to_furnace(self, state, priv):
        """打猎跑远了：按记下的熔炉位置走回去。"""
        offset = state.get("furnace_offset")
        if not offset:
            return self.wander(state, priv)
        yaw = math.degrees(math.atan2(-offset[0], offset[2]))
        turn = self._turn_towards(state, yaw)
        if turn:
            return turn
        return self._go(state, priv)

    # ------------------------------------------------------------ 任务入口
    def act(self, state, priv, mem, tracker):
        name = self.task.name
        self.mem = mem
        if state["air"] < 290 and name != "shelter":
            return self.go_home(state, mem)
        hungry = state["food"] < self.eat_below and state["meat"] >= 1
        if name == "explore":
            return self.wander(state, priv)
        if name == "wood":
            return self.chop_toward(state, priv)
        if name == "craft":
            if tracker.collected["log"] < 3 and state["planks"] < 6 and state["crafting_table"] < 1 \
                    and not tracker.events.get("table_placed"):
                return self.chop_toward(state, priv)
            return self.craft_chain(state, tracker=tracker) or self.chop_toward(state, priv)
        if name == "tools":
            if tracker.collected["log"] < 3 and state["wooden_pickaxe"] < 1:
                return self.chop_toward(state, priv)
            step = self.craft_chain(state, want_tools=True, tracker=tracker)
            if step:
                return step
            if state["wooden_sword"] >= 1 and state["wooden_pickaxe"] >= 1:
                return "equip_sword" if state["hand"] != "wooden_sword" else "noop"
            return self.chop_toward(state, priv)
        if name == "hunt":
            return self.get_sword(state, priv, tracker) or self.hunt(state, priv)
        if name == "eat":
            if state["meat"] >= 1 and state["food"] < 20:
                return self.eat(state)
            if state["meat"] >= 1:
                return self.sprint_jump(state, priv)  # 满饥饿值吃不下：疾跑跳跃消耗饥饿值
            return self.get_sword(state, priv, tracker) or self.hunt(state, priv)
        if name == "shelter":
            return self.shelter(state, tracker, priv)
        if name == "homing":
            return self.homing(state, priv, tracker)
        if name == "combat":
            return self.combat(state, priv, mem, tracker)
        if name == "cook":
            return self.cook(state, priv, tracker)
        if name == "torch":
            if tracker.collected["log"] < 3 and state["wooden_pickaxe"] < 1:
                return self.chop_toward(state, priv)
            step = self.craft_chain(state, want_tools=True, tracker=tracker)
            if state["wooden_pickaxe"] < 1:
                return step or self.chop_toward(state, priv)
            if state["coal"] >= 1 and state["stick"] >= 1 and state["torch"] < 1:
                return "craft_torch"
            if state["torch"] >= 1:
                return "place_torch"
            if state["hand"] != "wooden_pickaxe":
                return "equip_pickaxe"
            if self._blocks(priv, ("coal_ore",)):
                return self.chop_toward(state, priv, kinds=("coal_ore",))
            return "dig_down"
        # nightfall / survive_day
        return self.survival_strategy(state, priv, mem, tracker, hungry)

    def survival_strategy(self, state, priv, mem, tracker, hungry):
        """nightfall / survive_day 两个任务里老师怎么安排这一天，返回宏动作名。

        可用的技能：chop_toward / craft_chain / hunt / eat / shelter / wander / go_home。
        有用的量：mem.ticks（当前刻）、state["dist"]（离出生点）、priv["nearest_log"/"nearest_animal"]、
        tracker.collected / tracker.events、self.committed（已经挖下去了）。
        """
        raise NotImplementedError("TODO(学生)：设计老师的生存策略：白天在哪里做什么、什么时候躲、躲在哪里、饿了怎么办")
