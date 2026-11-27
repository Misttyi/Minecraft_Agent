"""不启动 Minecraft 的替身环境，只用于单元测试和本地冒烟。

接口对齐 SurviveDayWrapper：字典观测、action_space.no_op()、step 返回
(obs, reward, done, info)，info 里有 survival_steps / survived_day / life / food。
另外在 obs["privileged"] 里给出和 mc_rl.privileged.extract 相同格式的特权信息。
动力学是粗略近似：能砍树、合成、打猎、进食、挖坑封顶，夜里露天会掉血。
"""

import math

import numpy as np

from survive_day.constants import INVENTORY_ITEMS, MAX_EPISODE_STEPS, STEP_ALIVE_REWARD
from survive_day.wrappers import action_for_repeat


class _NoOpSpace:
    def no_op(self):
        return {
            "forward": 0, "back": 0, "left": 0, "right": 0, "jump": 0, "sneak": 0,
            "sprint": 0, "attack": 0, "use": 0,
            "camera": np.zeros(2, dtype=np.float32),
            "craft": "none", "nearbyCraft": "none", "nearbySmelt": "none",
            "equip": "none", "place": "none",
        }


RECIPES = {
    "planks": ({"log": 1}, 4),
    "stick": ({"planks": 2}, 4),
    "crafting_table": ({"planks": 4}, 1),
    "torch": ({"coal": 1, "stick": 1}, 4),
}
NEARBY = {
    "wooden_pickaxe": ({"planks": 3, "stick": 2}, 1),
    "wooden_sword": ({"planks": 2, "stick": 1}, 1),
    "furnace": ({"cobblestone": 8}, 1),
}


class MockSurviveEnv:
    def __init__(self, seed=0, action_repeat=8, privileged=True, max_episode_steps=MAX_EPISODE_STEPS,
                 night_damage=True):
        self.seed = int(seed)
        self.action_repeat = int(action_repeat)
        self.privileged = privileged
        self.max_episode_steps = int(max_episode_steps)
        self.night_damage = night_damage
        self.action_space = _NoOpSpace()

    # ------------------------------------------------------------ 世界
    def set_seed(self, seed):
        self.seed = int(seed)

    def reset(self):
        rng = np.random.default_rng(self.seed)
        self.rng = np.random.default_rng(self.seed + 7919)
        self.ticks = 0
        self.x, self.y, self.z = 0.5, 64.0, 0.5
        self.ground = 64.0
        self.yaw = float(rng.uniform(-180, 180))
        self.pitch = 0.0
        self.life, self.food, self.saturation = 20.0, 20.0, 5.0
        self.food_timer = 0
        self.inv = {item: 0 for item in INVENTORY_ITEMS}
        self.hand = "air"
        self.trees = [[float(rng.uniform(-14, 14)), float(rng.uniform(-14, 14)), 4] for _ in range(6)]
        self.animals = [[float(rng.uniform(-20, 20)), float(rng.uniform(-20, 20)),
                         str(rng.choice(["Cow", "Pig"])), 10.0] for _ in range(4)]
        self.items = []
        self.break_progress = 0
        self.use_progress = 0
        self.depth = 0
        self.sealed = False
        self.table = None
        self.furnace = None
        self.stone_hits = 0
        return self._obs()

    def _facing(self, tx, tz, limit_deg=14.0, reach=3.2):
        dx, dz = tx - self.x, tz - self.z
        dist = math.hypot(dx, dz)
        target = math.degrees(math.atan2(-dx, dz))
        diff = (target - self.yaw + 180) % 360 - 180
        return dist <= reach and abs(diff) <= limit_deg, dist, diff

    def _ray(self):
        if self.pitch > 70 and self.depth < 8:
            return {"type": "dirt" if self.depth < 4 else "stone", "distance": 1.5, "hit": "block"}
        for animal in self.animals:
            ok, dist, _ = self._facing(animal[0], animal[1])
            if ok and abs(self.pitch) < 40:
                return {"type": animal[2], "distance": dist, "hit": "entity"}
        for tree in self.trees:
            ok, dist, _ = self._facing(tree[0], tree[1])
            if ok and tree[2] > 0 and -60 < self.pitch < 45:
                return {"type": "log", "distance": dist, "hit": "block"}
        return None

    def _tick(self, action, first):
        cam = np.asarray(action.get("camera", (0, 0)), dtype=np.float32)
        self.pitch = float(np.clip(self.pitch + cam[0], -90, 90))
        self.yaw = (self.yaw + float(cam[1]) + 180) % 360 - 180
        if self.depth == 0 and (action.get("forward") or action.get("back")):
            sign = 1.0 if action.get("forward") else -0.5
            rad = math.radians(self.yaw)
            self.x += -math.sin(rad) * 0.2 * sign
            self.z += math.cos(rad) * 0.2 * sign
        ray = self._ray()
        if action.get("attack"):
            if ray and ray["hit"] == "block" and ray["type"] == "log":
                self.break_progress += 1
                if self.break_progress >= 50:
                    self.break_progress = 0
                    self.inv["log"] += 1
                    for tree in self.trees:
                        if self._facing(tree[0], tree[1])[0] and tree[2] > 0:
                            tree[2] -= 1
                            break
            elif ray and ray["hit"] == "block" and ray["type"] == "dirt":
                self.break_progress += 1
                if self.break_progress >= 16:
                    self.break_progress = 0
                    self.depth += 1
                    self.y -= 1
                    self.inv["dirt"] += 1
            elif ray and ray["hit"] == "block" and ray["type"] == "stone":
                if self.hand == "wooden_pickaxe":
                    self.break_progress += 1
                    if self.break_progress >= 20:
                        self.break_progress = 0
                        self.stone_hits += 1
                        self.inv["cobblestone"] += 1
                        if self.stone_hits % 3 == 0:
                            self.inv["coal"] += 1
            elif ray and ray["hit"] == "entity" and self.ticks % 8 == 0:
                for animal in self.animals:
                    if animal[2] == ray["type"] and self._facing(animal[0], animal[1])[0]:
                        animal[3] -= 4 if self.hand == "wooden_sword" else 2
                        if animal[3] <= 0:
                            self.items.append([animal[0], animal[1], "beef" if animal[2] == "Cow" else "porkchop"])
                            self.animals.remove(animal)
                        break
            else:
                self.break_progress = 0
        else:
            self.break_progress = 0
        for item in list(self.items):
            if math.hypot(item[0] - self.x, item[1] - self.z) < 1.5:
                self.inv[item[2]] += 1
                self.items.remove(item)
        if first:
            craft = action.get("craft", "none")
            if craft in RECIPES:
                self._craft(*RECIPES[craft], craft)
            nearby = action.get("nearbyCraft", "none")
            if nearby in NEARBY and self.table is not None:
                if math.hypot(self.table[0] - self.x, self.table[1] - self.z) < 4:
                    self._craft(*NEARBY[nearby], nearby)
            equip = action.get("equip", "none")
            if equip not in ("none",) and (equip == "air" or self.inv.get(equip, 0) > 0):
                self.hand = equip
            place = action.get("place", "none")
            if place == "crafting_table" and self.inv["crafting_table"] > 0 and self.pitch > 30:
                self.inv["crafting_table"] -= 1
                self.table = (self.x, self.z)
            if place == "dirt" and self.inv["dirt"] > 0 and self.pitch < -50 and self.depth >= 2:
                self.inv["dirt"] -= 1
                self.sealed = True
            if place == "torch" and self.inv["torch"] > 0 and self.pitch > 30:
                self.inv["torch"] -= 1
            if place == "furnace" and self.inv["furnace"] > 0 and self.pitch > 30:
                self.inv["furnace"] -= 1
                self.furnace = (self.x, self.z)
            smelt = action.get("nearbySmelt", "none")
            if smelt in ("cooked_beef", "cooked_porkchop") and self.furnace is not None \
                    and math.hypot(self.furnace[0] - self.x, self.furnace[1] - self.z) < 4:
                raw = "beef" if smelt == "cooked_beef" else "porkchop"
                fuel = next((f for f in ("coal", "planks", "log") if self.inv[f] > 0), None)
                if self.inv[raw] > 0 and fuel:
                    self.inv[raw] -= 1
                    self.inv[fuel] -= 1
                    self.inv[smelt] += 1
        if action.get("use") and self.hand in ("beef", "porkchop") and self.inv[self.hand] > 0 and self.food < 20:
            self.use_progress += 1
            if self.use_progress >= 32:
                self.use_progress = 0
                self.inv[self.hand] -= 1
                self.food = min(20.0, self.food + 3)
                self.saturation = min(self.food, self.saturation + 1.8)
                if self.inv[self.hand] == 0:
                    self.hand = "air"
        elif not action.get("use"):
            self.use_progress = 0
        self.ticks += 1
        self.food_timer += 1
        if self.food_timer >= 900:
            self.food_timer = 0
            if self.saturation > 0:
                self.saturation = max(0.0, self.saturation - 1)
            else:
                self.food = max(0.0, self.food - 1)
        if self.food <= 0 and self.ticks % 80 == 0:
            self.life -= 1
        night = 13000 <= self.ticks <= 23000
        if self.night_damage and night and not self.sealed and self.rng.random() < 1 / 120:
            self.life -= 3
        if self.ticks % 80 == 0 and self.food >= 18 and self.life < 20:
            self.life = min(20.0, self.life + 1)
        for animal in self.animals:
            animal[0] += float(self.rng.normal(0, 0.03))
            animal[1] += float(self.rng.normal(0, 0.03))

    def _craft(self, needs, amount, product):
        if all(self.inv.get(k, 0) >= v for k, v in needs.items()):
            for k, v in needs.items():
                self.inv[k] -= v
            self.inv[product] += amount

    def step(self, action):
        reward = 0.0
        done = False
        survived = False
        for index in range(self.action_repeat):
            self._tick(action_for_repeat(action, first=(index == 0)), index == 0)
            alive = self.life > 0
            if alive:
                reward += STEP_ALIVE_REWARD
                if self.ticks >= self.max_episode_steps:
                    done = survived = True
            else:
                done = True
            if done:
                break
        obs = self._obs()
        info = {
            "survival_steps": self.ticks,
            "survived_day": survived,
            "life": obs["life_stats"]["life"],
            "food": obs["life_stats"]["food"],
            "alive": self.life > 0,
        }
        return obs, reward, done, info

    def close(self):
        pass

    # ------------------------------------------------------------ 观测
    def _sun(self):
        t = self.ticks % 24000
        if t < 12000:
            return 1.0
        if t < 13500:
            return 1.0 - (t - 12000) / 1500
        if t < 22500:
            return 0.0
        return (t - 22500) / 1500

    def _pov(self):
        sun = self._sun()
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        horizon = int(np.clip(32 - self.pitch * 0.5, 0, 64))
        img[:horizon] = (np.array([120, 160, 255]) * (0.2 + 0.8 * sun)).astype(np.uint8)
        img[horizon:] = (np.array([70, 140, 50]) * (0.3 + 0.7 * sun)).astype(np.uint8)
        if self.sealed or self.depth >= 2:
            img[:] = (90, 70, 50)
        for tree in self.trees:
            ok, dist, diff = self._facing(tree[0], tree[1], limit_deg=35, reach=20)
            if ok and tree[2] > 0:
                col = int(32 + diff / 35 * 30)
                width = max(1, int(8 / max(dist, 1)))
                img[max(0, horizon - 20):horizon + 4, max(0, col - width):col + width] = (100, 75, 40)
        for animal in self.animals:
            ok, dist, diff = self._facing(animal[0], animal[1], limit_deg=35, reach=20)
            if ok:
                col = int(32 + diff / 35 * 30)
                width = max(1, int(6 / max(dist, 1)))
                color = (240, 160, 150) if animal[2] == "Pig" else (60, 40, 30)
                img[horizon:horizon + 6, max(0, col - width):col + width] = color
        return img

    def _privileged(self):
        logs = []
        for tree in self.trees:
            for level in range(tree[2]):
                logs.append((tree[0] - self.x, float(level) + 0.5, tree[1] - self.z))
        entities = [
            {"name": a[2], "dx": a[0] - self.x, "dy": 0.0, "dz": a[1] - self.z} for a in self.animals
        ] + [
            {"name": i[2], "dx": i[0] - self.x, "dy": 0.0, "dz": i[1] - self.z, "item": True}
            for i in self.items
        ]
        from mc_rl.privileged import summarize_privileged

        return summarize_privileged({"logs": logs, "entities": entities, "ray": self._ray()})

    def _obs(self):
        sky = not self.sealed  # 和真实游戏一样：1×1 竖井里仍能看到天空，封顶后才看不到
        obs = {
            "pov": self._pov(),
            "inventory": {k: np.array(v) for k, v in self.inv.items()},
            "equipped_items": {"mainhand": {"type": self.hand, "damage": 0, "maxDamage": 0}},
            "life_stats": {
                "is_alive": self.life > 0, "life": self.life, "food": self.food,
                "saturation": self.saturation, "air": 300, "xp": 0, "score": 0,
            },
            "location_stats": {
                "sun_brightness": self._sun(), "sky_light_level": self._sun(),
                "light_level": 15 * self._sun() if sky else 0, "can_see_sky": np.array([0, 1] if sky else [1, 0]),
                "xpos": self.x, "ypos": self.y, "zpos": self.z, "pitch": self.pitch, "yaw": self.yaw,
                "biome_id": 1, "is_raining": False,
            },
        }
        if self.privileged:
            obs["privileged"] = self._privileged()
        return obs
