"""宏动作 + 任务得分表 + 回放记录，套在 SurviveDayWrapper（或 mock）外面。

策略看到的观测：{"vec": float32[V], "pov": uint8[64,64,3], "mask": bool[A]}。
"""

from collections import deque

import numpy as np

from mc_rl.features import EpisodeMemory, pov_array, snapshot, vector_size, vectorize
from mc_rl.macros import MACROS, NUM_MACROS, action_mask, expand
from mc_rl.privileged import extract
from mc_rl.tasks import TASK_INDEX, TASKS, TaskTracker, classify_death, get_task

RECORD_KEYS = ("life", "food", "saturation", "sun", "light", "sky", "dy", "pitch", "yaw",
               "x", "y", "z", "log", "planks", "stick", "crafting_table", "wooden_sword",
               "wooden_pickaxe", "dirt", "meat", "coal", "torch", "hand", "ticks")


def merge_action(no_op, override):
    action = dict(no_op)
    for key, value in override.items():
        action[key] = value
    return action


class TaskEnv:
    def __init__(self, base_factory, action_repeat=8, record=False, record_every=1,
                 expert=None, pov_size=64):
        """base_factory(seed) -> 环境。环境若有 set_seed(seed) 就复用，否则重建。"""
        self.base_factory = base_factory
        self.action_repeat = int(action_repeat)
        self.record = record
        self.record_every = max(1, int(record_every))
        self.expert = expert
        self.pov_size = pov_size
        self.base = None
        self.seed = None
        self.num_tasks = len(TASKS)
        self.obs_size = vector_size(self.num_tasks)
        self.task = None
        self.history = deque(maxlen=8)

    # ------------------------------------------------------------ 生命周期
    def _ensure_base(self, seed):
        if self.base is not None and seed != self.seed:
            setter = getattr(self.base, "set_seed", None)
            if setter is not None and setter(seed) is not False:
                self.seed = seed
                return
            self.close()
        if self.base is None:
            self.base = self.base_factory(seed)
            self.seed = seed

    def reset(self, task=None, seed=0):
        self.task = get_task(task) if isinstance(task, str) else (task or TASKS[0])
        self.task_index = TASK_INDEX[self.task.name]
        self._ensure_base(seed)
        raw = self.base.reset()
        self.no_op = self.base.action_space.no_op()
        self.mem = EpisodeMemory()
        self.mem.observe(snapshot(raw))
        self.state = snapshot(raw, self.mem)
        self.priv = extract(raw)
        self.tracker = TaskTracker(self.task)
        self.tracker.start(self.state, self.mem)
        self.history.clear()
        self.history.append(self.state)
        self.episode_return = 0.0
        self.frames = []
        self.trace = []
        if self.expert is not None:
            self.expert.reset(self.task)
        self._record(raw, "reset", 0.0, [])
        return self.observation(raw)

    def close(self):
        if self.base is not None:
            try:
                self.base.close()
            finally:
                self.base = None

    # ------------------------------------------------------------ 观测
    def observation(self, raw):
        return {
            "vec": vectorize(self.state, self.task_index, self.num_tasks),
            "pov": pov_array(raw, self.pov_size),
            "mask": action_mask(self.state),
        }

    def expert_action(self):
        if self.expert is None:
            return -1
        name = self.expert.act(self.state, self.priv, self.mem, self.tracker)
        index = MACROS.index(name)
        if not action_mask(self.state)[index]:
            return MACROS.index("noop")
        return index

    # ------------------------------------------------------------ 推进
    def step(self, action_index):
        action_index = int(action_index)
        macro = MACROS[action_index]
        mask = action_mask(self.state)
        invalid = not mask[action_index]
        if invalid:
            macro = "noop"
        prev = self.state
        ticks_before = self.mem.ticks
        raw = None
        done = False
        info = {}
        for override in expand(macro, prev, self.action_repeat):
            raw, _r, done, info = self.base.step(merge_action(self.no_op, override))
            if done:
                break
        self.mem.ticks = int(info.get("survival_steps", ticks_before + self.action_repeat))
        self.mem.decisions += 1
        self.mem.last_macro = macro
        current = snapshot(raw, self.mem)
        if macro == "place_table" and current["crafting_table"] < prev["crafting_table"]:
            self.mem.place_table(prev)
        if macro == "place_furnace" and current["furnace"] < prev["furnace"]:
            self.mem.place_furnace(prev)
        if macro == "seal_above" and (current["dirt"] < prev["dirt"] or current["cobblestone"] < prev["cobblestone"]):
            self.mem.sealed += 1
        self.mem.observe(current)
        self.state = snapshot(raw, self.mem)
        self.priv = extract(raw)
        self.history.append(self.state)
        ticks = self.mem.ticks - ticks_before
        # 以包装器的 alive 为准。Malmo 会在 24000 刻前几十刻按时间结束，survived_day 几乎总是 False，不能拿来判死亡。
        died = self.state["alive"] < 0.5 or not info.get("alive", True)
        if died:
            self.state["alive"] = 0.0
        timeout = (not done) and self.mem.ticks >= self.task.horizon
        complete_stop = self.task.stop_on_success and self.tracker.complete
        final = done or timeout
        reward, newly = self.tracker.update(self.state, macro, ticks, self.priv, final=final)
        if not final and self.task.stop_on_success and self.tracker.complete:
            final = True
            complete_stop = True
            more, newer = self.tracker.update(self.state, "noop", 0, None, final=True)
            reward += more
            newly += newer
        if invalid:
            reward -= self.task.dense.get("invalid", 0.0)
        self.episode_return += reward
        self._record(raw, macro, reward, newly)
        out = {
            "macro": macro,
            "ticks": self.mem.ticks,
            "newly": newly,
            "truncated": bool(timeout and not (self.task.stop_on_success and self.tracker.complete)),
            "task": self.task.name,
        }
        if final:
            out["episode"] = self.episode_summary(died)
        return self.observation(raw), float(reward), bool(final), out

    def episode_summary(self, died):
        summary = self.tracker.summary()
        summary.update(
            seed=int(self.seed) if self.seed is not None else None,
            survival_steps=int(self.mem.ticks),
            decisions=int(self.mem.decisions),
            died=bool(died),
            death_cause=classify_death(list(self.history)) if died else None,
            life=float(self.state["life"]),
            food=float(self.state["food"]),
            return_=round(self.episode_return, 3),
        )
        return summary

    def _record(self, raw, macro, reward, newly):
        if not self.record or raw is None:
            return
        if self.mem.decisions % self.record_every and not newly and macro != "reset":
            return
        self.frames.append(pov_array(raw, self.pov_size).copy())
        entry = {key: self.state.get(key) for key in RECORD_KEYS}
        entry.update(macro=macro, reward=round(float(reward), 3), newly=list(newly),
                     score=self.tracker.score)
        if self.priv is not None:
            for key in ("nearest_log", "nearest_animal", "nearest_hostile"):
                value = self.priv.get(key)
                entry[key] = None if value is None else round(float(value), 2)
            ray = self.priv.get("ray")
            entry["ray"] = None if not ray else f"{ray['hit']}:{ray['type']}@{ray['distance']:.1f}"
        self.trace.append(entry)


def num_actions():
    return NUM_MACROS


def make_base_factory(kind="minerl", privileged=True, action_repeat=8, **mock_kwargs):
    """返回 base_factory(seed)。kind = minerl | mock。"""
    if kind == "mock":
        from mc_rl.mock_env import MockSurviveEnv

        def factory(seed):
            return MockSurviveEnv(seed=seed, action_repeat=action_repeat, privileged=privileged, **mock_kwargs)

        return factory

    def factory(seed):
        if privileged:
            from mc_rl.privileged import make_privileged_env

            env = make_privileged_env(seed, action_repeat=action_repeat)
        else:
            from survive_day import make_env

            env = make_env(seed, action_repeat=action_repeat, strip_reward=False)
        from mc_rl.privileged import set_world_seed

        def set_seed(new_seed):
            return set_world_seed(env, new_seed)

        env.set_seed = set_seed
        return env

    return factory


def as_batch(observations):
    return {
        key: np.stack([obs[key] for obs in observations]) for key in ("vec", "pov", "mask")
    }
