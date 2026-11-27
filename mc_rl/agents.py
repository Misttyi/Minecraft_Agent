"""评测接口 agent(observation, env) -> action 的几种实现。

- PolicyAgent：学到的网络（只用标准观测），内部自己维护 EpisodeMemory 和宏动作展开。
- DaytimeScript：作业要求的白天脚本基线（太阳亮时转向挥击到有原木，合成到工作台后蹲下）。
- RandomMacroAgent：在合法宏动作里均匀随机，作为“去掉学习模块”的消融对照。
宏动作会展开成多个底层动作；评测接口每次只要一个，所以这里排队逐个发出。
"""

import numpy as np

from mc_rl.features import EpisodeMemory, pov_array, snapshot, vectorize
from mc_rl.macros import MACROS, MACRO_INDEX, action_mask, expand
from mc_rl.task_env import merge_action
from mc_rl.tasks import TASK_INDEX, TASKS


TRAIN_REPEAT = 8  # 训练时的动作重复；评测环境的 action_repeat 不同时在这里补齐
HIGH_LEVEL = ("craft", "equip", "place", "nearbyCraft", "nearbySmelt")


class MacroAgent:
    """把“选宏动作”的策略包装成评测接口 agent(observation, env) -> action。

    - 每个新环境（survive_day.evaluate 每个种子新建一个环境）自动重置记忆，避免上一局的时钟、出生点、工作台位置串进来。
    - 宏动作按训练时的 8 刻/拍展开；评测环境 action_repeat=1 时把每拍拆成 8 个底层动作发出
      （合成/装备/放置只在第一刻发），所以 action_repeat=1 和 8 下行为一致。
    """

    def __init__(self, task="survive_day", action_repeat=TRAIN_REPEAT):
        self.task_index = TASK_INDEX[task]
        self.action_repeat = action_repeat
        self._env = None
        self.reset()

    def reset(self):
        self.mem = EpisodeMemory()
        self.queue = []
        self.prev_state = None
        self.pending = None
        self.macros = []

    def choose(self, observation, state):
        raise NotImplementedError

    def _emit(self, steps, repeat):
        factor = max(1, TRAIN_REPEAT // max(1, int(repeat)))
        out = []
        for override in steps:
            for copy in range(factor):
                if copy == 0:
                    out.append(override)
                else:
                    out.append({k: v for k, v in override.items() if k not in HIGH_LEVEL})
        return out

    def __call__(self, observation, env):
        if env is not self._env:
            self.reset()
            self._env = env
        repeat = getattr(env, "action_repeat", 1)
        if self.mem.spawn is None:
            self.mem.observe(snapshot(observation))
        if self.queue:
            return merge_action(env.action_space.no_op(), self.queue.pop(0))
        state = snapshot(observation, self.mem)
        if self.pending is not None and self.prev_state is not None:
            macro = self.pending
            if macro == "place_table" and state["crafting_table"] < self.prev_state["crafting_table"]:
                self.mem.place_table(self.prev_state)
            if macro == "place_furnace" and state["furnace"] < self.prev_state["furnace"]:
                self.mem.place_furnace(self.prev_state)
            if macro == "seal_above" and (state["dirt"] < self.prev_state["dirt"]
                                          or state["cobblestone"] < self.prev_state["cobblestone"]):
                self.mem.sealed += 1
            self.mem.observe(state)
            state = snapshot(observation, self.mem)
        name = self.choose(observation, state)
        self.macros.append(name)
        self.pending = name
        self.prev_state = state
        steps = expand(name, state, TRAIN_REPEAT)
        self.mem.ticks += TRAIN_REPEAT * len(steps)
        self.mem.decisions += 1
        actions = self._emit(steps, repeat)
        self.queue = actions[1:]
        return merge_action(env.action_space.no_op(), actions[0])


class PolicyAgent(MacroAgent):
    def __init__(self, model, task="survive_day", device="cpu", deterministic=False, seed=0):
        self.model = model.to(device).eval()
        self.device = device
        self.deterministic = deterministic
        self.rng = np.random.default_rng(seed)
        import torch

        torch.manual_seed(seed)  # 采样可复现
        super().__init__(task)

    def choose(self, observation, state):
        import torch

        from mc_rl.models import to_tensors

        batch = {
            "vec": vectorize(state, self.task_index, len(TASKS))[None],
            "pov": pov_array(observation)[None],
            "mask": action_mask(state)[None],
        }
        with torch.no_grad():
            action, _, _ = self.model.act(to_tensors(batch, self.device), deterministic=self.deterministic)
        return MACROS[int(action[0])]


class RandomMacroAgent(MacroAgent):
    def __init__(self, task="survive_day", seed=0):
        self.rng = np.random.default_rng(seed)
        super().__init__(task)

    def choose(self, observation, state):
        valid = np.nonzero(action_mask(state))[0]
        return MACROS[int(self.rng.choice(valid))]


class DaytimeScript(MacroAgent):
    """作业基线 2：sun > 0.5 时转向并挥击直到有原木；合成木板、木棍、工作台；然后蹲下。"""

    # 每个方向连续挥击两拍宏动作，转满一圈后挪一步再搜索。
    _SEARCH = ("chop", "chop", "turn_right") * 24 + ("forward",)

    def reset(self):
        super().reset()
        self.search_step = 0

    def choose(self, observation, state):
        mask = action_mask(state)
        idle = "sneak" if mask[MACRO_INDEX["sneak"]] else "noop"
        if state["crafting_table"] >= 1:
            return idle

        needs_sticks = state["stick"] < 1
        planks_needed = 6 if needs_sticks else 4  # 木棍用 2 块，工作台用 4 块。
        if state["planks"] < planks_needed and mask[MACRO_INDEX["craft_planks"]]:
            self.search_step = 0  # 木材不足时，优先从刚砍到原木的方向继续。
            return "craft_planks"
        if needs_sticks and mask[MACRO_INDEX["craft_stick"]]:
            return "craft_stick"
        if not needs_sticks and mask[MACRO_INDEX["craft_table"]]:
            return "craft_table"

        if state["sun"] > 0.5:
            macro = self._SEARCH[self.search_step]
            self.search_step = (self.search_step + 1) % len(self._SEARCH)
            if mask[MACRO_INDEX[macro]]:
                return macro
        return idle
