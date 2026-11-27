"""死亡即结束、存活奖励，以及高层动作的重复窗口。"""

try:
    import gym
except ImportError:  # 没有安装 MineRL 时，单元测试仍能跑包装器逻辑。
    gym = None

from survive_day.constants import (
    HIGH_LEVEL_ACTIONS,
    MAX_EPISODE_STEPS,
    STEP_ALIVE_REWARD,
    SURVIVE_DAY_BONUS,
)


class _PlainEnv:
    """gym 不存在时的最小包装器基类。"""

    def __init__(self, env):
        self.env = env
        self.action_space = getattr(env, "action_space", None)
        self.observation_space = getattr(env, "observation_space", None)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def close(self):
        closer = getattr(self.env, "close", None)
        if closer is not None:
            return closer()


_WrapperBase = gym.Wrapper if gym is not None else _PlainEnv


def is_alive(observation) -> bool:
    stats = observation.get("life_stats") or {}
    return bool(stats.get("is_alive", True))


def life_stat(observation, key):
    stats = observation.get("life_stats") or {}
    return stats.get(key)


def action_for_repeat(action, first: bool):
    """重复窗口里，合成、装备、放置、熔炼只在第一刻生效。"""
    if first:
        return action
    repeated = dict(action)
    for key in HIGH_LEVEL_ACTIONS:
        if key in repeated:
            repeated[key] = "none"
    return repeated


class SurviveDayWrapper(_WrapperBase):
    """在 MineRL 环境外包一层生存任务逻辑。

    - 每活过一个环境步 +0.1。
    - 活到时间上限再 +1000。死亡不加这两项。
    - `is_alive` 变为假时立刻结束。MineRL 没有死亡退出 handler。
    - `strip_reward=True` 时对外奖励恒为 0，info 仍记录存活步数。评测使用这个模式。
    """

    def __init__(
        self,
        env,
        action_repeat=1,
        strip_reward=False,
        max_episode_steps=MAX_EPISODE_STEPS,
    ):
        super().__init__(env)
        if action_repeat < 1:
            raise ValueError("action_repeat 必须 >= 1")
        self.action_repeat = int(action_repeat)
        self.strip_reward = bool(strip_reward)
        self.max_episode_steps = int(max_episode_steps)
        self.steps = 0

    def reset(self, **kwargs):
        self.steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        total_reward = 0.0
        survived = False
        done = False
        observation = None
        info = {}
        for index in range(self.action_repeat):
            current = action_for_repeat(action, first=(index == 0))
            observation, reward, done, info = self.env.step(current)
            self.steps += 1
            reward = float(reward)
            alive = is_alive(observation)
            if alive:
                reward += STEP_ALIVE_REWARD
                if done or self.steps >= self.max_episode_steps:
                    done = True
                    survived = self.steps >= self.max_episode_steps
                    if survived:
                        reward += SURVIVE_DAY_BONUS
            else:
                done = True
                survived = False
            total_reward += reward
            if done:
                break
        info = dict(info)
        info["survival_steps"] = self.steps
        info["survived_day"] = survived
        info["life"] = life_stat(observation, "life")
        info["food"] = life_stat(observation, "food")
        info["alive"] = is_alive(observation)
        if self.strip_reward:
            total_reward = 0.0
        return observation, total_reward, done, info

    def close(self):
        closer = getattr(self.env, "close", None)
        if closer is not None:
            return closer()
