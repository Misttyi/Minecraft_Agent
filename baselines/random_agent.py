"""随机空操作下界。每步调用 action_space.no_op()。"""


def act(observation, env):
    del observation
    return env.action_space.no_op()
