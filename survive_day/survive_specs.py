"""MineRLSurviveDay-v0。

写法对齐 MineRL 0.4.4 的 ObtainDiamond：继承 SimpleEmbodimentEnvSpec，
用 handler 拼出 Gym Dict，再由 herobraine 渲染成 Malmo 任务 XML。
"""

from typing import List

from minerl.herobraine.env_specs.simple_embodiment import SimpleEmbodimentEnvSpec
from minerl.herobraine.hero import handlers
from minerl.herobraine.hero.handler import Handler
from minerl.herobraine.hero.mc import INVERSE_KEYMAP, MS_PER_STEP

from survive_day.constants import (
    CRAFT_ITEMS,
    ENV_ID,
    EQUIP_ITEMS,
    EQUIPPED_OBSERVATION_ITEMS,
    INVENTORY_ITEMS,
    MAX_EPISODE_STEPS,
    MS_PER_STEP as EXPECTED_MS_PER_STEP,
    NEARBY_CRAFT_ITEMS,
    NEARBY_SMELT_ITEMS,
    PLACE_ITEMS,
    REWARD_SCHEDULE,
)
from survive_day.wrappers import SurviveDayWrapper

if MS_PER_STEP != EXPECTED_MS_PER_STEP:
    raise RuntimeError(
        "本作业只支持 minerl 0.4.4：MS_PER_STEP 应为 "
        f"{EXPECTED_MS_PER_STEP}，当前是 {MS_PER_STEP}。"
    )


class SeededWorldGenerator(handlers.DefaultWorldGenerator):
    """给 Malmo 的 DefaultWorldGenerator 补上 seed 属性。

    MineRL 0.4.4 自带的 handler 只写出 forceReset 和 generatorOptions。
    Malmo 在种子为空时会随机生成世界，Gym 的 env.seed() 也改不到地形。
    """

    def __init__(self, seed, force_reset=True, generator_options="{}"):
        super().__init__(force_reset=force_reset, generator_options=generator_options)
        self.seed = str(seed)

    def xml_template(self) -> str:
        return """<DefaultWorldGenerator
            seed="{{ seed }}"
            forceReset="{{ force_reset | string | lower }}"
            generatorOptions='{{ generator_options }}'/>"""


class SurviveDayEnvSpec(SimpleEmbodimentEnvSpec):
    """从清晨开始的生存世界。活过 24000 个游戏刻即过完一个日夜。"""

    def __init__(self, world_seed, resolution=(64, 64)):
        self.world_seed = str(world_seed)
        super().__init__(
            ENV_ID,
            max_episode_steps=MAX_EPISODE_STEPS,
            resolution=resolution,
        )

    def create_observables(self):
        return super().create_observables() + [
            handlers.FlatInventoryObservation(list(INVENTORY_ITEMS)),
            handlers.EquippedItemObservation(
                items=list(EQUIPPED_OBSERVATION_ITEMS),
                _default="air",
                _other="other",
            ),
            handlers.ObservationFromLifeStats(),
            handlers.ObservationFromCurrentLocation(),
        ]

    def create_actionables(self):
        return super().create_actionables() + [
            handlers.KeybasedCommandAction("use", INVERSE_KEYMAP["use"]),
            handlers.CraftAction(list(CRAFT_ITEMS), _other="none", _default="none"),
            handlers.CraftNearbyAction(
                list(NEARBY_CRAFT_ITEMS), _other="none", _default="none"
            ),
            handlers.SmeltItemNearby(
                list(NEARBY_SMELT_ITEMS), _other="none", _default="none"
            ),
            handlers.EquipAction(list(EQUIP_ITEMS), _other="none", _default="none"),
            handlers.PlaceBlock(list(PLACE_ITEMS), _other="none", _default="none"),
        ]

    def create_rewardables(self) -> List[Handler]:
        return [handlers.RewardForCollectingItemsOnce(list(REWARD_SCHEDULE))]

    def create_agent_start(self) -> List[Handler]:
        return []

    def create_agent_handlers(self) -> List[Handler]:
        # 死亡退出放在 SurviveDayWrapper 里。这里不挂“拥有某物品就结束”。
        return []

    def create_server_world_generators(self) -> List[Handler]:
        return [SeededWorldGenerator(self.world_seed, force_reset=True)]

    def create_server_quit_producers(self) -> List[Handler]:
        return [
            handlers.ServerQuitFromTimeUp(
                time_limit_ms=MAX_EPISODE_STEPS * MS_PER_STEP
            ),
            handlers.ServerQuitWhenAnyAgentFinishes(),
        ]

    def create_server_decorators(self) -> List[Handler]:
        return []

    def create_server_initial_conditions(self) -> List[Handler]:
        return [
            handlers.TimeInitialCondition(allow_passage_of_time=True, start_time=0),
            handlers.WeatherInitialCondition("clear"),
            handlers.SpawningInitialCondition(allow_spawning=True),
        ]

    def is_from_folder(self, folder: str) -> bool:
        return False

    def determine_success_from_rewards(self, rewards: list) -> bool:
        # 通关看是否活到时间上限，不看物品奖励总和。
        return False

    def get_docstring(self):
        return (
            "在随机生存世界里从清晨活过一个 Minecraft 日夜（24000 刻）。"
            "死亡即结束。评测种子见 survive_day.constants.EVAL_SEEDS。"
        )


def make_env(world_seed, action_repeat=1, strip_reward=False):
    """创建一个带死亡判定和存活奖励的环境。

    直接用 EnvSpec.make()，不经过 gym.make 的 TimeLimit。
    24000 刻的限制由 Malmo 的 ServerQuitFromTimeUp 和包装器共同保证。
    每个种子单独构造规格，因为世界生成器在规格初始化时就把种子写进 XML。
    """
    spec = SurviveDayEnvSpec(world_seed=world_seed)
    env = spec.make()
    return SurviveDayWrapper(
        env,
        action_repeat=action_repeat,
        strip_reward=strip_reward,
        max_episode_steps=MAX_EPISODE_STEPS,
    )
