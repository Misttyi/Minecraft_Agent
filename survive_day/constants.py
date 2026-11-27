"""生存一天作业的任务常数。不依赖 MineRL，方便在没有游戏的机器上核对规格。"""

ENV_ID = "MineRLSurviveDay-v0"

# minerl 0.4.4 的 minerl.herobraine.hero.mc.MS_PER_STEP。
# 环境规格在导入 MineRL 后会再断言一次，避免装错版本。
MS_PER_STEP = 50
MAX_EPISODE_STEPS = 24000

EVAL_SEEDS = (11, 23, 47, 83, 149, 211, 307, 401, 503, 607)

STEP_ALIVE_REWARD = 0.1
SURVIVE_DAY_BONUS = 1000.0

# 同一物品只奖一次。数量必须是 1，这是 RewardForCollectingItemsOnce 的限制。
REWARD_SCHEDULE = (
    {"type": "log", "amount": 1, "reward": 1},
    {"type": "planks", "amount": 1, "reward": 2},
    {"type": "stick", "amount": 1, "reward": 4},
    {"type": "crafting_table", "amount": 1, "reward": 4},
    {"type": "wooden_pickaxe", "amount": 1, "reward": 8},
    {"type": "wooden_sword", "amount": 1, "reward": 8},
    {"type": "beef", "amount": 1, "reward": 16},
    {"type": "porkchop", "amount": 1, "reward": 16},
    {"type": "coal", "amount": 1, "reward": 16},
    {"type": "torch", "amount": 1, "reward": 32},
    {"type": "cooked_beef", "amount": 1, "reward": 64},
    {"type": "cooked_porkchop", "amount": 1, "reward": 64},
)

INVENTORY_ITEMS = (
    "log",
    "planks",
    "stick",
    "crafting_table",
    "wooden_pickaxe",
    "wooden_sword",
    "dirt",
    "cobblestone",
    "coal",
    "torch",
    "furnace",
    "beef",
    "porkchop",
    "cooked_beef",
    "cooked_porkchop",
)

EQUIP_ITEMS = (
    "none",
    "air",
    "wooden_pickaxe",
    "wooden_sword",
    "torch",
    "beef",
    "porkchop",
    "cooked_beef",
    "cooked_porkchop",
)

# 主手观测比装备动作多一个 other，用来表示不在枚举里的物品。
EQUIPPED_OBSERVATION_ITEMS = EQUIP_ITEMS + ("other",)

# 个人合成（2×2）与工作台合成必须互斥，否则 MineRL 的 from_universal 会重复计奖。
CRAFT_ITEMS = ("none", "planks", "stick", "crafting_table", "torch")
NEARBY_CRAFT_ITEMS = ("none", "wooden_pickaxe", "wooden_sword", "furnace")
NEARBY_SMELT_ITEMS = ("none", "cooked_beef", "cooked_porkchop")
PLACE_ITEMS = ("none", "dirt", "cobblestone", "crafting_table", "furnace", "torch")

# 动作重复时，只有这些高层动作在窗口的第一刻发送。
HIGH_LEVEL_ACTIONS = ("craft", "equip", "place", "nearbyCraft", "nearbySmelt")

# 向量 PPO 使用的标量。pov 是三维图像，不能放进这个列表。
VECTOR_OBSERVATION_FIELDS = (
    ("life_stats", "life"),
    ("life_stats", "food"),
    ("life_stats", "saturation"),
    ("location_stats", "sun_brightness"),
    ("location_stats", "light_level"),
    ("location_stats", "can_see_sky"),
    ("location_stats", "ypos"),
)
