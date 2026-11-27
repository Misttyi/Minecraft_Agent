"""不启动 Minecraft，核对包装器、奖励表和评测汇总。"""

import unittest

from survive_day.constants import (
    CRAFT_ITEMS,
    EVAL_SEEDS,
    INVENTORY_ITEMS,
    MAX_EPISODE_STEPS,
    MS_PER_STEP,
    NEARBY_CRAFT_ITEMS,
    NEARBY_SMELT_ITEMS,
    REWARD_SCHEDULE,
    STEP_ALIVE_REWARD,
    SURVIVE_DAY_BONUS,
    VECTOR_OBSERVATION_FIELDS,
)
from survive_day.scoring import check_seeds, median, summarize
from survive_day.wrappers import SurviveDayWrapper, action_for_repeat


class FakeClock:
    """第 t 步之后，若 t > alive_for，则 is_alive 为假。内层不主动结束。"""

    def __init__(self, alive_for, inner_reward=1.0):
        self.alive_for = alive_for
        self.inner_reward = inner_reward
        self.t = 0
        self.actions = []
        self.action_space = None
        self.closed = False
        try:
            from gym.spaces import Box, Dict, Discrete

            self.observation_space = Dict(
                {"life_stats": Dict({"is_alive": Discrete(2), "life": Box(0, 20, (1,)), "food": Box(0, 20, (1,))})}
            )
            self.reward_range = (-float("inf"), float("inf"))
            self.metadata = {}
        except ImportError:
            pass

    def reset(self):
        self.t = 0
        self.actions = []
        return self._observation()

    def _observation(self):
        alive = self.t <= self.alive_for
        return {
            "life_stats": {
                "is_alive": alive,
                "life": 20.0 if alive else 0.0,
                "food": 17 if alive else 0,
            }
        }

    def step(self, action):
        self.actions.append(dict(action))
        self.t += 1
        return self._observation(), self.inner_reward, False, {"source": "fake"}

    def close(self):
        self.closed = True


class SpecificationTests(unittest.TestCase):
    def test_day_length_matches_minecraft_ticks(self):
        self.assertEqual(MAX_EPISODE_STEPS, 24000)
        self.assertEqual(MS_PER_STEP, 50)
        self.assertEqual(MAX_EPISODE_STEPS * MS_PER_STEP, 20 * 60 * 1000)

    def test_eval_seeds_are_the_ten_fixed_worlds(self):
        self.assertEqual(
            EVAL_SEEDS, (11, 23, 47, 83, 149, 211, 307, 401, 503, 607)
        )

    def test_item_rewards_are_once_each_and_match_the_assignment(self):
        rewards = {item["type"]: item["reward"] for item in REWARD_SCHEDULE}
        self.assertEqual(
            rewards,
            {
                "log": 1,
                "planks": 2,
                "stick": 4,
                "crafting_table": 4,
                "wooden_pickaxe": 8,
                "wooden_sword": 8,
                "beef": 16,
                "porkchop": 16,
                "coal": 16,
                "torch": 32,
                "cooked_beef": 64,
                "cooked_porkchop": 64,
            },
        )
        self.assertTrue(all(item["amount"] == 1 for item in REWARD_SCHEDULE))

    def test_craft_lists_do_not_overlap(self):
        self.assertFalse(set(CRAFT_ITEMS) & set(NEARBY_CRAFT_ITEMS) - {"none"})
        self.assertFalse(set(CRAFT_ITEMS) & set(NEARBY_SMELT_ITEMS) - {"none"})
        for item in ("beef", "porkchop", "cooked_beef", "log"):
            self.assertIn(item, INVENTORY_ITEMS)

    def test_vector_features_leave_out_the_image(self):
        flat_names = [name for _group, name in VECTOR_OBSERVATION_FIELDS]
        self.assertNotIn("pov", flat_names)
        self.assertIn("sun_brightness", flat_names)

    def test_unknown_seed_is_rejected_for_grading(self):
        check_seeds(EVAL_SEEDS)
        with self.assertRaises(ValueError):
            check_seeds([11, 999])


class WrapperTests(unittest.TestCase):
    def test_death_ends_the_episode_without_the_survival_bonus(self):
        env = SurviveDayWrapper(FakeClock(alive_for=1), action_repeat=1)
        env.reset()
        _observation, reward, done, info = env.step({"attack": 1, "craft": "planks"})
        self.assertFalse(done)
        self.assertEqual(reward, 1.0 + STEP_ALIVE_REWARD)
        observation, reward, done, info = env.step({"attack": 1, "craft": "stick"})
        self.assertTrue(done)
        self.assertFalse(info["survived_day"])
        self.assertEqual(reward, 1.0)
        self.assertEqual(info["survival_steps"], 2)
        self.assertEqual(info["life"], 0.0)
        self.assertNotEqual(reward, 1.0 + SURVIVE_DAY_BONUS)

    def test_living_until_the_time_limit_adds_the_bonus_once(self):
        env = SurviveDayWrapper(
            FakeClock(alive_for=100), action_repeat=2, max_episode_steps=4
        )
        env.reset()
        _observation, reward, done, info = env.step(
            {"forward": 1, "craft": "planks", "use": 1}
        )
        self.assertFalse(done)
        self.assertEqual(reward, 2 * (1.0 + STEP_ALIVE_REWARD))
        self.assertEqual(env.env.actions[0]["craft"], "planks")
        self.assertEqual(env.env.actions[1]["craft"], "none")
        self.assertEqual(env.env.actions[1]["use"], 1)
        observation, reward, done, info = env.step({"forward": 1, "craft": "stick"})
        self.assertTrue(done)
        self.assertTrue(info["survived_day"])
        self.assertEqual(info["survival_steps"], 4)
        self.assertEqual(reward, 2 * (1.0 + STEP_ALIVE_REWARD) + SURVIVE_DAY_BONUS)
        self.assertEqual(observation["life_stats"]["food"], 17)

    def test_eval_mode_drops_reward_but_keeps_the_step_count(self):
        env = SurviveDayWrapper(
            FakeClock(alive_for=100),
            action_repeat=1,
            strip_reward=True,
            max_episode_steps=2,
        )
        env.reset()
        env.step({"attack": 0})
        _observation, reward, done, info = env.step({"attack": 0})
        self.assertTrue(done)
        self.assertEqual(reward, 0.0)
        self.assertEqual(info["survival_steps"], 2)
        self.assertTrue(info["survived_day"])

    def test_repeat_helper_does_not_mutate_the_original_action(self):
        action = {"craft": "torch", "attack": 1}
        repeated = action_for_repeat(action, first=False)
        self.assertEqual(action["craft"], "torch")
        self.assertEqual(repeated["craft"], "none")
        self.assertEqual(repeated["attack"], 1)
        self.assertEqual(action_for_repeat(action, first=True)["craft"], "torch")

    def test_death_inside_a_repeat_window_does_not_send_more_actions(self):
        env = SurviveDayWrapper(FakeClock(alive_for=0), action_repeat=4)
        env.reset()
        _observation, reward, done, info = env.step({"craft": "planks", "forward": 1})
        self.assertTrue(done)
        self.assertEqual(len(env.env.actions), 1)
        self.assertFalse(info["survived_day"])
        self.assertEqual(reward, 1.0)

    def test_wrapper_closes_the_inner_env(self):
        inner = FakeClock(alive_for=1)
        env = SurviveDayWrapper(inner)
        env.close()
        self.assertTrue(inner.closed)


class ScoringTests(unittest.TestCase):
    def test_median_of_ten_sorted_survival_times(self):
        episodes = [
            {
                "seed": seed,
                "survival_steps": steps,
                "survived_day": steps >= MAX_EPISODE_STEPS,
            }
            for seed, steps in zip(EVAL_SEEDS, range(1000, 11000, 1000))
        ]
        report = summarize(episodes)
        self.assertEqual(report["median_survival_steps"], 5500.0)
        self.assertEqual(report["survived_day_count"], 0)
        self.assertEqual(len(report["seeds"]), 10)

    def test_odd_count_median_picks_the_middle(self):
        self.assertEqual(median([3, 1, 2]), 2.0)


if __name__ == "__main__":
    unittest.main()
