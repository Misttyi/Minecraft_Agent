"""mc_rl 的离线测试：只用替身环境，不启动 Minecraft。"""

import os
import tempfile
import unittest

import numpy as np

from mc_rl.expert import Expert, target_angles
from mc_rl.features import EpisodeMemory, snapshot, vector_size
from mc_rl.macros import MACRO_INDEX, MACROS, action_mask, expand
from mc_rl.mock_env import MockSurviveEnv
from mc_rl.task_env import TaskEnv, make_base_factory
from mc_rl.tasks import TASK_NAMES, TASKS, TaskTracker, classify_death, get_task, score_tables_markdown


def run_expert(task, seed=3, limit=4000):
    env = TaskEnv(make_base_factory("mock"), expert=Expert(seed=seed), record=True)
    env.reset(task, seed)
    done = False
    info = {}
    steps = 0
    while not done and steps < limit:
        _obs, _r, done, info = env.step(env.expert_action())
        steps += 1
    return env, info


def solution(test):
    """依赖学生实现的测试：学生版里遇到 NotImplementedError 就跳过。"""
    import functools

    @functools.wraps(test)
    def wrapper(self, *args, **kwargs):
        try:
            return test(self, *args, **kwargs)
        except NotImplementedError as error:
            self.skipTest(f"需要学生实现：{error}")

    return wrapper


class ScoreTableTest(unittest.TestCase):
    def test_every_table_sums_to_100(self):
        for task in TASKS:
            self.assertAlmostEqual(sum(m.points for m in task.milestones), 100.0, msg=task.name)

    def test_extension_tasks_keep_core_layout(self):
        from mc_rl.features import core_vector_size, vector_size
        from mc_rl.macros import MACROS
        from mc_rl.tasks import ALL_TASK_NAMES, EXTENSION_TASKS, TASK_INDEX

        self.assertEqual(ALL_TASK_NAMES[:len(TASK_NAMES)], TASK_NAMES)
        self.assertEqual(TASK_INDEX["survive_day"], 8)
        self.assertEqual(MACROS[28], "place_torch")  # 旧动作编号不变
        self.assertEqual(core_vector_size(), 52)
        self.assertGreater(vector_size(), 52)
        for task in EXTENSION_TASKS:
            self.assertAlmostEqual(sum(m.points for m in task.milestones), 100.0)

    def test_task_of_vectors_old_and_new_layout(self):
        from mc_rl.features import core_vector_size, task_of_vectors, vectorize
        from mc_rl.tasks import ALL_TASK_NAMES, TASK_INDEX

        mem = EpisodeMemory()
        obs = MockSurviveEnv(seed=1).reset()
        state = snapshot(obs, mem)
        state.update(dist=60.0, xp=50.0, ticks=15000.0, furnace_near=1.0)  # 让额外特征取较大值
        rows = np.stack([vectorize(state, TASK_INDEX[name]) for name in ALL_TASK_NAMES])
        np.testing.assert_array_equal(task_of_vectors(rows), [TASK_INDEX[n] for n in ALL_TASK_NAMES])
        core = rows[:len(TASK_NAMES), :core_vector_size()]  # 旧数据只有核心长度
        np.testing.assert_array_equal(task_of_vectors(core), np.arange(len(TASK_NAMES)))

    def test_load_demos_drops_tasks_in_full_layout(self):
        from mc_rl.features import vectorize
        from mc_rl.imitation import load_demos
        from mc_rl.macros import NUM_MACROS
        from mc_rl.tasks import TASK_INDEX

        mem = EpisodeMemory()
        state = snapshot(MockSurviveEnv(seed=1).reset(), mem)
        state.update(ticks=15000.0, furnace_near=1.0)
        names = ["hunt", "eat", "craft", "nightfall", "homing"]
        vec = np.stack([vectorize(state, TASK_INDEX[n]) for n in names])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "d.npz")
            np.savez(path, vec=vec, pov=np.zeros((len(names), 64, 64, 3), np.uint8),
                     mask=np.ones((len(names), NUM_MACROS), bool), labels=np.arange(len(names)))
            data = load_demos([path], drop_tasks=("hunt", "eat"), drop_from=[path])
        self.assertEqual(list(data["labels"]), [2, 3, 4])

    def test_old_checkpoint_upgrade_keeps_policy(self):
        try:
            import torch
        except ImportError:
            self.skipTest("没有 torch")
        from mc_rl.models import ActorCritic, load_checkpoint, save_checkpoint, to_tensors
        from mc_rl.task_env import as_batch

        old = ActorCritic(52, 29, arch="cnn")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "old.pt")
            save_checkpoint(path, old)
            same, _ = load_checkpoint(path)
            up, _ = load_checkpoint(path, upgrade=True)
        env = TaskEnv(make_base_factory("mock"))
        batch = to_tensors(as_batch([env.reset("wood", 1)]), "cpu")
        with torch.no_grad():
            a, _ = same(batch)
            b, _ = up(batch)
        pa, pb = torch.softmax(a, -1).numpy(), torch.softmax(b, -1).numpy()
        np.testing.assert_allclose(pa, pb, atol=2e-3)  # 新增且合法的动作（sprint_jump）初始概率 < 0.1%
        self.assertEqual(int(pa.argmax()), int(pb.argmax()))

    def test_markdown_lists_all_tasks(self):
        text = score_tables_markdown()
        for name in TASK_NAMES:
            self.assertIn(f"`{name}`", text)

    def test_prerequisites_block_out_of_order_milestones(self):
        task = get_task("craft")
        tracker = TaskTracker(task)
        mem = EpisodeMemory()
        env = MockSurviveEnv(seed=1)
        obs = env.reset()
        mem.observe(snapshot(obs))
        state = snapshot(obs, mem)
        tracker.start(state, mem)
        # 背包里凭空出现工作台，但没有原木和木板：不该给分。
        fake = dict(state, crafting_table=1.0)
        _reward, newly = tracker.update(fake, "noop", 8)
        self.assertNotIn("table", newly)

    def test_milestones_paid_once(self):
        task = get_task("wood")
        tracker = TaskTracker(task)
        mem = EpisodeMemory()
        obs = MockSurviveEnv(seed=1).reset()
        mem.observe(snapshot(obs))
        state = snapshot(obs, mem)
        tracker.start(state, mem)
        first, newly = tracker.update(dict(state, log=1.0), "chop", 48)
        self.assertEqual(newly, ["log1"])
        again, newly = tracker.update(dict(state, log=1.0), "chop", 48)
        self.assertEqual(newly, [])
        self.assertGreater(first, again)

    def test_death_cause(self):
        base = {"food": 10, "air": 300, "y": 60, "life": 10, "dy": 0, "sky": 1, "ticks": 5000, "sun": 1.0}
        self.assertEqual(classify_death([dict(base, food=0, life=0)]), "starvation")
        fall = [dict(base, y=70, life=10), dict(base, y=60, life=0)]
        self.assertEqual(classify_death(fall), "fall")
        self.assertEqual(classify_death([dict(base, ticks=15000, sun=0.0, life=0)]), "night_monster")
        self.assertEqual(classify_death([dict(base, dy=-10, sky=0, life=0)]), "cave")


class MacroTest(unittest.TestCase):
    def test_mask_blocks_impossible_crafts(self):
        obs = MockSurviveEnv(seed=0).reset()
        state = snapshot(obs, EpisodeMemory())
        mask = action_mask(state)
        for name in ("craft_planks", "craft_table", "place_table", "eat", "craft_sword", "seal_above"):
            self.assertFalse(mask[MACRO_INDEX[name]], name)
        for name in ("forward", "chop", "turn_left", "dig_down"):
            self.assertTrue(mask[MACRO_INDEX[name]], name)

    def test_camera_split_over_repeat(self):
        state = {"pitch": 0.0}
        steps = expand("turn_left", state, 8)
        self.assertAlmostEqual(float(steps[0]["camera"][1]) * 8, -15.0)
        self.assertEqual(len(expand("eat", state, 8)), 5)

    def test_angles(self):
        yaw, pitch, horiz = target_angles(0.0, 1.62, 3.0)
        self.assertAlmostEqual(yaw, 0.0)
        self.assertAlmostEqual(pitch, 0.0)
        yaw, _, _ = target_angles(-3.0, 0.0, 0.0)
        self.assertAlmostEqual(yaw, 90.0)


class TaskEnvTest(unittest.TestCase):
    def test_observation_shapes(self):
        env = TaskEnv(make_base_factory("mock"))
        obs = env.reset("wood", 0)
        self.assertEqual(obs["vec"].shape, (vector_size(len(TASKS)),))
        self.assertEqual(obs["pov"].shape, (64, 64, 3))
        self.assertEqual(obs["mask"].shape, (len(MACROS),))

    def test_expert_solves_easy_tasks_in_mock(self):
        for task in ("wood", "craft", "tools", "hunt", "eat", "shelter"):
            env, info = run_expert(task)
            self.assertIn("episode", info, task)
            self.assertTrue(info["episode"]["success"], (task, info["episode"]))

    def test_expert_solves_extension_tasks_in_mock(self):
        for task in ("homing", "cook"):
            env, info = run_expert(task, limit=3000)
            self.assertIn("episode", info, task)
            self.assertTrue(info["episode"]["success"], (task, info["episode"]))

    @solution
    def test_expert_survives_mock_day(self):
        env, info = run_expert("survive_day", limit=6000)
        episode = info["episode"]
        self.assertGreaterEqual(episode["survival_steps"], 23900, episode)
        self.assertGreaterEqual(episode["score"], 80, episode)

    def test_invalid_action_becomes_noop(self):
        env = TaskEnv(make_base_factory("mock"))
        env.reset("craft", 0)
        _obs, _r, _d, info = env.step(MACRO_INDEX["craft_table"])
        self.assertEqual(info["macro"], "noop")


class MyTaskTest(unittest.TestCase):
    def test_my_task_is_a_valid_score_table(self):
        self.skipTest("自创任务为可选扩展接口，不要求实现，不计入评分")


class AgentInterfaceTest(unittest.TestCase):
    def test_new_env_resets_memory(self):
        from mc_rl.agents import MacroAgent

        class Sneak(MacroAgent):
            def choose(self, observation, state):
                return "sneak"

        agent = Sneak()
        for seed in (1, 2):
            env = MockSurviveEnv(seed=seed, max_episode_steps=400)
            obs = env.reset()
            done = False
            while not done:
                obs, _r, done, _info = env.step(agent(obs, env))
            self.assertLessEqual(agent.mem.ticks, 400 + 8)
            self.assertAlmostEqual(agent.mem.spawn[0], 0.5)

    def test_action_repeat_one_matches_eight(self):
        from mc_rl.agents import MacroAgent

        class Turn(MacroAgent):
            def choose(self, observation, state):
                return "turn_left"

        yaws = []
        for repeat in (1, 8):
            agent = Turn()
            env = MockSurviveEnv(seed=3, action_repeat=repeat, max_episode_steps=64)
            obs = env.reset()
            start = env.yaw
            done = False
            while not done:
                obs, _r, done, _info = env.step(agent(obs, env))
            yaws.append((env.yaw - start + 180) % 360 - 180)
        self.assertAlmostEqual(yaws[0], yaws[1], places=3)


class PPOMathTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("没有 torch")

    @solution
    def test_gae_matches_hand_computation(self):
        from mc_rl.ppo import compute_gae

        adv, ret = compute_gae([1.0, 0.0, 2.0], [0.5, 0.5, 0.5], [0, 1, 0], last_value=1.0, gamma=0.9, lam=0.8)
        # t=2: 2 + 0.9*1 - 0.5 = 2.4；t=1 是终止：0 - 0.5 = -0.5；t=0: 1 + 0.45 - 0.5 + 0.72*(-0.5) = 0.59
        np.testing.assert_allclose(adv, [0.59, -0.5, 2.4], atol=1e-5)
        np.testing.assert_allclose(ret, np.array([0.59, -0.5, 2.4]) + 0.5, atol=1e-5)

    @solution
    def test_ppo_loss_is_finite_with_single_sample(self):
        import torch

        from mc_rl.ppo import ppo_loss

        logits = torch.zeros(1, 4, requires_grad=True)
        mask = torch.ones(1, 4, dtype=torch.bool)
        loss, stats = ppo_loss(logits, torch.zeros(1), mask, torch.tensor([1]), torch.tensor([-1.386]),
                               torch.tensor([1.0]), torch.tensor([0.5]), torch.tensor([2]), 0.2, 0.5, 0.01, 1.0)
        self.assertTrue(torch.isfinite(loss))
        for key in ("pg_loss", "vf_loss", "entropy", "approx_kl", "clip_frac", "bc_loss"):
            self.assertIn(key, stats)

    def test_best_checkpoint_selection_tracks_survive_day(self):
        from mc_rl.ppo import Curriculum, selection_score

        cur = Curriculum(list(TASK_NAMES), seed=0)
        per_task = {"wood": {"score": 90.0, "n": 10}}
        self.assertIsNone(selection_score(per_task, list(TASK_NAMES), cur))
        per_task["survive_day"] = {"score": 40.0, "n": 5}
        self.assertEqual(selection_score(per_task, list(TASK_NAMES), cur), 40.0)


class CurriculumTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("没有 torch")

    def test_stages_unlock_in_order(self):
        from mc_rl.ppo import Curriculum

        cur = Curriculum(list(TASK_NAMES), seed=0)
        probs = dict(zip(cur.names, cur.probs()))
        self.assertEqual(cur.unlocked, 0)
        self.assertEqual(probs["survive_day"], 0.0)
        self.assertGreater(probs["wood"], 0.0)
        for _ in range(5):
            cur.update("explore", True, 100)
            cur.update("wood", True, 100)
        self.assertEqual(cur.unlocked, 1)
        probs = dict(zip(cur.names, cur.probs()))
        self.assertGreater(probs["shelter"], 0.0)
        self.assertGreater(probs["wood"], 0.0)  # 早期阶段保留采样
        self.assertEqual(probs["nightfall"], 0.0)

    def test_partial_mastery_by_score_unlocks(self):
        from mc_rl.ppo import Curriculum

        cur = Curriculum(["tools", "hunt", "eat", "nightfall"], seed=0)
        for _ in range(4):
            cur.update("tools", True, 100)
            cur.update("hunt", False, 50)  # 拿到前几个里程碑但没成功
            cur.update("eat", False, 45)
        self.assertEqual(cur.unlocked, 3)

    def test_episode_cap_prevents_deadlock(self):
        from mc_rl.ppo import Curriculum
        from mc_rl.tasks import STAGE_UNLOCK_EPISODES

        cur = Curriculum(["hunt", "eat", "nightfall"], seed=0)
        self.assertEqual(cur.unlocked, 2)
        for _ in range(STAGE_UNLOCK_EPISODES * 3):
            cur.update("hunt", False, 0)
        self.assertEqual(cur.unlocked, 2)  # 上限按任务计：只练了打猎不算
        for _ in range(STAGE_UNLOCK_EPISODES):
            cur.update("eat", False, 0)
        self.assertEqual(cur.unlocked, 3)

    def test_sealed_only_inside_own_shaft(self):
        tracker = TaskTracker(get_task("shelter"))
        mem = EpisodeMemory()
        mem.sealed = 1
        base = {"sky": 0.0, "x": 0.5, "y": 60.0, "z": 0.5}
        tracker.mem = mem
        tracker.dig_anchor = (0.5, 63.0, 0.5)
        tracker.shaft_depth = 3.0
        tracker.state = dict(base)
        from mc_rl.tasks import sealed
        self.assertTrue(sealed(tracker))
        tracker.state = dict(base, x=6.5, y=60.0)  # 从竖井横着挖出去的洞里
        self.assertFalse(sealed(tracker))
        tracker.state = dict(base, y=63.0)  # 回到地面（树下 / 悬崖下）
        self.assertFalse(sealed(tracker))


class LearningTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("没有 torch")

    @solution
    def test_bc_then_ppo_smoke(self):
        from mc_rl.imitation import collect, train_bc
        from mc_rl.ppo import train_ppo
        from mc_rl.render import render_all_replays, render_dashboard
        from mc_rl.vec_env import SubprocVecEnv

        spec = {"kind": "mock", "privileged": True, "expert": True, "action_repeat": 8}
        vec = SubprocVecEnv(2, spec)
        with tempfile.TemporaryDirectory() as tmp:
            try:
                data, episodes = collect(vec, None, 120, ["wood", "craft"], [1, 2, 3], run_dir=tmp,
                                         log_fn=lambda *_: None)
                model, history = train_bc(data, arch="cnn", epochs=3, device="cpu", log_fn=lambda *_: None)
                self.assertLess(history[-1]["loss"], history[0]["loss"])  # 小样本验证集太噪，看训练损失下降
                cfg = {"num_envs": 2, "rollout_steps": 32, "minibatch": 32, "epochs": 1, "total_ticks": 1500,
                       "tasks": "wood,craft", "train_seeds": "1-5", "replay_every": 1, "save_every": 1}
                train_ppo(vec, cfg, tmp, init=model, device="cpu", log_fn=lambda *_: None)
            finally:
                vec.close()
            self.assertTrue(os.path.exists(os.path.join(tmp, "last.pt")))
            pages = render_all_replays(tmp)
            self.assertTrue(pages)
            out = render_dashboard([tmp], os.path.join(tmp, "dash.html"))
            with open(out, encoding="utf-8") as handle:
                self.assertIn("学习曲线", handle.read())

    @solution
    def test_policy_agent_matches_eval_interface(self):
        from mc_rl.agents import DaytimeScript, PolicyAgent
        from mc_rl.features import vector_size as vs
        from mc_rl.macros import NUM_MACROS
        from mc_rl.models import ActorCritic

        model = ActorCritic(vs(len(TASKS)), NUM_MACROS, arch="cnn")
        for agent in (PolicyAgent(model), DaytimeScript()):
            env = MockSurviveEnv(seed=2, max_episode_steps=800)
            obs = env.reset()
            done = False
            while not done:
                obs, _r, done, info = env.step(agent(obs, env))
            self.assertGreaterEqual(info["survival_steps"], 800)


if __name__ == "__main__":
    unittest.main()
