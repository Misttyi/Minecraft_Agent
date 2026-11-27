"""官方口径评测：未改动的 survive_day.make_env(strip_reward=True)，10 个固定种子并行。

输出字段和 survive_day.evaluate 相同（seed / survival_steps / survived_day / life / food），
额外加 death_cause、宏动作直方图，以及可选的回放（给 HTML 用）。
"""

import json
import multiprocessing as mp
import os
import time
from collections import Counter

import numpy as np

from survive_day.constants import EVAL_SEEDS


def _make_agent(kind, checkpoint, task, seed, device):
    from mc_rl.agents import DaytimeScript, PolicyAgent, RandomMacroAgent

    if kind == "policy":
        from mc_rl.models import load_checkpoint

        model, _ = load_checkpoint(checkpoint, device)
        return PolicyAgent(model, task=task, device=device, seed=seed)
    if kind == "random_macro":
        return RandomMacroAgent(task=task, seed=seed)
    if kind == "daytime":
        return DaytimeScript(task=task)
    if kind == "noop":
        return None
    raise ValueError(kind)


def run_eval_episode(args):
    kind, checkpoint, task, seed, action_repeat, record, out_dir, env_kind, env_vars = args
    os.environ.update(env_vars or {})
    from mc_rl.features import EpisodeMemory, pov_array, snapshot
    from mc_rl.tasks import classify_death

    if env_kind == "mock":
        from mc_rl.mock_env import MockSurviveEnv

        env = MockSurviveEnv(seed=seed, action_repeat=action_repeat, privileged=False)
    else:
        from survive_day import make_env

        env = make_env(seed, action_repeat=action_repeat, strip_reward=True)
    agent = _make_agent(kind, checkpoint, task, seed, "cpu")
    history, frames, trace = [], [], []
    mem = EpisodeMemory()
    info = {"survival_steps": 0, "survived_day": False, "life": None, "food": None}
    start = time.time()
    try:
        observation = env.reset()
        mem.observe(snapshot(observation))
        done = False
        step = 0
        while not done:
            action = env.action_space.no_op() if agent is None else agent(observation, env)
            observation, _reward, done, info = env.step(action)
            step += 1
            mem.ticks = int(info["survival_steps"])
            state = snapshot(observation, mem)
            mem.observe(state)
            history.append(state)
            history = history[-8:]
            if record and (step % 2 == 0 or done):
                frames.append(pov_array(observation).copy())
                entry = {k: state.get(k) for k in ("life", "food", "sun", "dy", "ticks", "log", "planks",
                                                   "stick", "crafting_table", "dirt", "meat", "hand", "pitch")}
                entry.update(macro=agent.pending if agent is not None else "noop", reward=0.0, newly=[], score=0)
                trace.append(entry)
    finally:
        env.close()
    died = not info.get("survived_day") and not info.get("alive", True)
    result = {
        "seed": int(seed),
        "survival_steps": int(info["survival_steps"]),
        "survived_day": bool(info["survived_day"]),
        "life": None if info.get("life") is None else float(np.asarray(info["life"]).reshape(-1)[0]),
        "food": None if info.get("food") is None else float(np.asarray(info["food"]).reshape(-1)[0]),
        "death_cause": classify_death(history) if died else None,
        "macro_hist": dict(Counter(getattr(agent, "macros", []))) if agent is not None else {},
        "wall_seconds": round(time.time() - start, 1),
    }
    if record and frames and out_dir:
        from mc_rl.render import save_replay

        episode = {"task": "survive_day", "score": 0, "success": result["survived_day"],
                   "survival_steps": result["survival_steps"], "died": died,
                   "death_cause": result["death_cause"]}
        save_replay(out_dir, f"eval_{kind}_seed{seed}", {"frames": np.stack(frames), "trace": trace,
                                                         "episode": episode, "seed": seed})
    return result


def evaluate_parallel(kind, checkpoint=None, seeds=EVAL_SEEDS, task="survive_day", action_repeat=8,
                      workers=10, record=False, out_dir=None, env_kind="minerl", env_vars=None):
    from survive_day.scoring import check_seeds, summarize

    seeds = tuple(seeds)
    if env_kind != "mock":
        check_seeds(seeds)
    jobs = [(kind, checkpoint, task, s, action_repeat, record, out_dir, env_kind, env_vars) for s in seeds]
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(workers, len(jobs))) as pool:
        episodes = pool.map(run_eval_episode, jobs)
    report = summarize(episodes)
    causes = Counter(e["death_cause"] for e in episodes if e["death_cause"])
    report["death_causes"] = dict(causes)
    report["agent"] = kind
    report["checkpoint"] = checkpoint
    report["action_repeat"] = action_repeat
    return report


def write_report(report, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
