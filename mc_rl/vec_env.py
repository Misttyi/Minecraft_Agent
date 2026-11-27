"""多进程并行环境。每个子进程各开一个 Minecraft（或 mock），出错时自动重建。

step 时主进程顺带给出“如果这局结束，下一局用哪个任务、哪个种子”，子进程自动重置。
结束时 info 里带 final_obs（用于截断时自举价值）和 episode 汇总；需要回放时带 replay。
"""

import multiprocessing as mp
import os
import time
import traceback

import numpy as np


def _safe_close(env):
    """关闭崩溃的 Minecraft 时自己也可能抛异常：吞掉，避免整个子进程退出、拖垮训练。"""
    try:
        env.close()
    except Exception:
        traceback.print_exc()


def _worker(remote, index, spec):
    os.environ.update(spec.get("env_vars", {}))
    # 几十个 Minecraft 同时启动会抢端口、挤爆 CPU：按编号错开。
    time.sleep(index * spec.get("launch_stagger", 0.0))
    if spec.get("display"):
        os.environ["DISPLAY"] = spec["display"]
    from mc_rl.expert import Expert
    from mc_rl.task_env import TaskEnv, make_base_factory

    factory = make_base_factory(
        spec.get("kind", "minerl"),
        privileged=spec.get("privileged", True),
        action_repeat=spec.get("action_repeat", 8),
        **spec.get("mock_kwargs", {}),
    )
    expert = Expert(seed=1000 + index) if spec.get("expert", True) else None
    env = TaskEnv(factory, action_repeat=spec.get("action_repeat", 8),
                  record=False, expert=expert, record_every=spec.get("record_every", 2))
    record_next = False
    current = None
    last_obs = None  # 最近一次正常的观测：Minecraft 崩溃时当作截断，用它自举价值

    def safe_reset(task, seed):
        nonlocal current
        for attempt in range(5):
            try:
                env.record = record_next
                obs = env.reset(task, seed)
                current = (task, seed)
                return obs
            except Exception:  # Minecraft 偶尔起不来：关掉重开
                traceback.print_exc()
                _safe_close(env)
                time.sleep(5 + 10 * attempt)
        raise RuntimeError(f"worker {index}: 连续 5 次重置失败")

    while True:
        try:
            cmd, data = remote.recv()
        except EOFError:
            break
        if cmd == "reset":
            task, seed, record_next = data
            obs = safe_reset(task, seed)
            last_obs = obs
            remote.send((obs, env.expert_action()))
        elif cmd == "step":
            action, next_task, next_seed, next_record = data
            try:
                obs, reward, done, info = env.step(action)
            except Exception:
                traceback.print_exc()
                _safe_close(env)
                # 崩溃不是这一局真的结束：按截断处理，用崩溃前最后一个观测自举价值。
                info = {"crash": True, "task": current[0] if current else None, "truncated": last_obs is not None}
                if last_obs is not None:
                    info["final_obs"] = last_obs
                obs, reward, done = None, 0.0, True
            if done:
                if obs is not None:
                    info["final_obs"] = obs
                if env.record and obs is not None:
                    info["replay"] = {"frames": np.stack(env.frames), "trace": env.trace,
                                      "episode": info.get("episode"), "seed": env.seed}
                record_next = next_record
                obs = safe_reset(next_task, next_seed)

            last_obs = obs
            remote.send((obs, float(reward), bool(done), info, env.expert_action()))
        elif cmd == "close":
            env.close()
            remote.close()
            break
        else:
            raise ValueError(cmd)


class SubprocVecEnv:
    def __init__(self, num_envs, spec, start_delay=0.0):
        ctx = mp.get_context("spawn")
        self.num_envs = num_envs
        self.remotes, self.processes = [], []
        for index in range(num_envs):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker, args=(child, index, spec), daemon=True)
            proc.start()
            child.close()
            self.remotes.append(parent)
            self.processes.append(proc)
            if start_delay:
                time.sleep(start_delay)

    def reset(self, tasks, seeds, records=None):
        records = records or [False] * self.num_envs
        for remote, task, seed, rec in zip(self.remotes, tasks, seeds, records):
            remote.send(("reset", (task, seed, rec)))
        results = [remote.recv() for remote in self.remotes]
        return [r[0] for r in results], np.array([r[1] for r in results])

    def step(self, actions, next_tasks, next_seeds, next_records):
        for remote, a, t, s, r in zip(self.remotes, actions, next_tasks, next_seeds, next_records):
            remote.send(("step", (int(a), t, s, r)))
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, dones, infos, labels = zip(*results)
        return list(obs), np.array(rewards, dtype=np.float32), np.array(dones), list(infos), np.array(labels)

    # ---- 异步接口：每个环境各走各的，重置只卡住它自己
    def step_one(self, index, action, next_task, next_seed, next_record=False):
        self.remotes[index].send(("step", (int(action), next_task, next_seed, bool(next_record))))

    def poll(self, waiting, timeout=None):
        """waiting: 正在等结果的环境编号集合。返回 [(编号, (obs, reward, done, info, label)), ...]。"""
        from multiprocessing.connection import wait

        lookup = {id(self.remotes[i]): i for i in waiting}
        ready = wait([self.remotes[i] for i in waiting], timeout)
        return [(lookup[id(conn)], conn.recv()) for conn in ready]

    def close(self):
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for proc in self.processes:
            proc.join(timeout=30)
            if proc.is_alive():
                proc.terminate()
