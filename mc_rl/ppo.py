"""PPO（带模仿辅助损失和自适应课程）。

- 截断（任务时长到了）时用 final_obs 的价值自举；死亡和任务完成按终止处理。
- 每个样本都附带专家标签（子进程里的特权专家给出）。bc_coef > 0 时加交叉熵，
  并按 bc_decay 逐次衰减：前期跟着老师走，后期由任务得分表的奖励接管。
- 课程：每个任务维护成功率滑动平均，采样概率 ∝ 权重 × (下限 + 4p(1-p))，
  学会的和完全不会的都少采，正在学的多采。
"""

import json
import os
import time
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn.functional as F

from mc_rl.models import ActorCritic, save_checkpoint, to_tensors
from mc_rl.render import save_replay  # noqa: F401  （旧接口）
from mc_rl.task_env import as_batch
from mc_rl.tasks import TASK_NAMES, TASKS, get_task

DEFAULTS = dict(
    num_envs=8,
    rollout_steps=256,  # 每个环境每次更新的决策步
    epochs=4,
    minibatch=256,
    gamma=0.99,
    lam=0.95,
    clip=0.2,
    lr=3e-4,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    bc_coef=0.5,
    bc_decay=0.97,
    bc_min=0.0,
    total_ticks=2_000_000,
    tasks=",".join(TASK_NAMES),
    curriculum_floor=0.15,
    arch="cnn",
    seed=0,
    train_seeds="1000-1999",
    save_every=10,
    replay_every=20,
    action_repeat=8,
    normalize_reward=True,
    critic_warmup=3,  # 从 BC 初始化时，前几次更新只训练价值头（BC 不训练价值头，随机价值会把策略带偏）
    target_kl=0.03,  # 近似 KL 超过就提前结束本次更新
    staged=True,  # 按阶段由易到难解锁任务
    start_stage=0,
)


def parse_seeds(text):
    if "-" in str(text):
        lo, hi = str(text).split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in str(text).split(",") if s]


class Curriculum:
    """任务采样。

    staged=True（默认）：按 tasks.py 的阶段由易到难解锁。只采已解锁阶段的任务；当前前沿阶段权重 ×frontier_boost，
    已掌握的早期阶段保留一部分采样防止遗忘。阶段内按 权重 × (下限 + 4p(1-p)) 偏向“正在学”的任务。
    解锁条件：前沿阶段各任务成功率均值 ≥ STAGE_UNLOCK_SUCCESS，或得分均值 ≥ STAGE_UNLOCK_SCORE，
    或该阶段每个任务都跑满 STAGE_UNLOCK_EPISODES 局（按任务计，短任务不会替长任务凑数）。
    fixed={任务: 权重} 时不做课程，按固定权重采样。
    """

    def __init__(self, names, floor=0.15, seed=0, fixed=None, staged=True, start_stage=0, frontier_boost=2.0):
        from mc_rl.tasks import STAGE_UNLOCK_EPISODES, STAGE_UNLOCK_SCORE, STAGE_UNLOCK_SUCCESS

        self.names = list(names)
        self.floor = floor
        self.rate = {name: 0.0 for name in self.names}
        self.score = {name: 0.0 for name in self.names}
        self.count = defaultdict(int)
        self.rng = np.random.default_rng(seed)
        self.fixed = fixed
        self.staged = staged and not fixed
        self.stages = sorted({get_task(n).stage for n in self.names})
        self.unlocked = max([st for st in self.stages if st <= start_stage] or [self.stages[0]])
        self.frontier_boost = frontier_boost
        self.unlock_success = STAGE_UNLOCK_SUCCESS
        self.unlock_score = STAGE_UNLOCK_SCORE
        self.unlock_episodes = STAGE_UNLOCK_EPISODES
        self.history = []  # (累计局数, 解锁到的阶段)

    def _stage_tasks(self, stage):
        return [n for n in self.names if get_task(n).stage == stage]

    def probs(self):
        if self.fixed:
            weights = np.array([self.fixed.get(n, 0.0) for n in self.names], dtype=np.float64)
            if weights.sum() <= 0:
                raise ValueError(f"task_weights {self.fixed} 没有覆盖 tasks {self.names}")
            return weights / weights.sum()
        weights = []
        for name in self.names:
            task = get_task(name)
            if self.staged and task.stage > self.unlocked:
                weights.append(0.0)
                continue
            p = self.rate[name]
            w = task.weight * (self.floor + 4.0 * p * (1.0 - p))
            if self.staged and task.stage == self.unlocked:
                w *= self.frontier_boost
            weights.append(w)
        weights = np.array(weights, dtype=np.float64)
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            weights = np.array([0.0 if (self.staged and get_task(n).stage > self.unlocked) else 1.0
                                for n in self.names])
        return weights / weights.sum()

    def sample(self):
        return str(self.rng.choice(self.names, p=self.probs()))

    def _maybe_unlock(self):
        if not self.staged:
            return
        later = [st for st in self.stages if st > self.unlocked]
        if not later:
            return
        frontier = self._stage_tasks(self.unlocked)
        played = min(self.count[n] for n in frontier)
        seen_all = all(self.count[n] >= 3 for n in frontier)
        mean_success = float(np.mean([self.rate[n] for n in frontier]))
        mean_score = float(np.mean([self.score[n] for n in frontier]))
        mastered = seen_all and (mean_success >= self.unlock_success or mean_score >= self.unlock_score)
        if mastered or played >= self.unlock_episodes:
            self.unlocked = later[0]
            self.history.append((int(sum(self.count.values())), self.unlocked))

    def update(self, name, success, score, alpha=0.1):
        self.count[name] += 1
        a = max(alpha, 1.0 / self.count[name])
        self.rate[name] += a * (float(success) - self.rate[name])
        self.score[name] += a * (float(score) - self.score[name])
        self._maybe_unlock()

    def state(self):
        return {"rate": dict(self.rate), "score": dict(self.score), "count": dict(self.count),
                "probs": dict(zip(self.names, map(float, self.probs()))),
                "stage": self.unlocked if self.staged else None, "unlock_history": list(self.history)}


class RunningStd:
    def __init__(self):
        self.count, self.mean, self.var = 1e-4, 0.0, 1.0

    def update(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        batch_mean, batch_var, n = x.mean(), x.var(), x.size
        delta = batch_mean - self.mean
        total = self.count + n
        self.mean += delta * n / total
        self.var = (self.var * self.count + batch_var * n + delta ** 2 * self.count * n / total) / total
        self.count = total

    @property
    def std(self):
        return float(np.sqrt(self.var) + 1e-8)


class Logger:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.metrics = open(os.path.join(run_dir, "metrics.jsonl"), "a")
        self.episodes = open(os.path.join(run_dir, "episodes.jsonl"), "a")

    def metric(self, row):
        self.metrics.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.metrics.flush()

    def episode(self, row):
        self.episodes.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.episodes.flush()


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """单个环境一段轨迹的 GAE。

    rewards/values/dones: 长度 T 的数组；dones[t]=1 表示第 t 步之后这一局结束（截断时奖励里已经加过自举价值）。
    last_value: 第 T 步（轨迹之后）那个观测的价值估计。返回 (advantages, returns)。
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    raise NotImplementedError("TODO(学生)：按时间倒序计算 delta_t = r_t + γ V(s_{t+1})(1-done_t) - V(s_t)，A_t = delta_t + γλ(1-done_t)A_{t+1}；returns = A + V")


def ppo_loss(logits, value, mask, actions, old_logp, adv, ret, labels, clip, vf_coef, ent_coef, bc_coef):
    """PPO 的一个小批量损失。返回 (总损失, 统计字典)。

    logits 已经被掩码处理过（非法动作是很大的负数）；adv 是原始优势（在函数里做标准化）；
    labels 是专家动作，-1 表示没有标签；bc_coef > 0 时加模仿交叉熵。
    统计字典至少要有 pg_loss / vf_loss / entropy / approx_kl / clip_frac，有模仿项时加 bc_loss / bc_acc。
    """
    raise NotImplementedError("TODO(学生)：裁剪策略目标 + 价值 MSE − 熵（只对合法动作）+ 可选 BC 交叉熵；小批量只有 1 个样本时不要除以 std")


def ppo_update(model, optimizer, data, cfg, bc_coef, device, value_only=False):
    """多轮小批量更新。value_only=True 时只训练价值头（共享特征 detach），用于从 BC 初始化后的预热。
    近似 KL 超过 1.5×target_kl 时跳过这一步并结束本次更新。"""
    n = len(data["actions"])
    idx = np.arange(n)
    stats = defaultdict(list)
    for _ in range(cfg["epochs"]):
        np.random.shuffle(idx)
        stop = False
        for start in range(0, n, cfg["minibatch"]):
            mb = idx[start:start + cfg["minibatch"]]
            if len(mb) < 2:
                continue
            obs = to_tensors({k: data[k][mb] for k in ("vec", "pov", "mask")}, device)
            actions = torch.as_tensor(data["actions"][mb], device=device)
            old_logp = torch.as_tensor(data["logp"][mb], device=device)
            adv = torch.as_tensor(data["adv"][mb], device=device)
            ret = torch.as_tensor(data["ret"][mb], device=device)
            labels = torch.as_tensor(data["labels"][mb], device=device)
            optimizer.zero_grad()
            if value_only:
                with torch.no_grad():
                    feats = model.features(obs)
                value = model.v(feats).squeeze(-1)
                vf = F.mse_loss(value, ret)
                (cfg["vf_coef"] * vf).backward()
                stats["vf_loss"].append(float(vf.detach()))
            else:
                logits, value = model(obs)
                loss, part = ppo_loss(logits, value, obs["mask"], actions, old_logp, adv, ret, labels,
                                      cfg["clip"], cfg["vf_coef"], cfg["ent_coef"], bc_coef)
                for key, val in part.items():
                    stats[key].append(val)
                if cfg.get("target_kl") and part["approx_kl"] > 1.5 * cfg["target_kl"]:
                    stats["early_stop"].append(1.0)
                    stop = True
                    break
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            optimizer.step()
        if stop:
            break
    return {k: float(np.mean(v)) for k, v in stats.items()}


def train_ppo(vec, cfg, run_dir, init=None, device="cuda", log_fn=print):
    cfg = {**DEFAULTS, **cfg}
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    names = [n for n in cfg["tasks"].split(",") if n]
    fixed = cfg.get("task_weights")
    curriculum = Curriculum(names, floor=cfg["curriculum_floor"], seed=cfg["seed"], fixed=fixed,
                            staged=cfg["staged"], start_stage=cfg["start_stage"])
    seeds = parse_seeds(cfg["train_seeds"])
    rng = np.random.default_rng(cfg["seed"] + 1)
    from mc_rl.features import vector_size
    from mc_rl.macros import NUM_MACROS

    if init is not None:
        model = init.to(device)
    else:
        model = ActorCritic(vector_size(len(TASKS)), NUM_MACROS, arch=cfg["arch"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], eps=1e-5)
    logger = Logger(run_dir)
    with open(os.path.join(run_dir, "config.json"), "w") as handle:
        json.dump(cfg, handle, indent=2, ensure_ascii=False)
    n_env, T = vec.num_envs, cfg["rollout_steps"]
    ret_std = RunningStd()
    running_ret = np.zeros(n_env)
    episodes_done = 0
    env0_episodes = 0
    prev_ticks = np.zeros(n_env, dtype=np.int64)

    tasks = [curriculum.sample() for _ in range(n_env)]
    records = [bool(cfg["replay_every"])] + [False] * (n_env - 1)
    obs, labels = vec.reset(tasks, [int(rng.choice(seeds)) for _ in range(n_env)], records)
    total_ticks = 0
    update = 0
    bc_coef = cfg["bc_coef"]
    recent = defaultdict(lambda: deque(maxlen=50))
    best_score = -1.0
    start = time.time()
    obs = list(obs)
    labels = list(labels)
    while total_ticks < cfg["total_ticks"]:
        # ---- 异步采样：每个环境各采 T 步。谁先回来先算谁的动作，Minecraft 重置只卡住它自己。
        buf = [defaultdict(list) for _ in range(n_env)]
        pending = {}  # 环境 -> 发出动作时的 (vec, pov, mask, action, logp, value, label)
        counts = np.zeros(n_env, dtype=np.int64)

        def send(indices):
            batch = as_batch([obs[i] for i in indices])
            with torch.no_grad():
                action, logp, value = model.act(to_tensors(batch, device))
            action, logp, value = action.cpu().numpy(), logp.cpu().numpy(), value.cpu().numpy()
            for j, i in enumerate(indices):
                pending[i] = (batch["vec"][j], batch["pov"][j], batch["mask"][j], action[j], logp[j], value[j],
                              labels[i])
                record = bool(cfg["replay_every"]) and i == 0 and (env0_episodes + 1) % cfg["replay_every"] == 0
                vec.step_one(i, action[j], curriculum.sample(), int(rng.choice(seeds)), record)

        send(list(range(n_env)))
        while pending:
            again = []
            for i, (new_obs, reward, done, info, new_label) in vec.poll(set(pending)):
                vec_i, pov_i, mask_i, a_i, logp_i, v_i, label_i = pending.pop(i)
                if "ticks" in info:
                    total_ticks += max(0, int(info["ticks"]) - int(prev_ticks[i]))
                    prev_ticks[i] = int(info["ticks"])
                raw = float(reward)
                running_ret[i] = running_ret[i] * cfg["gamma"] + raw
                if cfg["normalize_reward"]:
                    ret_std.update(running_ret[i:i + 1])
                scale = ret_std.std if cfg["normalize_reward"] else 1.0
                r = raw / scale
                if done and info.get("truncated") and "final_obs" in info:
                    with torch.no_grad():
                        _, v_final = model(to_tensors(as_batch([info["final_obs"]]), device))
                    r += cfg["gamma"] * float(v_final)
                b = buf[i]
                for key, value in (("vec", vec_i), ("pov", pov_i), ("mask", mask_i), ("actions", a_i),
                                   ("logp", logp_i), ("values", v_i), ("rewards", r), ("dones", float(done)),
                                   ("labels", label_i)):
                    b[key].append(value)
                if done:
                    prev_ticks[i] = 0
                    running_ret[i] = 0.0
                    if i == 0:
                        env0_episodes += 1
                    ep = info.get("episode")
                    if not info.get("crash") and ep is not None:
                        episodes_done += 1
                        curriculum.update(ep["task"], ep["success"], ep["score"])
                        recent[ep["task"]].append(ep)
                        logger.episode({"update": update, "ticks": total_ticks, **ep})
                        if "replay" in info:
                            path = save_replay(run_dir, f"u{update:04d}_{ep['task']}_{int(ep['score'])}",
                                               info["replay"])
                            log_fn(f"  回放已保存 {path}")
                obs[i], labels[i] = new_obs, new_label
                counts[i] += 1
                if counts[i] < T:
                    again.append(i)
            if again:
                send(again)
        # ---- 逐环境 GAE
        with torch.no_grad():
            _, last_value = model(to_tensors(as_batch(obs), device))
        last_value = last_value.cpu().numpy()
        parts = defaultdict(list)
        for i in range(n_env):
            b = buf[i]
            adv, ret = compute_gae(b["rewards"], b["values"], b["dones"], float(last_value[i]),
                                   cfg["gamma"], cfg["lam"])
            parts["adv"].append(adv)
            parts["ret"].append(ret)
            for key in ("vec", "pov", "mask", "actions", "logp", "labels"):
                parts[key].append(np.asarray(b[key]))
        cat = lambda key: np.concatenate(parts[key])  # noqa: E731
        data = {
            "vec": cat("vec").astype(np.float32), "pov": cat("pov"), "mask": cat("mask").astype(bool),
            "actions": cat("actions").astype(np.int64), "logp": cat("logp").astype(np.float32),
            "adv": cat("adv").astype(np.float32), "ret": cat("ret").astype(np.float32),
            "labels": cat("labels").astype(np.int64),
        }
        warm = init is not None and update < cfg["critic_warmup"]
        # 这批成绩是更新前的策略打出来的：best.pt 存更新前的权重
        acting_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        stats = ppo_update(model, optimizer, data, cfg, bc_coef, device, value_only=warm)
        stats["value_only"] = float(warm)
        update += 1
        if not warm:
            bc_coef = max(cfg["bc_min"], bc_coef * cfg["bc_decay"])
        per_task = {}
        for name in names:
            eps = list(recent[name])
            if eps:
                per_task[name] = {
                    "score": float(np.mean([e["score"] for e in eps])),
                    "success": float(np.mean([e["success"] for e in eps])),
                    "return": float(np.mean([e["return_"] for e in eps])),
                    "survival": float(np.mean([e["survival_steps"] for e in eps])),
                    "n": len(eps),
                }
        mean_score = float(np.mean([v["score"] for v in per_task.values()])) if per_task else 0.0
        row = {
            "update": update, "ticks": total_ticks, "episodes": episodes_done,
            "elapsed": round(time.time() - start, 1), "bc_coef": bc_coef,
            "fps_ticks": round(total_ticks / max(time.time() - start, 1e-6), 1),
            "mean_score": mean_score, "tasks": per_task, "curriculum": curriculum.state(), **stats,
        }
        log_fn(
            f"[{update:4d}] ticks {total_ticks:>9d} eps {episodes_done:5d} score {mean_score:5.1f} "
            f"ent {stats.get('entropy', 0):.2f} kl {stats.get('approx_kl', 0):.3f} "
            f"bc {stats.get('bc_acc', float('nan')):.2f} fps {row['fps_ticks']:.0f} | "
            + " ".join(f"{k}:{v['score']:.0f}/{v['success']:.2f}" for k, v in per_task.items())
        )
        extra = {"update": update, "ticks": total_ticks, "cfg": cfg, "curriculum": curriculum.state()}
        if update % cfg["save_every"] == 0:
            save_checkpoint(os.path.join(run_dir, "last.pt"), model, extra)
        select = selection_score(per_task, names, curriculum)
        row["select_score"] = select
        if select is not None and select > best_score:
            best_score = select
            save_checkpoint(os.path.join(run_dir, "best.pt"), model,
                            {**extra, "mean_score": mean_score, "select_score": select}, state_dict=acting_state)
        logger.metric(row)
    save_checkpoint(os.path.join(run_dir, "last.pt"), model, {"update": update, "ticks": total_ticks, "cfg": cfg})
    return model


def selection_score(per_task, names, curriculum, task="survive_day", min_episodes=3):
    """选 best.pt 的指标，对齐评测目标：有 survive_day 就用它最近若干局的平均任务分（解锁且至少 min_episodes 局）。
    没有 survive_day 的任务集合，等课程解锁到最后阶段（或不分阶段）后用平均任务分。还不能比较时返回 None。"""
    if task in names:
        stat = per_task.get(task)
        if stat and stat["n"] >= min_episodes:
            return float(stat["score"])
        return None
    if curriculum.staged and curriculum.unlocked < max(curriculum.stages):
        return None
    if not per_task or any(n not in per_task for n in names):
        return None
    return float(np.mean([v["score"] for v in per_task.values()]))
