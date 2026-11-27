"""模仿学习：专家示范采集、行为克隆（BC）和 DAgger。

示范只存学生能看到的东西（vec、pov、mask）和专家标签。DAgger 第 k 轮以概率 beta
让专家执行、否则让学生执行，但所有状态都用专家打标签，修正学生自己走偏后的分布。
"""

import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from mc_rl.macros import MACROS, NUM_MACROS
from mc_rl.models import ActorCritic, to_tensors
from mc_rl.ppo import Curriculum, parse_seeds
from mc_rl.render import save_replay
from mc_rl.task_env import as_batch
from mc_rl.tasks import TASKS


def collect(vec, policy, steps, tasks, seeds, beta=1.0, device="cpu", seed=0, out_path=None,
            run_dir=None, log_fn=print, record_first=True, task_weights=None, save_every=600):
    """每个环境跑 steps 个决策步（异步：谁先回来先给谁发动作），返回数据字典和逐局汇总。

    以概率 beta 执行专家动作，否则执行学生动作；标签永远是专家动作。每 save_every 步落盘一次。
    """
    rng = np.random.default_rng(seed)
    curriculum = Curriculum(tasks, seed=seed, fixed=task_weights or {t: 1.0 for t in tasks})
    n_env = vec.num_envs
    cur_tasks = [curriculum.sample() for _ in range(n_env)]
    records = [record_first] + [False] * (n_env - 1)
    obs_list, labels = vec.reset(cur_tasks, [int(rng.choice(seeds)) for _ in range(n_env)], records)
    obs_list = list(obs_list)
    labels = list(labels)
    data = defaultdict(list)
    episodes = []
    counts = np.zeros(n_env, dtype=np.int64)
    record_left = 3 if record_first else 0
    start = time.time()
    total = 0

    def act(indices):
        batch = as_batch([obs_list[i] for i in indices])
        lab = np.array([labels[i] for i in indices])
        use_expert = rng.random(len(indices)) < beta
        if policy is not None and not use_expert.all():
            with torch.no_grad():
                student, _, _ = policy.act(to_tensors(batch, device))
            student = student.cpu().numpy()
        else:
            student = lab
        actions = np.where(use_expert & (lab >= 0), lab, student)
        for j, i in enumerate(indices):
            data["vec"].append(batch["vec"][j])
            data["pov"].append(batch["pov"][j])
            data["mask"].append(batch["mask"][j])
            data["labels"].append(lab[j])
            vec.step_one(i, actions[j], curriculum.sample(), int(rng.choice(seeds)), i < 2 and rng.random() < 0.15)

    def dump():
        if not out_path or not data["labels"]:
            return
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        np.savez_compressed(out_path, **{key: np.stack(value) for key, value in data.items()})
        with open(os.path.splitext(out_path)[0] + "_episodes.jsonl", "w") as handle:
            for ep in episodes:
                handle.write(json.dumps(ep, ensure_ascii=False) + "\n")

    waiting = set(range(n_env))
    act(list(range(n_env)))
    while waiting:
        ready = vec.poll(waiting)
        again = []
        for i, (obs, _reward, done, info, label) in ready:
            waiting.discard(i)
            counts[i] += 1
            total += 1
            obs_list[i], labels[i] = obs, label
            if done and info.get("episode"):
                ep = info["episode"]
                episodes.append(ep)
                if run_dir and "replay" in info:
                    save_replay(run_dir, f"demo_{len(episodes):04d}_{ep['task']}_{int(ep['score'])}", info["replay"])
                log_fn(f"  局 {len(episodes):4d} {ep['task']:12s} 分 {ep['score']:5.1f} 成功 {ep['success']} "
                       f"存活 {ep['survival_steps']} 死因 {ep['death_cause']}")
            if counts[i] < steps:
                again.append(i)
            if total % 1000 == 0:
                log_fn(f"  采样 {total}/{steps * n_env} 步，{time.time() - start:.0f}s，"
                       f"{total / max(time.time() - start, 1e-6):.1f} 步/s")
            if save_every and total % (save_every * n_env) == 0:
                dump()
        if again:
            act(again)
            waiting.update(again)
    out = {key: np.stack(value) for key, value in data.items()}
    dump()
    return out, episodes


def load_demos(paths, drop_tasks=(), drop_from=()):
    """合并示范。drop_from 里的文件去掉 drop_tasks 这些任务的样本（例如旧专家在生存任务上的冒险策略）。
    任务由向量里的任务 one-hot 识别（features.task_of_vectors，兼容新旧两种向量长度）。"""
    from mc_rl.features import task_of_vectors
    from mc_rl.tasks import TASK_INDEX

    drop_idx = [TASK_INDEX[t] for t in drop_tasks]
    drop_from = {os.path.abspath(p) for p in drop_from}
    parts = defaultdict(list)
    for path in paths:
        with np.load(path) as data:
            keep = np.ones(len(data["labels"]), dtype=bool)
            if drop_idx and os.path.abspath(path) in drop_from:
                keep = ~np.isin(task_of_vectors(data["vec"]), drop_idx)
            for key in ("vec", "pov", "mask", "labels"):
                parts[key].append(data[key][keep])
    # 旧版本的示范向量更短、动作更少：在末尾补 0 / False（新增特征和动作都追加在末尾）
    from mc_rl.features import vector_size
    from mc_rl.macros import NUM_MACROS

    width = {"vec": max([vector_size()] + [a.shape[1] for a in parts["vec"]]),
             "mask": max([NUM_MACROS] + [a.shape[1] for a in parts["mask"]])}
    for key, fill in (("vec", 0.0), ("mask", False)):
        parts[key] = [np.concatenate([a, np.full((a.shape[0], width[key] - a.shape[1]), fill, dtype=a.dtype)], 1)
                      if a.shape[1] < width[key] else a for a in parts[key]]
    return {key: np.concatenate(value) for key, value in parts.items()}


def train_bc(data, model=None, arch="cnn", epochs=20, batch_size=256, lr=3e-4, device="cuda",
             balance=0.5, log_fn=print, val_frac=0.05, seed=0):
    """带类别重加权的交叉熵。balance=0 不加权，1 完全按类别频率倒数加权。"""
    from mc_rl.features import vector_size

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    keep = data["labels"] >= 0
    data = {key: value[keep] for key, value in data.items()}
    n = len(data["labels"])
    if model is None:
        model = ActorCritic(vector_size(len(TASKS)), NUM_MACROS, arch=arch)
    model = model.to(device)
    # 验证集按连续块划分：相邻帧几乎一样，逐帧随机划分会让验证准确率虚高、选不出真正最好的轮次。
    # 块按固定编号取（每 1/val_frac 块取 1 块）：DAgger 每轮把新数据接在后面，旧数据的验证块保持不变，
    # 热启动的模型不会在“上一轮训练过的验证集”上自我评估。
    block = 256 if n >= 256 * 20 else max(1, n // 20)
    starts = np.arange(0, n, block)
    period = max(2, int(round(1.0 / max(val_frac, 1e-3))))
    is_val = (np.arange(len(starts)) % period) == period // 2
    if not is_val.any():
        is_val[0] = True
    val_idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts[is_val]])
    train_idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts[~is_val]]) \
        if (~is_val).any() else val_idx.copy()
    counts = np.bincount(data["labels"][train_idx], minlength=NUM_MACROS).astype(np.float64)
    freq = counts / max(counts.sum(), 1)
    weights = np.where(counts > 0, (1.0 / np.maximum(freq, 1e-6)) ** balance, 0.0)
    weights = weights / max((weights * freq).sum(), 1e-8)
    class_w = torch.as_tensor(weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    best_score, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        rng.shuffle(train_idx)
        losses = []
        for start in range(0, len(train_idx), batch_size):
            mb = train_idx[start:start + batch_size]
            obs = to_tensors({k: data[k][mb] for k in ("vec", "pov", "mask")}, device)
            labels = torch.as_tensor(data["labels"][mb], device=device)
            losses.append(bc_step(model, optimizer, obs, labels, class_w))
        acc = evaluate_bc(model, data, val_idx, device)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), **acc})
        log_fn(f"  BC epoch {epoch + 1:3d} loss {np.mean(losses):.3f} val_acc {acc['val_acc']:.3f} "
               f"rare_acc {acc['val_rare_acc']:.3f}")
        # 稀有动作（合成、放置、进食……）才是技能的关键，和总体准确率一起作为选模型的依据
        score = acc["val_acc"] + acc["val_rare_acc"]
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            history[-1]["best"] = True
    if best_state is not None:
        model.load_state_dict(best_state)
        best = max((h for h in history if h.get("best")), key=lambda h: h["epoch"])
        log_fn(f"  BC 选用第 {best['epoch']} 轮（val_acc {best['val_acc']:.3f} rare_acc {best['val_rare_acc']:.3f}）")
    return model, history


def bc_step(model, optimizer, obs, labels, class_w, max_grad_norm=1.0):
    """行为克隆的一步：带类别权重的交叉熵（logits 已按动作掩码处理），梯度裁剪，返回损失值。"""
    raise NotImplementedError("TODO(学生)：logits = model(obs)[0]；带 class_w 的交叉熵；反传、裁剪梯度、更新；返回 float 损失")


@torch.no_grad()
def evaluate_bc(model, data, idx, device, batch_size=1024):
    model.eval()
    preds = []
    for start in range(0, len(idx), batch_size):
        mb = idx[start:start + batch_size]
        logits, _ = model(to_tensors({k: data[k][mb] for k in ("vec", "pov", "mask")}, device))
        preds.append(logits.argmax(-1).cpu().numpy())
    model.train()
    pred = np.concatenate(preds) if preds else np.zeros(0, dtype=np.int64)
    labels = data["labels"][idx]
    counts = np.bincount(data["labels"], minlength=NUM_MACROS)
    rare = counts[labels] < 0.02 * len(data["labels"])
    per_class = {}
    for cls in np.unique(labels):
        sel = labels == cls
        per_class[MACROS[cls]] = float((pred[sel] == cls).mean())
    return {
        "val_acc": float((pred == labels).mean()) if len(labels) else 0.0,
        "val_rare_acc": float((pred[rare] == labels[rare]).mean()) if rare.any() else 1.0,
        "per_class": per_class,
    }


def run_dagger(vec, model, rounds, steps, tasks, seeds, run_dir, device="cuda", epochs=10,
               beta0=1.0, beta_decay=0.5, demo_paths=(), log_fn=print, task_weights=None, lr=3e-4):
    """第 0 轮纯专家，之后 beta 每轮乘 beta_decay。数据集累积。"""
    os.makedirs(run_dir, exist_ok=True)
    paths = list(demo_paths)
    beta = beta0
    history = []
    for k in range(rounds):
        path = os.path.join(run_dir, f"dagger_{k:02d}.npz")
        log_fn(f"DAgger 第 {k} 轮：beta={beta:.2f}，每环境 {steps} 步")
        _, episodes = collect(vec, model, steps, tasks, seeds, beta=beta,
                              device=device, seed=k, out_path=path, run_dir=run_dir, log_fn=log_fn,
                              task_weights=task_weights)
        paths.append(path)
        data = load_demos(paths)
        model, bc_hist = train_bc(data, model=model, epochs=epochs, device=device, log_fn=log_fn, lr=lr)
        summary = summarize_episodes(episodes)
        history.append({"round": k, "beta": beta, "samples": int(len(data["labels"])), "episodes": summary,
                        "bc": next((h for h in reversed(bc_hist) if h.get("best")), bc_hist[-1])})
        with open(os.path.join(run_dir, "dagger_history.json"), "w") as handle:
            json.dump(history, handle, indent=2, ensure_ascii=False)
        torch.save({"config": model.config(), "state_dict": model.state_dict(), "extra": {"dagger_round": k}},
                   os.path.join(run_dir, "bc.pt"))
        beta *= beta_decay
    return model, history


def summarize_episodes(episodes):
    out = defaultdict(lambda: {"n": 0, "score": 0.0, "success": 0.0, "survival": 0.0})
    for ep in episodes:
        row = out[ep["task"]]
        row["n"] += 1
        row["score"] += ep["score"]
        row["success"] += float(ep["success"])
        row["survival"] += ep["survival_steps"]
    for row in out.values():
        for key in ("score", "success", "survival"):
            row[key] /= max(row["n"], 1)
    return dict(out)
