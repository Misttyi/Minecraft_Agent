"""策略网络。

- arch="mlp"：作业规定的向量 PPO（两层 128，只看向量），不看 pov。
- arch="cnn"：向量 MLP + 64×64 pov 小卷积，用来自己找树、找动物。
两种都输出被掩码处理过的宏动作 logits 和状态价值。
"""

import torch
from torch import nn
from torch.distributions import Categorical

NEG_INF = -1e8


def _init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class PovEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            _init(nn.Conv2d(3, 32, 8, stride=4), 2 ** 0.5), nn.ReLU(),
            _init(nn.Conv2d(32, 64, 4, stride=2), 2 ** 0.5), nn.ReLU(),
            _init(nn.Conv2d(64, 64, 3, stride=1), 2 ** 0.5), nn.ReLU(),
            nn.Flatten(),
            _init(nn.Linear(64 * 4 * 4, out_dim), 2 ** 0.5), nn.ReLU(),
        )

    def forward(self, pov):
        # uint8 NHWC -> float NCHW
        x = pov.permute(0, 3, 1, 2).float() / 255.0
        return self.net(x)


class ActorCritic(nn.Module):
    def __init__(self, vec_dim, num_actions, arch="cnn", hidden=256, vec_hidden=128):
        super().__init__()
        self.arch = arch
        self.vec_dim = vec_dim
        self.num_actions = num_actions
        self.vec = nn.Sequential(
            _init(nn.Linear(vec_dim, vec_hidden), 2 ** 0.5), nn.Tanh(),
            _init(nn.Linear(vec_hidden, vec_hidden), 2 ** 0.5), nn.Tanh(),
        )
        if arch == "cnn":
            self.pov = PovEncoder(hidden)
            trunk_in = hidden + vec_hidden
            self.trunk = nn.Sequential(_init(nn.Linear(trunk_in, hidden), 2 ** 0.5), nn.ReLU())
            feat = hidden
        elif arch == "mlp":
            self.pov = None
            self.trunk = nn.Identity()
            feat = vec_hidden
        else:
            raise ValueError(arch)
        self.pi = _init(nn.Linear(feat, num_actions), 0.01)
        self.v = _init(nn.Linear(feat, 1), 1.0)

    def features(self, obs):
        # 向量特征向后兼容：新增特征接在末尾，旧模型只读前 vec_dim 维
        h = self.vec(obs["vec"][..., :self.vec_dim])
        if self.pov is not None:
            h = self.trunk(torch.cat([self.pov(obs["pov"]), h], dim=-1))
        return h

    def forward(self, obs):
        h = self.features(obs)
        logits = self.pi(h)
        mask = obs.get("mask")
        if mask is not None and mask.shape[-1] > logits.shape[-1]:
            # 旧模型没有后来追加的宏动作：这些动作的 logit 视为负无穷
            pad = torch.full(logits.shape[:-1] + (mask.shape[-1] - logits.shape[-1],), NEG_INF,
                             dtype=logits.dtype, device=logits.device)
            logits = torch.cat([logits, pad], dim=-1)
        if mask is not None:
            logits = torch.where(mask, logits, torch.full_like(logits, NEG_INF))
        return logits, self.v(h).squeeze(-1)

    def act(self, obs, deterministic=False):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value

    def config(self):
        return {"vec_dim": self.vec_dim, "num_actions": self.num_actions, "arch": self.arch}


def to_tensors(batch, device):
    return {
        "vec": torch.as_tensor(batch["vec"], dtype=torch.float32, device=device),
        "pov": torch.as_tensor(batch["pov"], dtype=torch.uint8, device=device),
        "mask": torch.as_tensor(batch["mask"], dtype=torch.bool, device=device),
    }


def save_checkpoint(path, model, extra=None, state_dict=None):
    """state_dict 不给就存模型当前权重（例如 best.pt 要存产生这批成绩的、更新前的权重）。"""
    torch.save({"config": model.config(), "state_dict": state_dict if state_dict is not None else model.state_dict(),
                "extra": extra or {}}, path)


def load_checkpoint(path, device="cpu", upgrade=False):
    """upgrade=True：把旧检查点扩展到当前的完整向量长度和动作数，用于继续训练。
    新增的输入列权重为 0、新增动作的 logit 偏置为 -8：原有动作的概率分布几乎不变（新动作初始概率 < 0.1%），
    新动作仍然可以在继续训练中学会。"""
    data = torch.load(path, map_location=device)
    config = dict(data["config"])
    model = ActorCritic(**config).to(device)
    model.load_state_dict(data["state_dict"])
    if not upgrade:
        return model, data.get("extra", {})
    from mc_rl.features import vector_size
    from mc_rl.macros import NUM_MACROS

    if config["vec_dim"] >= vector_size() and config["num_actions"] >= NUM_MACROS:
        return model, data.get("extra", {})
    new = ActorCritic(vector_size(), NUM_MACROS, arch=config["arch"]).to(device)
    state = new.state_dict()
    for key, value in model.state_dict().items():
        target = state[key]
        if target.shape == value.shape:
            state[key] = value.clone()
        elif key == "vec.0.weight":
            target.zero_()
            target[:, :value.shape[1]] = value
        elif key == "pi.weight":
            target.zero_()
            target[:value.shape[0]] = value
        elif key == "pi.bias":
            target.fill_(-8.0)
            target[:value.shape[0]] = value
        else:
            raise ValueError(f"无法扩展参数 {key}: {tuple(value.shape)} -> {tuple(target.shape)}")
    new.load_state_dict(state)
    return new, data.get("extra", {})
