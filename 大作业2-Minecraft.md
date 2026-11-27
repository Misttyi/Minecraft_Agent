# 大作业2：在 Minecraft 里活过一天

用 MineRL 0.4.4 训练一个智能体，在真实 Minecraft 生存世界里从清晨活到第二天（24000 游戏刻）。你可以用强化学习、模仿学习，或两者结合。

本文件包含环境搭建、任务说明、要写什么代码、怎么训练评测、提交什么、怎么评分。

---

## 零、运行环境

### 0.1 你需要什么样的机器

| 项    | 需求                   | 说明                                                   |
| ---- | -------------------- | ---------------------------------------------------- |
| 系统   | **Linux 或 WSL2**     | minerl 0.4.4 不支持 Windows 原生 Python。Windows 用户走 0.3 节 |
| CPU  | **至少 4 核，推荐 8 核以上**  | 单个 Minecraft 实例约 3 核；一局约 7–20 分钟，10 种子串行约 70–200 分钟  |
| 内存   | **至少 8 GB，推荐 16 GB** | 单实例约 2.3 GB；想并行开多个实例就按 2.5 GB/个加                     |
| GPU  | **不需要**              | PPO/BC 在 CPU 上可跑。GPU 只加速神经网络，不是必需                    |
| 磁盘   | 至少 10 GB             | minerl 客户端数 GB，权重和录像另计                               |
| Java | **JDK 8**（必须精确 8）    | 9+ 会导致 Minecraft 1.11.2 启动失败                         |

**结论：CPU 就够。** 有 Linux 服务器就用服务器；只有 Windows PC 就装 WSL2。Course 服务器和 WSL2 两条路下面都写了，任选一条。

### 0.2 方案 A：Linux 服务器

登录、传文件按课程提供的服务器说明操作。之后所有命令都在服务器终端执行。先加载 Conda：

```bash
module load miniconda3 2>/dev/null || true   # 若服务器用 module
conda create -n mc python=3.8 -y
conda activate mc
```

### 0.3 方案 B：Windows + WSL2

以管理员身份打开 PowerShell，执行：

```powershell
wsl --install -d Ubuntu-22.04
```

重启后从开始菜单打开 Ubuntu，设好用户名密码，然后：

```bash
sudo apt update
sudo apt install -y build-essential git curl wget \
    libxml2-dev libxslt1-dev gfortran libopenblas-dev xvfb
```

安装 Miniconda：

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
echo 'source ~/miniconda3/etc/profile.d/conda.sh' >> ~/.bashrc
echo 'conda activate mc' >> ~/.bashrc
source ~/.bashrc
conda create -n mc python=3.8 -y
conda activate mc
```

Windows 的盘在 WSL2 里是 `/mnt/c`、`/mnt/d`。把工程放进 WSL 家目录（`~`）而非 `/mnt/c`，否则文件系统性能很差：

```bash
cp -r /mnt/d/你的路径/大作业2/release2 ~/mc_hw
cd ~/mc_hw
```

### 0.4 安装 JDK 8

```bash
conda install -y -c conda-forge openjdk=8
java -version     # 必须显示 1.8.x
```

若 `java -version` 不是 1.8，检查 `JAVA_HOME` 是否指向 `$CONDA_PREFIX`，并用 conda 环境里的 java：

```bash
export JAVA_HOME="$CONDA_PREFIX"
export PATH="$JAVA_HOME/bin:$PATH"
java -version
```

建议把这两行写进 `~/.bashrc`。

### 0.5 安装 minerl 0.4.4

minerl 0.4.4 的 PyPI 源码包有上游构建缺陷（引用了一个已下线的 JitPack 坐标），直接 `pip install minerl==0.4.4` 会卡在 Gradle 报错。**不要从零编译。** 发布包已附带预构建 wheel（`minerl-0.4.4-cp38-cp38-linux_x86_64.whl`，约 538 MB），仅适用于 **Linux x86_64 + Python 3.8**：

```bash
# 在工程根目录执行
python -m pip install ./minerl-0.4.4-cp38-cp38-linux_x86_64.whl
```

可选校验完整性（SHA-256）：

```bash
sha256sum minerl-0.4.4-cp38-cp38-linux_x86_64.whl
# 应为 23cc9124b72fcb6641a8666174aabac4b98f73ece37645e16193f272770c205d
```

若平台不符（如 ARM、非 Python 3.8），需自己从源码构建：下载 `minerl-0.4.4.tar.gz`，修 `minerl/Malmo/Minecraft/build.gradle` 里失效的 JitPack 坐标（改成可用的本地 Maven 仓库或镜像），再用 `pip install .`。这一步容易卡在 Gradle，优先用配套 wheel。

游戏端首次启动会下载 Minecraft 客户端和依赖，可能要十几分钟。之后会缓存。

### 0.6 修复 NumPy 兼容

mc 环境的 numpy 若是 1.24+，minerl 0.4.4 用到的 `np.bool` / `np.int` / `np.float` 已被删除，真实环境一启动就崩。发布包提供了补丁脚本：

```bash
python scripts/install_compat.py
python -c "import numpy as np; print(np.bool is bool, np.int is int)"
```

第二行应输出 `True True`。该脚本只在当前环境的 site-packages 写入 `sitecustomize.py`（若有多个 site-packages 目录会逐个写入），不改动已装包；删除对应文件即可回滚。

### 0.7 安装其余依赖

```bash
python -m pip install -r requirements.txt
```

### 0.8 验证安装（不启动游戏）

```bash
python -m unittest tests.test_survival
python -m unittest tests.test_mc_rl
python scripts/mcrl.py tables
```

应看到 `tests.test_survival` 14 项通过，`tests.test_mc_rl` 23 项通过、6 项跳过（共 29 项）。跳过项 = 你未选路线涉及的 TODO + 可选自创任务接口；补全当前路线的 TODO 后，对应测试会自动跑，未选路线继续跳过是正常的。

### 0.9 验证真实环境

无显示器的机器必须用 `xvfb-run -a`：

```bash
export JAVA_HOME="$CONDA_PREFIX"
export PATH="$JAVA_HOME/bin:$PATH"
xvfb-run -a python - <<'PY'
from survive_day import make_env
env = make_env(11, action_repeat=8, strip_reward=True)
obs = env.reset()
print("REAL_ENV_RESET_OK", sorted(obs.keys()))
env.close()
PY
```

看到 `REAL_ENV_RESET_OK` 说明整条链路通了。

> **Kaggle / Colab 能用吗？** 可 pip 安装，但免费额度只有 4 核 / 30 GB、会话 12 小时、sudo/apt 受限，JDK8 + Xvfb 不一定装得上。只适合体验和跑 mock，不建议作为主力环境。

---

## 一、任务概述

从自然出生点开始，让智能体在 Minecraft 生存世界里活过 24000 游戏刻（20 分钟游戏时间，一个完整日夜）。世界会时间流逝、天黑刷怪、饥饿掉血。

作业提供完整的训练基础设施：任务得分表、宏动作、观测特征、老师技能、模仿学习、PPO、并行环境、评测和可视化。**你要做的是按所选路线完成对应的 `TODO(学生)`，用它们训练出策略，并证明学习组件真的起作用。** 四条路线的分工见第三章；公共必做是白天脚本基线。自创任务接口保留在框架里，但不要求实现，也不计入评分。

四条技术路线任选：

1. **路线 1 模仿学习**：老师示范 → 行为克隆（BC）→ DAgger。
2. **路线 2 强化学习**：纯向量 PPO（不看图像）。
3. **路线 3 混合**：模仿初始化 + PPO + 持续模仿损失（DAPG）。
4. **路线 4 完全自定义策略**：跳出框架自己写 policy，评测接口兼容，详见第五章。

---

## 二、工程里已经给了什么

| 模块                    | 内容                                            |
| --------------------- | --------------------------------------------- |
| `survive_day/`        | 评测环境与包装器（**禁止修改**）；死因分类在 `mc_rl/tasks.py`     |
| `mc_rl/tasks.py`      | 13 个任务的得分表、5 阶段课程、成就、结算器；另保留自创任务接口（不要求实现、不计分） |
| `mc_rl/macros.py`     | 29 个核心宏动作 + 4 个扩展宏动作，共 33 个，含动作掩码             |
| `mc_rl/features.py`   | 观测→归一化向量，智能体记忆                                |
| `mc_rl/task_env.py`   | 宏动作 + 得分表 + 回放记录                              |
| `mc_rl/privileged.py` | 训练专用特权观测（方块/实体/准星），只给老师                       |
| `mc_rl/expert.py`     | 老师的基础技能；**生存策略是 TODO**                        |
| `mc_rl/models.py`     | CNN(pov) + MLP(向量) Actor-Critic               |
| `mc_rl/ppo.py`        | PPO 训练循环、课程、日志；**GAE 与损失是 TODO**              |
| `mc_rl/imitation.py`  | 示范采集与 DAgger；**BC 单步是 TODO**                  |
| `mc_rl/agents.py`     | 评测接口；**白天脚本是 TODO**                           |
| `mc_rl/evaluate.py`   | 10 固定种子并行评测                                   |
| `mc_rl/render.py`     | 回放 HTML 与训练看板                                 |
| `mc_rl/mock_env.py`   | 不开游戏的替身环境                                     |
| `mc_rl/vec_env.py`    | 多进程并行，崩溃自动重建                                  |
| `scripts/mcrl.py`     | 命令行入口                                         |
| `scripts/sweep.py`    | 多组参数并行对比                                      |

定位所有 TODO：

```bash
grep -rn "TODO(学生)" --include="*.py" .
```

---

## 三、你要实现什么（5 个 TODO，按路线选做）

| #   | 文件:行                     | 任务       | 必做路线          | 提示                                                                                                       |
| --- | ------------------------ | -------- | ------------- | -------------------------------------------------------------------------------------------------------- |
| 1   | `mc_rl/agents.py:133`    | 白天脚本基线   | 1 / 2 / 3 / 4 | 太阳亮时转向+挥击直到有原木；合成木板、木棍、工作台；然后蹲下。不进食、不打怪、不挖洞                                                              |
| 2   | `mc_rl/ppo.py:193`       | GAE 优势估计 | 2 / 3         | 按时间倒序，注意 `dones` 截断与最后一步自举价值                                                                             |
| 3   | `mc_rl/ppo.py:203`       | PPO 损失   | 2 / 3         | 裁剪策略目标 + 价值 MSE − 熵（只对合法动作）+ 可选 BC 交叉熵；单样本 minibatch 不要除以 std                                            |
| 4   | `mc_rl/imitation.py:193` | BC 单步训练  | 1 / 3         | 带类别权重的交叉熵，梯度裁剪，返回 float 损失                                                                               |
| 5   | `mc_rl/expert.py:514`    | 老师的生存策略  | 1 / 3         | 白天做什么、何时躲、躲哪里、饿了怎么办。可用 `chop_toward` / `craft_chain` / `hunt` / `eat` / `shelter` / `wander` / `go_home` |

- **路线 1（模仿学习）**：做 1、4、5。
- **路线 2（纯 PPO）**：做 1、2、3。
- **路线 3（混合 DAPG）**：1–5 全做。
- **路线 4（完全自定义策略）**：只做公共必做的 1；其余 TODO 可留空，评测兼容边界见第五章。

第 1 项四条路线都要做，因为报告必须有三基线对比。未选路线涉及的 TODO 可以留空，对应测试会继续跳过，不影响评分。

**自创任务不要求做**：`mc_rl/tasks.py` 中的 `make_my_task()` 接口仅作扩展演示保留，不要求实现，也不计入评分。

**注意**：第 5 项里打猎要“按下-松开”攻击，并先做木剑；徒手伤害 1，动物会被击退逃跑。

---

## 四、从零到结果的流程

### 4.1 第 1 步：先在 mock 上调通

**任何代码改动都先在替身环境验证**，不要一上来就开 Minecraft。

```bash
python -m unittest tests.test_mc_rl
python -m unittest tests.test_survival
python scripts/mcrl.py tables
python scripts/mcrl.py ppo --env mock --run runs/mock \
    --num-envs 4 --total-ticks 50000 --device cpu
```

### 4.2 第 2 步：跑三个基线（真实环境）

```bash
export JAVA_HOME="$CONDA_PREFIX"; export PATH="$JAVA_HOME/bin:$PATH"

# 随机空操作（下界）
xvfb-run -a python scripts/mcrl.py eval --agent noop --workers 10 \
    --out runs/eval/noop.json

# 白天脚本（你实现的 TODO 1）
xvfb-run -a python scripts/mcrl.py eval --agent daytime --workers 10 \
    --out runs/eval/daytime.json

# 随机合法宏动作（消融用，零学习）
xvfb-run -a python scripts/mcrl.py eval --agent random_macro --workers 10 \
    --out runs/eval/random_macro.json
```

`--workers` 是并行种子数。机器弱就调小（每个 worker 一个 Minecraft，约 3 核 / 2.3 GB）。

### 4.3 第 3 步：训练策略

**路线 1：模仿学习**

```bash
xvfb-run -a python scripts/mcrl.py demo --run runs/demo --num-envs 8 --steps 1500
xvfb-run -a python scripts/mcrl.py bc --run runs/bc --demos runs/demo/demo_0.npz
xvfb-run -a python scripts/mcrl.py dagger --run runs/dagger \
    --init runs/bc/bc.pt --rounds 3 --beta0 0.5 --beta-decay 0.5
```

**路线 2：纯 PPO**

```bash
xvfb-run -a python scripts/mcrl.py ppo --run runs/ppo --arch mlp \
    --num-envs 8 --total-ticks 100000 --device cpu
```

`--arch mlp` 表示不看图像、只用向量（作业基础配方）。

**路线 3：混合 DAPG（推荐）**

```bash
xvfb-run -a python scripts/mcrl.py ppo --run runs/dapg \
    --init runs/dagger/bc.pt --num-envs 8 --total-ticks 100000 \
    --bc-coef 0.5 --bc-decay 0.98 --lr 1e-4 --device cpu
```

**路线 4：完全自定义策略**

没有统一训练命令。你自己的训练脚本和依赖放在 `code/`，权重放在 `weights/`；自测直接用 `evaluate(agent)`，不要用 CLI 的 `--agent policy`。接口约束、动作格式、跨种子重置和回放要求见第 5.3 节“自定义策略的兼容边界”。

多组参数对比：

```bash
python scripts/sweep.py --preset v2 --root runs/sweep \
    --bc runs/dagger/bc.pt --gpus 0 --num-envs 8 --total-ticks 100000
```

`--dry-run` 可先打印将要跑的组。预设定义在 `scripts/sweep.py` 的 `PRESETS`。`v2` 有 7 组，按示例同时开 8 个环境会占约 56 个 Minecraft 实例，适合服务器选跑；普通小组把 `--num-envs` 降到 2，或只跑自己需要的单组命令。

### 4.4 第 4 步：评测

评测有两个入口，用途不同，不要混用：

| 入口                                    | 用途                                               | 谁用             |
| ------------------------------------- | ------------------------------------------------ | -------------- |
| CLI `--agent policy --checkpoint ...` | 复现**随包模型结构**的训练检查点；并行 10 种子，`--record` 生成回放 HTML | 学生自测、调试、准备报告配图 |
| `evaluate(agent)`（Python API）         | 评测**学生自定义策略**，统一的正式评分入口                          | 助教统一评测         |

**入口一：CLI（学生自测）**

```bash
xvfb-run -a python scripts/mcrl.py eval \
    --agent policy --checkpoint runs/dapg/best.pt \
    --workers 10 --record --out runs/eval/policy.json
```

固定种子：`11, 23, 47, 83, 149, 211, 307, 401, 503, 607`。`--record` 会生成逐帧回放 HTML。此入口只认框架自带的模型结构（`ActorCritic` + `PolicyAgent`）；若你的策略换了结构，用它跑不通是正常的，改用入口二。`evaluate(agent)` 也只接受这 10 个固定种子，单种子调试请用 `--agent` CLI 或 `mc_rl/mock_env.py`，不要给 `evaluate` 传自定义种子列表。

**入口二：`evaluate(agent)`（助教正式评测）**

助教不运行你的训练脚本，也不解析你的 CLI 输出。助教只执行：

```python
from submission import agent
from survive_day.eval import evaluate, write_report
report = evaluate(agent)
write_report(report, "policy.json")
```

你提交的 `submission.py` 必须让上面三行直接可用——详见第五章。此入口在 `survive_day/` 原样保留的评测环境上串行跑 10 个固定种子，单局约 7–20 分钟，全程约 70–200 分钟；不生成回放 HTML，回放由你随作业提交，助教另行核验。

**判定口径**：用 `survival_steps >= 23800` 判定“活过一天”。Malmo 常在 24000 刻前几十刻按时间结束，`survived_day` 字段几乎恒为 `False`，不要用该字段判分。

### 4.5 第 5 步：可视化

```bash
python scripts/mcrl.py render --run runs/eval --npz runs/eval/replays/eval_policy_seed11.npz
python scripts/mcrl.py dashboard --runs 'runs/sweep/*' \
    --evals 'runs/eval/*.json' --out runs/dashboard.html
```

回放 HTML 能逐帧看视角、生命/饥饿曲线、宏动作和里程碑打勾时刻，用来找失败原因、给报告配图。

---

## 五、提交材料

### 5.1 压缩包结构

- 每组同学提交**一个**压缩包：`组长学号.zip`。
- 解压后顶层目录用学号命名，内部文件名统一英文：

```
学号/
├── submission.py             ← 必交：助教评测的唯一入口
├── weights/                  ← 必交：submission.py 需要的权重
│   └── best.pt
├── code/                     ← 你修改过的全部源码
│   ├── mc_rl/ppo.py
│   ├── mc_rl/imitation.py
│   ├── mc_rl/expert.py
│   └── mc_rl/agents.py
├── results/
│   ├── policy.json           ← 10 种子评测结果（你本地复现用）
│   ├── noop.json             ← 三个基线
│   ├── daytime.json
│   ├── random_macro.json
│   ├── dashboard.html        ← 训练曲线看板
│   └── replay/               ← 至少 1 局回放 HTML
├── pre.pptx                  ← 课堂展示 PPT
└── report.pdf                ← 项目报告
```

- **不要放**：`__pycache__`、`.ipynb_checkpoints`、`runs/` 下的临时检查点、Minecraft 客户端、数据集。权重只放 `submission.py` 实际加载的那些。
- **不要改**：`survive_day/` 和任务规格文件 `mc_rl/tasks.py`；`code/` 只放你实际改过的算法文件。

### 5.2 `submission.py` 接口约定（关键）

助教在**自己的机器**上运行下面三行，不会执行你的训练脚本，也不会改你的源码：

```python
from submission import agent
from survive_day.eval import evaluate, write_report
report = evaluate(agent)
write_report(report, "policy.json")
```

因此 `submission.py` 必须满足：

1. **模块级对象**：`import submission` 后存在 `agent`，类型为 callable，签名 `agent(observation, env) -> action`（与 `survive_day/eval.py` 一致，每调用一次返回一个底层动作）。
2. **自加载权重**：加载权重的代码由你写在 `submission.py` 里（例如 `torch.load` / `load_checkpoint`），路径相对 `submission.py` 所在目录，用 `Path(__file__).parent / "weights" / "best.pt"`，不要写死助教机器上的绝对路径。
3. **无训练副作用**：导入时只做“建网络 + 读权重”，不得启动 Minecraft、不得训练、不得联网、不得改 `survive_day/`。`survive_day/` 在助教侧原样保留。
4. **只用标准观测**：评测环境的观测由 `make_env(strip_reward=True)` 给出，不要用特权观测（方块网格、实体、准星）。

**助教侧工作目录**（保证上述三行能跑起来）：助教机器保留一份未修改的发布包作为基线，把学生压缩包解压后按下面方式覆盖，再从根目录运行评测。`survive_day/` 始终用基线版本，不被学生文件替换。

```
发布包根目录/
├── submission.py        ← 由学生包提供
├── weights/             ← 由学生包提供
├── mc_rl/ppo.py 等      ← 由学生包 code/ 里的同名文件覆盖（若有）
├── survive_day/         ← 基线原样保留
├── scripts/ tests/ ...
```

框架自带策略的最小模板（用你自己训练出的权重替换 `best.pt`）：

```python
# submission.py
from pathlib import Path
from mc_rl.agents import PolicyAgent
from mc_rl.models import load_checkpoint

HERE = Path(__file__).resolve().parent
model, _ = load_checkpoint(str(HERE / "weights" / "best.pt"), device="cpu")
agent = PolicyAgent(model, device="cpu", deterministic=True)
```

覆盖后 `mc_rl/`、`survive_day/` 与 `submission.py` 同级，`import` 直接可用，无需改 `sys.path`。若你完全自带一套 `mc_rl/`，则在文件顶部加 `sys.path.insert(0, str(HERE / "code"))`，并确认其中不含对 `survive_day/` 的改动。

### 5.3 自定义策略的兼容边界

若完全自己写策略，只要导出一个满足 `agent(observation, env) -> action` 的对象即可；宏动作展开、记忆维护都要放在这个对象内部。**评测不要求你使用 `mc_rl/` 的模型、宏动作或 TODO 框架**。

| 评测会提供                                   | 评测不会提供 / 不允许                        |
| --------------------------------------- | ----------------------------------- |
| `observation`：标准观测（`strip_reward=True`） | 特权观测（方块网格、实体、准星）                    |
| `env.action_space` 和当前 env 对象           | 被改过的 `survive_day/`；训练脚本；助教机器上的绝对路径 |
| 每个种子新建一个 env                            | 自动跨种子重置你的内部状态（你要自己按 `env` 身份变化重置）   |

另外注意：

- `env.step()` 返回的 reward 被评测包装器丢掉，恒为 0；策略不能依赖环境奖励，要自己算内在奖励或只用状态输入。
- 每次调用返回的是**底层动作 dict**，必须与 `env.action_space` 兼容。若使用宏动作，展开逻辑放在 agent 内部（可复用 `mc_rl.macros.expand`，也可自己写）。
- 训练常用 `action_repeat=8`，正式评测默认 `action_repeat=1`。如果策略按“8 刻一决策”设计，要自己按 `getattr(env, "action_repeat", 1)` 补齐节奏。
- CLI `--agent policy --checkpoint` 只认框架自带的 `ActorCritic + PolicyAgent`。自定义结构请用 `evaluate(agent)` 自测；回放 HTML 由你随包提交，助教另行核验。
- 未选路线涉及的 TODO 可以留空，对应测试跳过不影响评分；但三基线、消融、报告和展示要求不变。

### 5.4 提交前自检

先创建 `submission.py`，再在项目根目录执行；不启动游戏也能验接口：

```bash
python -c "from submission import agent; print(callable(agent))"
```

输出 `True` 才算接口合格。

### 5.5 报告建议结构

1. **方法**：路线选择（1/2/3/4），当前路线涉及的 TODO 各自怎么实现；路线 2/3 写 GAE 与 PPO 损失，路线 1 写 BC / DAgger 的关键更新，路线 4 写清楚自定义策略的结构与动作输出。
2. **训练配置**：超参数、网络结构、课程设置、实际用了多少步。
3. **结果**：三基线 + 学习策略的中位存活步数对比表、学习曲线、10 种子明细。
4. **消融**：关掉学习组件后中位数是否下降（这是高分评价的重要依据）。
5. **里程碑技能**：回放里哪些技能学到了、哪些没有，附得分表截图。
6. **失败分析**：按死因归类（饥饿 / 摔落 / 夜间怪物 / 掉进洞穴 / 溺水）。
7. **局限性**：哪里没做好，为什么。

---

## 六、评分标准

总分 100 分，按**生存结果、里程碑技能、课堂展示**三块综合评定，参考权重约为 **40 / 40 / 20**。以下是评价关注点，不是固定换算表；助教会根据当年整体完成情况、报告质量和现场展示做最终调整，**以助教评定为准**。

### 6.1 生存结果（参考 40 分）

主要依据助教运行 `evaluate(agent)` 在固定 10 个种子上的复跑结果；你本地用 CLI 或 `evaluate(agent)` 得到的结果用于报告对照，不直接决定分数。关注点：

- 评测能否完整跑通、结果是否有效；
- 学习策略是否明显优于随机空操作和白天脚本等基线；
- 存活表现是否稳定，消融是否能说明学习组件真的有贡献；
- 是否只靠固定套路取巧，而不是学到了可迁移的决策。

具体分档和阈值不预先公开，由助教结合结果分布确定。

### 6.2 里程碑技能（参考 40 分）

从随包提交的回放与报告中看技能链完成度。统计时以**技能类里程碑**（砍树、合成、庇护、工具、进食、火把等）为主；`t6000`～`t24000`、`dusk`、`night` 等纯存活计时项不重复计分。关注点：

- 前期采集、合成与庇护是否稳定；
- 工具、进食、火把等进阶技能是否真正学到；
- 关键技能是否在多个种子中复现，而不是单局偶然触发；
- 是否存在挖坑封顶蹲一夜之类的刷分套路。

本块独立观察回放与报告，不依赖 6.1 的数值输出。自创任务接口保留在 `mc_rl/tasks.py` 中，但不要求实现、不计分。

> 只靠“挖坑封顶蹲一夜”的策略：生存分可以很高，但技能分很低。这是设计使然——**光活下来拿不到高分**。

### 6.3 课堂展示（参考 20 分）

关注是否讲清方法选择、关键实现、结果对比、消融与失败分析，以及能否回答随访问题。具体分值由助教现场掌握。

### 6.4 总体参考（非硬档位）

| 大致水平   | 典型表现                       |
| ------ | -------------------------- |
| 达到基本要求 | 流程完整、基线齐全、策略能跑通，报告与展示清楚    |
| 良好     | 在基线对比和消融中能证明学习组件有效，技能链较完整  |
| 优秀     | 生存稳定、技能链扎实，分析与展示能解释设计取舍与局限 |

以上只作自查参考，不对应固定分数。

### 6.5 任务与里程碑得分表

工程提供 13 个任务（10 核心 + 3 扩展），每个任务的里程碑分合计 100。里程碑必须按前置依次达成；任务内得分用于训练奖励塑形、课程解锁和回放显示，**不直接进最终评分**（最终评分只用 6.1–6.3 三块）。

评分口径上：**核心任务里的技能类里程碑通常作为 6.2 的观察依据**（砍树、合成、庇护、工具、进食、火把等，含进阶技能），但纯存活计时项不重复计分；**3 个扩展任务**（回家、夜战、烤肉）默认不启用，可用 `--tasks homing,combat,cook` 训练，但不单独加分；**自创任务不计分，也不要求做**。

课程分 5 阶段由易到难解锁：前沿阶段成功率 ≥ 0.5、得分均值 ≥ 50，或该阶段每个任务已跑满 12 局，满足其一即进下一阶段。`survive_day` 权重最高（2.0），`torch` 最低（0.3）。

| #   | 阶段  | 任务                 | 时长    | 成功条件       | 里程碑（得分；括号内为前置）                                                                                                                   |
| --- | --- | ------------------ | ----- | ---------- | -------------------------------------------------------------------------------------------------------------------------------- |
| 0   | 0   | 探索 `explore`       | 1600  | 走出 32 格    | dist8(25) → dist16(25,dist8) → dist32(30,dist16) → unhurt(20,dist8,局末)                                                           |
| 1   | 0   | 采原木 `wood`         | 2400  | 收 3 原木     | log1(40) → log3(35,log1) → log5(25,log3)                                                                                         |
| 2   | 1   | 基础合成 `craft`       | 3600  | 放下工作台      | log1(10) → log3(10,log1) → planks(15,log1) → stick(15,planks) → table(20,planks) → placed(30,table)                              |
| 3   | 2   | 木制工具 `tools`       | 6000  | 合成木镐       | log3(10) → table(10,log3) → placed(10,table) → sword(25,placed) → pickaxe(25,placed) → wield(20,sword)                           |
| 4   | 2   | 狩猎 `hunt`          | 6000  | 得 1 块肉     | meat1(50) → meat3(25,meat1) → healthy(25,meat1,局末)                                                                               |
| 5   | 2   | 进食 `eat`           | 7200  | 吃下一块肉      | meat1(30) → hold(15,meat1) → ate(35,hold) → fed(20,ate,局末)                                                                       |
| 6   | 1   | 挖坑封顶 `shelter`     | 2400  | 头顶封住       | dig1(15) → dig2(15,dig1) → dig3(20,dig2) → sealed(30,dig2) → stay(20,sealed)                                                     |
| 7   | 3   | 日落前庇护 `nightfall`  | 16000 | 入夜前藏好      | log3(5) → table(5,log3) → shelter(35) → dusk(20) → night(35,dusk)                                                                |
| 8   | 3   | 生存一天 `survive_day` | 24000 | 活到任务结束     | log1(5) → table(5,log1) → shelter(10) → t6000(10) → t12000(15,t6000) → t18000(20,t12000) → t24000(25,t18000) → fed(10,t12000,局末) |
| 9   | 4   | 煤炭火把 `torch`       | 12000 | 合成火把       | pickaxe(20) → cobble(20,pickaxe) → coal(25,pickaxe) → torch(20,coal) → light(15,torch)                                           |
| 10  | 1   | 外出回家 `homing`（扩展）  | 3200  | 回到出生点 4 格内 | out24(30) → home(50,out24) → unhurt(20,out24,局末)                                                                                 |
| 11  | 4   | 夜战 `combat`（扩展）    | 20000 | 夜里持木剑击杀怪物  | sword(15) → armed(15,sword) → kill1(35,sword) → kill3(15,kill1) → survive(20)                                                    |
| 12  | 4   | 烤肉 `cook`（扩展）      | 12000 | 烤出熟肉       | pickaxe(15) → cobble8(20,pickaxe) → furnace(15,cobble8) → placed(15,furnace) → meat(15) → cooked(20,placed+meat)                 |

每个任务另有**成就**（速通、无伤、点亮庇护所等），不计入 100 分，只在回放里显示徽章。完整训练奖励与稠密塑形参数写在 `mc_rl/tasks.py`；需要完整得分表文件时运行 `python scripts/mcrl.py tables --out <目录>`。

**死因分类**（报告用）：`starvation` 饥饿（饥饿值归零）、`fall` 摔落（高度降 ≥4 且掉血 ≥3）、`night_monster` 夜间怪物（≥12500 刻或太阳亮度 <0.4）、`cave` 掉进洞穴（低于出生点 6 格且看不见天空）、`drown` 溺水（氧气 <30）、`other` 其他。判定看最后 6 个决策步。

---

## 七、禁止事项

- 禁止修改 `survive_day/` 下的任何文件（评测口径）。
- 禁止关闭怪物、关闭饥饿、开启创造模式，或在评测时读取任务 XML 以外的游戏内存。
- 禁止在评测时使用特权观测（方块网格、实体、准星）；特权观测只允许训练老师。
- 禁止抄袭同学代码。可以讨论思路，不能提交高度相同的源码。
- 训练预算：策略训练预算为 10 万决策步或 100 局，先到为准（`ticks` 即一次 agent 决策，底层动作由 `action_repeat` 展开）。示范采集、三基线评测和消融评测不计入该预算，但必须在报告里写明实际规模。超预算需在报告里说明。

---

## 八、框架模板说明

工程里的 5 个 `TODO(学生)` 按路线选做，就是模板。你可以：

- 直接补全 TODO 的函数体；
- 修改接口，但要保持功能完整；
- 不用现有框架自己写，只要满足交付要求。

模板只是起点。最终提交的代码必须是你自己写的。

---

## 九、常见问题

**Q：Windows 能直接跑吗？** 不行。minerl 0.4.4 依赖 Linux 版 Minecraft 客户端和 Xvfb，Windows 必须走 WSL2。

**Q：一定要 GPU 吗？** 不。CPU 足够，只是慢一些。

**Q：一局要多久？** 真实 Minecraft 单种子约 7–20 分钟（取决于存活时长和 `action_repeat`），10 种子串行约 70–200 分钟；8 并行在 32 核机器上约 10–30 分钟。

**Q：`survived_day` 为什么总是 False？** Malmo 在 24000 刻前几十刻按时间结束，包装器判断的是“达到 24000”，所以字段几乎恒 False。用 `survival_steps >= 23800`。

**Q：打猎/进食为什么 0 分？** 攻击宏动作已改成“按下-松开”，老师要先做木剑。徒手伤害 1，动物被击退逃跑。照这个改。

**Q：跑 mock 和真实结果差很多？** 正常。mock 只是替身，用来验证代码逻辑；技能要在真实环境里学。

---

## 十、联系方式

请大家尽早开始完成大作业，以免后期遇到问题来不及解决。

如遇到难以解决的问题，大家可以在课程微信群中讨论，互帮互助，也可以联系负责本次大作业的助教。

助教-黄恺奕  微信：MorphUES000  邮箱：sjtu5292760@sjtu.edu.cn

助教-张亚洲  微信：ZYZ15231806222  邮箱：zhangyz2022@sjtu.edu.cn

助教-李毅恒  微信：13671933226  邮箱：liyiheng@sjtu.edu.cn

助教-夏贤贤  微信：-Deuxmille  邮箱：urutoramarin2000@sjtu.edu.cn
