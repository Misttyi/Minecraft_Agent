# Minecraft RL IL 大作业四人分工与 Agent 执行说明

本文用于四人小组选择路线 3“混合 DAPG”完成 Minecraft 生存大作业。四个执行 Agent 必须先完整阅读原始作业说明，再按照各自任务书工作。本文规定文件所有权、输入输出、验收标准和交接顺序；原始作业说明中的接口、评分口径和禁止事项具有更高优先级。

## 1. 唯一规格来源

所有 Agent 开始工作前必须完整阅读：

```text
大作业2-RL、IL\大作业2-Minecraft.md
```

同时检查自己负责文件中的 `TODO(学生)`、相邻函数、调用方和对应测试。不得只根据本报告猜测接口。

统一选择路线 3：

```text
老师示范 -> 行为克隆 BC -> DAgger -> PPO 微调 + 持续模仿损失 DAPG
```

路线 3 需要完成全部 5 个 TODO：

1. `mc_rl/agents.py`：白天脚本基线。
2. `mc_rl/ppo.py`：GAE。
3. `mc_rl/ppo.py`：PPO 损失。
4. `mc_rl/imitation.py`：BC 单步训练。
5. `mc_rl/expert.py`：老师的生存策略。

## 2. 总体目标与验收口径

本项目按老师给出的三个评分块准备材料：生存结果约 40 分、里程碑技能约 40 分、课堂展示约 20 分。具体阈值不预先公开，最终分数以助教评定为准；本说明的目标是覆盖每个评分关注点和提交规范，避免因缺材料、接口错误或展示不完整失分。

最终系统应完成以下闭环：

1. 三个基线均能在真实环境运行并产生 JSON：`noop`、`daytime`、`random_macro`。
2. 老师能在 mock 环境完成生存任务，并在真实环境生成包含关键技能动作的示范。
3. BC 训练损失下降，稀有动作准确率可报告。
4. DAgger 能生成修正后的策略检查点。
5. DAPG 能从 DAgger 检查点初始化并完成训练。
6. 最终 `submission.py` 可被助教的三行评测代码直接导入执行。
7. 固定 10 个种子的正式评测完整结束，并生成报告、回放和训练看板。

“活过一天”的统一判断是：

```text
survival_steps >= 23800
```

不得使用通常为 `False` 的 `survived_day` 字段判分。

最终报告必须同时展示：

- 三个零学习基线；
- 至少一个模仿学习检查点，例如 BC 或 DAgger；
- 最终 DAPG；
- 至少一个消融，例如关闭持续 BC 损失的 PPO 微调；
- 10 种子明细、存活步数中位数、成功种子数、技能出现次数和死因。

最终课堂展示必须讲清：路线选择、5 个 TODO 的关键实现、训练配置、三基线、消融、技能链、失败分析、局限性，并准备回答随访问题。

## 3. 文件所有权与禁止交叉规则

### 3.1 独占所有权

| 人员 | 独占修改范围 | 独占生成目录 |
| --- | --- | --- |
| 1号 Agent：基线、评测、提交接口与打包 | `mc_rl/agents.py`、`submission.py` | `results/`、最终压缩包、`handoff/agent1.md` |
| 2号 Agent：老师、示范与技能证据 | `mc_rl/expert.py` | `runs/demo/`、`evidence/agent2/`、`handoff/agent2.md` |
| 3号 Agent：BC、DAgger、看板与最终报告 | `mc_rl/imitation.py`、报告源文件 | `runs/bc/`、`runs/dagger/`、`deliverables/report.pdf`、`deliverables/dashboard.html`、`handoff/agent3.md` |
| 4号 Agent：PPO、DAPG、消融与课堂展示 | `mc_rl/ppo.py`、PPT 源文件 | `runs/dapg/`、`weights/best.pt`、`deliverables/pre.pptx`、`presentation/`、`handoff/agent4.md` |

任何 Agent 不得修改其他人的独占文件。需要修改时，只能在交接说明中提出，由该文件的所有者实施。

### 3.2 统一锁定文件

以下内容默认只读：

```text
survive_day/
mc_rl/tasks.py
mc_rl/macros.py
mc_rl/features.py
mc_rl/task_env.py
mc_rl/models.py
mc_rl/evaluate.py
scripts/mcrl.py
tests/
requirements.txt
```

其中 `survive_day/` 和 `mc_rl/tasks.py` 属于明确禁止修改的评测规格。其他锁定文件如果确有缺陷，由1号 Agent 记录问题并组织确认；未确认前任何人不得修改。

### 3.3 文档零交叉

最终文档也实行独占编辑：3号 Agent 是 `report.pdf` 和 `dashboard.html` 的唯一编辑者，4号 Agent 是 `pre.pptx` 和答辩提纲的唯一编辑者。1号只把已经验收的最终文件复制进提交包，不修改其内容。2号只写 `evidence/agent2/skill_analysis.md` 和自己的证据图片。所有人通过各自独立的 `handoff/agentN.md` 交付文字、数值和图表路径，不得多人同时编辑同一个报告或 PPT。

### 3.4 Git 建议

四个分支分别为：

```text
agent1-baseline-eval
agent2-expert-demo
agent3-imitation
agent4-ppo-dapg
```

每个分支只提交自己的独占文件。1号 Agent 负责最终集成；其他人不得在集成分支直接修改文件。

### 3.5 任务量平衡与评分责任

| Agent | 算法与工程工作 | 实验与材料工作 | 主要评分责任 |
| --- | --- | --- | --- |
| xky | 白天脚本、`submission.py`、评测兼容 | 三基线、CLI回放、正式10种子评测、最终打包 | 生存结果可复现、提交接口与目录规范 |
| lwy | 老师生存策略 | 示范采集、技能复现统计、死因与回放证据 | 里程碑技能40分的证据完整性 |
| lqr | BC单步、BC与DAgger | 训练看板、结果整合、最终项目报告 | 报告质量、曲线与消融证据表达 |
| hyw | GAE、PPO损失、DAPG | 主训练、消融、PPT | 最终学习策略、消融与展示20分 |

4号虽然负责两个数学 TODO，但二者位于同一文件和同一训练链；1号需要承担耗时的四类评测和打包，2号需要承担示范与逐回放技能核验，3号需要承担最终报告，因此四人的总体工作量按“代码难度 + 真实环境运行时间 + 材料责任”平衡，而不是按 TODO 数量机械平均。

## 4. 总体执行与交接顺序

### 阶段 A：可以并行进行

1. 1号实现白天脚本和评测骨架。
2. 2号实现老师的生存策略，并在 mock 中验证。
3. 3号实现 BC 单步训练，并进行最小张量测试。
4. 4号实现 GAE 与 PPO 损失，并通过数学测试。

### 阶段 B：必须串行交接

```text
2号生成 demo_0.npz
3号读取 demo_0.npz，训练 BC 和 DAgger，生成 bc.pt
4号读取 bc.pt，训练 DAPG，生成 best.pt
1号读取 best.pt，生成 submission.py，完成 CLI 回放评测和正式10种子评测
2号只读最终评测和回放，生成技能复现与死因证据
3号汇总四人的交接材料，生成 dashboard.html 和 report.pdf
4号读取最终报告与证据，生成 pre.pptx
1号只复制已验收产物，生成最终压缩包
```

后一个 Agent 不得用口头描述代替前一个 Agent 的真实产物。文件不存在、无法读取或交接说明不完整时，应停止下游真实训练，先完成接口修复。

## 5. 1号 Agent 任务书：基线、评测与交付

### 5.1 负责范围

1号 Agent 独占：

```text
mc_rl/agents.py
submission.py
results/
最终压缩包
handoff/agent1.md
```

不得修改 `expert.py`、`imitation.py` 或 `ppo.py`。

### 5.2 输入

- 原始作业说明；
- 4号交付的 `weights/best.pt`；
- 3号交付的 `deliverables/report.pdf` 和 `deliverables/dashboard.html`；
- 4号交付的 `deliverables/pre.pptx`；
- 2～4号的交接说明。

### 5.3 任务一：实现白天脚本

在 `DaytimeScript.choose()` 中实现作业要求的固定脚本：

1. 太阳亮度大于 0.5 且没有获得足够原木时，通过转向和挥击搜索并砍树。
2. 有原木后按库存和动作掩码依次合成木板、木棍、工作台。
3. 完成工作台后持续蹲下或保持安全空操作。
4. 该基线禁止主动进食、攻击怪物或挖洞。
5. 必须使用现有宏动作名称和 `action_mask`，不得返回不存在或非法的动作。

实现前检查 `mc_rl/macros.py` 中的宏动作表和 `MacroAgent` 的展开机制。不得另写一套底层动作队列。

### 5.4 任务二：运行三个基线

真实环境中运行：

```bash
xvfb-run -a python scripts/mcrl.py eval --agent noop --workers 10 --out runs/eval/noop.json
xvfb-run -a python scripts/mcrl.py eval --agent daytime --workers 10 --out runs/eval/daytime.json
xvfb-run -a python scripts/mcrl.py eval --agent random_macro --workers 10 --out runs/eval/random_macro.json
```

机器资源不足时可将 `--workers` 降到 2 或 4，但必须最终覆盖固定 10 个种子。不要在 16 GB 内存电脑上同时启动 8～10 个 Minecraft 实例。

固定种子只能使用：

```text
11, 23, 47, 83, 149, 211, 307, 401, 503, 607
```

### 5.5 任务三：制作 submission.py

等待4号交付 `weights/best.pt` 后，按作业说明提供的框架模板实现：

```python
from pathlib import Path
from mc_rl.agents import PolicyAgent
from mc_rl.models import load_checkpoint

HERE = Path(__file__).resolve().parent
model, _ = load_checkpoint(str(HERE / "weights" / "best.pt"), device="cpu")
agent = PolicyAgent(model, device="cpu", deterministic=True)
```

要求：

- 模块级存在 callable `agent`；
- 权重路径相对 `submission.py`；
- 导入时只建网络和加载权重；
- 不启动 Minecraft、不训练、不联网；
- 只使用标准观测；
- 不修改 `survive_day/`。

### 5.6 任务四：最终评测与提交材料

先使用 CLI 进行学生自测并生成回放：

```bash
xvfb-run -a python scripts/mcrl.py eval \
  --agent policy --checkpoint weights/best.pt \
  --workers 10 --record --out runs/eval/policy.json
```

该入口用于框架自带的 `ActorCritic + PolicyAgent` 检查点和回放配图。随后再运行助教使用的正式接口：

```python
from submission import agent
from survive_day.eval import evaluate, write_report
report = evaluate(agent)
write_report(report, "policy.json")
```

两个入口不得混用。正式 `evaluate(agent)` 串行评测固定 10 个种子，不生成回放 HTML，全程可能需要约 70～200 分钟。单种子调试使用 CLI 或 mock，不得向 `evaluate` 传自定义种子列表。

最终整理：

每组只提交一个 `组长学号.zip`，解压后的顶层目录也使用学号，内部文件名统一英文：

```text
学号/
├── submission.py
├── weights/best.pt
├── code/                 只放四人实际修改的算法文件
├── results/
│   ├── policy.json
│   ├── noop.json
│   ├── daytime.json
│   ├── random_macro.json
│   ├── dashboard.html
│   └── replay/
├── pre.pptx
└── report.pdf
```

1号不得修改3号和4号交付的 `report.pdf`、`dashboard.html`、`pre.pptx`，只负责检查文件存在、可打开并复制到最终结构。`code/` 只放四个实际修改的算法文件。不得放整个 `runs/`、数据集、Minecraft 客户端、临时 checkpoint、`__pycache__` 或 `.ipynb_checkpoints`。

### 5.7 验收标准

- `DaytimeScript` 能完整跑完 mock 环境，不抛异常。
- 三个基线 JSON 均存在且包含全部固定种子结果。
- CLI 策略评测生成 `policy.json` 和至少一局回放 HTML。
- `python -c "from submission import agent; print(callable(agent))"` 输出 `True`。
- 正式 `evaluate(agent)` 能在10个种子上完整结束。
- `report.pdf`、`pre.pptx`、`dashboard.html` 均可打开且来自对应唯一所有者。
- 最终包内没有临时 checkpoint、Minecraft 客户端、数据集或缓存目录。

### 5.8 交付物

```text
mc_rl/agents.py
submission.py
results/noop.json
results/daytime.json
results/random_macro.json
results/policy.json
results/dashboard.html
results/replay/*.html
最终压缩包
handoff/agent1.md
```

## 6. 2号 Agent 任务书：老师策略与示范采集

### 6.1 负责范围

2号 Agent 独占：

```text
mc_rl/expert.py
runs/demo/
evidence/agent2/
handoff/agent2.md
```

不得修改 `agents.py`、`imitation.py` 或 `ppo.py`。

### 6.2 输入

- 原始作业说明；
- `mc_rl/expert.py` 中已实现的基础技能；
- `mc_rl/tasks.py` 中任务和里程碑定义，只读；
- mock 环境和现有测试。

### 6.3 任务一：实现 survival_strategy

必须复用现有技能：

```text
chop_toward
craft_chain
hunt
eat
shelter
wander
go_home
```

推荐按优先级和时间阶段组织，而不是堆叠大量互相冲突的条件。

#### 全局高优先级

1. 已经开始庇护流程且 `self.committed` 为真时，优先完成庇护，避免中途退出导致暴露。
2. 饥饿达到危险阈值且有可食用物品时，优先进食。
3. 临近日落且未庇护时，停止远距离探索或狩猎。

#### 建议阶段

1. 早期采集：获得至少 3～5 个原木。
2. 基础合成：木板、木棍、工作台，并尽量放置工作台。
3. 工具准备：优先木剑，再根据资源合成木镐。
4. 白天中段：在安全时狩猎获得肉，饥饿时进食。
5. 日落准备：距离出生点不远时返回；时间不足时就地庇护。
6. 夜晚：保持封闭、减少无意义移动，并处理饥饿。

老师不能只追求生存时长。示范必须尽可能包含砍树、合成、工作台、工具、狩猎、进食、庇护和火把等技能类里程碑，否则后续 BC 会被普通移动动作主导。火把任务已有独立任务逻辑时应保留对应示范，不要求把所有技能强行塞进单局 `survive_day`。

打猎必须遵守作业说明：先做木剑，攻击宏动作要形成按下和松开的过程。不得徒手追逐动物作为主要方案。

### 6.4 任务二：mock 验证

运行老师相关测试，并特别检查：

```bash
python -m unittest tests.test_mc_rl.TaskEnvTest.test_expert_survives_mock_day
```

目标：

```text
survival_steps >= 23900
score >= 80
```

同时检查简单任务 `wood`、`craft`、`tools`、`hunt`、`eat`、`shelter` 没有被新策略破坏。

### 6.5 任务三：真实示范采集

通过真实环境采集：

```bash
xvfb-run -a python scripts/mcrl.py demo --run runs/demo --num-envs 8 --steps 1500
```

资源不足时降低 `--num-envs`，但不要减少示范质量检查。采集结束后至少报告：

- 总示范步数；
- 成功和失败局数；
- 各宏动作样本数量；
- 合成、放置、狩猎、进食、庇护等稀有动作是否出现；
- 明显失败模式。

如果示范几乎只有移动、攻击或蹲下，应先改老师或增加采集，不得把低质量示范直接交给3号。

### 6.6 任务四：最终技能与失败证据

等待1号完成最终策略的10种子评测和回放后，只读 `results/`，不得修改其中任何文件。生成 `evidence/agent2/skill_analysis.md`，至少统计：

- 砍树、木板、木棍、工作台、工作台放置；
- 木剑、木镐、持有工具；
- 狩猎、获得肉、进食；
- 庇护、日落前藏好；
- 煤炭、火把、放置光源；
- 每项技能完成种子数、首次完成时刻和是否能在多个种子稳定复现；
- `starvation`、`fall`、`night_monster`、`cave`、`drown`、`other` 的死因数量；
- 至少一局成功回放和一局失败回放的关键帧说明。

`t6000`～`t24000`、`dusk`、`night` 等纯存活计时项不得作为技能成果重复统计。`homing`、`combat`、`cook` 是扩展任务，不单独加分；自创任务不要求实现、不计分。2号只提供事实、表格和图片路径，不编辑最终报告或 PPT。

### 6.7 验收标准

- 老师通过 mock 生存测试。
- 不使用最终评测禁止的游戏内存；特权观测只用于训练老师。
- `runs/demo/demo_0.npz` 可被 `scripts/mcrl.py bc` 正常读取。
- 示范包含多种技能类宏动作，而不是单一庇护策略。
- `handoff/agent2.md` 写明采集命令、步数、种子、成功率和动作分布。
- `evidence/agent2/skill_analysis.md` 区分技能里程碑与纯计时里程碑，并包含多种子复现和死因统计。

### 6.8 交给3号和4号的产物

```text
runs/demo/demo_0.npz
evidence/agent2/skill_analysis.md
evidence/agent2/ 中的回放截图或图表
handoff/agent2.md
mc_rl/expert.py 对应的提交或 commit hash
```

第一次交接必须明确告诉3号示范文件的绝对或项目相对路径；最终评测后再把技能证据交给3号写报告、交给4号做展示。不得只说“已经跑好了”。

## 7. 3号 Agent 任务书：行为克隆与 DAgger

### 7.1 负责范围

3号 Agent 独占：

```text
mc_rl/imitation.py
runs/bc/
runs/dagger/
deliverables/report_source.md
deliverables/report.pdf
deliverables/dashboard.html
handoff/agent3.md
```

不得修改 `agents.py`、`expert.py` 或 `ppo.py`。

### 7.2 输入

- 原始作业说明；
- 2号交付的 `runs/demo/demo_0.npz`；
- 现有 `ActorCritic` 模型和训练循环，只读。
- 1号最终交付的基线、正式评测和回放；
- 2号最终交付的技能与死因证据；
- 4号交付的 DAPG、消融和训练日志。

### 7.3 任务一：实现 bc_step

保持现有函数签名：

```python
bc_step(model, optimizer, obs, labels, class_w, max_grad_norm=1.0)
```

正确顺序：

1. 清空梯度；
2. `logits = model(obs)[0]`；
3. 使用 `class_w` 计算带类别权重的交叉熵；
4. 反向传播；
5. 使用 `max_grad_norm` 裁剪模型参数梯度；
6. 更新优化器；
7. 返回 Python `float` 损失。

伪代码：

```python
optimizer.zero_grad(set_to_none=True)
logits = model(obs)[0]
loss = F.cross_entropy(logits, labels.long(), weight=class_w)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
optimizer.step()
return float(loss.item())
```

不得修改模型输出、动作掩码语义或训练函数接口。

### 7.4 任务二：BC 训练

读取2号示范：

```bash
xvfb-run -a python scripts/mcrl.py bc --run runs/bc --demos runs/demo/demo_0.npz
```

记录每轮：

- training loss；
- validation accuracy；
- rare action accuracy；
- 最佳轮次；
- 各类别样本数量。

总体准确率高但稀有动作准确率接近零时，不得直接宣布完成。应检查类别权重、示范分布和模型是否只预测常见移动动作。

### 7.5 任务三：DAgger

运行：

```bash
xvfb-run -a python scripts/mcrl.py dagger --run runs/dagger \
  --init runs/bc/bc.pt --rounds 3 --beta0 0.5 --beta-decay 0.5
```

三轮建议混合系数约为：

```text
0.5
0.25
0.125
```

比较 BC 与 DAgger：

- 验证集总体准确率；
- 稀有动作准确率；
- mock 或真实环境中的存活步数；
- 是否减少因策略偏离老师轨迹造成的失败。

### 7.6 任务四：训练看板和最终报告

3号是最终 `report.pdf` 和 `dashboard.html` 的唯一编辑者。其他人只提供交接材料，不直接修改报告。

使用真实日志生成训练看板，命令示例：

```bash
python scripts/mcrl.py dashboard \
  --runs 'runs/*' --evals 'runs/eval/*.json' \
  --out deliverables/dashboard.html
```

如实际目录结构不同，只调整输入 glob，不修改看板程序。最终报告至少包含：

1. 方法：说明选择路线3的理由，以及白天基线、老师策略、BC、DAgger、GAE、PPO损失和持续BC损失的关键实现。
2. 训练配置：网络结构、超参数、课程设置、种子、实际训练决策步数和局数。
3. 结果：三个基线、BC或DAgger、DAPG和消融的中位存活步数、10种子明细、成功种子数及学习曲线。
4. 消融：在尽量相同的训练步数、种子和结构下，说明关闭学习组件后中位数是否下降。
5. 技能：使用2号证据展示技能类里程碑、完成种子数、首次完成时刻和得分表截图；不得把纯计时里程碑当技能。
6. 失败分析：按规定死因分类，并使用回放解释至少一个失败案例。
7. 局限性：说明未稳定学会的技能、泛化不足、训练预算和环境限制。

报告中的关键数值必须能追溯到 JSON、日志或回放，不得手工编造。若模型未学会火把或其他进阶技能，应如实说明，不得删去该项。

### 7.7 验收标准

- `bc_step` 返回有限的 Python float。
- 训练过程中无 NaN 或 Inf。
- BC 的训练损失总体下降。
- DAgger 完成预定轮次。
- 最终检查点能够被4号的 PPO 初始化逻辑加载。
- 交接说明包含模型结构、训练参数、示范路径和最佳检查点路径。
- `dashboard.html` 可打开，曲线来源路径明确。
- `report.pdf` 覆盖老师建议的七部分，并包含40分生存、40分技能对应证据。

### 7.8 交给4号和1号的产物

```text
runs/dagger/bc.pt
runs/bc/ 中的训练历史
runs/dagger/ 中的 DAgger 历史
deliverables/dashboard.html
deliverables/report.pdf
handoff/agent3.md
mc_rl/imitation.py 对应的提交或 commit hash
```

4号只读取报告和模仿学习产物制作展示，不修改 `imitation.py` 或报告；1号只把最终 `report.pdf` 和 `dashboard.html` 复制到提交包。

## 8. 4号 Agent 任务书：PPO 数学与 DAPG 训练

### 8.1 负责范围

4号 Agent 独占：

```text
mc_rl/ppo.py
runs/dapg/
weights/best.pt
presentation/
deliverables/pre.pptx
handoff/agent4.md
```

不得修改 `agents.py`、`expert.py` 或 `imitation.py`。

### 8.2 输入

- 原始作业说明；
- 3号交付的 `runs/dagger/bc.pt`；
- PPO 现有训练循环、模型和动作掩码，只读。
- 1号最终评测结果、2号技能证据和3号最终报告，用于课堂展示。

### 8.3 任务一：实现 GAE

保持函数签名不变。按时间倒序计算：

```text
delta_t = reward_t + gamma * next_value * (1 - done_t) - value_t
advantage_t = delta_t + gamma * lambda * (1 - done_t) * advantage_(t+1)
return_t = advantage_t + value_t
```

关键要求：

- 最后一步的 `next_value` 使用 `last_value`；
- 向前移动后使用 `values[t]` 作为下一轮的 `next_value`；
- `done[t] = 1` 时同时切断价值自举和优势递推；
- 返回与输入长度相同的 `float32` 数组。

### 8.4 任务二：实现 PPO 损失

损失应包含：

```text
总损失 = 策略损失 + vf_coef * 价值损失 - ent_coef * 熵 + bc_coef * 模仿损失
```

实现要点：

1. 从已掩码的 logits 构造动作分布。
2. 计算新动作 log probability 和 `ratio = exp(new_logp - old_logp)`。
3. 策略损失使用原目标和裁剪目标两者的较小值。
4. 价值损失使用预测价值和 return 的 MSE。
5. 熵只统计合法动作；不得让非法动作影响熵奖励。
6. `labels >= 0` 的样本才参与 BC 交叉熵。
7. minibatch 多于1个样本时才能用标准差标准化优势；单样本不得除以 std。
8. 统计字典至少返回：`pg_loss`、`vf_loss`、`entropy`、`approx_kl`、`clip_frac`；存在模仿项时还要有 `bc_loss` 和 `bc_acc`。

### 8.5 任务三：数学和 mock 验证

首先运行：

```bash
python -m unittest tests.test_mc_rl.PPOMathTest
```

然后运行 mock smoke test：

```bash
python scripts/mcrl.py ppo --env mock --run runs/mock \
  --num-envs 4 --total-ticks 50000 --device cpu
```

确认：

- GAE 手算测试匹配；
- 单样本 PPO loss 有限；
- loss、KL、entropy 和 value loss 不出现 NaN；
- 能保存 `last.pt` 或对应检查点。

### 8.6 任务四：DAPG 主训练

等待3号交付后运行：

```bash
xvfb-run -a python scripts/mcrl.py ppo --run runs/dapg \
  --init runs/dagger/bc.pt --num-envs 8 --total-ticks 100000 \
  --bc-coef 0.5 --bc-decay 0.98 --lr 1e-4 --device cpu
```

资源不足时将 `--num-envs` 降至2或4。训练预算必须完全按照老师原文执行：策略训练预算为10万决策步或100局，先到为准；示范采集、三个基线评测和消融评测不计入该预算，但必须在报告中写明实际规模；任何超预算情况都必须说明。不得自行解释成“每个主训练可以各跑10万步”。

至少保存：

- 最佳检查点；
- 最后检查点；
- 每次更新的策略损失、价值损失、熵、KL、clip fraction、BC loss；
- 各任务得分和课程阶段；
- 实际训练步数和局数。

### 8.7 任务五：消融

至少提供一个可以说明学习组件贡献的对照。推荐优先级：

1. 相同 DAgger 初始化，但 PPO 微调时 `bc_coef=0`；
2. DAgger 检查点直接评测，与 DAPG 比较；
3. 纯 PPO，与 DAPG 比较。

消融应尽量保持训练步数、种子和网络结构一致。不能用完全不同预算的模型直接下结论。

### 8.8 任务六：课堂展示

4号是 `pre.pptx` 和答辩提纲的唯一编辑者。PPT必须使用最终真实结果，至少包含：

1. 问题定义、路线3选择理由和完整流程。
2. 四人分工及5个TODO的关键实现，不展示大段源码。
3. GAE、PPO裁剪损失、BC类别权重、DAgger和持续模仿损失的核心逻辑。
4. 训练配置、网络结构、课程和训练预算。
5. 三个基线、BC或DAgger、DAPG和消融的结果对比。
6. 10种子生存稳定性、技能多种子复现、火把等未学会技能的如实说明。
7. 成功回放、失败回放、死因、局限性和下一步改进。

展示前至少组织一次全员模拟答辩。每位成员都应能回答自己模块的问题；4号整理 `presentation/qa.md`，至少准备以下问题：

- 为什么选择路线3而不是纯PPO或纯模仿学习；
- DAgger解决什么分布偏移问题；
- 为什么BC需要类别权重；
- `done` 如何影响GAE；
- PPO裁剪、动作掩码和熵如何计算；
- 持续BC损失是否真的有贡献，消融证据是什么；
- 为什么不能只挖坑过夜；
- 最终策略为什么没有使用特权观测；
- 结果是否在多个种子稳定复现。

### 8.9 验收标准

- PPO 数学测试全部通过。
- mock PPO 能训练并保存检查点。
- DAPG 能正确读取 `runs/dagger/bc.pt`。
- 主训练无 NaN，日志和 checkpoint 完整。
- 最佳模型不是只根据某个简单任务选择，应优先关注 `survive_day`。
- 将最终检查点复制或导出为 `weights/best.pt`，并验证能够通过 `load_checkpoint` 加载。
- `pre.pptx` 覆盖方法、实现、结果、消融、技能和失败分析，且所有数值与最终报告一致。
- `presentation/qa.md` 完整，至少完成一次模拟答辩并记录无法回答的问题。

### 8.10 交给1号和3号的产物

```text
weights/best.pt
runs/dapg/ 中的日志和训练曲线数据
至少一个消融结果
deliverables/pre.pptx
presentation/qa.md
handoff/agent4.md
mc_rl/ppo.py 对应的提交或 commit hash
```

3号使用训练日志和消融结果编写最终报告；1号使用 `best.pt`、`pre.pptx` 和交接说明完成评测与打包。两人都只读4号产物，不修改 `ppo.py`、PPT或权重。

## 9. 统一交接文件模板

每个 Agent 完成后必须创建自己的 `handoff/agentN.md`，使用以下结构：

````markdown
# Agent N 交接说明

## 修改范围
- 修改文件：
- 未修改但读取的关键文件：
- Git 分支与 commit：

## 实现内容
- 完成的 TODO：
- 核心算法或状态机：
- 与原接口保持一致的说明：

## 执行命令
```bash
# 按真实顺序列出运行过的命令
```

## 验收结果
- 测试通过情况：
- 关键数值：
- 已知失败或限制：

## 交付产物
- 文件路径：
- 文件用途：
- 下游 Agent 应如何加载：

## 下游注意事项
- 必须保持的参数或模型结构：
- 不能假设的内容：
- 如果加载失败应先检查什么：
````

不得写“测试已通过”而不提供命令和结果摘要；不得写“权重在目录里”而不提供确切路径。

## 10. 集成检查清单

### 10.1 不启动 Minecraft 的检查

```bash
python -m unittest tests.test_survival
python -m unittest tests.test_mc_rl
python scripts/mcrl.py tables
python -c "from submission import agent; print(callable(agent))"
```

路线 3 的相关 TODO 测试必须实际运行并通过；只有可选自创任务测试可以继续跳过。

### 10.2 真实环境检查

```bash
export JAVA_HOME="$CONDA_PREFIX"
export PATH="$JAVA_HOME/bin:$PATH"
```

确认真实环境能够 reset，然后按顺序完成：

1. 三个基线；
2. 示范采集；
3. BC；
4. DAgger；
5. DAPG；
6. 消融；
7. CLI固定10种子评测并使用 `--record` 生成回放；
8. 正式 `evaluate(agent)` 固定10种子评测；
9. 技能与死因证据；
10. dashboard、报告、PPT和模拟答辩。

固定种子为 `11, 23, 47, 83, 149, 211, 307, 401, 503, 607`。正式评测不生成回放，回放必须来自CLI的 `--record`。

回放与看板命令示例：

```bash
python scripts/mcrl.py render \
  --run runs/eval \
  --npz runs/eval/replays/eval_policy_seed11.npz

python scripts/mcrl.py dashboard \
  --runs 'runs/*' --evals 'runs/eval/*.json' \
  --out deliverables/dashboard.html
```

### 10.3 最终结果表建议

| 模型 | 中位存活步数 | 活过一天种子数 | 砍树 | 工作台 | 工具 | 进食 | 庇护 | 主要死因 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| noop |  |  |  |  |  |  |  |  |
| daytime |  |  |  |  |  |  |  |  |
| random_macro |  |  |  |  |  |  |  |  |
| BC 或 DAgger |  |  |  |  |  |  |  |  |
| DAPG |  |  |  |  |  |  |  |  |
| 消融 |  |  |  |  |  |  |  |  |

技能部分另做多种子复现表，不能只在总结果表中打勾：

| 技能类里程碑 | 完成种子数/10 | 首次完成时刻中位数 | 对应回放证据 | 是否稳定复现 |
| --- | ---: | ---: | --- | --- |
| 原木 |  |  |  |  |
| 木板与木棍 |  |  |  |  |
| 工作台与放置 |  |  |  |  |
| 木剑与木镐 |  |  |  |  |
| 狩猎与获得肉 |  |  |  |  |
| 进食 |  |  |  |  |
| 庇护 |  |  |  |  |
| 煤炭、火把与照明 |  |  |  |  |

## 11. 全员必须遵守的注意事项

1. 禁止修改 `survive_day/` 和 `mc_rl/tasks.py`。
2. 最终策略只能使用标准观测，不能使用老师的特权观测。
3. 正式评测中的环境 reward 恒为0，最终策略不能依赖评测 reward。
4. 训练常用 `action_repeat=8`，正式评测默认是1；框架的 `MacroAgent` 已处理展开，不要重复实现节奏补偿。
5. 不得用 `survived_day` 判定成功，使用 `survival_steps >= 23800`。
6. 不得只提交最终分数；必须保留基线、消融、技能里程碑和失败分析。
7. 不得只靠挖坑封顶过夜；该方案可能提高生存分，但技能分会很低。
8. `submission.py` 不得写死绝对路径，不得在导入时训练、联网或启动游戏。
9. 权重只提交 `submission.py` 实际使用的文件。
10. 真实训练前必须先通过 mock 和数学测试。
11. 禁止关闭怪物、关闭饥饿或开启创造模式。
12. 禁止在评测时读取任务 XML 以外的游戏内存。
13. 禁止抄袭其他同学代码；可以讨论思路，但不能提交高度相同的源码。
14. 策略训练预算为10万决策步或100局，先到为准；示范、三基线和消融评测不计入预算，但实际规模必须写入报告，超预算必须说明。
15. 纯存活计时里程碑不得重复作为技能成果，扩展任务和自创任务不单独加分。

## 12. 完成定义

只有同时满足以下条件，整个小组任务才算完成：

- 四个 TODO 文件分别由对应所有者完成；
- 所有路线 3 相关测试通过；
- 三个基线结果完整；
- 示范、BC、DAgger、DAPG 产物沿规定顺序完成交接；
- 最终策略能被正式 `evaluate(agent)` 接口加载并跑完10个种子；
- 报告包含方法、配置、基线、消融、技能、死因和局限性；
- 至少一局回放 HTML 和训练 dashboard 可正常打开；
- PPT覆盖方法选择、关键实现、结果对比、消融和失败分析，并完成至少一次模拟答辩；
- 生存、技能、展示三个评分块都有可核验材料；
- 提交包命名为 `组长学号.zip`，顶层目录为学号，内部文件名为英文且结构与作业说明一致；
- 训练预算和示范、基线、消融的实际规模已写入报告；
- 没有人修改其他人的独占文件或评测规格文件。


## 13. 1号与2号职责边界

1号和2号在早期游戏行为上有部分相似，但在代码职责、目的和产物上不冲突。

| 对比项 | 1号：白天脚本基线 | 2号：老师生存策略 |
|---|---|---|
| 文件 | `mc_rl/agents.py` | `mc_rl/expert.py` |
| 用途 | 无学习对照组 | 生成模仿学习示范 |
| 观测 | 仅标准观测 | 训练时允许特权观测 |
| 行为范围 | 砍到原木、合成木板/木棍/工作台、随后蹲下 | 完整完成采集、工具、狩猎、进食、庇护和夜间生存 |
| 后续产物 | `daytime.json` | `demo_0.npz` |
| 下游 | 用于最终基线比较 | 交给3号训练BC/DAgger |
| 是否互相调用 | 否 | 否 |

关键边界是：

### 1号只能完成固定基线

1号完成工作台后就停止扩展技能，持续蹲下或安全空操作。

明确禁止1号实现：

- 放置工作台
- 制作木剑、木镐
- 狩猎
- 进食
- 挖洞或建庇护所
- 返回出生点
- 夜间决策

这样才能保证 `daytime` 是一个能力有限的非学习基线。

### 2号实现完整老师

2号的“早期采集、基础合成”只是完整生存技能链的前置阶段。没有原木和工作台，后续工具、狩猎、进食和庇护都无法完成。

2号必须：

- 使用 `expert.py` 已有的 `chop_toward`、`craft_chain` 等技能
- 不调用或复制 `DaytimeScript`
- 不修改 `agents.py`
- 不负责生成 `daytime.json`
- 最终交付带完整技能链的 `demo_0.npz`

因此，两人可以完全并行工作，也不会产生 Git 文件冲突。准确表述应当是：

> 1号实现“刻意受限的固定基线”；2号实现“用于示范采集的完整老师”。二者早期游戏行为相似，但算法、文件、目的和产物完全独立。

## 14. 依据公开资料加入的轻量优化要求

本节只补强现有路线，不增加新的技术路线、第三方模型或跨人员文件修改。四个 Agent 不需要复现完整 Plan4MC，不引入 MineCLIP、在线大模型、循环网络或新的宏动作，也不额外要求 CNN/MLP 对比。先完成老师指定的 5 个 TODO、测试和正式交付，再做以下小幅优化。

### 14.1 1号：增加一张技能链结果表

除最终存活步数外，1号从已有 JSON 和回放整理以下条件结果：

```text
获得原木 -> 完成木板/木棍 -> 工作台 -> 工具 -> 食物/进食 -> 庇护 -> 活过一天
```

每一项只统计“完成种子数/10”，不开发新的评测程序。至少比较 `daytime`、BC或DAgger、DAPG、关闭持续BC损失的消融。固定 10 个正式种子尽量在模型冻结后统一运行；训练中调试优先使用 mock 或训练种子，避免反复根据正式种子调参。

### 14.2 2号：在正常示范中加入少量恢复示范

老师策略仍使用现有阶段和基础技能，不改成复杂规划器。真实示范除正常成功流程外，尽量覆盖以下四类恢复状态：

1. 目标树或动物离开视野后重新搜索；
2. 合成材料不足时返回采集，而不是重复无效合成；
3. 临近日落时中止远距离狩猎并转向庇护；
4. 庇护未封闭或饥饿危险时优先补救。

不需要故意执行危险的随机宏动作；可通过不同种子、不同目标方向或小幅视角偏移获得恢复样本。交给3号前只增加一张动作分布表，列出每个宏动作的样本数，并确认合成、放置、装备、攻击、进食和封顶等稀有动作确实出现。

### 14.3 3号：DAgger优先收集错误状态

仍按现有命令运行 3 轮 DAgger，混合系数保持约 `0.5 -> 0.25 -> 0.125`，不增加轮数。每轮额外记录：

- 新增样本数；
- 学习策略与老师动作不一致的比例；
- 稀有动作准确率；
- 一个最常见的失败或恢复案例。

优先保留技能切换点和失败前后的状态，例如“已有木材但没有继续合成”“目标丢失后持续空转”“天黑仍在远处探索”。如果示范缺少某类稀有动作，由3号在交接说明中把缺口退回2号补采，3号不得直接修改 `expert.py`。

### 14.4 4号：保持PPO简单稳定

4号不额外更换网络或损失定义，继续使用现有 CNN、分阶段课程、critic warmup、优势标准化、梯度裁剪、目标 KL 和逐步衰减的 BC 损失。主训练日志必须保留：

```text
pg_loss, vf_loss, entropy, approx_kl, clip_frac, bc_loss, bc_acc,
bc_coef, total_ticks, episodes, curriculum_stage, per_task_success
```

若 KL 突然升高、熵很快接近零或任务成功率连续下降，优先降低学习率或停止继续更新，不同时修改网络、奖励和课程。`best.pt` 依据 `survive_day` 表现和技能复现选择，不能只看训练 loss。消融优先采用“相同 DAgger 初始化、PPO阶段令 `bc_coef=0`”，避免再训练一套完全不同的模型。

### 14.5 轻量优化验收

以上优化完成的最低证据为：

- 1号：一张技能链完成种子数表；
- 2号：一张示范动作分布表，并说明是否包含恢复示范；
- 3号：三轮 DAgger 的样本数、动作分歧率和稀有动作准确率；
- 4号：DAPG 与 `bc_coef=0` 消融的同口径结果及完整训练日志；
- 3号最终把上述内容各用一张表或一段文字放入报告，4号从报告中选取最关键结果放入 PPT。

这些材料全部由现有日志、JSON 和回放生成，不要求新增网页、数据格式或外部依赖。

### 14.6 公开资料依据

- MineRL 2019 竞赛复盘表明，成绩较好的方案普遍使用动作缩减、分层技能和示范数据，因此保留现有宏动作、课程和模仿初始化。
- DAgger 原论文强调收集学习策略实际访问状态上的专家标签，因此只要求3号优先补充错误和恢复状态。
- DAPG及MineRL示范微调工作支持“示范初始化后继续强化学习，并保留模仿约束”，因此继续使用逐步衰减的 BC 损失。
- PPO公开实现建议持续观察 KL、熵、clip fraction、策略损失和价值损失，因此这里只补充日志和异常处理，不扩展复杂算法。

参考链接：

- <https://proceedings.mlr.press/v123/milani20a/milani20a.pdf>
- <https://proceedings.mlr.press/v15/ross11a.html>
- <https://arxiv.org/abs/1709.10087>
- <https://proceedings.mlr.press/v123/scheller20a.html>
- <https://docs.cleanrl.dev/rl-algorithms/ppo/>
