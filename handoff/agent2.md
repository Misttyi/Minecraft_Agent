# 2 号 lwy 交接说明

## 修改范围

算法变更在 `mc_rl/expert.py` 的 `Expert.survival_strategy`。接口依据为 `task_env.py`、`features.py`、`macros.py`、`tasks.py`、`imitation.py` 及 `tests/test_mc_rl.py`。本分支按项目相对路径交接。

## 策略

老师先收集三份原木，按库存合成木板、木棍、工作台和工具。木剑先于木镐；狩猎复用装备木剑与按下再松开的攻击宏。木材不够时重新找树，目标消失时复用探索技能。饥饿值低于 15 且有肉时装备并进食。获得四份储备且至少过了 2400 刻后可提前庇护；受重伤也可提前庇护。到 9000 刻结束狩猎，在距出生点 4 至 24 格的范围内尝试返程，至 10800 刻就地开挖。挖井后继续封顶，已封闭时按需进食。火把等独立课程沿用原有老师。训练数据由原有采集器输出，学生只接收标准观测及合法宏标签。

## 执行命令与验收

在包含 `scripts/` 的项目根目录执行。Windows 验证使用以下命令：

```powershell
$env:PYTHONPATH='tests;.'
python -m unittest -v test_mc_rl.TaskEnvTest.test_expert_survives_mock_day test_mc_rl.TaskEnvTest.test_expert_solves_easy_tasks_in_mock
python -m unittest discover -s tests -v
```

两个指定测试均实际通过。2 号独立开发时，全套 43 项，38 项通过，5 项跳过；在含 1 号代码的干净检出中复测 43 项，39 项通过，4 项跳过（BC/PPO 综合训练、GAE、PPO 损失、可选自创任务），零失败。完整读取示范、逐行核对标签与掩码、核对 SHA-256，并用课程 CLI 从进食回放成功生成 HTML。种子 3 的 mock 生存结果、训练种子 1001 的复验及恢复分支测试见 `evidence/agent2/skill_analysis.md`。另用临时目录运行 `demo --env mock --tasks wood,craft,tools,hunt,eat,shelter,torch --num-envs 2 --steps 400`，经 BC 使用的 `load_demos` 读回 800 个样本，逐行验证标签在动作掩码内。

## 真实采集环境

本机 Ubuntu-24.04（WSL2），可用内存约 15 GiB，采用 3 个实例。用户目录中的独立 Python 3.8 和 JDK 8、CPU 版 PyTorch 2.4.1、课程 MineRL 0.4.4 wheel、NumPy 兼容补丁及 Xvfb 用于此次采集。下面的复跑示例把主批写入单独目录，避免覆盖已交付的合并示范。LWJGL 启动还需要 `xrandr`，Ubuntu 软件包名为 `x11-xserver-utils`。

```bash
d="$HOME/.local/share/mc-lwy"
export JAVA_HOME="$d/jdk8"
export PATH="$d/py38/bin:$JAVA_HOME/bin:$PATH"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
xvfb-run -a python scripts/mcrl.py demo --env minerl --run runs/recollect/base --num-envs 3 --steps 1500 --device cpu
```

## 最终评测证据

待 1 号交付同一检查点对应的十种子正式评测 JSON、CLI JSON 与全部种子的原始回放。当前交接的技能核验记录位于 `evidence/agent2/skill_analysis.md`。报告和展示请分别标明训练老师、学习策略，以及 mock、真实环境的来源。

## 采样种子、补采命令与版本

主批调度随机种子 0，训练世界从 1000–1999 抽取。三个实例的世界切换记录分别为 1850→1872、1636→1873→1865→1947→1435→1511、1511→1480→1130→1785→1756→1208。客户端为下一局准备世界也可能出现在日志中。正式十种子全部留给评测。

```bash
xvfb-run -a python scripts/mcrl.py demo --env minerl --run runs/demo/extra_eat --tasks eat --train-seeds 1872 --num-envs 1 --steps 800 --seed 1 --device cpu
xvfb-run -a python scripts/mcrl.py demo --env minerl --run runs/demo/extra_day --tasks survive_day --train-seeds 1872 --num-envs 1 --steps 750 --seed 2 --device cpu
xvfb-run -a python scripts/mcrl.py demo --env minerl --run runs/demo/extra_torch --tasks torch --train-seeds 1130 --num-envs 1 --steps 550 --seed 3 --device cpu
PYTHONPATH=. xvfb-run -a python runs/demo/selected_eat/collect_eat.py
xvfb-run -a python scripts/mcrl.py demo --run runs/demo/extra_tools --tasks tools --train-seeds 1130 --num-envs 1 --steps 160 --seed 5 --device cpu
xvfb-run -a python scripts/mcrl.py demo --run runs/demo/extra_eat_recovery --tasks eat --train-seeds 1112 --num-envs 1 --steps 900 --seed 6 --device cpu
```

进食补采脚本在 1100–1123 的自然出生点检查教师可见的树木及附近牛猪。满足条件的 1112、1118 各跑一局，其余 22 次只 reset，未保存样本。早期轨迹反映旧进食行为：满饥饿值时继续疾跑跳跃。最后一个补采批次使用调整后的进食老师，获肉先装备、遇阻重新寻路、可进食时执行 eat。两版都是课程真实环境的观测、合法动作和真实结果。这个定向补采集合用于提供稀有标签，成功比例受任务与出生点选择影响。

## 真实示范质量与交接

3 号加载[合并示范](../runs/demo/demo_0.npz)。下表记录采集机上的原始分批路径。仓库提供合并示范、[逐局记录](../runs/demo/demo_0_episodes.jsonl)及证据引用的原始回放。此文件合并以下真实 MineRL 批次，每行一个决策标签。共 8520 条，经原版 BC 的 `mc_rl.imitation.load_demos` 实际读回；向量 61 维，图像 64 × 64 × 3，动作掩码 33 维。逐行检查输入长度、向量有限性、标签范围及标签对应动作合法性。

| 原始批次 | 决策步 | 已结束成功局 | 已结束失败局 | 未结算局的决策步 |
| --- | ---: | ---: | ---: | ---: |
| `runs/demo/base/demo_0.npz` | 4500 | 3 | 8 | 1360 |
| `runs/demo/extra_day/demo_2.npz` | 750 | 0 | 0 | 750 |
| `runs/demo/extra_eat/demo_1.npz` | 800 | 0 | 1 | 250 |
| `runs/demo/extra_eat_recovery/demo_6.npz` | 900 | 1 | 1 | 83 |
| `runs/demo/extra_tools/demo_5.npz` | 160 | 2 | 0 | 65 |
| `runs/demo/extra_torch/demo_3.npz` | 550 | 0 | 1 | 232 |
| `runs/demo/selected_eat/demo_1112.npz` | 400 | 0 | 1 | 0 |
| `runs/demo/selected_eat/demo_1118.npz` | 460 | 0 | 1 | 0 |

已结束 19 局，按各自课程的成功条件统计，6 局成功，13 局失败（成功率 31.6%）。未结算轨迹贡献的示范保留，未计入成功率。已结算局死亡 0 次。各原始批次有同名 `_episodes.jsonl`；仓库中供小组核对的合并版位于 `runs/demo/demo_0_episodes.jsonl`。

各任务样本数：explore 165，wood 204，craft 751，tools 160，hunt 1054，eat 2560，shelter 409，nightfall 1211，survive_day 878，torch 1128。

| 宏动作 | 样本数 | 宏动作 | 样本数 |
| --- | ---: | --- | ---: |
| noop | 0 | craft_table | 18 |
| forward | 2575 | craft_torch | 0 |
| forward_jump | 1052 | place_table | 49 |
| back | 10 | craft_pickaxe | 8 |
| turn_left | 175 | craft_sword | 16 |
| turn_right | 198 | equip_sword | 11 |
| turn_left_big | 294 | equip_pickaxe | 3 |
| turn_right_big | 555 | equip_meat | 2 |
| look_up | 243 | eat | 1 |
| look_down | 62 | dig_down | 586 |
| look_level | 175 | seal_above | 108 |
| attack | 98 | place_torch | 0 |
| chop | 656 | craft_furnace | 0 |
| use | 0 | place_furnace | 0 |
| sneak | 873 | smelt_meat | 0 |
| craft_planks | 60 | sprint_jump | 674 |
| craft_stick | 18 |  |  |

成品 SHA-256 为 `fc3194e4cbbc38e44545b33bea986351258fa761d3040149e2c4780dcd0911ea`。BC 输入只需该合并文件，同时传入分批原件会重复计算样本。
