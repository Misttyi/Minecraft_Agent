# Agent 1 交接说明 · xky

## 修改范围

- 算法代码 `mc_rl/agents.py`，提交入口 `submission.py`。
- 本地诊断 `results/mock/{noop,daytime,random_macro}.json`。
- 核对了老师的 PPT 第 6–10 页、项目说明第 3–7 节和小组分工第 5、10、14 节。同时核对了宏动作、训练计数、CLI、正式评测、回放和现有测试。
- 当前目录是无 Git 元数据的发布包，暂无分支和 commit。

## 实现内容

- `DaytimeScript` 在太阳亮度大于 0.5 时按固定顺序搜索。每个方向连续执行两个 `chop` 宏，再转 15°，扫描一圈后前移一步。砍到原木后优先就地继续补齐木材。
- 合成木棍和工作台总共需要 6 块木板，通常要两块原木。先用有效库存合成木板和木棍，材料不足时继续补采。合成工作台后保持蹲下。动作由现有掩码校验，只通过 `MacroAgent` 展开。新环境会重置搜索进度。
- `submission.py` 依照课程模板提前写好，导入时从同级 `weights/best.pt` 加载模型，在 CPU 上按确定性策略返回底层动作。

## 执行命令

本机 mock 验证使用已有的 `Dofbot_2026` Conda 环境（Python 3.10，带 PyTorch）。以下命令均在项目根目录运行。

```bash
python -m unittest -v tests.test_survival tests.test_mc_rl
python scripts/mcrl.py tables
python scripts/mcrl.py eval --env mock --agent noop --workers 1 --run results/mock --out results/mock/noop.json
python scripts/mcrl.py eval --env mock --agent daytime --workers 1 --run results/mock --out results/mock/daytime.json
python scripts/mcrl.py eval --env mock --agent random_macro --workers 1 --run results/mock --out results/mock/random_macro.json
python -c "from survive_day import make_env; env=make_env(11, action_repeat=8, strip_reward=True); env.close()"
```

另以临时用例检查了半亮阈值、动作掩码、换环境重置，以及 `action_repeat=1` 和 `8` 下合成工作台后持续蹲下。提交接口由隔离目录内的临时 ActorCritic 检查点验证，正式权重由 4 号交付。

## 验收结果

- 原有单元测试 43 项，38 项通过、5 项跳过，包括 BC/PPO 综合测试、GAE、PPO 损失、老师生存策略及可选自创任务。白天脚本相关的现有接口测试及 3 项临时用例通过。
- 隔离目录中的提交入口可从不同工作目录导入，`callable(agent)` 为真。将正式 API 的环境工厂替换为 mock 后，按默认参数跑完 10 个固定种子并写出 JSON。CLI 另外用同一临时权重在 mock 种子 11 上生成了 HTML 回放。测试权重和回放随临时目录清理。
- 三份 mock JSON 均覆盖固定种子 `11, 23, 47, 83, 149, 211, 307, 401, 503, 607`。下表按单局 `survival_steps >= 23800` 计数。

| mock 基线 | 中位存活游戏刻 | 活过一天 |
| --- | ---: | ---: |
| noop | 13902.5 | 0/10 |
| daytime | 13967.5 | 0/10 |
| random_macro | 24000 | 10/10 |

白天脚本在 mock 的种子 11、23、83、211、607 上得到原木并完成木板、木棍和工作台，共 5/10。统计来自每局库存变化。真实 Minecraft 的三条基线和四组技能链对比均待测。

- 真实环境的 reset 检查报 `ModuleNotFoundError: No module named 'minerl'`。当前机器运行 Windows，`wsl -l -q` 返回失败，所给工程中找不到课程的 Linux/Python 3.8 预编译 MineRL wheel。
- 缺少正式权重 `weights/best.pt`、BC/DAgger 与消融检查点、`deliverables/{dashboard.html,report.pdf,pre.pptx}`、组长学号。正式评测和提交包均待完成。

## 交付产物

| 文件 | 用途 |
| --- | --- |
| `mc_rl/agents.py` | 在项目根目录用 CLI 的 `--agent daytime` 评测 |
| `submission.py` | 放在发布包根目录，加入正式权重后用 `from submission import agent` 加载 |
| `results/mock/*.json` | 本地冒烟结果，用来验证框架入口和种子覆盖 |

## 下游注意事项

### 小组分工需要补齐的细节

1. **提交规则有冲突。** PPT 第 9 页写“每位同学提交一个压缩包”，课程项目说明 §5.1 写“每组一个 `组长学号.zip`”。分工沿用了项目说明的组包口径。提交份数和上传人以助教确认的要求为准。组包内部目录按课程原文保留 `code/mc_rl/`。
2. **训练预算需要显式计数。** 现有 PPO 的 `--total-ticks` 累加底层游戏刻，外层循环仅按这个值退出，且整批 rollout 可以越过阈值。该参数无法自动执行“10 万决策步或 100 局先到即停”。请 4 号在负责的 `ppo.py` 实际计数并记录停训点，3 号提供 BC/DAgger 的样本、环境交互决策和局数，以便汇总实际规模并判断预算。消融评测与训练应分别记账。
3. **学会技能要靠实际观测确认。** CLI JSON 的 `macro_hist` 是动作尝试次数。现有 CLI 评测回放的 `newly` 恒为空，库存字段也仅覆盖部分材料，工具、火把、进食和封顶有时需要逐帧确认。请给 `daytime`、BC/DAgger、DAPG 和消融各保留 10 种子的观测证据，按库存变化或可见效果统计完成种子数。证据不足的项记为未核实，若需要结构化证据，由小组先确认对锁定的 CLI 记录器的变更。若没有成功局，回放和报告如实标明。
4. **报告的存活成功数要重算。** 框架写出的 `survived_day_count` 仍按 `survived_day` 汇总。读取每个种子的 `survival_steps` 用 23800 阈值重算。CLI 默认采样、动作重复 8，提交入口是确定性、正式评测默认动作重复 1。可用同一套 CLI 参数比较模型，并单列正式评测。
5. **3 号需要留存额外指标。** DAgger 当前日志有累计样本和验证历史，缺少学生与老师的实际动作分歧。可在 `imitation.py` 采样处统计实际发生的分歧，或注明仅统计执行学生策略的样本。稀有类准确率同时记录样本量。看板 CLI 按 `metrics.jsonl` 发现训练目录，可从 BC/DAgger 的独立 JSON 补齐报告曲线。交付看板时核对其中的回放相对路径在 `results/dashboard.html` 位置仍可用。

### 真实环境就绪后的评测与打包

先在 Linux / WSL2 中按项目说明完成真实环境 reset。并行度按可用资源调整，以下示例用 2 个 worker，保留全部固定种子。

```bash
xvfb-run -a python scripts/mcrl.py eval --agent noop --workers 2 --out runs/eval/noop.json
xvfb-run -a python scripts/mcrl.py eval --agent daytime --workers 2 --record --run runs/eval --out runs/eval/daytime.json
xvfb-run -a python scripts/mcrl.py eval --agent random_macro --workers 2 --out runs/eval/random_macro.json

python -c "from submission import agent; print(callable(agent))"
xvfb-run -a python scripts/mcrl.py eval --agent policy --checkpoint weights/best.pt --workers 2 --record --run runs/eval --out runs/eval/policy.json

mkdir -p results/replay
xvfb-run -a python - <<'PY'
from submission import agent
from survive_day.eval import evaluate, write_report
report = evaluate(agent)
write_report(report, "results/policy.json")
PY
```

- 用相同 CLI 参数对 BC/DAgger 和消融检查点评测并记录回放，分别设置 `--run runs/eval/dagger`、`--run runs/eval/ablation` 避免覆盖最终策略回放。对比结果可分别落在 `runs/eval/dagger.json`、`runs/eval/ablation.json`。
- 将三个真实基线 JSON 复制到 `results/`。交给 2、3 号的证据应包含 CLI JSON 与 `runs/eval/replays/` 下的回放。正式结果位于 `results/policy.json`。
- 最终包的明确范围是 `submission.py`、`weights/best.pt`、`code/mc_rl/{agents,expert,imitation,ppo}.py`、四份正式结果 JSON、`results/dashboard.html`、`results/replay/` 下的正式策略 HTML、`pre.pptx`、`report.pdf`。材料先检查可打开，再从各自所有者的交付路径原样复制。
- 用组长学号命名顶层目录和 ZIP。检查 ZIP 条目、四个评测 JSON 的种子覆盖和回放链接。将 `code/mc_rl/` 中的文件覆盖到一份原始发布包，按助教的导入方式复验。提交包仅包含课程要求的真实产物。
