# Minecraft 生存智能体

小组采用路线 3：用脚本老师采集真实 MineRL 示范，以 BC 和 DAgger 训练学生策略，再用带持续模仿损失的 PPO（DAPG）微调。环境安装、预算与评分口径见[课程说明](大作业2-Minecraft.md)，文件所有权与验收顺序见[四人分工](handoff/team-plan.md)。

## 成员交接

| 成员 | 读取 | 交付给下一位 |
| --- | --- | --- |
| 2 号 lwy · 老师与示范 | 原版训练环境和任务规格 | 已交付 [8520 条真实示范](runs/demo/demo_0.npz)、[逐局记录](runs/demo/demo_0_episodes.jsonl)、[动作分布与采集命令](handoff/agent2.md)。3 号可以开始训练 |
| 3 号 lqr · BC 与 DAgger | 2 号的合并示范 | 完成 `mc_rl/imitation.py` 后训练，向 4 号交付 `runs/dagger/bc.pt`、模型配置与训练记录 |
| 4 号 hyw · DAPG 与消融 | 3 号的 DAgger 权重 | 向 1 号交付最终的 `weights/best.pt`，向 3 号交付训练日志与消融结果 |
| 1 号 xky · 评测与集成 | 4 号的最终权重 | [白天基线与提交入口已就绪](handoff/agent1.md)。正式结果写入 `results/policy.json`，同权重的 CLI 结果写入 `runs/eval/policy.json`，10 个种子的原始回放保存在 `runs/eval/replays/` |

1 号交付真实十种子评测后，2 号据此更新[技能与失败分析](evidence/agent2/skill_analysis.md)。3 号汇总报告与看板，4 号制作展示，1 号集成提交包。当前证据页中的真实回放均来自训练老师。

## 读取示范与启动 BC

在仓库根目录、装好课程依赖的 Python 环境中运行。读取和校验示范可以离线完成。

```bash
python -c "from mc_rl.imitation import load_demos; d=load_demos(['runs/demo/demo_0.npz']); print({k: v.shape for k, v in d.items()})"
```

应读到 8520 行。每行有一张 64 × 64 的 RGB 图像、61 维向量、33 项动作掩码和一个老师动作标签。合并文件的 SHA-256 是 `fc3194e4cbbc38e44545b33bea986351258fa761d3040149e2c4780dcd0911ea`。数据中有 1 条实际进食标签，制作和放置火把均为 0 条，报告稀有动作准确率时应同时列出样本数。

3 号实现 `bc_step` 后运行：

```bash
python scripts/mcrl.py bc --demos runs/demo/demo_0.npz --run runs/bc --device cpu
```

## 测试与回放

```bash
python -m unittest discover -s tests -v
python scripts/mcrl.py render --npz runs/demo/extra_eat_recovery/replays/demo_0001_eat_100.npz
```

第二条命令生成可在浏览器打开的老师回放。该局的[关键帧](evidence/agent2/teacher_eat_1112.png)记录了从猎获到进食的过程。小组最终按各固定种子的 `survival_steps >= 23800` 统计生存成功，CLI 回放用于核验每种技能的实际表现。
