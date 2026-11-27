"""助教评测入口，加载小组训练导出的最终策略。"""

from pathlib import Path

from mc_rl.agents import PolicyAgent
from mc_rl.models import load_checkpoint


HERE = Path(__file__).resolve().parent
model, _ = load_checkpoint(str(HERE / "weights" / "best.pt"), device="cpu")
agent = PolicyAgent(model, device="cpu", deterministic=True)
