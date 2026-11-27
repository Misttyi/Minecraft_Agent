"""Collect whole training episodes from natural spawns with nearby livestock and trees."""
import json
from pathlib import Path

import numpy as np

from mc_rl.expert import Expert
from mc_rl.render import save_replay
from mc_rl.task_env import TaskEnv, make_base_factory


def main():
    root = Path(__file__).resolve().parent
    env = TaskEnv(make_base_factory("minerl"), expert=Expert(seed=4), record=True)
    try:
        for seed in range(1100, 1124):
            obs = env.reset("eat", seed)
            log, animal = (env.priv.get(k) for k in ("nearest_log", "nearest_animal"))
            print("spawn", seed, log, animal, flush=True)
            if log is None or animal is None or animal > 24:
                continue
            rows = {"vec": [], "mask": [], "pov": [], "labels": []}
            done = False
            while not done:
                label = env.expert_action()
                for key in ("vec", "mask", "pov"):
                    rows[key].append(obs[key].copy())
                rows["labels"].append(label)
                obs, _, done, info = env.step(label)
            ep = info["episode"]
            print(json.dumps(ep), flush=True)
            path = root / f"demo_{seed}.npz"
            np.savez_compressed(path, **{key: np.stack(values) for key, values in rows.items()})
            path.with_name(path.stem + "_episodes.jsonl").write_text(json.dumps(ep) + "\n")
            if ep["success"]:
                save_replay(str(root), f"eat_{seed}", {"episode": ep, "seed": seed,
                            "frames": np.stack(env.frames), "trace": env.trace})
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
