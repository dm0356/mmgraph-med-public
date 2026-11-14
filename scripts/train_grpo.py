"""GPU template: GRPO training of the retrieval-and-answer policy.

This script is a *template*. Reproducing the paper's trained policy needs a GPU
stack (torch + transformers), a Qwen2.5-VL-3B checkpoint and rollout data.
The reward used here is ``mmgraph_agent.policy.compute_reward`` (paper Eq. (6)).

Usage (after `pip install -e '.[gpu]'`):
    python scripts/train_grpo.py --config configs/default.toml --output-dir out/grpo
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse  # noqa: E402

from mmgraph_agent.config import load_config  # noqa: E402
from mmgraph_agent.policy import GRPOTrainer, compute_reward  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="GRPO training template (requires GPU stack).")
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--output-dir", default="out/grpo")
    args = parser.parse_args()

    config = load_config(args.config)

    # Demonstrate the reward function works on the CPU without torch.
    print("reward sanity check:")
    print("  1 retrieval, f1=1.0 ->", compute_reward(1.0, 1, True, config))
    print("  2 retrievals, f1=1.0 ->", compute_reward(1.0, 2, True, config))
    print("  0 retrievals, f1=1.0 ->", compute_reward(1.0, 0, False, config))

    trainer = GRPOTrainer(config)
    if not trainer.available:
        print(
            "\nTorch is not installed, so GRPO training cannot run here.\n"
            "Install the GPU extras and implement GRPOTrainer.train() in "
            "src/mmgraph_agent/policy/grpo.py."
        )
        return 1

    print("\nGPU stack detected. Implement GRPOTrainer.train() around a Qwen2.5-VL-3B policy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
