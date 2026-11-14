"""Evaluate a registered dataset.

Usage:  python scripts/evaluate.py --dataset smoke --config configs/smoke.toml --verbose
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmgraph_agent.cli import evaluate  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(evaluate())
