"""Build the zero-API multimodal hypergraph from a JSONL corpus.

Usage:  python scripts/build_hypergraph.py --config configs/smoke.toml --out out/mmkg.json
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmgraph_agent.cli import build  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(build())
