"""Render the constructed multimodal hypergraph to an interactive HTML page.

Usage:
    python scripts/render_hypergraph.py --config configs/smoke.toml --data-root data/smoke --out out/mmkg.html
    python scripts/render_hypergraph.py --config configs/smoke_zh.toml --data-root data/smoke_zh --out out/mmkg_zh.html
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmgraph_agent.cli import viz  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(viz())
