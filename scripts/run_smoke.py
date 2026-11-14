"""Run the end-to-end CPU smoke demo (no GPU, no model download).

Usage:  python scripts/run_smoke.py --verbose
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmgraph_agent.cli import smoke  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(smoke())
