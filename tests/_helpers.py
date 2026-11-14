"""Shared helpers for the test suite."""

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mmgraph_agent.config import DEFAULTS  # noqa: E402


def make_config() -> dict:
    """Deterministic config: regex parser, hashing embedder."""
    cfg = copy.deepcopy(DEFAULTS)
    cfg["hypergraph"]["parser"] = "regex"
    cfg["hypergraph"]["embedder"] = "hashing"
    return cfg
