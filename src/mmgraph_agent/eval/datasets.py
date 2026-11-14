"""Dataset loaders.

The smoke dataset is a tiny synthetic multimodal corpus committed to the repo.
The 23-dataset MMGraph-Agent benchmark list is documented here as a roadmap;
real loaders are intentionally left as explicit stubs so a CPU-only user gets a
clear error instead of a silent failure.
"""

from __future__ import annotations

import json
from pathlib import Path

# Roadmap: the paper evaluates on 23 datasets across three regimes.
K_ONLY = ["DocBench", "MMLongBench-Doc", "ViDoSeek", "2Wiki", "HotpotQA", "Musique", "NQ", "PopQA", "TriviaQA"]
WEB_ONLY = ["FVQA", "InfoSeek", "MMSearch", "SimpleVQA", "LiveVQA"]
K_WEB = [
    "MedXpertQA",
    "FinMME",
    "We-Math2.0",
    "RBench",
    "MicroVQA",
    "EEmo-Bench",
    "ArtQuest",
    "AstroLLaVA",
    "GeoGrid-Bench",
]
ALL_DATASETS = K_ONLY + WEB_ONLY + K_WEB


def _read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_docs(path: str | Path) -> list[dict]:
    return _read_jsonl(path)


def load_qa(path: str | Path) -> list[dict]:
    return _read_jsonl(path)


def load_smoke(root: str | Path = "data/smoke") -> tuple[list[dict], list[dict]]:
    root = Path(root)
    docs = load_docs(root / "corpus.jsonl")
    qa = load_qa(root / "qa.jsonl")
    return docs, qa


def get_dataset(name: str) -> tuple[list[dict], list[dict]]:
    """Return ``(docs, qa)`` for a dataset name.

    Only ``smoke`` is implemented for the zero-dependency reproduction path.
    """
    if name == "smoke":
        return load_smoke()
    if name in ALL_DATASETS:
        raise NotImplementedError(
            f"Dataset {name!r} is on the MMGraph-Agent benchmark list but has no "
            "local loader yet. Download its raw multimodal corpus, convert it to "
            "the JSONL schema used by data/smoke (doc_id, text, images/tables/"
            "diagrams), then implement a loader in eval/datasets.py."
        )
    raise ValueError(f"Unknown dataset {name!r}. Try 'smoke'.")


__all__ = [
    "load_docs",
    "load_qa",
    "load_smoke",
    "get_dataset",
    "ALL_DATASETS",
    "K_ONLY",
    "WEB_ONLY",
    "K_WEB",
]
