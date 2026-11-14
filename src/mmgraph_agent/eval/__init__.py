"""Evaluation harness (paper §4: token F1 across regimes)."""

from .datasets import ALL_DATASETS, get_dataset, load_docs, load_qa, load_smoke
from .metrics import answer_recall, exact_match, token_f1
from .runner import evaluate, run_episode
from .validation import validate_corpus, validate_qa

__all__ = [
    "load_docs",
    "load_qa",
    "load_smoke",
    "get_dataset",
    "ALL_DATASETS",
    "token_f1",
    "exact_match",
    "answer_recall",
    "run_episode",
    "evaluate",
    "validate_corpus",
    "validate_qa",
]
