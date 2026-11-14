"""Evaluation metrics (paper: token-level F1 is the headline metric)."""

from __future__ import annotations

from ..utils import tokenize


def token_f1(prediction: str, gold: str) -> float:
    """SQuAD-style token F1 between a prediction and a gold answer."""
    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(gold)
    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0

    remaining: dict[str, int] = {}
    for token in pred_tokens:
        remaining[token] = remaining.get(token, 0) + 1

    overlap = 0
    for token in gold_tokens:
        if remaining.get(token, 0) > 0:
            remaining[token] -= 1
            overlap += 1

    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if tokenize(prediction) == tokenize(gold) else 0.0


def answer_recall(answer: str, context: str) -> float:
    """Fraction of gold-answer tokens that appear anywhere in the context."""
    gold_tokens = tokenize(answer)
    if not gold_tokens:
        return 1.0
    context_tokens = set(tokenize(context))
    hits = sum(1 for token in gold_tokens if token in context_tokens)
    return hits / len(gold_tokens)


__all__ = ["token_f1", "exact_match", "answer_recall"]
