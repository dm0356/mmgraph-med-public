"""Clinical evaluation metrics for warning models (stdlib + numpy).

Implemented without scikit-learn so the harness runs on the cluster's base env:
AUROC, AUPRC, sensitivity/specificity/PPV/NPV, Brier score, ECE, lead time and
bootstrap confidence intervals.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence

import numpy as np


def _arrays(labels: Sequence[int], scores: Sequence[float]):
    y = np.asarray(list(labels), dtype=float)
    s = np.asarray(list(scores), dtype=float)
    if y.shape != s.shape:
        raise ValueError("labels and scores must have the same length")
    mask = np.isfinite(y) & np.isfinite(s)
    return y[mask], s[mask]


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    y, s = _arrays(labels, scores)
    positives, negatives = int((y == 1).sum()), int((y == 0).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    sorted_scores = s[order]
    start = 0
    while start < len(s):
        end = start + 1
        while end < len(s) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def auprc(labels: Sequence[int], scores: Sequence[float]) -> float:
    y, s = _arrays(labels, scores)
    if int((y == 1).sum()) == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(1, int((y == 1).sum()))
    points = np.concatenate([[0.0], recall, [1.0]])
    values = np.concatenate([[precision[0] if len(precision) else 1.0], precision,
                             [precision[-1] if len(precision) else 1.0]])
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(values, points))


def threshold_at_specificity(labels: Sequence[int], scores: Sequence[float],
                             specificity: float = 0.9) -> float:
    y, s = _arrays(labels, scores)
    negatives = s[y == 0]
    if len(negatives) == 0:
        return float("nan")
    # threshold such that `specificity` fraction of negatives fall below it
    return float(np.quantile(negatives, specificity))


def _binary_at(labels: Sequence[int], scores: Sequence[float], threshold: float):
    y, s = _arrays(labels, scores)
    predicted = (s >= threshold).astype(int)
    tp = int(((predicted == 1) & (y == 1)).sum())
    fp = int(((predicted == 1) & (y == 0)).sum())
    tn = int(((predicted == 0) & (y == 0)).sum())
    fn = int(((predicted == 0) & (y == 1)).sum())
    return tp, fp, tn, fn


def sensitivity_specificity(labels, scores, threshold: float) -> tuple[float, float]:
    tp, fp, tn, fn = _binary_at(labels, scores, threshold)
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    return sensitivity, specificity


def ppv_npv(labels, scores, threshold: float) -> tuple[float, float]:
    tp, fp, tn, fn = _binary_at(labels, scores, threshold)
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    return ppv, npv


def brier_score(labels, probs) -> float:
    y, p = _arrays(labels, probs)
    return float(np.mean((p - y) ** 2)) if len(y) else float("nan")


def expected_calibration_error(labels, probs, bins: int = 10) -> float:
    y, p = _arrays(labels, probs)
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (p >= lower) & (p < upper if index < bins - 1 else p <= upper)
        if mask.sum() == 0:
            continue
        total += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(total)


def bootstrap_ci(metric: Callable[[Sequence[int], Sequence[float]], float],
                 labels, scores, n_boot: int = 500, seed: int = 0) -> tuple[float, float]:
    y = np.asarray(list(labels), dtype=float)
    s = np.asarray(list(scores), dtype=float)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        index = rng.integers(0, len(y), len(y))
        value = metric(y[index], s[index])
        if math.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def lead_time_hours(alert_times: Iterable[str], event_times: Iterable[str]) -> dict:
    """Lead time for alerts that fired before the event (hours)."""
    from datetime import datetime

    def parse(value):
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")) if value else None

    leads = []
    for alert, event in zip(alert_times, event_times):
        alert_dt, event_dt = parse(alert), parse(event)
        if alert_dt is None or event_dt is None:
            continue
        hours = (event_dt - alert_dt).total_seconds() / 3600.0
        if hours > 0:
            leads.append(hours)
    if not leads:
        return {"n": 0, "median_hours": float("nan"), "iqr_hours": float("nan")}
    array = np.asarray(leads)
    return {
        "n": len(leads),
        "median_hours": float(np.median(array)),
        "iqr_hours": float(np.percentile(array, 75) - np.percentile(array, 25)),
        "p90_hours": float(np.percentile(array, 90)),
    }


def evaluate_predictions(rows: list[dict], specificity: float = 0.9,
                         n_boot: int = 500) -> dict:
    labels = [row["label"] for row in rows]
    scores = [row["score"] for row in rows]
    threshold = threshold_at_specificity(labels, scores, specificity)
    sensitivity, spec = sensitivity_specificity(labels, scores, threshold)
    ppv, npv = ppv_npv(labels, scores, threshold)
    auc_lo, auc_hi = bootstrap_ci(auroc, labels, scores, n_boot=n_boot)
    sens_lo, sens_hi = bootstrap_ci(
        lambda y, s: sensitivity_specificity(y, s, threshold)[0], labels, scores, n_boot=n_boot
    )
    return {
        "n": len(rows),
        "positives": int(sum(1 for label in labels if label == 1)),
        "event_rate": float(np.mean(labels)) if labels else float("nan"),
        "auroc": auroc(labels, scores),
        "auroc_ci95": [auc_lo, auc_hi],
        "auprc": auprc(labels, scores),
        "threshold_at_specificity": threshold,
        "specificity": spec,
        "sensitivity": sensitivity,
        "sensitivity_ci95": [sens_lo, sens_hi],
        "ppv": ppv,
        "npv": npv,
        "brier": brier_score(labels, scores),
        "ece": expected_calibration_error(labels, scores),
        "lead_time": lead_time_hours(
            [row.get("alert_time") for row in rows],
            [row.get("event_time") for row in rows],
        ),
    }
