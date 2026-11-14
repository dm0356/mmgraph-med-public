"""Human-review workflow for structured feature extraction.

1. ``make_review_rows`` selects uncertain / out-of-range / low-confidence records
   and produces a CSV sheet with empty reviewer columns.
2. ``score_review`` joins the completed sheet with the extraction output and
   computes field-level status accuracy, present/absent F1, numeric error and
   Cohen's kappa.
"""

from __future__ import annotations

import random
from typing import Any

REVIEW_COLUMNS = [
    "patient_id", "variable", "rule_value", "rule_status", "confidence",
    "snippet", "time", "source", "extractor", "needs_review",
    "review_value", "review_status", "reviewer", "comment",
]


def make_review_rows(records: list[dict], max_rows: int = 200,
                     low_confidence: float = 0.8, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    priority: list[tuple[str, dict]] = []
    rest: list[dict] = []
    for record in records:
        status = record.get("status")
        if status in ("uncertain", "out_of_range"):
            priority.append((status, record))
        elif status == "present" and float(record.get("confidence", 1.0)) < low_confidence:
            priority.append(("low_confidence", record))
        else:
            rest.append(record)
    rng.shuffle(rest)
    priority_records = [record for _reason, record in priority]
    if len(priority_records) >= max_rows:
        selected = priority_records[:max_rows]
    else:
        selected = priority_records + rest[: max_rows - len(priority_records)]
    reason_of = {id(record): reason for reason, record in priority}

    rows = []
    for record in selected:
        rows.append({
            "patient_id": record.get("patient_id", ""),
            "variable": record.get("variable", ""),
            "rule_value": "" if record.get("value") is None else record.get("value"),
            "rule_status": record.get("status", ""),
            "confidence": record.get("confidence", 0.0),
            "snippet": record.get("snippet", ""),
            "time": record.get("time", ""),
            "source": record.get("source", ""),
            "extractor": record.get("extractor", "rule"),
            "needs_review": reason_of.get(id(record), "random_sample"),
            "review_value": "",
            "review_status": "",
            "reviewer": "",
            "comment": "",
        })
    return rows


def _kappa(y1: list[str], y2: list[str]) -> float:
    n = len(y1)
    if n == 0:
        return 0.0
    labels = sorted(set(y1) | set(y2))
    observed = sum(a == b for a, b in zip(y1, y2)) / n
    expected = sum((y1.count(label) / n) * (y2.count(label) / n) for label in labels)
    return (observed - expected) / (1 - expected) if expected < 1 else 0.0


def _parse_value(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def score_review(records: list[dict], reviewed_rows: list[dict],
                 tolerance: float = 0.01) -> dict:
    indexed = {(record.get("patient_id"), record.get("variable")): record for record in records}
    y_pred, y_gold = [], []
    per_variable: dict[str, dict] = {}
    tp = fp = fn = tn = 0
    numeric_errors: list[float] = []
    status_correct = 0
    used = 0

    for row in reviewed_rows:
        gold_status = str(row.get("review_status") or "").strip()
        if not gold_status:
            continue
        record = indexed.get((row.get("patient_id"), row.get("variable")))
        if record is None:
            continue
        used += 1
        pred_status = record.get("status", "missing")
        y_pred.append(pred_status)
        y_gold.append(gold_status)
        status_correct += int(pred_status == gold_status)

        pred_pos = pred_status == "present"
        gold_pos = gold_status == "present"
        if pred_pos and gold_pos:
            tp += 1
            pred_value = _parse_value(record.get("value"))
            gold_value = _parse_value(row.get("review_value"))
            if pred_value is not None and gold_value is not None:
                numeric_errors.append(abs(pred_value - gold_value))
        elif pred_pos and not gold_pos:
            fp += 1
        elif not pred_pos and gold_pos:
            fn += 1
        else:
            tn += 1

        key = record.get("variable", "unknown")
        bucket = per_variable.setdefault(key, {"n": 0, "correct": 0, "mae": []})
        bucket["n"] += 1
        bucket["correct"] += int(pred_status == gold_status)
        if pred_pos and gold_pos:
            pred_value = _parse_value(record.get("value"))
            gold_value = _parse_value(row.get("review_value"))
            if pred_value is not None and gold_value is not None:
                bucket["mae"].append(abs(pred_value - gold_value))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    for bucket in per_variable.values():
        bucket["status_accuracy"] = bucket["correct"] / bucket["n"] if bucket["n"] else 0.0
        bucket["mae"] = sum(bucket["mae"]) / len(bucket["mae"]) if bucket["mae"] else None
        del bucket["correct"]

    return {
        "reviewed": used,
        "status_accuracy": status_correct / used if used else 0.0,
        "present_precision": precision,
        "present_recall": recall,
        "present_f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "numeric_mae": sum(numeric_errors) / len(numeric_errors) if numeric_errors else None,
        "numeric_within_tolerance": (
            sum(1 for error in numeric_errors if error <= tolerance) / len(numeric_errors)
            if numeric_errors else None
        ),
        "cohen_kappa": _kappa(y_gold, y_pred),
        "per_variable": per_variable,
    }


__all__ = ["REVIEW_COLUMNS", "make_review_rows", "score_review"]
