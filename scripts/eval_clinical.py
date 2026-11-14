"""Clinical warning evaluation harness.

Input predictions JSONL (one row per case):
    {"case_id": "...", "patient_id": "...", "score": 0.82,
     "label": 1, "alert_time": "...", "event_time": "..."}

Usage:
    # synthetic demo (no hospital data)
    python scripts/eval_clinical.py --synthetic 400 --out out/clinical_eval.json

    # real predictions
    python scripts/eval_clinical.py --predictions preds.jsonl --out out/clinical_eval.json
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmgraph_agent.eval.clinical_metrics import evaluate_predictions  # noqa: E402


def synthetic(n: int = 400, event_rate: float = 0.2, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    base = datetime(2026, 1, 1, 0, 0, 0)
    for index in range(n):
        positive = rng.random() < event_rate
        score = float(rng.beta(5, 2) if positive else rng.beta(2, 5))
        row = {
            "case_id": f"SYN-{index:04d}",
            "patient_id": f"P{index // 3:04d}",
            "score": round(score, 4),
            "label": 1 if positive else 0,
        }
        if positive:
            event = base + timedelta(hours=float(rng.uniform(6, 72)))
            lead = float(rng.uniform(1, 24))
            row["event_time"] = event.isoformat()
            row["alert_time"] = (event - timedelta(hours=lead)).isoformat()
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--synthetic", type=int, default=0)
    parser.add_argument("--specificity", type=float, default=0.9)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--out", default="out/clinical_eval.json")
    args = parser.parse_args()

    if args.predictions:
        rows = [json.loads(line) for line in Path(args.predictions).read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        rows = synthetic(args.synthetic or 400)
        print(f"using {len(rows)} synthetic predictions (no hospital data)")

    summary = evaluate_predictions(rows, specificity=args.specificity, n_boot=args.bootstrap)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
