"""Score a completed review sheet against extraction output.

Usage:
    python scripts/score_extraction.py \
        --features out/clinical_features/features.jsonl \
        --review out/clinical_features/review_sheet.csv \
        --out out/clinical_features/review_metrics.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmgraph_agent.eval.clinical_review import score_review  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--out", default="out/clinical_features/review_metrics.json")
    parser.add_argument("--tolerance", type=float, default=0.01)
    args = parser.parse_args()

    records = [json.loads(line) for line in Path(args.features).read_text(encoding="utf-8").splitlines() if line.strip()]
    with Path(args.review).open(encoding="utf-8-sig", newline="") as handle:
        reviewed = list(csv.DictReader(handle))

    summary = score_review(records, reviewed, tolerance=args.tolerance)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
