"""Create a human-review CSV from extracted features.

Usage:
    python scripts/make_review_sheet.py \
        --features out/clinical_features/features.jsonl \
        --out out/clinical_features/review_sheet.csv --max 200
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmgraph_agent.eval.clinical_review import REVIEW_COLUMNS, make_review_rows  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", default="out/clinical_features/review_sheet.csv")
    parser.add_argument("--max", type=int, default=200)
    parser.add_argument("--low-confidence", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    records = [json.loads(line) for line in Path(args.features).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = make_review_rows(records, max_rows=args.max,
                            low_confidence=args.low_confidence, seed=args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"review rows: {len(rows)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
