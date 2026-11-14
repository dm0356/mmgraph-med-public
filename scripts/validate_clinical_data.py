"""Validate a clinical case export against docs/CLINICAL_SCHEMA.md.

Usage:
    python scripts/validate_clinical_data.py --cases data/clinical_demo/cases.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmgraph_agent.eval.clinical_schema import validate_case, window_label  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    args = parser.parse_args()

    path = Path(args.cases)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    failures = 0
    for row in rows:
        errors = validate_case(row)
        label = window_label(row)
        if errors:
            failures += 1
            print(f"[FAIL] {row.get('case_id', '?')}: {errors}")
        else:
            print(f"[OK]   {row.get('case_id', '?')}: label={label}")
    print(f"validated {len(rows)} cases, failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
