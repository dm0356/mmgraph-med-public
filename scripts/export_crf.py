"""Export the CRF variable dictionary, blank form and demo rows.

Usage:
    python scripts/export_crf.py \
        --crf configs/crf_variables.json \
        --out-dir data/crf \
        --cases data/clinical_demo/cases.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmgraph_agent.eval.crf import load_crf, validate_crf, write_crf_exports  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crf", default="configs/crf_variables.json")
    parser.add_argument("--out-dir", default="data/crf")
    parser.add_argument("--cases", default=None, help="optional JSONL cases for demo rows")
    parser.add_argument("--features", default=None, help="optional extracted features JSONL")
    args = parser.parse_args()

    crf = load_crf(args.crf)
    errors = validate_crf(crf)
    if errors:
        print("CRF validation errors:")
        for error in errors:
            print(f"- {error}")
        return 1

    cases = []
    if args.cases:
        cases = [json.loads(line) for line in Path(args.cases).read_text(encoding="utf-8").splitlines() if line.strip()]
    features_by_patient: dict[str, list[dict]] = {}
    if args.features:
        for line in Path(args.features).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            features_by_patient.setdefault(record.get("patient_id", ""), []).append(record)

    summary = write_crf_exports(crf, args.out_dir, cases=cases,
                                features_by_patient=features_by_patient)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
