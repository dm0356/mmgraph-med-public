"""Extract structured variables from clinical report text (rules + optional LLM).

Usage:
    python scripts/extract_clinical_features.py \
        --reports data/clinical_demo/reports.jsonl \
        --out-dir out/clinical_features [--use-llm]

Outputs:
    features.jsonl     long format (patient x variable, provenance, extractor)
    features_wide.csv  patient x variable values (blank = missing)
    conflicts.json     rule/LLM disagreements (if LLM enabled)
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmgraph_agent.config import load_config  # noqa: E402
from mmgraph_agent.eval.clinical_features import extract_report, to_wide_csv  # noqa: E402
from mmgraph_agent.eval.clinical_features_llm import (  # noqa: E402
    LLMFeatureExtractor,
    extract_report_hybrid,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", required=True)
    parser.add_argument("--out-dir", default="out/clinical_features")
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--config", default="configs/cluster_local.toml")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-timeout", type=float, default=180)
    args = parser.parse_args()

    llm = None
    if args.use_llm:
        base_url, model = args.llm_base_url, args.llm_model
        if not base_url:
            config = load_config(args.config)
            extractor = config.get("extractor", {})
            base_url = extractor.get("base_url")
            model = model or extractor.get("model")
        if base_url:
            llm = LLMFeatureExtractor(base_url, model or "local-model", timeout=args.llm_timeout)
        else:
            print("LLM disabled: no [extractor].base_url / --llm-base-url")

    rows = [json.loads(line) for line in Path(args.reports).read_text(encoding="utf-8").splitlines() if line.strip()]
    records: list[dict] = []
    conflicts: list[dict] = []
    for row in rows:
        if llm is not None:
            extracted, row_conflicts = extract_report_hybrid(
                row.get("text", ""),
                patient_id=row.get("patient_id", ""),
                report_time=row.get("report_time", ""),
                source=row.get("source", ""),
                llm=llm,
            )
            conflicts.extend(row_conflicts)
        else:
            extracted = extract_report(
                row.get("text", ""),
                patient_id=row.get("patient_id", ""),
                report_time=row.get("report_time", ""),
                source=row.get("source", ""),
            )
        for record in extracted:
            record.setdefault("extractor", "rule")
        records.extend(extracted)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "features.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records), encoding="utf-8"
    )
    (out / "features_wide.csv").write_text(to_wide_csv(records), encoding="utf-8")
    if conflicts:
        (out / "conflicts.json").write_text(
            json.dumps(conflicts, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    counts = Counter(record["status"] for record in records)
    by_extractor = Counter(record.get("extractor", "rule") for record in records)
    found = Counter(record["variable"] for record in records if record["status"] == "present")
    print(json.dumps({
        "reports": len(rows),
        "records": len(records),
        "llm_enabled": llm is not None,
        "extractor": dict(by_extractor),
        "status": dict(counts),
        "conflicts": len(conflicts),
        "top_variables": found.most_common(10),
        "out_dir": str(out),
    }, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
