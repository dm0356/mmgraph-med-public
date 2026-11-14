"""Build RAG corpus/QA from clinical cases (patient x time-window).

Usage:
    python scripts/build_clinical_corpus.py \
        --cases data/clinical_demo/cases.jsonl \
        --out-root data/clinical_demo/corpus
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmgraph_agent.eval.clinical_schema import cases_to_corpus, validate_case  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()

    cases = [json.loads(line) for line in Path(args.cases).read_text(encoding="utf-8").splitlines() if line.strip()]
    valid = []
    for case in cases:
        errors = validate_case(case)
        if errors:
            print(f"skip {case.get('case_id', '?')}: {errors}")
        else:
            valid.append(case)

    docs, qa = cases_to_corpus(valid)
    out = Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "corpus.jsonl").write_text(
        "\n".join(json.dumps(doc, ensure_ascii=False) for doc in docs), encoding="utf-8"
    )
    (out / "qa.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in qa), encoding="utf-8"
    )
    print(f"cases={len(valid)} docs={len(docs)} qa={len(qa)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
