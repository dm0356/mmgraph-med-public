"""Validate a clinical corpus / QA JSONL pair against the data spec.

Usage:
    python scripts/validate_data.py --corpus data/smoke_zh/corpus.jsonl --qa data/smoke_zh/qa.jsonl
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse  # noqa: E402

from mmgraph_agent.eval.validation import validate_corpus, validate_qa  # noqa: E402


def _print_report(label: str, report: dict) -> None:
    print(f"[{label}] {json.dumps(report['stats'], ensure_ascii=False)}")
    for message in report["errors"]:
        print("  ERROR:", message)
    for message in report["warnings"]:
        print("  WARN :", message)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate corpus/QA JSONL files.")
    parser.add_argument("--corpus", default="data/smoke_zh/corpus.jsonl")
    parser.add_argument("--qa", default=None)
    args = parser.parse_args()

    ok = True
    corpus_report = validate_corpus(args.corpus)
    _print_report("corpus", corpus_report)
    ok = ok and corpus_report["ok"]

    if args.qa:
        qa_report = validate_qa(args.qa)
        _print_report("qa", qa_report)
        ok = ok and qa_report["ok"]

    print("\nOK" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
