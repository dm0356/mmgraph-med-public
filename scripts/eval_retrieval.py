"""Retrieval-focused evaluation: answer-in-context recall over QA pairs.

This exercises the MMGraph-Agent memory + multi-hop retrieval without needing a
generative answerer (useful for Chinese data where the heuristic policy is weak).

    python scripts/eval_retrieval.py --config configs/cluster_local.toml \
        --store /path/to/deidentified_subset/mmkg_subset.json \
        --data-root /path/to/deidentified_subset \
        --out /path/to/deidentified_subset/retrieval_eval.json
"""

import argparse
import json
from pathlib import Path

from mmgraph_agent.config import load_config
from mmgraph_agent.eval.datasets import load_docs, load_qa
from mmgraph_agent.hypergraph import Store, build_hypergraph, get_embedder
from mmgraph_agent.memory import TwoTierMemory
from mmgraph_agent.utils import tokenize


def _answer_in_context(answer: str, context: str) -> float:
    tokens = tokenize(answer)
    if not tokens:
        return 1.0
    lowered = context.lower()
    hits = sum(1 for token in tokens if token in lowered)
    return hits / len(tokens)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--store", default=None, help="reuse an existing store JSON")
    parser.add_argument("--out", default="out/retrieval_eval.json")
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(args.data_root)
    docs = load_docs(root / "corpus.jsonl")
    qa = load_qa(root / "qa.jsonl") if (root / "qa.jsonl").exists() else []

    store = Store.load(args.store) if args.store else build_hypergraph(docs, config)
    embedder = get_embedder(config)

    rows = []
    for item in qa:
        memory = TwoTierMemory(store, config, embedder)
        evidence = memory.retrieve(item["question"])
        context = " ".join(ev.text for ev in evidence)
        recall = _answer_in_context(item["answer"], context)
        rows.append({
            "qid": item.get("qid"),
            "question": item["question"],
            "gold": item["answer"],
            "answer_recall": recall,
            "evidence_ids": [ev.id for ev in evidence],
        })

    average = sum(row["answer_recall"] for row in rows) / len(rows) if rows else 0.0
    result = {"qa": len(rows), "avg_answer_recall": average, "rows": rows}
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"qa": len(rows), "avg_answer_recall": average}, ensure_ascii=False, indent=2))
    for row in rows:
        print(f"  {row['qid']} recall={row['answer_recall']:.2f}  {row['question']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
