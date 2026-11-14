"""Command-line entry points.

``mmgraph-build``     build the zero-API hypergraph from a JSONL corpus
``mmgraph-smoke``     run the end-to-end CPU smoke evaluation
``mmgraph-evaluate``  evaluate any dataset registered in ``eval.datasets``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .eval import evaluate as run_evaluation
from .eval import get_dataset, load_docs, load_smoke
from .hypergraph import build_hypergraph
from .viz import render_hypergraph_html


def build(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mmgraph-build", description="Build the zero-API multimodal hypergraph.")
    parser.add_argument("--config", default="configs/smoke.toml")
    parser.add_argument("--data-root", default="data/smoke")
    parser.add_argument("--out", default="out/mmkg.json")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    docs, _ = load_smoke(args.data_root)
    store = build_hypergraph(docs, config, base_dir=args.data_root)
    store.save(args.out)
    print(
        f"store: {len(store.entities)} entities, {len(store.images)} images, "
        f"{len(store.hyperedges)} hyperedges -> {args.out}"
    )
    return 0


def viz(argv: list[str] | None = None) -> int:
    """Render the multimodal hypergraph to a standalone interactive HTML file."""
    parser = argparse.ArgumentParser(
        prog="mmgraph-viz", description="Render the multimodal hypergraph as an interactive HTML page."
    )
    parser.add_argument("--config", default="configs/smoke.toml")
    parser.add_argument("--data-root", default="data/smoke")
    parser.add_argument("--out", default="out/mmkg.html")
    parser.add_argument("--title", default="MMGraph-Agent 多模态知识超图")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    docs = load_docs(Path(args.data_root) / "corpus.jsonl")
    store = build_hypergraph(docs, config, base_dir=args.data_root)
    stats = render_hypergraph_html(store, args.out, args.title)
    print(
        f"rendered {stats['entities']} entities, {stats['units']} units, "
        f"{stats['images']} images, {stats['edges']} edges -> {args.out}"
    )
    return 0


def smoke(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mmgraph-smoke", description="Run the end-to-end CPU smoke demo.")
    parser.add_argument("--config", default="configs/smoke.toml")
    parser.add_argument("--data-root", default="data/smoke")
    parser.add_argument("--verbose", action="store_true", help="print per-question rows")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    docs, qa = load_smoke(args.data_root)
    result = run_evaluation(docs, qa, config, base_dir=args.data_root)
    summary = {
        key: result[key]
        for key in ("store", "token_f1", "answer_recall", "avg_rounds", "avg_prompt_tokens", "avg_context_tokens")
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.verbose:
        for row in result["rows"]:
            print(
                f"[{row['qid']}] {row['question']}\n"
                f"    pred={row['prediction']!r} gold={row['gold']!r} "
                f"f1={row['f1']:.2f} recall={row['answer_recall']:.2f} "
                f"actions={row['actions']}"
            )
    return 0


def evaluate(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mmgraph-evaluate", description="Evaluate a registered dataset.")
    parser.add_argument("--config", default="configs/smoke.toml")
    parser.add_argument("--dataset", default="smoke")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    docs, qa = get_dataset(args.dataset)
    result = run_evaluation(docs, qa, config)
    summary = {
        key: result[key]
        for key in ("store", "token_f1", "answer_recall", "avg_rounds", "avg_prompt_tokens", "avg_context_tokens")
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.verbose:
        for row in result["rows"]:
            print(f"[{row['qid']}] pred={row['prediction']!r} f1={row['f1']:.2f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(smoke())
