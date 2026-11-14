"""Evaluate a GRPO/SFT policy per question category (greedy decoding).

Usage:
    PYTHONPATH=/path/to/pylibs python scripts/eval_grpo_policy.py \
        --qa-eval /path/to/public_pilot/pilot_big/qa_eval.jsonl \
        --per-category 12 --adapter /path/to/public_pilot/sft_adapter
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmgraph_agent.config import load_config  # noqa: E402
from mmgraph_agent.eval.metrics import token_f1  # noqa: E402
from mmgraph_agent.hypergraph import Store, get_embedder  # noqa: E402
from mmgraph_agent.policy.agent_env import RetrievalEnv  # noqa: E402
from mmgraph_agent.policy.grpo_trainer import GRPOTrainer  # noqa: E402


def normalize(text: str) -> str:
    return re.sub(r"[\s，。,.!?;:：]+", "", (text or "").strip().lower())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/public_pilot_bge.toml")
    parser.add_argument("--store", default="/path/to/public_pilot/pilot_big/mmkg_big_bge.json")
    parser.add_argument("--qa-eval", default="/path/to/public_pilot/pilot_big/qa_eval.jsonl")
    parser.add_argument("--model", default="/path/to/qwen-instruct")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--per-category", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--out", default="/path/to/public_pilot/eval_policy.json")
    args = parser.parse_args()

    config = load_config(args.config)
    store = Store.load(args.store)
    embedder = get_embedder(config)
    env = RetrievalEnv(store, embedder, config, max_rounds=3, top_k=3)
    trainer = GRPOTrainer(config, env, args.model, device=args.device,
                          max_new_tokens=args.max_new_tokens)
    print("loading policy ...", flush=True)
    trainer.load(args.adapter)

    rows = [json.loads(line) for line in Path(args.qa_eval).read_text(encoding="utf-8").splitlines() if line.strip()]
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_category[row.get("category", "unknown")].append(row)
    selected = []
    for category, items in sorted(by_category.items()):
        selected.extend(items[: args.per_category])
    print(f"evaluating {len(selected)} prompts across {len(by_category)} categories", flush=True)

    results = []
    for row in selected:
        rollout = trainer.rollout(row["question"], row["answer"], greedy=True)
        exact = 1.0 if normalize(rollout.answer) == normalize(row["answer"]) else 0.0
        results.append({
            "category": row.get("category", "unknown"),
            "question": row["question"],
            "gold": row["answer"],
            "answer": rollout.answer,
            "exact": exact,
            "f1": token_f1(rollout.answer, row["answer"]),
            "answered": 1.0 if rollout.answer else 0.0,
            "searches": rollout.searches,
        })

    def summarize(items):
        return {
            "n": len(items),
            "exact": sum(item["exact"] for item in items) / len(items) if items else 0.0,
            "f1": sum(item["f1"] for item in items) / len(items) if items else 0.0,
            "answered": sum(item["answered"] for item in items) / len(items) if items else 0.0,
        }

    summary = {"overall": summarize(results), "by_category": {}}
    for category in sorted({item["category"] for item in results}):
        summary["by_category"][category] = summarize(
            [item for item in results if item["category"] == category]
        )
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    Path(args.out).write_text(
        json.dumps({"summary": summary, "details": results}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
