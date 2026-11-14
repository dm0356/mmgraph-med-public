"""End-to-end episode runner + evaluation loop.

``run_episode`` replays the paper's multi-round agent loop on the CPU with the
heuristic policy: decide an action, retrieve through the L1/L2 memory, repeat,
then answer. ``evaluate`` aggregates token F1 and answer recall.
"""

from __future__ import annotations

from typing import Any

from ..hypergraph import BaseEmbedder, build_hypergraph
from ..memory import Evidence, TwoTierMemory
from ..policy import ANSWER, HeuristicPolicy
from ..retrieval import NullWebSearch, UnifiedRetriever, WebSearchChannel
from ..utils import estimate_tokens
from .metrics import answer_recall, token_f1


def run_episode(
    question: str,
    config: dict,
    store: Any,
    embedder: BaseEmbedder,
    has_image: bool = False,
    web: WebSearchChannel | None = None,
    policy: HeuristicPolicy | None = None,
) -> dict:
    memory = TwoTierMemory(store, config, embedder)
    retriever = UnifiedRetriever(store, memory, config, embedder, web or NullWebSearch())
    policy = policy or HeuristicPolicy(config)
    policy.store = store  # lets the heuristic answerer ground spans to a document

    evidence: list[Evidence] = []
    actions: list[str] = []
    rounds = 0

    for round_idx in range(policy.max_rounds):
        action, payload = policy.decide(question, evidence, has_image=has_image, round_idx=round_idx)
        actions.append(action)
        if action == ANSWER:
            return {
                "answer": payload or "",
                "evidence": evidence,
                "actions": actions,
                "rounds": rounds,
                "prompt_tokens": memory.prompt_tokens(),
            }
        query = payload or question
        evidence = retriever.retrieve(query, action=action)
        rounds += 1
        if not evidence:
            break

    return {
        "answer": policy.answer(question, evidence),
        "evidence": evidence,
        "actions": actions,
        "rounds": rounds,
        "prompt_tokens": memory.prompt_tokens(),
    }


def evaluate(
    docs: list[dict],
    qa: list[dict],
    config: dict,
    embedder: BaseEmbedder | None = None,
    web: WebSearchChannel | None = None,
    base_dir: str | None = None,
) -> dict:
    from ..hypergraph import get_embedder

    embedder = embedder or get_embedder(config)
    store = build_hypergraph(docs, config, embedder, base_dir=base_dir)
    policy = HeuristicPolicy(config)

    rows: list[dict] = []
    for item in qa:
        result = run_episode(
            item["question"],
            config,
            store,
            embedder,
            has_image=bool(item.get("has_image", False)),
            web=web,
            policy=policy,
        )
        evidence_text = " ".join(ev.text for ev in result["evidence"])
        rows.append(
            {
                "qid": item.get("qid", ""),
                "question": item["question"],
                "gold": item.get("answer", ""),
                "prediction": result["answer"],
                "f1": token_f1(result["answer"], item.get("answer", "")),
                "answer_recall": answer_recall(item.get("answer", ""), evidence_text),
                "actions": result["actions"],
                "rounds": result["rounds"],
                "prompt_tokens": result["prompt_tokens"],
                "context_tokens": estimate_tokens(evidence_text),
            }
        )

    def _mean(key: str) -> float:
        return sum(row[key] for row in rows) / len(rows) if rows else 0.0

    return {
        "store": {
            "entities": len(store.entities),
            "images": len(store.images),
            "hyperedges": len(store.hyperedges),
            "units": len(store.units),
        },
        "rows": rows,
        "token_f1": _mean("f1"),
        "answer_recall": _mean("answer_recall"),
        "avg_rounds": _mean("rounds"),
        "avg_prompt_tokens": _mean("prompt_tokens"),
        "avg_context_tokens": _mean("context_tokens"),
    }


__all__ = ["run_episode", "evaluate"]
