"""Evaluate CLIP text->image retrieval vs caption-space image retrieval.

Usage:
    python scripts/eval_image_retrieval.py \
        --config configs/image_clip.toml \
        --store out/image_demo/mmkg.json \
        --queries data/image_demo/queries.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmgraph_agent.config import load_config  # noqa: E402
from mmgraph_agent.hypergraph import Store, get_embedder  # noqa: E402
from mmgraph_agent.hypergraph.encoder import get_image_text_encoder  # noqa: E402


def _rank(hits: list[dict], gold: str) -> int | None:
    for index, hit in enumerate(hits, start=1):
        if hit["image"]["id"] == gold:
            return index
    return None


def _summary(ranks: list[int | None]) -> dict:
    total = len(ranks)
    return {
        "n": total,
        "hit@1": sum(1 for rank in ranks if rank == 1) / total if total else 0.0,
        "hit@5": sum(1 for rank in ranks if rank and rank <= 5) / total if total else 0.0,
        "mrr": sum(1.0 / rank for rank in ranks if rank) / total if total else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/image_clip.toml")
    parser.add_argument("--store", default="out/image_demo/mmkg.json")
    parser.add_argument("--queries", default="data/image_demo/queries.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config)
    store = Store.load(args.store)
    text_encoder = get_embedder(config)
    clip_encoder = get_image_text_encoder(config)
    if clip_encoder is None:
        print("CLIP image encoder unavailable; set [image].embedder = 'clip'")
        return 1

    queries = [json.loads(line) for line in Path(args.queries).read_text(encoding="utf-8").splitlines() if line.strip()]
    clip_ranks, caption_ranks = [], []
    rows = []
    for item in queries:
        query = item["query"]
        gold = item["gold_image_id"]
        clip_vec = clip_encoder.encode_texts([query])[0]
        clip_hits = store.search_images_clip(clip_vec, top_k=args.top_k)
        caption_vec = text_encoder.encode_texts([query])[0]
        caption_hits = store.search_images(caption_vec, top_k=args.top_k)
        clip_rank = _rank(clip_hits, gold)
        caption_rank = _rank(caption_hits, gold)
        clip_ranks.append(clip_rank)
        caption_ranks.append(caption_rank)
        rows.append({"query": query, "gold": gold, "clip_rank": clip_rank,
                     "caption_rank": caption_rank})

    summary = {"clip": _summary(clip_ranks), "caption": _summary(caption_ranks)}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    for row in rows:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
