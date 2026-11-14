"""Cache-inspired L1/L2 memory (Module 2, paper §3.2).

Mirrors the paper's CPU cache hierarchy:

* ``L2`` is the persistent full hypergraph store.
* ``L1`` is a per-trajectory cache accumulating evidence retrieved so far.
* Every admitted evidence block is truncated to ``L_max`` words and condensed
  to an ``L_dig``-word digest, keeping the prompt bounded uniformly over the
  retrieval horizon; ``L_cap`` is the hard prompt-token budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..hypergraph import Store
from ..hypergraph.encoder import BaseEmbedder
from ..utils import cosine, digest, estimate_tokens, truncate_words


@dataclass
class Evidence:
    id: str
    source: str  # "offline" | "web"
    modality: str
    text: str
    score: float = 0.0
    doc_id: str = ""
    entities: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    embedding: Any = None


class TwoTierMemory:
    def __init__(self, store: Store, config: dict, embedder: BaseEmbedder,
                 image_text_encoder: Any | None = None) -> None:
        self.store = store
        self.embedder = embedder
        mem = config["memory"]
        self.C = int(mem["C"])
        self.L_max = int(mem["L_max"])
        self.L_dig = int(mem["L_dig"])
        self.L_cap = int(mem["L_cap"])
        self.hops = int(config.get("retrieval", {}).get("hops", 0))
        self.l1: dict[str, Evidence] = {}
        self.cross_modal = bool(
            config.get("retrieval", {}).get("cross_modal", False)
            or config.get("image", {}).get("text_query", False)
        )
        if self.cross_modal and image_text_encoder is None:
            try:
                from ..hypergraph.encoder import get_image_text_encoder

                image_text_encoder = get_image_text_encoder(config)
            except Exception:
                image_text_encoder = None
        self.image_text_encoder = image_text_encoder if self.cross_modal else None

    # -- helpers ---------------------------------------------------------
    def encode(self, text: str) -> np.ndarray:
        return self.embedder.encode_texts([text])[0]

    def reset_trajectory(self) -> None:
        self.l1.clear()

    def prompt_tokens(self) -> int:
        return sum(estimate_tokens(ev.text) for ev in self.l1.values())

    def _to_evidence(self, hit: dict) -> Evidence:
        if hit["kind"] == "hyperedge":
            edge = hit["hyperedge"]
            return Evidence(
                id=edge["id"],
                source="offline",
                modality=edge["modality"],
                text=edge["text"],
                score=hit["score"],
                doc_id=edge["doc_id"],
                entities=self.store.entity_names(edge.get("entities", [])),
                images=edge.get("images", []),
            )
        image = hit["image"]
        return Evidence(
            id=image["id"],
            source="offline",
            modality="image",
            text=image["caption"],
            score=hit["score"],
            doc_id=image.get("doc_id", ""),
        )

    def admit(self, evidence: Evidence) -> None:
        """Truncate + digest one evidence block, insert it into L1."""
        evidence.text = digest(truncate_words(evidence.text, self.L_max), self.L_dig)
        evidence.embedding = self.encode(evidence.text)
        self.l1[evidence.id] = evidence
        self._enforce_cap()

    def _enforce_cap(self) -> None:
        while self.l1 and self.prompt_tokens() > self.L_cap:
            weakest = min(self.l1.values(), key=lambda e: e.score)
            del self.l1[weakest.id]

    def _search_l1(self, query_vec: np.ndarray, action: str, k: int) -> list[Evidence]:
        # The paper serves from a *unified* L1 index, so both text and image
        # evidence compete by query similarity regardless of the action used.
        scored = sorted(self.l1.values(), key=lambda e: cosine(query_vec, e.embedding), reverse=True)
        return scored[:k]

    def serve(self, query_vec: np.ndarray, action: str = "text") -> list[Evidence]:
        """Return merged Top-C evidence from the unified L1 index."""
        return self._search_l1(query_vec, action, self.C)

    def retrieve(self, query: str, action: str = "text", extra: list[Evidence] | None = None) -> list[Evidence]:
        query_vec = self.encode(query)
        hits = self.store.search(query_vec, action=action, top_k=self.C * 2, threshold=0.0)
        fresh = [self._to_evidence(hit) for hit in hits]

        # CLIP text -> image retrieval (cross-modal fusion)
        if self.image_text_encoder is not None:
            try:
                clip_vec = self.image_text_encoder.encode_texts([query])[0]
                for hit in self.store.search_images_clip(clip_vec, top_k=self.C, threshold=0.0):
                    fresh.append(self._to_evidence(hit))
            except Exception:
                pass

        # multi-hop expansion through shared entities (M3KG-RAG / MMGraph-Agent)
        if self.hops and action == "text":
            seed_ids = [hit["hyperedge"]["id"] for hit in hits if hit["kind"] == "hyperedge"]
            for edge in self.store.neighbor_hyperedges(seed_ids, self.hops):
                score = cosine(query_vec, np.asarray(edge["embedding"], dtype=np.float32))
                fresh.append(
                    self._to_evidence({"kind": "hyperedge", "score": score, "hyperedge": edge})
                )

        merged: dict[str, Evidence] = {}
        for evidence in fresh:
            merged.setdefault(evidence.id, evidence)
        for evidence in merged.values():
            self.admit(evidence)
        if action == "text" and extra:
            for evidence in extra:
                self.admit(evidence)
        return self.serve(query_vec, action)


__all__ = ["Evidence", "TwoTierMemory"]
