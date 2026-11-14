"""Persistent multimodal hypergraph store (Module 1, paper §3.1).

The store is a plain, JSON-serialisable container of entities, image units,
raw units and n-ary hyperedges. Retrieval operates on *vectors* supplied by
the caller so the same store works with the hashing fallback and with CLIP.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..utils import cosine


class Store:
    def __init__(
        self,
        meta: dict | None = None,
        entities: list[dict] | None = None,
        images: list[dict] | None = None,
        units: list[dict] | None = None,
        hyperedges: list[dict] | None = None,
    ) -> None:
        self.meta = meta or {}
        self.entities = entities or []
        self.images = images or []
        self.units = units or []
        self.hyperedges = hyperedges or []
        self._entity_by_id = {ent["id"]: ent for ent in self.entities}
        self._edge_by_id = {edge["id"]: edge for edge in self.hyperedges}

    # -- persistence -----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": self.meta,
            "entities": self.entities,
            "images": self.images,
            "units": self.units,
            "hyperedges": self.hyperedges,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Store":
        return cls(
            meta=data.get("meta") or {},
            entities=data.get("entities") or [],
            images=data.get("images") or [],
            units=data.get("units") or [],
            hyperedges=data.get("hyperedges") or [],
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Store":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # -- accessors -------------------------------------------------------
    @property
    def embed_dim(self) -> int:
        return int(self.meta.get("embed_dim", 256))

    def entity(self, entity_id: str) -> dict | None:
        return self._entity_by_id.get(entity_id)

    def entity_names(self, entity_ids: list[str]) -> list[str]:
        return [e["name"] for e in (self._entity_by_id.get(i) for i in entity_ids) if e]

    def hyperedges_for_image(self, image_id: str) -> list[dict]:
        return [h for h in self.hyperedges if image_id in (h.get("images") or [])]

    def neighbor_hyperedges(self, edge_ids: list[str], hops: int = 1) -> list[dict]:
        """Multi-hop expansion: hyperedges reachable through shared entities.

        Mirrors the "multi-hop" idea used by M3KG-RAG / MMGraph-Agent: starting
        from seed hyperedges, walk ``hops`` steps along entity co-occurrence.
        """
        frontier = {eid for eid in edge_ids if eid in self._edge_by_id}
        seen = set(frontier)
        for _ in range(max(0, int(hops))):
            entity_ids: set[str] = set()
            for edge_id in frontier:
                entity_ids.update(self._edge_by_id[edge_id].get("entities", []))
            next_ids: set[str] = set()
            for edge in self.hyperedges:
                if edge["id"] in seen:
                    continue
                if entity_ids & set(edge.get("entities", [])):
                    next_ids.add(edge["id"])
            seen.update(next_ids)
            frontier = next_ids
            if not next_ids:
                break
        return [edge for edge in self.hyperedges if edge["id"] in seen]

    # -- vector retrieval -------------------------------------------------
    def search_text(self, query_vec: np.ndarray, top_k: int = 5, threshold: float = 0.0) -> list[dict]:
        scored = []
        for edge in self.hyperedges:
            score = cosine(query_vec, np.asarray(edge["embedding"], dtype=np.float32))
            if score >= threshold:
                scored.append((score, edge))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [{"score": s, "kind": "hyperedge", "hyperedge": e} for s, e in scored[:top_k]]

    def search_images(self, query_vec: np.ndarray, top_k: int = 5, threshold: float = 0.0) -> list[dict]:
        """Search images in the *caption text* space (legacy behaviour)."""
        scored = []
        for image in self.images:
            score = cosine(query_vec, np.asarray(image["embedding"], dtype=np.float32))
            if score >= threshold:
                scored.append((score, image))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [{"score": s, "kind": "image", "image": img} for s, img in scored[:top_k]]

    def search_images_clip(self, query_vec: np.ndarray, top_k: int = 5,
                           threshold: float = 0.0) -> list[dict]:
        """Search images in the CLIP space using a CLIP text query vector."""
        scored = []
        for image in self.images:
            vector = image.get("image_embedding")
            if not vector:
                continue
            score = cosine(query_vec, np.asarray(vector, dtype=np.float32))
            if score >= threshold:
                scored.append((score, image))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [{"score": s, "kind": "image", "image": img, "space": "clip"} for s, img in scored[:top_k]]

    def search(
        self,
        query_vec: np.ndarray,
        action: str = "text",
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[dict]:
        if action == "image":
            return self.search_images(query_vec, top_k, threshold)
        return self.search_text(query_vec, top_k, threshold)
