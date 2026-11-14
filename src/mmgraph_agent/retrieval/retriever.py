"""Unified retrieval: offline hypergraph + optional online web channel (§3.2/3.3).

The paper hides the evidence *source* behind a single retrieval token so an
offline-trained policy transfers online unchanged. This module implements that
interface: both channels return :class:`Evidence` objects with a ``source`` tag,
and the memory merges them into the unified L1 index.
"""

from __future__ import annotations

from typing import Any

from ..hypergraph import Store
from ..hypergraph.encoder import BaseEmbedder
from ..memory import Evidence, TwoTierMemory


class WebSearchChannel:
    """Interface for the online channel. Returns ``Evidence`` with source='web'."""

    def search(self, query: str, top_k: int = 5) -> list[Evidence]:  # pragma: no cover - interface
        return []


class NullWebSearch(WebSearchChannel):
    """Disabled online channel (matches the paper: web off in training)."""

    def search(self, query: str, top_k: int = 5) -> list[Evidence]:
        return []


class CannedWebSearch(WebSearchChannel):
    """Deterministic fake web results, useful for offline smoke tests."""

    def __init__(self, snippets: list[str]) -> None:
        self.snippets = list(snippets)

    def search(self, query: str, top_k: int = 5) -> list[Evidence]:
        out: list[Evidence] = []
        for index, snippet in enumerate(self.snippets[:top_k]):
            out.append(
                Evidence(
                    id=f"web:{index}",
                    source="web",
                    modality="text",
                    text=snippet,
                    score=0.5,
                )
            )
        return out


class UnifiedRetriever:
    def __init__(
        self,
        store: Store,
        memory: TwoTierMemory,
        config: dict,
        embedder: BaseEmbedder,
        web: WebSearchChannel | None = None,
    ) -> None:
        self.store = store
        self.memory = memory
        self.config = config
        self.embedder = embedder
        self.web = web or NullWebSearch()

    def retrieve(self, query: str, action: str = "text") -> list[Evidence]:
        """Route a retrieval action through the unified cache."""
        extra: list[Evidence] = []
        if action == "text" and self.config["retrieval"].get("enable_web"):
            extra = self.web.search(query, top_k=self.memory.C)
        return self.memory.retrieve(query, action=action, extra=extra)

    def reset(self) -> None:
        self.memory.reset_trajectory()


__all__ = ["WebSearchChannel", "NullWebSearch", "CannedWebSearch", "UnifiedRetriever"]
