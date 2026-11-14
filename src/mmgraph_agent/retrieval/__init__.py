"""Unified retrieval over offline + online sources (paper §3.2/3.3)."""

from .retriever import (
    CannedWebSearch,
    NullWebSearch,
    UnifiedRetriever,
    WebSearchChannel,
)

__all__ = ["UnifiedRetriever", "WebSearchChannel", "NullWebSearch", "CannedWebSearch"]
