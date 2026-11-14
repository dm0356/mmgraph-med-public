"""Heuristic, CPU-only policy used for the smoke pipeline.

This is *not* the paper's GRPO-trained Qwen2.5-VL policy. It is a deterministic
stand-in that exercises the same action space (text search / image search /
answer) so the retrieval + memory machinery can be validated without a GPU.
"""

from __future__ import annotations

import re
from typing import Any

from .actions import ANSWER, IMAGE_SEARCH, TEXT_SEARCH
from ..memory import Evidence

_STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "was",
    "are",
    "were",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "and",
    "or",
    "what",
    "which",
    "who",
    "where",
    "when",
    "how",
    "does",
    "do",
    "did",
    "it",
    "its",
    "this",
    "that",
    "be",
    "been",
    "has",
    "have",
    "had",
    "by",
    "with",
    "from",
    "as",
    "than",
}

_MEASURE_RE = re.compile(
    r"(\d[\d,\.]*\s*(?:metres?|meters?|kilometres?|kilometers?|km|cm|mm|m|"
    r"years?|months?|days?|hours?|minutes?|seconds?|kg|kgs|degrees?|percent|%))",
    re.I,
)
_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_CAP_RE = re.compile(r"([A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+)*)")
_WHERE_RE = re.compile(r"\b(?:in|at|near|from)\s+([A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+)*)")
_BY_RE = re.compile(r"\bby\s+([A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+)*)")

_CAP_STOP = {
    "the",
    "a",
    "an",
    "it",
    "she",
    "he",
    "they",
    "this",
    "that",
    "we",
    "i",
    "its",
    "his",
    "her",
    "their",
}

_MEASURE_TRIGGER = re.compile(
    r"how (tall|long|much|many|old|far|large|big|heavy|high|deep|wide)|"
    r"melting point|boiling point|temperature|height|length|area|population|weighs?",
    re.I,
)


def _content_tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[A-Za-z0-9]+", text.lower()) if t not in _STOPWORDS]


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _overlap(question: str, text: str) -> int:
    return len(set(_content_tokens(question)) & set(_content_tokens(text)))


def best_sentence(question: str, evidence: list[Evidence]) -> str:
    """The evidence sentence sharing the most content tokens with the question."""
    best, best_score = "", -1
    for item in evidence:
        for sentence in _sentences(item.text):
            score = _overlap(question, sentence)
            if score > best_score:
                best_score = score
                best = sentence
    return best


def pick_best(question: str, evidence: list[Evidence]) -> Evidence | None:
    """The evidence item most lexically aligned with the question."""
    best, best_score = None, -1
    for item in evidence:
        score = _overlap(question, item.text)
        if score > best_score:
            best_score = score
            best = item
    return best


def short_answer(question: str, sentence: str) -> str | None:
    """Extractive short-span heuristics applied to one sentence."""
    q = question.lower().strip()

    # numeric / measurement questions
    if _MEASURE_TRIGGER.search(q):
        match = _MEASURE_RE.search(sentence)
        if match:
            return match.group(1)

    # temporal questions
    if "year" in q or q.startswith("when"):
        match = _YEAR_RE.search(sentence)
        if match:
            return match.group(1)

    # location questions
    if q.startswith("where"):
        match = _WHERE_RE.search(sentence)
        if match:
            return match.group(1)

    # person / agent questions
    if q.startswith("who"):
        match = _BY_RE.search(sentence)
        if match:
            return match.group(1)
        for phrase_match in _CAP_RE.finditer(sentence):
            phrase = phrase_match.group(1)
            if phrase.lower() in _CAP_STOP or phrase_match.start() == 0:
                continue
            return phrase

    return None


class HeuristicPolicy:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.max_rounds = int(config["policy"]["max_rounds"])
        self.store: Any = None  # set by the runner for document-grounded spans

    def decide(
        self,
        question: str,
        evidence: list[Evidence],
        has_image: bool = False,
        round_idx: int = 0,
    ) -> tuple[str, str | None]:
        if not evidence and round_idx == 0:
            return TEXT_SEARCH, question
        if (
            has_image
            and round_idx == 1
            and not any(e.modality == "image" for e in evidence)
            and self.max_rounds > 2
        ):
            return IMAGE_SEARCH, None
        return ANSWER, self.answer(question, evidence)

    def _document_pool(self, question: str, evidence: list[Evidence]) -> list[Evidence]:
        """Evidence plus every stored hyperedge from the best-matching document."""
        pool = list(evidence)
        best = pick_best(question, evidence)
        if self.store is not None and best is not None:
            for edge in self.store.hyperedges:
                if edge["doc_id"] == best.doc_id:
                    pool.append(
                        Evidence(
                            id=edge["id"],
                            source="offline",
                            modality=edge["modality"],
                            text=edge["text"],
                            score=0.0,
                            doc_id=edge["doc_id"],
                        )
                    )
        seen: set[str] = set()
        unique: list[Evidence] = []
        for item in pool:
            if item.id in seen:
                continue
            seen.add(item.id)
            unique.append(item)
        return unique

    def answer(self, question: str, evidence: list[Evidence]) -> str:
        pool = self._document_pool(question, evidence)
        best = pick_best(question, evidence)

        # 1) ground a short span to the best-matching document first
        if best is not None:
            for item in pool:
                if item.doc_id != best.doc_id:
                    continue
                for sentence in _sentences(item.text):
                    short = short_answer(question, sentence)
                    if short:
                        return short

        # 2) fall back to any evidence sentence with a usable span
        for item in pool:
            for sentence in _sentences(item.text):
                short = short_answer(question, sentence)
                if short:
                    return short

        return best_sentence(question, pool)


__all__ = ["HeuristicPolicy", "best_sentence", "pick_best", "short_answer"]
