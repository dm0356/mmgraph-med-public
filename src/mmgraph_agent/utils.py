"""Small, dependency-light utilities shared by every module.

Everything here must work with only the Python standard library and NumPy so
that the CPU smoke pipeline runs without spaCy, torch or any model download.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable

import numpy as np

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*|[\u4e00-\u9fff]")
# Split after Chinese sentence punctuation unconditionally, but after an
# English .!? only when whitespace follows — otherwise decimals such as
# ``8.74`` or ``650.0-655.0`` would be cut in half.
_SENT_RE = re.compile(r"(?<=[。！？])\s*|(?<=[.!?])\s+")
_CAP_RE = re.compile(r"[A-Z][A-Za-z0-9\-]*(?:\s+[A-Z][A-Za-z0-9\-]*)*")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens used for F1, hashing and overlap."""
    return _WORD_RE.findall(text.lower())


def split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries without pulling in nltk/spaCy."""
    text = text.strip()
    if not text:
        return []
    return [p.strip() for p in _SENT_RE.split(text) if p.strip()]


def hash_token(token: str, dim: int) -> int:
    """Stable token -> bucket index (md5 so runs are reproducible)."""
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16) % dim


def hash_embed(text: str, dim: int = 256) -> np.ndarray:
    """Deterministic bag-of-words hash embedding, L2 normalised.

    This is the ``auto`` fallback so the pipeline is usable on a CPU with no
    model weights. In the real reproduction replace it with a frozen CLIP text
    encoder (see :class:`mmgraph_agent.hypergraph.encoder.CLIPEmbedder`).
    """
    vec = np.zeros(dim, dtype=np.float32)
    tokens = tokenize(text)
    if not tokens:
        return vec
    for tok in tokens:
        vec[hash_token(tok, dim)] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two 1-D arrays (safe against zero vectors)."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def word_count(text: str) -> int:
    # Chinese has no whitespace between words; fall back to character count.
    if _CJK_RE.search(text):
        return len(text)
    return len(text.split())


def truncate_words(text: str, n: int) -> str:
    """Keep at most ``n`` whitespace words, or ``n`` characters for CJK text."""
    if _CJK_RE.search(text):
        return text[:n]
    words = text.split()
    return " ".join(words[:n])


def digest(text: str, n: int) -> str:
    """Naive extractive digest: leading sentences within a length budget.

    The paper condenses evidence with the policy itself; a first-sentence /
    first-words digest is a deterministic stand-in for the CPU smoke path.
    """
    sentences = split_sentences(text)
    if not sentences:
        return truncate_words(text, n)
    out: list[str] = []
    used = 0
    for sent in sentences:
        size = word_count(sent)
        if used + size > n:
            break
        out.append(sent)
        used += size
    if not out:
        return truncate_words(text, n)
    return " ".join(out)


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
    "in",
    "on",
    "at",
    "and",
    "or",
    "but",
    "of",
    "to",
    "for",
    "with",
    "by",
    "from",
    "as",
    "than",
}


def proper_noun_phrases(text: str) -> list[str]:
    """Regex fallback entity extractor (capitalised phrases)."""
    seen: dict[str, None] = {}
    for match in _CAP_RE.finditer(text):
        words = match.group(0).strip().split()
        while words and words[0].lower() in _CAP_STOP:
            words = words[1:]
        if not words:
            continue
        phrase = " ".join(words)
        if len(phrase) >= 2 and not phrase.isupper():
            seen.setdefault(phrase, None)
    return list(seen)


def estimate_tokens(text: str) -> int:
    """Cheap prompt-size estimate (~1.3 tokens per whitespace word)."""
    return int(word_count(text) * 1.3) + 1


def first_n_words(text: str, n: int) -> str:
    return " ".join(text.split()[:n])
