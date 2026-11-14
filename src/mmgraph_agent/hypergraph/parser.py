"""Modality-aware parsing + local entity extraction (Module 1, paper §3.1).

The paper's construction is *zero API cost*: a rule-based parser slices a
document into modality-tagged units and a local extractor (spaCy NER) pulls
entities. This module implements that contract with a pure-regex fallback so
the CPU smoke path needs no downloaded model.
"""

from __future__ import annotations

from typing import Any

from ..utils import proper_noun_phrases, split_sentences

# spaCy is an optional dependency; load once and remember the outcome.
_SPACY: Any = "unloaded"
_TEXT_MODALITIES = ("text", "table", "diagram")
_ENTITY_LABELS = {
    "PERSON",
    "ORG",
    "GPE",
    "LOC",
    "PRODUCT",
    "EVENT",
    "WORK_OF_ART",
    "FAC",
    "NORP",
}

# Small seed lexicon used as a *demo* fallback for Chinese text: Chinese has no
# capitalisation cue for the regex extractor. For real Chinese / clinical data,
# replace with an LLM-based extractor or a Chinese NER model.
_ZH_LEXICON = {
    "糖尿病",
    "高血压",
    "冠心病",
    "肺炎",
    "发热",
    "咳嗽",
    "胸闷",
    "心电图",
    "血糖",
    "空腹血糖",
    "白细胞",
    "血常规",
    "体温",
    "胸部CT",
    "CT",
    "MRI",
    "X线",
    "患者",
    "医生",
    "医院",
    "降糖药",
    "窦性心律",
}


def _extract_zh_terms(text: str) -> list[str]:
    found = [(text.find(term), term) for term in _ZH_LEXICON if term in text]
    found.sort()
    return [term for _, term in found]


def _get_spacy() -> Any:
    global _SPACY
    if _SPACY == "unloaded":
        try:  # pragma: no cover - depends on the environment
            import spacy  # type: ignore

            _SPACY = spacy.load("en_core_web_sm")
        except Exception:  # pragma: no cover - fallback path
            _SPACY = None
    return _SPACY


def _regex_sents(text: str) -> list[str]:
    return split_sentences(text)


def extract_entities(text: str, config: dict) -> list[str]:
    """Return entity mentions for ``text`` (spaCy when available, else regex)."""
    parser = config["hypergraph"].get("parser", "auto")
    entities: list[str] = []

    if parser in ("auto", "spacy"):
        nlp = _get_spacy()
        if nlp is not None:
            doc = nlp(text)
            entities = [
                ent.text.strip()
                for ent in doc.ents
                if ent.label_ in _ENTITY_LABELS and ent.text.strip()
            ]

    if not entities:
        entities = proper_noun_phrases(text)
    entities = entities + _extract_zh_terms(text)

    seen: set[str] = set()
    out: list[str] = []
    for ent in entities:
        key = ent.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(ent.strip())
    cap = int(config["hypergraph"].get("max_entities_per_unit", 12))
    return out[:cap]


def parse_documents(docs: list[dict], config: dict) -> list[dict]:
    """Flatten documents into modality-tagged units.

    Each unit has the shape::

        {"unit_id", "doc_id", "modality", "text", "image_id"?}

    Text is split on sentence boundaries; figures / tables / diagrams become
    single units whose ``text`` is the caption or flattened content.
    """
    units: list[dict] = []
    for doc in docs:
        doc_id = str(doc.get("doc_id", "doc"))
        idx = 0

        text = (doc.get("text") or "").strip()
        for sent in _regex_sents(text):
            units.append(
                {"unit_id": f"{doc_id}:u{idx}", "doc_id": doc_id, "modality": "text", "text": sent}
            )
            idx += 1

        for image in doc.get("images") or []:
            caption = (image.get("caption") or "").strip()
            units.append(
                {
                    "unit_id": f"{doc_id}:u{idx}",
                    "doc_id": doc_id,
                    "modality": "image",
                    "text": caption,
                    "image_id": str(image.get("image_id", f"{doc_id}:img{idx}")),
                    "path": image.get("path"),
                }
            )
            idx += 1

        for table in doc.get("tables") or []:
            units.append(
                {
                    "unit_id": f"{doc_id}:u{idx}",
                    "doc_id": doc_id,
                    "modality": "table",
                    "text": (table.get("text") or "").strip(),
                    "image_id": str(table.get("table_id", "")),
                }
            )
            idx += 1

        for diagram in doc.get("diagrams") or []:
            units.append(
                {
                    "unit_id": f"{doc_id}:u{idx}",
                    "doc_id": doc_id,
                    "modality": "diagram",
                    "text": (diagram.get("text") or "").strip(),
                    "image_id": str(diagram.get("diagram_id", "")),
                }
            )
            idx += 1

    return units


def is_text_unit(unit: dict) -> bool:
    return unit["modality"] in _TEXT_MODALITIES
