"""Zero-API multimodal hypergraph builder (Module 1, paper §3.1).

Pipeline: parse documents into modality-tagged units -> extract entities with a
local extractor -> embed entities and units -> create n-ary hyperedges and link
each hyperedge to the image units of its document.

The resulting graph keeps the paper's *coverage* property: every hyperedge is
linked to at least one multimodal item whenever the document contains one.

The extractor is pluggable: the default ``RegexExtractor`` is zero-API, while a
configured ``LLMExtractor`` (DeepSeek / Qwen / any OpenAI-compatible endpoint)
handles Chinese and clinical text where capitalisation heuristics fail.
"""

from __future__ import annotations

from pathlib import Path

from .captioner import get_captioner
from .construction import get_construction_pipeline
from .encoder import BaseEmbedder, get_embedder, get_image_embedder
from .llm_extractor import Extractor, get_extractor
from .parser import is_text_unit, parse_documents
from .store import Store

_PLACE_HINTS = {
    "paris",
    "france",
    "china",
    "beijing",
    "london",
    "united states",
    "usa",
    "japan",
    "tokyo",
    "germany",
    "india",
    "egypt",
    "italy",
    "rome",
}


def _valid_entity_name(name: str) -> bool:
    """Reject table fragments, pure numbers and over-long strings as entities."""
    text = (name or "").strip()
    if not text or len(text) > 40:
        return False
    if "\n" in text or "|" in text:
        return False
    compact = text.replace(".", "").replace("-", "").replace(":", "").replace(" ", "")
    if compact.isdigit():
        return False
    return True


def _guess_type(name: str) -> str:
    low = name.lower()
    if low in _PLACE_HINTS:
        return "place"
    if any(ch.isdigit() for ch in name):
        return "measure"
    return "concept"


def _merge_duplicate_entities(entities: list[dict], hyperedges: list[dict]) -> list[dict]:
    """Merge entities that collapsed to the same canonical name.

    Dictionary/LLM normalization can map several surface forms to one canonical
    name, leaving duplicate entity rows that all point at the same concept. Keep
    the first row, keep the longest description, and remap hyperedge references.
    """
    by_name: dict[str, dict] = {}
    remap: dict[str, str] = {}
    merged: list[dict] = []
    for entity in entities:
        key = (entity.get("name") or "").strip().lower()
        if not key:
            continue
        if key in by_name:
            target = by_name[key]
            remap[entity["id"]] = target["id"]
            if len(entity.get("description") or "") > len(target.get("description") or ""):
                target["description"] = entity["description"]
        else:
            by_name[key] = entity
            merged.append(entity)
    for edge in hyperedges:
        new_ids: list[str] = []
        for entity_id in edge.get("entities", []):
            entity_id = remap.get(entity_id, entity_id)
            if entity_id not in new_ids:
                new_ids.append(entity_id)
        edge["entities"] = new_ids
    return merged


def _resolve_image_path(raw: str | None, base_dir: str | Path | None) -> Path | None:
    if not raw:
        return None
    path = Path(str(raw))
    candidates = [path]
    if not path.is_absolute():
        if base_dir:
            candidates.append(Path(base_dir) / path)
        candidates.append(Path.cwd() / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def build_hypergraph(
    docs: list[dict],
    config: dict,
    embedder: BaseEmbedder | None = None,
    captioner=None,
    extractor: Extractor | None = None,
    construction=None,
    base_dir: str | Path | None = None,
) -> Store:
    """Construct a multimodal knowledge hypergraph from raw documents."""
    embedder = embedder or get_embedder(config)
    image_encoder = get_image_embedder(config)
    if captioner is None:
        captioner = get_captioner(config)
    if extractor is None:
        extractor = get_extractor(config)
    if construction is None:
        construction = get_construction_pipeline(config)
    units = parse_documents(docs, config)

    # 1a) M3KG-style rewriter: context-enriched text per document
    contexts: dict[str, str] = {}
    if construction is not None:
        for doc in docs:
            doc_id = str(doc.get("doc_id", "doc"))
            try:
                contexts[doc_id] = construction.rewrite(doc)
            except Exception:
                contexts[doc_id] = doc.get("text") or ""

    # 1b) extract entity/relation mentions once per text-like unit. The LLM
    # extractor caches responses on disk, so repeated runs are cheap.
    rich_by_unit: dict[str, dict] = {}
    for unit in units:
        if is_text_unit(unit) and unit["text"].strip():
            if construction is not None:
                rich_by_unit[unit["unit_id"]] = extractor.extract_rich(
                    unit["text"], context=contexts.get(unit["doc_id"])
                )
            else:
                rich_by_unit[unit["unit_id"]] = extractor.extract_rich(unit["text"])

    names: dict[str, str] = {}
    for rich in rich_by_unit.values():
        for ent in rich.get("entities", []):
            if _valid_entity_name(ent):
                names.setdefault(ent.strip().lower(), ent.strip())

    entities: list[dict] = []
    entity_id_of: dict[str, str] = {}
    for index, (key, display) in enumerate(names.items()):
        entity_id = f"e{index:04d}"
        entity_id_of[key] = entity_id
        entities.append(
            {
                "id": entity_id,
                "name": display,
                "type": _guess_type(display),
                "description": "",
                "embedding": None,
            }
        )

    if entities:
        vecs = embedder.encode_texts([ent["name"] for ent in entities])
        for ent, vec in zip(entities, vecs):
            ent["embedding"] = vec.tolist()

    # 2) image units become multimodal items with their own embedding
    images: list[dict] = []
    for unit in units:
        if unit["modality"] == "image":
            if captioner is not None:
                described = captioner.describe(unit)
                if described:
                    unit["text"] = described
            # caption text stays in the *text* embedding space so that the
            # existing text-query retrieval keeps working
            vec = embedder.encode_captions([unit["text"]])[0] if unit["text"] else embedder.encode_texts([""])[0]
            resolved = _resolve_image_path(unit.get("path"), base_dir)
            entry = {
                "id": unit.get("image_id") or unit["unit_id"],
                "doc_id": unit["doc_id"],
                "caption": unit["text"],
                "path": str(resolved) if resolved else unit.get("path"),
                "embedding": vec.tolist(),
                "embed_dim": int(vec.shape[0]),
                "embedder": embedder.name,
            }
            # pixel-level CLIP vector (a separate, shared image-text space)
            if image_encoder is not None and resolved is not None:
                try:
                    image_vec = image_encoder.encode_images([str(resolved)])[0]
                    entry["image_embedding"] = image_vec.tolist()
                    entry["image_embed_dim"] = int(image_vec.shape[0])
                    entry["image_embedder"] = image_encoder.name
                except Exception:
                    pass
            images.append(entry)

    # 3) n-ary hyperedges for text / table / diagram units
    hyperedges: list[dict] = []
    for unit in units:
        if not is_text_unit(unit) or not unit["text"].strip():
            continue
        rich = rich_by_unit.get(unit["unit_id"], {"entities": [], "relations": []})
        edge_entities: list[str] = []
        for mention in rich.get("entities", []):
            if not _valid_entity_name(mention):
                continue
            entity_id = entity_id_of.get(mention.strip().lower())
            if entity_id and entity_id not in edge_entities:
                edge_entities.append(entity_id)
        # Keep entity-less units as well: facts such as measurements may have
        # no proper noun but are still retrievable by their text embedding.

        vec = embedder.encode_texts([unit["text"]])[0]
        linked_images = [img["id"] for img in images if img["doc_id"] == unit["doc_id"]]
        hyperedges.append(
            {
                "id": f"h{len(hyperedges):05d}",
                "doc_id": unit["doc_id"],
                "unit_id": unit["unit_id"],
                "modality": unit["modality"],
                "text": unit["text"],
                "entities": edge_entities,
                "relations": rich.get("relations", []) or [],
                "images": linked_images,
                "embedding": vec.tolist(),
            }
        )

        # first occurrence provides a cheap zero-API description
        for entity_id in edge_entities:
            ent = next(x for x in entities if x["id"] == entity_id)
            if not ent["description"]:
                ent["description"] = unit["text"]

    # 4) M3KG-style grounding + context-aware description refinement
    if construction is not None and entities:
        for doc in docs:
            doc_id = str(doc.get("doc_id", "doc"))
            doc_names: list[str] = []
            for unit in units:
                if unit["doc_id"] == doc_id and unit["unit_id"] in rich_by_unit:
                    doc_names.extend(rich_by_unit[unit["unit_id"]].get("entities", []))
            try:
                canonical, descriptions = construction.enrich_entities(doc, doc_names, contexts.get(doc_id, ""))
            except Exception:
                continue
            for original, canon in canonical.items():
                entity_id = entity_id_of.get(original.lower())
                if not entity_id:
                    continue
                entity = next((e for e in entities if e["id"] == entity_id), None)
                if entity is None:
                    continue
                entity["name"] = canon
                if descriptions.get(canon):
                    entity["description"] = descriptions[canon]
        # canonicalization may collapse several rows to the same name -> merge
        entities = _merge_duplicate_entities(entities, hyperedges)
        # names changed -> recompute entity embeddings
        vecs = embedder.encode_texts([entity["name"] for entity in entities])
        for entity, vec in zip(entities, vecs):
            entity["embedding"] = vec.tolist()

    meta = {
        "embedder": embedder.name,
        "embed_dim": embedder.dim,
        "image_embedder": getattr(image_encoder, "name", "none"),
        "image_dim": int(getattr(image_encoder, "dim", 0)),
        "extractor": getattr(extractor, "name", type(extractor).__name__),
        "construction": "on" if construction is not None else "off",
        "num_docs": len(docs),
        "num_units": len(units),
    }
    return Store(meta=meta, entities=entities, images=images, units=units, hyperedges=hyperedges)
