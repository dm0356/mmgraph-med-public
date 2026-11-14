"""JSONL data validation for the clinical corpus and QA sets.

Kept in the package (not only in the script) so it is unit-testable and can be
called from CI or a notebook.
"""

from __future__ import annotations

import json
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
REGIMES = {"k_only", "web_only", "k_web"}


def _iter_jsonl(path: str | Path):
    with open(Path(path), "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield lineno, json.loads(line), None
            except json.JSONDecodeError as exc:
                yield lineno, None, str(exc)


def validate_corpus(path: str | Path) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    stats = {
        "docs": 0,
        "text_units": 0,
        "images": 0,
        "with_image_path": 0,
        "tables": 0,
        "diagrams": 0,
    }
    seen: set[str] = set()

    for lineno, obj, err in _iter_jsonl(path):
        if err:
            errors.append(f"line {lineno}: invalid JSON ({err})")
            continue
        if not isinstance(obj, dict):
            errors.append(f"line {lineno}: expected a JSON object")
            continue

        doc_id = obj.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id.strip():
            errors.append(f"line {lineno}: missing/invalid doc_id")
            continue
        doc_id = doc_id.strip()
        if doc_id in seen:
            errors.append(f"line {lineno}: duplicate doc_id {doc_id!r}")
        seen.add(doc_id)
        stats["docs"] += 1

        text = obj.get("text") or ""
        if not isinstance(text, str):
            errors.append(f"line {lineno}: text must be a string")
            text = ""
        if text.strip():
            stats["text_units"] += 1

        images = obj.get("images") or []
        if not isinstance(images, list):
            errors.append(f"line {lineno}: images must be a list")
            images = []
        for index, image in enumerate(images):
            if not isinstance(image, dict):
                errors.append(f"line {lineno}: images[{index}] must be an object")
                continue
            image_id = image.get("image_id")
            if not isinstance(image_id, str) or not image_id.strip():
                errors.append(f"line {lineno}: images[{index}] missing image_id")
            stats["images"] += 1

            image_path = image.get("path")
            caption = str(image.get("caption") or "")
            if image_path:
                stats["with_image_path"] += 1
                candidate = Path(str(image_path))
                if not candidate.exists():
                    errors.append(f"line {lineno}: image path not found: {image_path}")
                elif candidate.suffix.lower() not in IMAGE_EXTENSIONS:
                    warnings.append(f"line {lineno}: unusual image extension: {image_path}")
            elif not caption.strip():
                warnings.append(f"line {lineno}: image {image_id!r} has neither path nor caption")

        for key in ("tables", "diagrams"):
            items = obj.get(key) or []
            if not isinstance(items, list):
                errors.append(f"line {lineno}: {key} must be a list")
                continue
            for index, item in enumerate(items):
                if not isinstance(item, dict) or not str(item.get("text") or "").strip():
                    errors.append(f"line {lineno}: {key}[{index}] needs non-empty text")
                    continue
                stats[key] += 1

        if not text.strip() and not images and not (obj.get("tables") or obj.get("diagrams")):
            warnings.append(f"line {lineno}: document {doc_id!r} has no content")

    return {"ok": not errors, "errors": errors, "warnings": warnings, "stats": stats}


def validate_qa(path: str | Path) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    stats = {"qa": 0, "has_image": 0}
    seen: set[str] = set()

    for lineno, obj, err in _iter_jsonl(path):
        if err:
            errors.append(f"line {lineno}: invalid JSON ({err})")
            continue
        if not isinstance(obj, dict):
            errors.append(f"line {lineno}: expected a JSON object")
            continue

        qid = obj.get("qid")
        if not isinstance(qid, str) or not qid.strip():
            errors.append(f"line {lineno}: missing/invalid qid")
            continue
        qid = qid.strip()
        if qid in seen:
            errors.append(f"line {lineno}: duplicate qid {qid!r}")
        seen.add(qid)
        stats["qa"] += 1

        for field in ("question", "answer"):
            if not str(obj.get(field) or "").strip():
                errors.append(f"line {lineno}: {field} is empty")

        if bool(obj.get("has_image", False)):
            stats["has_image"] += 1

        regime = obj.get("regime")
        if regime and regime not in REGIMES:
            warnings.append(f"line {lineno}: unknown regime {regime!r}")

    return {"ok": not errors, "errors": errors, "warnings": warnings, "stats": stats}


__all__ = ["validate_corpus", "validate_qa"]
