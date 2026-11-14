"""M3KG-style construction stages (rewriter -> grounding -> refinement -> inspector).

The paper's construction is a multi-stage pipeline:

1. **Context-enriched triplet extraction** - rewrite a semantic caption into a
   knowledge-intensive text before extracting entities/relations.
2. **Knowledge grounding** - normalize entity mentions to canonical concepts and
   attach encyclopaedic descriptions.
3. **Context-aware description refinement** - select/rewrite the description that
   best fits the surrounding context.
4. **Self-reflection** - an inspector scores descriptions and flags the weak ones.

This module implements the stages as LLM calls over the existing OpenAI-compatible
client (DeepSeek/Qwen/local vLLM). Every stage is optional and controlled by the
``[construction]`` config section. A custom ``chat`` callable can be injected in
tests so no network is needed.
"""

from __future__ import annotations

import json
import os

from ..http_utils import post_json
from .knowledge_searcher import KnowledgeSearcher, get_searcher

SYSTEM = "你是严谨的医学知识工程助手。只输出要求的格式，不要解释，不要输出 Markdown 代码块。"

REWRITE_PROMPT = """请把下面的医学文本改写为“知识密集型”描述：保留全部事实，补全隐含主语，
不要编造新事实，不要输出列表或解释。只输出改写后的中文文本。

文本：
{text}
"""

NORMALIZE_PROMPT = """把下列实体名称规范化为简洁的医学概念名：去掉不必要的修饰语，
统一全称/简称，保留关键限定词（如部位、类型）。输出 JSON：
{{"mapping": {{"原名称": "规范名"}}}}

实体列表：{entities}
"""

GROUND_PROMPT = """基于给定上下文，为每个实体写一句简洁、客观、不编造的中文百科式描述（20~40 字）。
输出 JSON：{{"descriptions": {{"规范名": "描述"}}}}

上下文：
{context}

实体列表：{entities}
"""

REFINE_PROMPT = """下面是候选描述列表（可能来自本地知识库、维基百科或模型生成）。请结合上下文，
选择最贴合、最精确的一条；若都不贴切，就基于上下文改写一条（不编造）。输出 JSON：
{{"descriptions": {{"实体": "最终描述"}}}}

上下文：
{context}

候选：
{candidates}
"""

INSPECT_PROMPT = """请给下面每条描述打分（0-10：是否贴合上下文、是否编造），并列出需要重写的实体。
输出 JSON：{{"scores": {{"实体": 8}}, "rewrite": ["需要重写的实体"]}}

上下文：
{context}

描述：
{descriptions}
"""


class LLMClient:
    """Minimal OpenAI-compatible chat client used by the construction stages."""

    def __init__(self, config: dict) -> None:
        ext = config.get("extractor", {})
        self.base_url = (ext.get("base_url") or "").strip()
        self.model = ext.get("model") or ""
        self.api_key_env = ext.get("api_key_env") or "DEEPSEEK_API_KEY"
        self.temperature = float(ext.get("temperature", 0.0))
        self.timeout = float(ext.get("timeout", 60))
        self.max_retries = int(ext.get("max_retries", 2))
        self.max_new_tokens = int(ext.get("max_new_tokens", 1024))

    @property
    def ready(self) -> bool:
        return bool(self.base_url)

    def chat(self, prompt: str) -> str:
        if not self.ready:
            raise RuntimeError("construction stages require [extractor].base_url")
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
        }
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(self.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        last_error: Exception | None = None
        for _ in range(self.max_retries + 1):
            try:
                data = post_json(self.base_url, body, headers, self.timeout)
                return data["choices"][0]["message"]["content"]
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        raise RuntimeError(f"construction LLM call failed: {last_error}")


def _parse_json(content: str | None) -> dict:
    if not isinstance(content, str):
        return {}
    text = content.strip()
    text = text.removeprefix("```json").removeprefix("```")
    if text.endswith("```"):
        text = text[:-3]
    try:
        value = json.loads(text.strip())
        return value if isinstance(value, dict) else {}
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                value = json.loads(text[start : end + 1])
                return value if isinstance(value, dict) else {}
            except Exception:
                return {}
        return {}


class ConstructionPipeline:
    def __init__(self, config: dict, chat=None, searcher: KnowledgeSearcher | None = None) -> None:
        section = config.get("construction", {})
        self.enabled = bool(section.get("enabled", False))
        self.do_rewrite = bool(section.get("rewriter", True))
        self.do_grounding = bool(section.get("grounding", True))
        self.do_refinement = bool(section.get("refinement", True))
        self.do_inspector = bool(section.get("inspector", False))
        self.client = LLMClient(config)
        self._chat = chat or self.client.chat
        self.searcher = searcher if searcher is not None else get_searcher(config)

    def _safe_chat(self, prompt: str) -> str:
        try:
            return self._chat(prompt)
        except Exception:
            return ""

    def _safe_json(self, prompt: str) -> dict:
        return _parse_json(self._safe_chat(prompt))

    @staticmethod
    def _batches(items: list, size: int):
        for start in range(0, len(items), size):
            yield items[start:start + size]

    def _clean_names(self, names: list[str]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for name in names:
            text = (name or "").strip()
            if not text or len(text) > 40 or "\n" in text or "|" in text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(text)
        return unique

    # -- stages ----------------------------------------------------------
    def rewrite(self, doc: dict) -> str:
        text = (doc.get("text") or "").strip()
        if not self.do_rewrite or not text:
            return text
        title = (doc.get("title") or "").strip()
        prompt = REWRITE_PROMPT.format(text=(title + "\n" + text).strip())
        rewritten = self._safe_chat(prompt).strip()
        return rewritten or text

    def enrich_entities(self, doc: dict, names: list[str], rewritten: str) -> tuple[dict, dict]:
        """Return ``(canonical_map, refined_descriptions)`` for the given names."""
        unique = self._clean_names(names)
        canonical = {name: name for name in unique}
        descriptions: dict[str, str] = {}
        if not unique:
            return canonical, descriptions

        # dictionary-first normalization (e.g. ICD-10 preferred names)
        if hasattr(self.searcher, "canonical"):
            for name in unique:
                try:
                    canonical_name = self.searcher.canonical(name)
                except Exception:
                    canonical_name = None
                if canonical_name:
                    canonical[name] = canonical_name

        if self.do_grounding:
            for batch in self._batches(unique, 20):
                data = self._safe_json(
                    NORMALIZE_PROMPT.format(entities=json.dumps(batch, ensure_ascii=False))
                )
                mapping = data.get("mapping") or {}
                if isinstance(mapping, dict):
                    for name in batch:
                        candidate = mapping.get(name)
                        if isinstance(candidate, str) and candidate.strip():
                            canonical[name] = candidate.strip()

        targets = list(dict.fromkeys(canonical.values()))

        if self.do_grounding:
            for batch in self._batches(targets, 10):
                data = self._safe_json(
                    GROUND_PROMPT.format(
                        context=rewritten[:2000],
                        entities=json.dumps(batch, ensure_ascii=False),
                    )
                )
                found = data.get("descriptions") or {}
                if isinstance(found, dict):
                    for key, value in found.items():
                        if isinstance(value, str) and value.strip():
                            descriptions[key.strip()] = value.strip()

        # external knowledge grounding (local glossary / Wikipedia)
        external: dict[str, list[str]] = {}
        for target in targets:
            try:
                external[target] = self.searcher.search(target) or []
            except Exception:
                external[target] = []

        if self.do_refinement:
            for batch in self._batches(targets, 10):
                candidates_payload: dict[str, list[str]] = {}
                for target in batch:
                    items = list(external.get(target, []))
                    if descriptions.get(target):
                        items.append(descriptions[target])
                    if items:
                        candidates_payload[target] = items
                if not candidates_payload:
                    continue
                data = self._safe_json(
                    REFINE_PROMPT.format(
                        context=rewritten[:2000],
                        candidates=json.dumps(candidates_payload, ensure_ascii=False),
                    )
                )
                refined = data.get("descriptions") or {}
                if isinstance(refined, dict):
                    for key, value in refined.items():
                        if isinstance(value, str) and value.strip():
                            descriptions[key.strip()] = value.strip()
            # if the LLM returned nothing, keep the best external candidate
            for target in targets:
                if not descriptions.get(target) and external.get(target):
                    descriptions[target] = external[target][0]
        else:
            for target in targets:
                if external.get(target):
                    descriptions[target] = external[target][0]

        if self.do_inspector and descriptions:
            for batch in self._batches(list(descriptions.keys()), 20):
                payload = {key: descriptions[key] for key in batch}
                data = self._safe_json(
                    INSPECT_PROMPT.format(
                        context=rewritten[:2000],
                        descriptions=json.dumps(payload, ensure_ascii=False),
                    )
                )
                scores = data.get("scores") or {}
                if isinstance(scores, dict):
                    for key, score in scores.items():
                        try:
                            if float(score) < 6 and key in descriptions:
                                descriptions[key] = ""
                        except (TypeError, ValueError):
                            continue

        # dictionary-first authoritative descriptions (e.g. ICD-10 code + name)
        if hasattr(self.searcher, "preferred"):
            for target in targets:
                try:
                    preferred = self.searcher.preferred(target)
                except Exception:
                    preferred = None
                if preferred:
                    descriptions[target] = preferred

        # map description keys back to canonical names (the LLM sometimes answers
        # with the original surface form instead of the canonical name)
        canonical_values = set(canonical.values())
        normalized: dict[str, str] = {}
        for key, value in descriptions.items():
            if key in canonical:
                normalized[canonical[key]] = value
            elif key in canonical_values:
                normalized[key] = value
        return canonical, normalized


def get_construction_pipeline(config: dict) -> ConstructionPipeline | None:
    if not config.get("construction", {}).get("enabled"):
        return None
    return ConstructionPipeline(config)


__all__ = ["LLMClient", "ConstructionPipeline", "get_construction_pipeline"]
