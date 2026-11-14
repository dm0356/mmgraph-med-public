"""LLM-based entity / relation extraction (optional path).

Ported and adapted from the ``ai-knowledge-graph`` extraction prompts
(Apache-2.0) for Chinese clinical text. The zero-API path keeps the local
regex / NER extractor; when ``[extractor].mode = "llm"`` the builder asks an
OpenAI-compatible chat endpoint (DeepSeek, Qwen/DashScope compatible mode, a
local vLLM, ...) for a strict JSON payload:

.. code-block:: json

    {"entities": ["糖尿病", "空腹血糖"],
     "relations": [{"head": "糖尿病", "relation": "表现为", "tail": "空腹血糖升高"}]}

Responses are cached on disk, retried on failure, and degrade to the regex
extractor so a flaky endpoint cannot abort a build.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from ..http_utils import post_json

SYSTEM_ZH = "你是一个严谨的知识抽取器。只输出 JSON，不要解释，不要输出 Markdown 代码块。"
PROMPT_ZH = """从下面的中文医学文本中抽取知识。
要求：
1. 实体：疾病、症状、检查、药品、身体部位、人物、机构、数值指标等专有名词，保持原文表述并去重。
2. 关系：用简洁的中文谓词连接两个实体；没有明确关系时 relations 返回空数组。
3. 严格输出 JSON，格式为：
{{"entities": ["..."], "relations": [{{"head": "...", "relation": "...", "tail": "..."}}]}}

文本：
{text}
"""

SYSTEM_EN = "You are a rigorous knowledge extractor. Output JSON only, no prose, no Markdown fences."
PROMPT_EN = """Extract knowledge from the text below.
Rules:
1. Entities: named concepts (diseases, symptoms, exams, drugs, body parts, people,
   organisations, measurements), keep the surface form, deduplicate.
2. Relations: a short predicate linking two entities; [] when none is explicit.
3. Return strict JSON: {{"entities": ["..."], "relations": [{{"head": "...", "relation": "...", "tail": "..."}}]}}

Text:
{text}
"""

EMPTY = {"entities": [], "relations": []}


class Extractor:
    name = "extractor"

    def extract(self, text: str, context: str | None = None) -> list[str]:  # pragma: no cover - interface
        raise NotImplementedError

    def extract_rich(self, text: str, context: str | None = None) -> dict:
        return {"entities": self.extract(text), "relations": []}


class RegexExtractor(Extractor):
    """Zero-API extractor wrapping the local regex / lexicon extractor."""

    name = "regex"

    def __init__(self, config: dict) -> None:
        self.config = config

    def extract(self, text: str, context: str | None = None) -> list[str]:
        from .parser import extract_entities

        return extract_entities(text, self.config)


class LLMExtractor(Extractor):
    """OpenAI-compatible chat endpoint extractor (DeepSeek / Qwen / local)."""

    name = "llm"

    def __init__(self, config: dict) -> None:
        ext = config.get("extractor", {})
        self.base_url = (ext.get("base_url") or "").strip()
        self.model = ext.get("model") or "deepseek-chat"
        self.api_key_env = ext.get("api_key_env") or "MMGRAPH_API_KEY"
        self.temperature = float(ext.get("temperature", 0.0))
        self.timeout = float(ext.get("timeout", 60))
        self.max_retries = int(ext.get("max_retries", 2))
        self.max_new_tokens = int(ext.get("max_new_tokens", 1024))
        self.language = str(ext.get("language") or "zh").lower()
        self.cache_dir = ext.get("cache_dir") or "out/.extract_cache"
        self.max_entities = int(config["hypergraph"].get("max_entities_per_unit", 12))
        self._fallback = RegexExtractor(config)

    # -- public API ------------------------------------------------------
    def extract(self, text: str, context: str | None = None) -> list[str]:
        return self.extract_rich(text, context)["entities"]

    def extract_rich(self, text: str, context: str | None = None) -> dict:
        text = (text or "").strip()
        if not text:
            return dict(EMPTY)
        cached = self._cache_get(text, context)
        if cached is not None:
            return cached

        prompt = (PROMPT_ZH if "zh" in self.language else PROMPT_EN).format(text=text)
        if context:
            prompt = f"参考上下文（用于消歧，不要照抄）：\n{context[:2000]}\n\n" + prompt
        try:
            content = self._chat(prompt)
        except Exception:
            # never abort a build on a flaky endpoint: fall back to the local
            # zero-API extractor so construction still succeeds.
            return {"entities": self._fallback.extract(text), "relations": []}

        payload = _parse_json_payload(content)
        if not isinstance(payload, dict):
            return {"entities": self._fallback.extract(text), "relations": []}

        entities: list[str] = []
        seen: set[str] = set()
        for entity in payload.get("entities") or []:
            if not isinstance(entity, str):
                continue
            entity = entity.strip()
            key = entity.lower()
            if entity and key not in seen:
                seen.add(key)
                entities.append(entity)
        entities = entities[: self.max_entities]

        relations: list[dict] = []
        for rel in payload.get("relations") or []:
            if not isinstance(rel, dict) or not rel.get("head") or not rel.get("tail"):
                continue
            relations.append(
                {
                    "head": str(rel["head"]).strip(),
                    "relation": str(rel.get("relation") or "相关").strip(),
                    "tail": str(rel["tail"]).strip(),
                }
            )

        result = {"entities": entities, "relations": relations}
        self._cache_put(text, context, result)
        return result

    def ping(self) -> str:
        """Send a tiny probe request; raises on failure (used by check_api)."""
        return self._chat("请只回复：OK")

    # -- HTTP ------------------------------------------------------------
    def _chat(self, prompt: str) -> str:
        if not self.base_url:
            raise RuntimeError(
                "LLM extraction is not configured: set [extractor].base_url in the config."
            )
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_ZH if "zh" in self.language else SYSTEM_EN},
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
            except Exception as exc:  # noqa: BLE001 - retry any transport/parse error
                last_error = exc
        raise RuntimeError(f"LLM extraction failed after {self.max_retries + 1} attempt(s): {last_error}")

    # -- disk cache ------------------------------------------------------
    def _cache_key(self, text: str, context: str | None = None) -> str:
        raw = self.model + "\x00" + text + "\x00" + (context or "")
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _cache_get(self, text: str, context: str | None = None) -> dict | None:
        try:
            path = Path(self.cache_dir) / f"{self._cache_key(text, context)}.json"
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return None

    def _cache_put(self, text: str, context: str | None, value: dict) -> None:
        try:
            directory = Path(self.cache_dir)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"{self._cache_key(text, context)}.json").write_text(
                json.dumps(value, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass


def _parse_json_payload(content: str | None) -> dict | None:
    """Extract a JSON object from a model reply (tolerates fences / prose)."""
    if not isinstance(content, str):
        return None
    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except Exception:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            value = json.loads(text[start : end + 1])
            return value if isinstance(value, dict) else None
        except Exception:
            return None
    return None


def get_extractor(config: dict) -> Extractor:
    """Build the configured extractor, defaulting to the zero-API regex path."""
    ext = config.get("extractor", {})
    mode = str(ext.get("mode", "auto")).lower()
    if mode == "regex":
        return RegexExtractor(config)
    if mode == "llm":
        return LLMExtractor(config)
    # auto: use the API only when an endpoint has been configured
    if (ext.get("base_url") or "").strip():
        return LLMExtractor(config)
    return RegexExtractor(config)


__all__ = ["Extractor", "RegexExtractor", "LLMExtractor", "get_extractor"]
