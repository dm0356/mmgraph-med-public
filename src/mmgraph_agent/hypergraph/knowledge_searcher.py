"""External knowledge grounding: look up candidate descriptions for entities.

This is the "searcher" stage of the M3KG-RAG construction pipeline. Two sources
are supported, both pluggable behind :class:`KnowledgeSearcher`:

* :class:`LocalGlossarySearcher` - a local TSV/JSONL medical glossary. Recommended
  for hospital data: only the entity name is looked up locally, nothing leaves the
  machine.
* :class:`WikipediaSearcher` - Chinese (or other) Wikipedia via the MediaWiki API.
  Useful for general concepts; **medical accuracy must be verified**.

The construction pipeline feeds these candidates, together with LLM-generated
descriptions, into the context-aware refinement stage.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path


class KnowledgeSearcher:
    name = "none"

    def search(self, term: str) -> list[str]:  # pragma: no cover - interface
        return []


class NullSearcher(KnowledgeSearcher):
    name = "none"


class LocalGlossarySearcher(KnowledgeSearcher):
    """Look up descriptions from a local TSV or JSONL glossary.

    TSV format::

        2型糖尿病\t一种以胰岛素抵抗为特征的慢性代谢性疾病。

    JSONL format (supports aliases)::

        {"term": "2型糖尿病", "description": "...", "aliases": ["T2DM", "二型糖尿病"]}
    """

    name = "local"

    def __init__(self, path: str | Path, max_candidates: int = 1) -> None:
        self.path = Path(path)
        self.max_candidates = max(1, int(max_candidates))
        self._index: dict[str, list[str]] = {}
        self._load()

    def _add(self, term: str, description: str) -> None:
        term = (term or "").strip().lower()
        description = (description or "").strip()
        if not term or not description:
            return
        bucket = self._index.setdefault(term, [])
        if description not in bucket:
            bucket.append(description)

    def _add_record(self, item: dict) -> None:
        term = item.get("term") or item.get("name") or ""
        description = item.get("description") or ""
        if not isinstance(description, str):
            return
        self._add(term, description)
        for alias in item.get("aliases") or []:
            if isinstance(alias, str):
                self._add(alias, description)

    def _load(self) -> None:
        if not self.path.exists():
            return
        suffix = self.path.suffix.lower()
        if suffix in (".tsv", ".txt"):
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip() or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    self._add(parts[0], parts[1])
            return

        # .jsonl / .json
        text = self.path.read_text(encoding="utf-8")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = [json.loads(line) for line in text.splitlines() if line.strip()]

        if isinstance(data, dict):
            if "term" in data or "name" in data:
                self._add_record(data)
            else:
                for term, description in data.items():
                    if isinstance(description, str):
                        self._add(term, description)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    self._add_record(item)

    def search(self, term: str) -> list[str]:
        return self._index.get((term or "").strip().lower(), [])[: self.max_candidates]


class WikipediaSearcher(KnowledgeSearcher):
    """Fetch the intro extract of a Wikipedia page as a candidate description."""

    name = "wikipedia"

    API = "https://{lang}.wikipedia.org/w/api.php"

    def __init__(self, config: dict) -> None:
        grounding = config.get("grounding", {})
        self.lang = grounding.get("lang") or "zh"
        self.timeout = float(grounding.get("timeout", 10))
        self.max_candidates = max(1, int(grounding.get("max_candidates", 1)))
        self.max_chars = int(grounding.get("max_chars", 200))
        self._url = self.API.format(lang=self.lang)
        self._cache: dict[str, list[str]] = {}

    def search(self, term: str) -> list[str]:
        term = (term or "").strip()
        if not term:
            return []
        if term in self._cache:
            return self._cache[term]

        extract = self._fetch_extract(term)
        if not extract:
            title = self._search_title(term)
            if title and title != term:
                extract = self._fetch_extract(title)

        result = [self._clean(extract)] if extract else []
        self._cache[term] = result
        return result

    def _clean(self, text: str) -> str:
        text = " ".join(text.split())
        return text[: self.max_chars]

    def _fetch_extract(self, title: str) -> str:
        data = self._query(
            {
                "action": "query",
                "format": "json",
                "prop": "extracts",
                "exintro": 1,
                "explaintext": 1,
                "redirects": 1,
                "titles": title,
            }
        )
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            extract = (page.get("extract") or "").strip()
            if extract:
                return extract
        return ""

    def _search_title(self, term: str) -> str:
        data = self._query(
            {"action": "query", "format": "json", "list": "search", "srsearch": term, "srlimit": 1}
        )
        hits = data.get("query", {}).get("search", [])
        return hits[0].get("title", "") if hits else ""

    def _query(self, params: dict) -> dict:
        try:
            return self._get_json(params)
        except Exception:
            return {}

    def _get_json(self, params: dict) -> dict:  # pragma: no cover - network
        url = self._url + "?" + urllib.parse.urlencode(params)
        request = urllib.request.Request(
            url, headers={"User-Agent": "mmgraph-agent/0.1 (research; contact: local)"}
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8", "replace"))


class ICD10Searcher(KnowledgeSearcher):
    """ICD-10 Chinese dictionary: canonical names + authoritative codes.

    The data file is a JSON tree (``value`` / ``label`` / ``children``), e.g.
    https://github.com/chaseliu/ICD-10-CN. It is not vendored in the repo;
    download it into ``data/icd10/diseases.json`` (see docs/MEDICAL_TERMINOLOGY.md).

    Note: that public dataset is a **3-character category list** (E11, I48, ...),
    not the full 6-digit clinical coding. ``ALIASES`` maps common clinical terms
    to readable names and ``HINTS`` bridges them to the category labels.
    """

    name = "icd10"

    # clinical abbreviation / synonym -> readable standard name
    ALIASES = {
        "冠心病": "冠状动脉粥样硬化性心脏病",
        "心梗": "心肌梗死",
        "心衰": "心力衰竭",
        "房颤": "心房颤动",
        "室速": "室性心动过速",
        "高血压": "高血压病",
    }

    # readable standard name -> substrings that identify the ICD category label
    HINTS = {
        "冠状动脉粥样硬化性心脏病": ["慢性缺血性心脏病", "心绞痛", "动脉粥样硬化"],
        "心肌梗死": ["急性心肌梗死"],
        "急性非ST段抬高型心肌梗死": ["急性心肌梗死"],
        "不稳定型心绞痛": ["心绞痛"],
        "2型糖尿病": ["非胰岛素依赖型糖尿病"],
        "1型糖尿病": ["胰岛素依赖型糖尿病"],
        "心房颤动": ["心房纤颤"],
        "室性心动过速": ["阵发性心动过速"],
        "高脂血症": ["脂血症", "脂蛋白代谢紊乱"],
        "颈动脉粥样硬化": ["动脉粥样硬化"],
        "高血压病": ["原发性高血压", "高血压病"],
    }

    def __init__(self, path: str | Path, max_candidates: int = 3) -> None:
        self.path = Path(path)
        self.max_candidates = max(1, int(max_candidates))
        self.entries: list[tuple[str, str, bool]] = []  # (code, label, is_leaf)
        self._by_label: dict[str, tuple[str, str]] = {}
        self._load()

    # suffixes that indicate a diagnosis rather than an organ / attribute
    DIAGNOSIS_HINTS = (
        "病", "炎", "梗死", "硬化", "衰竭", "综合征", "症", "癌", "瘤", "感染",
        "损伤", "溃疡", "结石", "栓塞", "血栓", "狭窄", "不全", "反流", "失常",
        "过速", "过缓", "颤动", "扑动", "早搏", "阻滞", "贫血", "糖尿病", "高血压",
        "高脂血症", "息肉", "囊肿", "畸形", "骨折", "中毒", "变性", "萎缩", "肥大",
        "坏死", "出血", "缺血", "积水", "疝", "痔", "癣", "疹", "痛",
    )

    @staticmethod
    def _strip_modifiers(text: str) -> str:
        result = (text or "").strip()
        for suffix in ("史", "病史", "一级", "二级", "三级", "1级", "2级", "3级"):
            if result.endswith(suffix) and len(result) > len(suffix):
                result = result[: -len(suffix)]
                break
        return result

    def _load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8-sig"))

        def walk(node):
            code = node.get("value")
            label = (node.get("label") or "").strip()
            children = node.get("children") or []
            if code and label:
                self.entries.append((str(code), label, not children))
                self._by_label.setdefault(label, (str(code), label))
            for child in children:
                walk(child)

        if isinstance(data, list):
            for node in data:
                walk(node)
        elif isinstance(data, dict):
            walk(data)

    def _resolve_alias(self, term: str) -> str:
        text = (term or "").strip()
        if text in self.ALIASES:
            return self.ALIASES[text]
        for key, value in self.ALIASES.items():
            if key in text:
                return value
        return text

    @staticmethod
    def _rank(candidates, target: str):
        # prefer leaf categories over range nodes, then the shortest / most
        # specific category label (I21 before "I23 ...近期并发症")
        return sorted(candidates, key=lambda item: ("-" in item[0], len(item[1]), -len(item[0])))

    def _find_entry(self, term: str) -> tuple[str, str] | None:
        text = (term or "").strip()
        if not text or not self.entries:
            return None
        base = self._strip_modifiers(text)
        if text in self._by_label:
            return self._by_label[text]
        if base in self._by_label:
            return self._by_label[base]
        resolved = self._resolve_alias(base)
        if resolved in self._by_label:
            return self._by_label[resolved]
        # short generic concepts (血压 / 血糖 / 用药 / 手术) must not be coded
        # by fuzzy matching; they are attributes/procedures, not ICD diagnoses
        if len(base) <= 2:
            return None
        hints = (
            list(self.HINTS.get(resolved, []))
            + list(self.HINTS.get(base, []))
            + list(self.HINTS.get(text, []))
        )
        # fuzzy fallback only for diagnosis-like terms (not "甲状腺"/"下肢静脉")
        if not hints and not any(suffix in base for suffix in self.DIAGNOSIS_HINTS):
            return None
        for hint in hints:
            candidates = [(c, l, f) for c, l, f in self.entries if hint in l and "*" not in c]
            if candidates:
                ranked = self._rank(candidates, resolved or base)
                return ranked[0][0], ranked[0][1]
        candidates = [
            (c, l, f)
            for c, l, f in self.entries
            if "*" not in c and (base in l or l in base or resolved in l or l in resolved)
        ]
        if candidates:
            ranked = self._rank(candidates, base)
            return ranked[0][0], ranked[0][1]
        return None

    def canonical(self, term: str) -> str | None:
        if not self.entries:
            return None
        text = (term or "").strip()
        resolved = self._resolve_alias(text)
        if resolved != text:
            return resolved
        return text if text in self._by_label else None

    def preferred(self, term: str) -> str | None:
        hit = self._find_entry(term)
        return f"ICD-10 {hit[0]} {hit[1]}" if hit else None

    def search(self, term: str) -> list[str]:
        preferred = self.preferred(term)
        return [preferred] if preferred else []


def get_searcher(config: dict) -> KnowledgeSearcher:
    grounding = config.get("grounding", {})
    kind = str(grounding.get("searcher", "none") or "none").lower()
    if kind == "local":
        path = grounding.get("local_path") or ""
        return LocalGlossarySearcher(path, int(grounding.get("max_candidates", 1)))
    if kind == "wikipedia":
        return WikipediaSearcher(config)
    if kind == "icd10":
        path = grounding.get("icd10_path") or "data/icd10/diseases.json"
        return ICD10Searcher(path, int(grounding.get("max_candidates", 3)))
    return NullSearcher()


__all__ = [
    "KnowledgeSearcher",
    "NullSearcher",
    "LocalGlossarySearcher",
    "WikipediaSearcher",
    "ICD10Searcher",
    "get_searcher",
]
