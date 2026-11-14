"""LLM fallback for structured variable extraction.

The rule extractor (clinical_features.py) is transparent and fast but misses
free-form phrasing. This module asks a local OpenAI-compatible LLM to fill only
the variables the rules left ``missing``/``uncertain``, then validates every
answer against the source text, the variable dictionary, units and ranges.

Nothing is trusted blindly: if the LLM value is not supported by the report text
or falls outside the plausible range, it is rejected and the rule result is kept.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from ..http_utils import post_json
from .clinical_features import VARIABLES, UNIT_FACTORS, extract_report

STATUSES = {"present", "absent", "uncertain", "missing"}


def _parse_json(content: str) -> dict:
    text = (content or "").strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                value = json.loads(text[start:end + 1])
                return value if isinstance(value, dict) else {}
            except Exception:
                return {}
    return {}


def _bool_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("true", "1", "是", "有", "阳性", "present"):
        return True
    if text in ("false", "0", "否", "无", "阴性", "absent"):
        return False
    return None


def _supported(text: str, variable: str, spec: dict, raw_value: Any, evidence: str = "") -> bool:
    aliases = sorted(spec["aliases"], key=len, reverse=True)
    # an LLM paraphrase is acceptable only if its cited evidence is verbatim in
    # the report *and* contains the variable's own alias (or its numeric value)
    if evidence and evidence in text:
        if spec["kind"] == "numeric":
            try:
                number = float(str(raw_value).lstrip("<>"))
            except ValueError:
                return False
            if variable == "age" and "岁" in evidence and f"{number:g}" in evidence:
                return True
            return f"{number:g}" in evidence and any(alias in evidence for alias in aliases)
        return any(alias in evidence for alias in aliases)
    for alias in aliases:
        pattern = re.escape(alias)
        if alias.isascii():
            pattern = rf"(?<![A-Za-z]){pattern}(?![A-Za-z])"
        if spec["kind"] == "numeric":
            try:
                number = float(str(raw_value).lstrip("<>"))
            except ValueError:
                continue
            if re.search(pattern + r"[^0-9<>]{0,14}" + re.escape(f"{number:g}"), text):
                return True
        else:
            if re.search(pattern, text):
                return True
    return False


class LLMFeatureExtractor:
    def __init__(self, base_url: str, model: str, timeout: float = 180,
                 api_key_env: str = "LOCAL_LLM_KEY", temperature: float = 0.0) -> None:
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.api_key_env = api_key_env
        self.temperature = temperature

    def _prompt(self, text: str, variables: list[str]) -> str:
        specs = []
        for variable in variables:
            spec = VARIABLES[variable]
            if spec["kind"] == "numeric":
                detail = f"数值，单位 {spec.get('unit', '')}，合理范围 {spec.get('range')}"
            elif spec["kind"] == "boolean":
                detail = "布尔（true/false；未见=absent；可疑=uncertain；未提及=missing）"
            else:
                detail = f"类别，取值 {spec.get('values')}"
            specs.append(f"- {variable}: {detail}")
        return (
            "你是临床研究变量抽取器。根据报告文本抽取指定变量，只输出 JSON，不要解释。\n"
            "status 只能是 present / absent / uncertain / missing。\n"
            "每个变量格式：{\"value\": 值或null, \"unit\": \"单位\", \"status\": \"...\", \"evidence\": \"原文片段\"}\n"
            "变量定义：\n" + "\n".join(specs) + "\n\n报告文本：\n" + text[:4000]
        )

    def _call(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(self.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是严谨的临床变量抽取器，只输出 JSON。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": 1024,
        }
        data = post_json(self.base_url, body, headers, self.timeout)
        return data["choices"][0]["message"]["content"]

    def extract(self, text: str, variables: list[str], patient_id: str = "",
                report_time: str = "", source: str = "") -> list[dict]:
        if not variables:
            return []
        try:
            payload = _parse_json(self._call(self._prompt(text, variables)))
        except Exception:
            return []

        records: list[dict] = []
        for variable in variables:
            item = payload.get(variable)
            if not isinstance(item, dict):
                continue
            spec = VARIABLES[variable]
            status = str(item.get("status", "missing")).lower()
            if status not in STATUSES:
                status = "missing"
            raw_value = item.get("value")
            evidence = str(item.get("evidence") or "")
            if evidence and evidence not in text:
                evidence = ""

            if status == "present" and evidence and any(
                word in evidence for word in ("可疑", "考虑", "可能", "？", "?")
            ):
                status = "uncertain"
            if status in ("present", "absent") and not _supported(
                text, variable, spec, raw_value, evidence
            ):
                status = "missing"

            value: Any = None
            if status == "present":
                if spec["kind"] == "numeric":
                    try:
                        value = float(str(raw_value).lstrip("<>"))
                    except ValueError:
                        status, value = "missing", None
                    else:
                        unit = str(item.get("unit") or spec.get("unit", "")).lower()
                        value *= UNIT_FACTORS.get((unit, spec.get("unit", "").lower()), 1.0)
                        low, high = spec.get("range", (float("-inf"), float("inf")))
                        if not low <= value <= high:
                            status = "out_of_range"
                elif spec["kind"] == "boolean":
                    value = _bool_value(raw_value)
                    if value is None:
                        status = "missing"
                else:
                    value = str(raw_value) if raw_value in spec["values"] else None
                    if value is None:
                        status = "missing"
            elif status == "absent" and spec["kind"] == "boolean":
                value = False
            elif status == "absent":
                value = None

            records.append({
                "variable": variable,
                "value": value,
                "unit": spec.get("unit", ""),
                "status": status,
                "confidence": 0.75 if status == "present" else 0.5,
                "snippet": evidence,
                "patient_id": patient_id,
                "time": report_time,
                "source": source,
                "kind": spec["kind"],
                "extractor": "llm",
            })
        return records


def extract_report_hybrid(text: str, patient_id: str = "", report_time: str = "",
                          source: str = "", llm: LLMFeatureExtractor | None = None
                          ) -> tuple[list[dict], list[dict]]:
    """Rules first; LLM fills missing/uncertain; conflicts are reported separately."""
    rule_records = extract_report(text, patient_id=patient_id, report_time=report_time, source=source)
    for record in rule_records:
        record.setdefault("extractor", "rule")
    if llm is None:
        return rule_records, []

    needs = [record["variable"] for record in rule_records
             if record["status"] in ("missing", "uncertain")]
    llm_records = []
    # batch the prompt: long variable lists make the JSON output unreliable
    for start in range(0, len(needs), 12):
        llm_records.extend(
            llm.extract(text, needs[start:start + 12], patient_id, report_time, source)
        )
    llm_by_variable = {record["variable"]: record for record in llm_records}

    merged: list[dict] = []
    conflicts: list[dict] = []
    for record in rule_records:
        candidate = llm_by_variable.get(record["variable"])
        if candidate and record["status"] in ("missing", "uncertain"):
            if candidate["status"] in ("present", "absent", "out_of_range"):
                candidate["upgraded_by_llm"] = record["status"] == "uncertain"
                merged.append(candidate)
                if record["status"] == "uncertain" and candidate["status"] == "absent":
                    conflicts.append({
                        "patient_id": patient_id, "variable": record["variable"],
                        "rule_status": record["status"], "llm_status": candidate["status"],
                        "note": "rule uncertain vs LLM absent",
                    })
                continue
        merged.append(record)
    return merged, conflicts


__all__ = ["LLMFeatureExtractor", "extract_report_hybrid"]
