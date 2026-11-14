"""Structured variable extraction from Chinese clinical reports (prototype).

Design goals:
  * MAARS-aligned variable dictionary (EHR + cardiac-imaging report variables);
  * unit normalisation (cm -> mm, L -> ml, ...);
  * negation ("未见/无/阴性") and uncertainty ("可疑/考虑") semantics;
  * provenance: every value carries time, source, snippet and confidence;
  * missing is explicit (status="missing"), never silently zero.

This is a rule + dictionary prototype. A constrained LLM extractor can be added
later for free-form phrasing, using the same output schema.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

# ---------------------------------------------------------------------------
# Variable dictionary (subset of MAARS EHR + CIR variables)
# ---------------------------------------------------------------------------
VARIABLES: dict[str, dict[str, Any]] = {
    # demographics / history
    "age": {"aliases": ["年龄"], "kind": "numeric", "unit": "岁", "range": (0, 120)},
    "BMI": {"aliases": ["BMI", "体重指数"], "kind": "numeric", "unit": "kg/m2", "range": (10, 60)},
    "hypertension": {"aliases": ["高血压"], "kind": "boolean"},
    "diabetes": {"aliases": ["糖尿病"], "kind": "boolean"},
    "AF": {"aliases": ["心房颤动", "房颤", "AF"], "kind": "boolean"},
    "NSVT": {"aliases": ["NSVT", "非持续性室速", "短阵室速"], "kind": "boolean"},
    "syncope": {"aliases": ["晕厥"], "kind": "boolean"},
    "presyncope": {"aliases": ["先兆晕厥", "黑矇"], "kind": "boolean"},
    "angina": {"aliases": ["心绞痛"], "kind": "boolean"},
    "dyspnea": {"aliases": ["呼吸困难", "气短"], "kind": "boolean"},
    "family_history_SCD": {"aliases": ["猝死家族史", "家族史"], "kind": "boolean"},
    "dyslipidemia": {"aliases": ["高脂血症", "血脂异常"], "kind": "boolean"},
    # echo / CMR report variables
    "LVEF": {"aliases": ["左心室射血分数", "左室射血分数", "LVEF", "EF", "射血分数"],
             "kind": "numeric", "unit": "%", "range": (5, 90)},
    "IVS_thickness": {"aliases": ["室间隔基底段厚度", "室间隔厚度", "室间隔", "IVSd", "IVS"],
                       "kind": "numeric", "unit": "mm", "range": (3, 40)},
    "LA_diameter": {"aliases": ["左房内径", "左心房内径", "LA直径", "LA"], "kind": "numeric",
                    "unit": "mm", "range": (20, 80)},
    "LVEDV": {"aliases": ["左室舒张末容积", "LVEDV"], "kind": "numeric", "unit": "ml",
              "range": (40, 400)},
    "LVESV": {"aliases": ["左室收缩末容积", "LVESV"], "kind": "numeric", "unit": "ml",
              "range": (10, 300)},
    "LVOT_gradient": {"aliases": ["左室流出道压差", "LVOT压差", "LVOT梯度"], "kind": "numeric",
                      "unit": "mmHg", "range": (0, 200)},
    "LGE_percent": {"aliases": ["LGE百分比", "LGE%", "延迟强化百分比"], "kind": "numeric",
                    "unit": "%", "range": (0, 80)},
    "LGE_present": {"aliases": ["LGE", "延迟强化", "钆延迟强化"], "kind": "boolean"},
    "SAM": {"aliases": ["SAM", "二尖瓣收缩期前向运动"], "kind": "boolean"},
    "thrombus": {"aliases": ["血栓"], "kind": "boolean"},
    "pericardial_effusion": {"aliases": ["心包积液"], "kind": "boolean"},
    "mitral_regurgitation": {"aliases": ["二尖瓣反流", "二尖瓣返流"], "kind": "categorical",
                             "values": ["轻度", "中度", "重度"]},
    "TR_severity": {"aliases": ["三尖瓣反流", "三尖瓣返流"], "kind": "categorical",
                    "values": ["轻度", "中度", "重度"]},
    "LGE_pattern": {"aliases": ["强化模式", "强化部位", "LGE分布"], "kind": "categorical",
                    "values": ["心内膜下", "中层", "心外膜下", "透壁", "斑片状"]},
    "wall_motion_abnormal": {"aliases": ["室壁运动异常", "节段性室壁运动异常"], "kind": "boolean"},
    "RV_dysfunction": {"aliases": ["右室功能减低", "右室收缩功能减低"], "kind": "boolean"},
    # labs / ECG
    "potassium": {"aliases": ["血钾", "钾离子", "K+"], "kind": "numeric", "unit": "mmol/L",
                  "range": (1.5, 8.0)},
    "NT_proBNP": {"aliases": ["NT-proBNP", "NT-proBNP", "脑钠肽", "BNP"], "kind": "numeric",
                  "unit": "pg/mL", "range": (0, 50000)},
    "eGFR": {"aliases": ["eGFR", "肾小球滤过率"], "kind": "numeric", "unit": "ml/min/1.73m2",
             "range": (1, 150)},
    "HbA1c": {"aliases": ["糖化血红蛋白", "HbA1c"], "kind": "numeric", "unit": "%",
              "range": (3, 20)},
    "LDL": {"aliases": ["LDL-C", "低密度脂蛋白"], "kind": "numeric", "unit": "mmol/L",
            "range": (0.2, 15)},
    "HR": {"aliases": ["心率", "HR"], "kind": "numeric", "unit": "次/分", "range": (20, 250)},
    "QRS": {"aliases": ["QRS时限", "QRS"], "kind": "numeric", "unit": "ms", "range": (40, 250)},
    "QTc": {"aliases": ["QTc", "QTc间期"], "kind": "numeric", "unit": "ms", "range": (250, 700)},
    "PVC": {"aliases": ["室性早搏", "室早", "PVC"], "kind": "boolean"},
}

NEGATION = ("未见", "无", "没有", "阴性", "排除", "未发现")
UNCERTAIN = ("可疑", "考虑", "可能", "？", "?")
UNIT_FACTORS = {
    ("cm", "mm"): 10.0,
    ("m", "mm"): 1000.0,
    ("l", "ml"): 1000.0,
    ("ng/l", "pg/ml"): 1.0,
}


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "")


def _alias_pattern(alias: str) -> str:
    escaped = re.escape(alias)
    if alias.isascii():
        return rf"(?<![A-Za-z]){escaped}(?![A-Za-z])"
    return escaped


def _snippet(text: str, start: int, end: int) -> str:
    return text[max(0, start - 18): min(len(text), end + 18)]


def _window(text: str, start: int, end: int, radius: int = 14) -> str:
    return text[max(0, start - radius): min(len(text), end + radius)]


def _clause(text: str, start: int, end: int) -> str:
    """Return the punctuation-delimited clause containing [start, end)."""
    left = 0
    for mark in "。；;，,\n":
        position = text.rfind(mark, 0, start)
        if position >= 0:
            left = max(left, position + 1)
    right = len(text)
    for mark in "。；;，,\n":
        position = text.find(mark, end)
        if position >= 0:
            right = min(right, position)
    return text[left:right]


def _negation_status(window: str) -> str:
    if any(word in window for word in NEGATION):
        return "absent"
    if any(word in window for word in UNCERTAIN):
        return "uncertain"
    return "present"


def _numeric(text: str, variable: str, spec: dict) -> dict | None:
    aliases = sorted(spec["aliases"], key=len, reverse=True)
    for alias in aliases:
        pattern = re.compile(
            _alias_pattern(alias)
            + r"[^0-9<>]{0,14}([<>]?\d+(?:\.\d+)?)\s*([A-Za-z%/]+|次/分|毫秒|毫米|厘米|毫升|岁)?"
        )
        match = pattern.search(text)
        if not match:
            continue
        raw = match.group(1)
        unit = (match.group(2) or "").lower()
        target = spec.get("unit", "")
        try:
            value = float(raw.lstrip("<>"))
        except ValueError:
            continue
        detected_unit = unit or target.lower()
        factor = UNIT_FACTORS.get((detected_unit, target.lower()), 1.0)
        value *= factor
        low, high = spec.get("range", (float("-inf"), float("inf")))
        status = "present" if low <= value <= high else "out_of_range"
        return {
            "variable": variable,
            "value": round(value, 3),
            "unit": target,
            "status": status,
            "confidence": 0.95 if status == "present" else 0.5,
            "snippet": _snippet(text, match.start(), match.end()),
        }
    return None


def _boolean(text: str, variable: str, spec: dict) -> dict | None:
    aliases = sorted(spec["aliases"], key=len, reverse=True)
    for alias in aliases:
        for match in re.finditer(_alias_pattern(alias), text):
            window = _clause(text, match.start(), match.end())
            status = _negation_status(window)
            value = None if status == "uncertain" else status == "present"
            return {
                "variable": variable,
                "value": value,
                "unit": "",
                "status": status,
                "confidence": 0.9 if status in ("present", "absent") else 0.6,
                "snippet": _snippet(text, match.start(), match.end()),
            }
    return None


def _categorical(text: str, variable: str, spec: dict) -> dict | None:
    aliases = sorted(spec["aliases"], key=len, reverse=True)
    for alias in aliases:
        for match in re.finditer(_alias_pattern(alias), text):
            window = _clause(text, match.start(), match.end())
            found = [value for value in spec["values"] if value in window]
            if found:
                status = _negation_status(window)
                value = None if status == "uncertain" else (found[0] if status == "present" else None)
                return {
                    "variable": variable,
                    "value": value,
                    "unit": "",
                    "status": status if status != "absent" else "absent",
                    "confidence": 0.8,
                    "snippet": _snippet(text, match.start(), match.end()),
                }
    return None


def extract_report(text: str, patient_id: str = "", report_time: str = "",
                   source: str = "") -> list[dict]:
    """Extract every dictionary variable from one report (missing = explicit)."""
    text = _normalize(text)
    records: list[dict] = []
    for variable, spec in VARIABLES.items():
        if spec["kind"] == "numeric":
            found = _numeric(text, variable, spec)
        elif spec["kind"] == "boolean":
            found = _boolean(text, variable, spec)
        else:
            found = _categorical(text, variable, spec)
        if found is None:
            found = {
                "variable": variable,
                "value": None,
                "unit": spec.get("unit", ""),
                "status": "missing",
                "confidence": 0.0,
                "snippet": "",
            }
        found.update({
            "patient_id": patient_id,
            "time": report_time,
            "source": source,
            "kind": spec["kind"],
        })
        records.append(found)
    return records


def to_wide_csv(records: list[dict]) -> str:
    """Build a patient x variable CSV (values only; blanks mean missing)."""
    patients = sorted({record["patient_id"] for record in records})
    variables = list(VARIABLES)
    lines = ["patient_id," + ",".join(variables)]
    for patient in patients:
        row = {record["variable"]: record for record in records if record["patient_id"] == patient}
        cells = []
        for variable in variables:
            value = row.get(variable, {}).get("value")
            cells.append("" if value is None else str(value))
        lines.append(patient + "," + ",".join(cells))
    return "\n".join(lines)


__all__ = ["VARIABLES", "extract_report", "to_wide_csv"]
