"""CRF variable dictionary: load, validate, and map clinical cases to CRF rows.

The CRF is the single source of truth shared by:
  * data export (HIS/LIS/PACS/ECG),
  * report extraction (clinical_features),
  * risk-model feature tables,
  * RAG evidence granularity.

See docs/CRF_MAPPING.md and configs/crf_variables.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .clinical_features import VARIABLES

CATEGORIES = {"META", "EHR", "CIR", "ECG", "CMR", "OUTCOME"}
TYPES = {"string", "numeric", "boolean", "categorical", "datetime"}
SOURCES = {"structured", "report", "image"}


def load_crf(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_crf(crf: dict) -> list[str]:
    errors: list[str] = []
    variables = crf.get("variables")
    if not isinstance(variables, list) or not variables:
        return ["crf.variables must be a non-empty list"]

    names: set[str] = set()
    for index, variable in enumerate(variables):
        name = variable.get("name")
        prefix = f"variables[{index}]"
        if not name:
            errors.append(f"{prefix}: missing name")
            continue
        if name in names:
            errors.append(f"{prefix}: duplicate name {name}")
        names.add(name)
        for field in ("label_cn", "category", "type", "source"):
            if not variable.get(field):
                errors.append(f"{name}: missing {field}")
        if variable.get("category") not in CATEGORIES:
            errors.append(f"{name}: category must be one of {sorted(CATEGORIES)}")
        if variable.get("type") not in TYPES:
            errors.append(f"{name}: type must be one of {sorted(TYPES)}")
        if variable.get("source") not in SOURCES:
            errors.append(f"{name}: source must be one of {sorted(SOURCES)}")
        extraction = variable.get("extraction_variable")
        if extraction and extraction not in VARIABLES:
            errors.append(f"{name}: extraction_variable {extraction} not in clinical_features.VARIABLES")
        if variable.get("type") == "categorical" and not variable.get("allowed_values"):
            errors.append(f"{name}: categorical variable needs allowed_values")
        if variable.get("type") == "numeric" and not variable.get("unit"):
            errors.append(f"{name}: numeric variable needs unit")
    return errors


def _diagnosis_text(case: dict) -> str:
    parts = []
    for item in case.get("diagnoses") or []:
        parts.append(str(item.get("code", "")))
        parts.append(str(item.get("name", "")))
    return " ".join(parts)


def _has_diagnosis(case: dict, code_prefixes: tuple[str, ...], keywords: tuple[str, ...]) -> bool | None:
    diagnoses = case.get("diagnoses") or []
    if not diagnoses:
        return None
    text = _diagnosis_text(case)
    for item in diagnoses:
        code = str(item.get("code", ""))
        if any(code.startswith(prefix) for prefix in code_prefixes):
            return True
    if any(keyword in text for keyword in keywords):
        return True
    return False


def _lab_value(case: dict, keywords: tuple[str, ...]) -> float | None:
    for item in case.get("labs") or []:
        name = str(item.get("name", ""))
        if any(keyword in name for keyword in keywords):
            try:
                return float(item.get("value"))
            except (TypeError, ValueError):
                return None
    return None


def _ecg_values(case: dict) -> dict[str, Any]:
    rows = case.get("ecg") or []
    if not rows:
        return {}
    record = rows[0]
    mapping = {
        "HR": record.get("hr"),
        "QRS": record.get("qrs_ms"),
        "QTc": record.get("qtc_ms"),
        "rhythm": record.get("rhythm"),
        "PVC": record.get("pvc"),
    }
    return {key: value for key, value in mapping.items() if value is not None}


def _echo_values(case: dict) -> dict[str, Any]:
    rows = case.get("echo") or []
    if not rows:
        return {}
    record = rows[0]
    mapping = {
        "LVEF": record.get("lvef"),
        "IVS_thickness": record.get("ivs"),
        "LA_diameter": record.get("la"),
        "LVEDV": record.get("lvedv"),
        "LVESV": record.get("lvesv"),
        "LVOT_gradient": record.get("lvot"),
        "SAM": record.get("sam"),
        "thrombus": record.get("thrombus"),
        "pericardial_effusion": record.get("pericardial_effusion"),
        "mitral_regurgitation": record.get("mr"),
        "TR_severity": record.get("tr"),
        "wall_motion_abnormal": record.get("wall_motion"),
        "RV_dysfunction": record.get("rv"),
    }
    return {key: value for key, value in mapping.items() if value is not None}


def _cmr_values(case: dict) -> dict[str, Any]:
    rows = case.get("cmr") or []
    if not rows:
        return {}
    record = rows[0]
    mapping = {
        "LGE_present": record.get("lge"),
        "LGE_percent": record.get("lge_percent"),
        "LGE_pattern": record.get("lge_pattern"),
        "LGE_image_available": bool(record.get("path")),
    }
    return {key: value for key, value in mapping.items() if value is not None}


def _medication_flags(case: dict) -> tuple[bool, bool, bool, bool | None]:
    medications = [str(item.get("name", "")) for item in case.get("medications") or []]
    text = " ".join(medications)
    if not medications:
        return None, None, None, None
    beta = any(word in text for word in ("美托洛尔", "比索洛尔", "卡维地洛", "β"))
    amiodarone = "胺碘酮" in text
    icd = any(word in text for word in ("ICD", "CRT-D", "植入式心律转复"))
    return beta, amiodarone, icd, icd


def _nyha(case: dict) -> str | None:
    text = _diagnosis_text(case)
    match = re.search(r"(?:NYHA|心功能)\s*([IV]+)\s*级?", text)
    return match.group(1) if match else None


def case_to_crf_row(case: dict, features: list[dict] | None = None) -> dict:
    """Map one clinical case + optional extracted features to a CRF row."""
    row: dict[str, Any] = {"patient_id": case.get("patient_id"), "encounter_id": case.get("encounter_id")}
    row.update({
        "split": case.get("split"),
        "age": (case.get("demographics") or {}).get("age"),
        "sex": (case.get("demographics") or {}).get("sex"),
        "BMI": (case.get("demographics") or {}).get("bmi"),
        "window_start": (case.get("window") or {}).get("start"),
        "window_end": (case.get("window") or {}).get("end"),
    })

    row["hypertension"] = _has_diagnosis(case, ("I10", "I11", "I12", "I13", "I15"), ("高血压",))
    row["diabetes"] = _has_diagnosis(case, ("E10", "E11", "E12", "E13", "E14"), ("糖尿病",))
    row["AF"] = _has_diagnosis(case, ("I48",), ("心房颤动", "房颤"))
    row["dyslipidemia"] = _has_diagnosis(case, ("E78",), ("高脂血症", "血脂异常"))
    row["prior_MI"] = _has_diagnosis(case, ("I21", "I22", "I25.2"), ("心肌梗死",))
    row["prior_stroke"] = _has_diagnosis(case, ("I63", "I64", "I69"), ("脑梗死", "脑卒中"))
    row["NYHA"] = _nyha(case)

    row["potassium"] = _lab_value(case, ("血钾", "钾离子"))
    row["NT_proBNP"] = _lab_value(case, ("NT-proBNP", "NT-proBNP", "脑钠肽", "BNP"))
    row["eGFR"] = _lab_value(case, ("eGFR", "肾小球滤过率"))
    row["HbA1c"] = _lab_value(case, ("糖化血红蛋白", "HbA1c"))
    row["LDL"] = _lab_value(case, ("LDL-C", "低密度脂蛋白"))

    row.update(_echo_values(case))
    row.update(_ecg_values(case))
    row.update(_cmr_values(case))

    beta, amiodarone, icd, _ = _medication_flags(case)
    row["beta_blocker"] = beta
    row["amiodarone"] = amiodarone
    row["ICD_device"] = icd

    labels = case.get("labels") or {}
    event = labels.get("event")
    row["SCDA_event"] = event not in (None, "none")
    row["event_type"] = event or "none"
    row["event_time"] = labels.get("event_time")
    row["event_source"] = labels.get("event_source")
    row["follow_up_days"] = labels.get("follow_up_days")

    # merge report-extracted variables (rule/LLM), without overriding direct fields
    if features:
        for record in features:
            if record.get("status") not in ("present", "absent"):
                continue
            variable = record.get("variable")
            if variable in row and row[variable] is not None:
                continue
            if record.get("value") is not None:
                row[variable] = record.get("value")
    return row


def crf_names(crf: dict) -> list[str]:
    return [variable["name"] for variable in crf["variables"]]


def crf_dictionary_rows(crf: dict) -> list[dict]:
    rows = []
    for variable in crf["variables"]:
        rows.append({
            "name": variable["name"],
            "label_cn": variable["label_cn"],
            "category": variable["category"],
            "type": variable["type"],
            "unit": variable.get("unit", ""),
            "allowed_values": "|".join(variable.get("allowed_values", [])),
            "required": variable.get("required", False),
            "source": variable["source"],
            "maars_branch": variable.get("maars_branch", ""),
            "extraction_variable": variable.get("extraction_variable", ""),
            "schema_path": variable.get("schema_path", ""),
            "missing_rule": variable.get("missing_rule", ""),
            "notes": variable.get("notes", ""),
        })
    return rows


def write_crf_exports(crf: dict, out_dir: str | Path,
                      cases: list[dict] | None = None,
                      features_by_patient: dict[str, list[dict]] | None = None) -> dict:
    """Write the variable dictionary, blank CRF form and optional demo rows."""
    import csv

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = crf_names(crf)

    dictionary_rows = crf_dictionary_rows(crf)
    with (out / "crf_variable_dictionary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dictionary_rows[0]))
        writer.writeheader()
        writer.writerows(dictionary_rows)

    with (out / "crf_form_template.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        writer.writerow(["" for _ in names])

    mapping = {
        variable["name"]: {
            "label_cn": variable["label_cn"],
            "source": variable["source"],
            "maars_branch": variable.get("maars_branch", ""),
            "schema_path": variable.get("schema_path", ""),
            "extraction_variable": variable.get("extraction_variable", ""),
        }
        for variable in crf["variables"]
    }
    (out / "crf_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    rows_written = 0
    if cases:
        rows = []
        for case in cases:
            features = (features_by_patient or {}).get(case.get("patient_id"), [])
            mapped = case_to_crf_row(case, features=features)
            rows.append({name: mapped.get(name) for name in names})
        with (out / "crf_demo_rows.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=names)
            writer.writeheader()
            writer.writerows(rows)
        rows_written = len(rows)

    return {
        "variables": len(names),
        "categories": {
            category: sum(1 for variable in crf["variables"] if variable["category"] == category)
            for category in sorted(CATEGORIES)
        },
        "demo_rows": rows_written,
        "out_dir": str(out),
    }


__all__ = [
    "load_crf", "validate_crf", "case_to_crf_row", "crf_names",
    "crf_dictionary_rows", "write_crf_exports",
]
