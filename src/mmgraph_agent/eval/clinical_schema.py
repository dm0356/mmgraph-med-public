"""Clinical case schema: patient x time-window, labels and RAG document conversion.

See docs/CLINICAL_SCHEMA.md. This module is dependency-light (stdlib only) so it
can validate exports before any model runs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

EVENTS = {"none", "VT", "VF", "SCD", "ICD_shock"}
MALIGNANT = {"VT", "VF", "SCD", "ICD_shock"}


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def validate_case(case: dict) -> list[str]:
    """Return a list of human-readable validation errors (empty = valid)."""
    errors: list[str] = []
    for field in ("case_id", "patient_id", "window", "labels"):
        if not case.get(field):
            errors.append(f"missing {field}")
    if errors:
        return errors

    window = case.get("window") or {}
    start = _parse_time(window.get("start"))
    end = _parse_time(window.get("end"))
    if start is None or end is None:
        errors.append("window.start/end must be ISO 8601")
    elif end <= start:
        errors.append("window.end must be after window.start")

    labels = case.get("labels") or {}
    event = labels.get("event")
    if event not in EVENTS:
        errors.append(f"labels.event must be one of {sorted(EVENTS)}")
    if event in MALIGNANT and _parse_time(labels.get("event_time")) is None:
        errors.append("labels.event_time is required for a malignant event")
    horizon = labels.get("outcome_horizon_hours")
    if not isinstance(horizon, (int, float)) or horizon <= 0:
        errors.append("labels.outcome_horizon_hours must be a positive number")

    if case.get("deidentified") is not True:
        errors.append("case must be deidentified=true before entering the pipeline")

    demographics = case.get("demographics") or {}
    if not isinstance(demographics.get("age"), (int, float)):
        errors.append("demographics.age must be numeric")
    if demographics.get("sex") not in ("M", "F"):
        errors.append("demographics.sex must be M or F")

    for list_field in ("diagnoses", "labs", "medications", "ecg", "echo", "cmr", "notes"):
        value = case.get(list_field, [])
        if value is not None and not isinstance(value, list):
            errors.append(f"{list_field} must be a list")
    return errors


def window_label(case: dict) -> int | None:
    """Return 1 (pre-event), 0 (negative) or None (exclude).

    Positive: event occurs within ``outcome_horizon_hours`` after the window end.
    Excluded: event already occurred inside the window, or follow-up too short.
    """
    window = case.get("window") or {}
    end = _parse_time(window.get("end"))
    labels = case.get("labels") or {}
    horizon = float(labels.get("outcome_horizon_hours", 48))
    event = labels.get("event")
    if end is None:
        return None
    if event == "none":
        return 0
    event_time = _parse_time(labels.get("event_time"))
    if event_time is None:
        return None
    delta_hours = (event_time - end).total_seconds() / 3600.0
    if delta_hours < 0:
        return None
    return 1 if delta_hours <= horizon else 0


def _fmt(value: Any) -> str:
    return "" if value is None else str(value)


def case_to_documents(case: dict) -> list[dict]:
    """Convert one clinical case into RAG documents (same schema as corpus.jsonl)."""
    docs: list[dict] = []
    patient = case.get("patient_id", "?")
    window = case.get("window") or {}
    span = f"{_fmt(window.get('start'))} 至 {_fmt(window.get('end'))}"
    label = window_label(case)
    label_text = {1: "阳性（前驱窗）", 0: "阴性", None: "排除"}[label]

    demographics = case.get("demographics") or {}
    diagnoses = case.get("diagnoses") or []
    diag_text = "；".join(f"{d.get('code', '')} {d.get('name', '')}".strip() for d in diagnoses) or "无记录"
    docs.append({
        "doc_id": f"{case['case_id']}-diagnosis",
        "text": (
            f"患者 {patient}，{_fmt(demographics.get('age'))} 岁，"
            f"性别 {_fmt(demographics.get('sex'))}，时间窗 {span}。"
            f"诊断：{diag_text}。标签：{label_text}。"
        ),
        "images": [],
        "tables": [],
        "meta": {"case_id": case["case_id"], "patient_id": patient, "modality": "diagnosis"},
    })

    labs = case.get("labs") or []
    if labs:
        parts = [
            f"{_fmt(item.get('time'))} {_fmt(item.get('name'))} {_fmt(item.get('value'))} {_fmt(item.get('unit'))}"
            for item in labs
        ]
        docs.append({
            "doc_id": f"{case['case_id']}-labs",
            "text": f"患者 {patient} 时间窗 {span} 检验：{'；'.join(parts)}。",
            "images": [],
            "tables": [],
            "meta": {"case_id": case["case_id"], "patient_id": patient, "modality": "labs"},
        })

    medications = case.get("medications") or []
    if medications:
        parts = [
            f"{_fmt(item.get('name'))} {_fmt(item.get('dose'))} {_fmt(item.get('route'))}"
            for item in medications
        ]
        docs.append({
            "doc_id": f"{case['case_id']}-medications",
            "text": f"患者 {patient} 时间窗 {span} 用药：{'；'.join(parts)}。",
            "images": [],
            "tables": [],
            "meta": {"case_id": case["case_id"], "patient_id": patient, "modality": "medications"},
        })

    ecg = case.get("ecg") or []
    for index, item in enumerate(ecg):
        text = (
            f"患者 {patient} 时间窗 {span} 心电图 {_fmt(item.get('time'))}："
            f"心率 {_fmt(item.get('hr'))} 次/分，QRS {_fmt(item.get('qrs_ms'))} ms，"
            f"QTc {_fmt(item.get('qtc_ms'))} ms，节律 {_fmt(item.get('rhythm'))}。"
        )
        images = []
        if item.get("path"):
            images.append({"image_id": f"{case['case_id']}-ecg-{index}", "caption": "12 导联心电图", "path": item["path"]})
        docs.append({
            "doc_id": f"{case['case_id']}-ecg-{index}",
            "text": text,
            "images": images,
            "tables": [],
            "meta": {"case_id": case["case_id"], "patient_id": patient, "modality": "ecg"},
        })

    echo = case.get("echo") or []
    for index, item in enumerate(echo):
        text = (
            f"患者 {patient} 时间窗 {span} 超声心动图 {_fmt(item.get('time'))}："
            f"LVEF {_fmt(item.get('lvef'))}%，{_fmt(item.get('notes'))}。"
        )
        docs.append({
            "doc_id": f"{case['case_id']}-echo-{index}",
            "text": text,
            "images": [],
            "tables": [],
            "meta": {"case_id": case["case_id"], "patient_id": patient, "modality": "echo"},
        })

    for index, item in enumerate(case.get("notes") or []):
        docs.append({
            "doc_id": f"{case['case_id']}-note-{index}",
            "text": f"患者 {patient} 时间窗 {span} 病程记录：{_fmt(item.get('text'))}。",
            "images": [],
            "tables": [],
            "meta": {"case_id": case["case_id"], "patient_id": patient, "modality": "note"},
        })
    return docs


def case_to_qa(case: dict) -> dict:
    window = case.get("window") or {}
    horizon = (case.get("labels") or {}).get("outcome_horizon_hours", 48)
    label = window_label(case)
    return {
        "qid": f"{case['case_id']}-risk",
        "question": (
            f"患者 {case['patient_id']} 在 {_fmt(window.get('start'))} 至 {_fmt(window.get('end'))} "
            f"时间窗，未来 {horizon} 小时内是否会发生恶性室性心律失常？"
        ),
        "answer": "是" if label == 1 else "否",
        "has_image": False,
        "regime": "k_only",
        "meta": {"case_id": case["case_id"], "patient_id": case["patient_id"], "label": label},
    }


def cases_to_corpus(cases: list[dict]) -> tuple[list[dict], list[dict]]:
    docs, qa = [], []
    for case in cases:
        docs.extend(case_to_documents(case))
        qa.append(case_to_qa(case))
    return docs, qa
