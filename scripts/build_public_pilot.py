"""Build the public ECG pilot corpus from VFDB + PTB-XL.

VFDB (open PhysioNet) has rhythm-change annotations:
``(N``, ``(VT``, ``(VFL``, ``(VF``, ``(VFIB``, ``(NOISE`` ...
We cut 5-second windows, extract simple ECG features, and label each window:

* ``pre``      — window ends within 60 s before a VT/VF onset;
* ``va``       — window overlaps a VT/VF/VFL/VFIB segment;
* ``baseline`` — normal rhythm and no VT/VF onset within 120 s;
* ``other``    — discarded.

The output is the same JSONL schema used by ``scripts/build_hypergraph.py``.

Usage:
    PYTHONPATH=/path/to/pylibs python scripts/build_public_pilot.py \
        --data-root data/public_pilot --out-root data/public_pilot
"""

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.stats import kurtosis

MALIGNANT = {"VT", "VFL", "VF", "VFIB"}
NORMAL = {"N", "NSR", "BI", "B"}


def _rhythm_timeline(ann) -> list[tuple[int, str]]:
    timeline = []
    for sample, note in zip(ann.sample, ann.aux_note):
        label = str(note).strip("\x00").lstrip("(").strip()
        if label:
            timeline.append((int(sample), label))
    return timeline


def _segments(timeline: list[tuple[int, str]], sig_len: int) -> list[tuple[int, int, str]]:
    segments = []
    for index, (sample, label) in enumerate(timeline):
        end = timeline[index + 1][0] if index + 1 < len(timeline) else sig_len
        if end > sample:
            segments.append((sample, end, label))
    return segments


def _features(signal: np.ndarray, fs: float) -> dict:
    signal = np.asarray(signal, dtype=float)
    signal = signal - np.nanmean(signal)
    if not np.all(np.isfinite(signal)):
        signal = np.nan_to_num(signal)

    low, high = 5.0, 15.0
    b, a = butter(3, [low / (fs / 2), high / (fs / 2)], btype="band")
    filtered = filtfilt(b, a, signal)
    peaks, _ = find_peaks(
        filtered,
        distance=max(1, int(0.25 * fs)),
        prominence=max(1e-6, float(np.std(filtered)) * 0.5),
    )
    rr = np.diff(peaks) / fs if len(peaks) > 1 else np.array([])

    hr = float(60.0 / np.mean(rr)) if len(rr) else 0.0
    rr_sd = float(np.std(rr) * 1000.0) if len(rr) else 0.0
    rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000.0) if len(rr) > 1 else 0.0
    pnn50 = float(np.mean(np.abs(np.diff(rr)) > 0.05)) if len(rr) > 1 else 0.0

    frequencies, power = welch(signal, fs=fs, nperseg=min(len(signal), 1024))
    band = (frequencies >= 1.0) & (frequencies <= 45.0)
    total = float(np.sum(power[band])) or 1e-12
    hf = float(np.sum(power[(frequencies >= 20.0) & (frequencies <= 45.0)]))
    hf_ratio = hf / total
    probabilities = power[band] / total
    entropy = float(-np.sum(probabilities * np.log2(probabilities + 1e-12)))

    return {
        "hr": hr,
        "rr_sd": rr_sd,
        "rmssd": rmssd,
        "pnn50": pnn50,
        "hf_ratio": hf_ratio,
        "entropy": entropy,
        "amp_std": float(np.std(signal)),
        "kurtosis": float(kurtosis(signal)),
    }


def _label_window(start: int, end: int, segments, onsets, fs: float) -> str:
    for seg_start, seg_end, label in segments:
        if label in MALIGNANT and start < seg_end and end > seg_start:
            return "va"
    for onset in onsets:
        if 0 <= onset - end <= 60 * fs:
            return "pre"
    in_normal = any(
        seg_start <= start and end <= seg_end and label in NORMAL
        for seg_start, seg_end, label in segments
    )
    near_event = any(abs(onset - (start + end) / 2) <= 120 * fs for onset in onsets)
    if in_normal and not near_event:
        return "baseline"
    return "other"


def build_vfdb(vfdb_root: Path, window_s: float = 5.0, stride_s: float = 2.5,
               per_class: int = 120, seed: int = 0) -> tuple[list[dict], list[dict]]:
    sys.path.insert(0, "/path/to/pylibs")
    import wfdb  # noqa: PLC0415

    records = sorted({path.stem for path in vfdb_root.glob("*.hea")})
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {"pre": [], "baseline": [], "va": []}

    for name in records:
        record = wfdb.rdrecord(str(vfdb_root / name))
        ann = wfdb.rdann(str(vfdb_root / name), "atr")
        fs = float(record.fs)
        signal = record.p_signal[:, 0]
        timeline = _rhythm_timeline(ann)
        segments = _segments(timeline, record.sig_len)
        onsets = [sample for sample, label in timeline if label in MALIGNANT]

        window = int(window_s * fs)
        stride = int(stride_s * fs)
        for start in range(0, record.sig_len - window + 1, stride):
            end = start + window
            label = _label_window(start, end, segments, onsets, fs)
            if label not in by_label:
                continue
            feats = _features(signal[start:end], fs)
            rhythm = next(
                (label_ for seg_start, seg_end, label_ in segments if seg_start <= start < seg_end),
                "NA",
            )
            distance = min((abs(onset - end) for onset in onsets), default=None)
            text = (
                f"VFDB 记录 {name}，窗口 {start / fs:.1f}-{end / fs:.1f} 秒："
                f"节律 {rhythm}，心率 {feats['hr']:.0f} 次/分，"
                f"RR 标准差 {feats['rr_sd']:.0f} 毫秒，RMSSD {feats['rmssd']:.0f} 毫秒，"
                f"PNN50 {feats['pnn50']:.2f}，高频能量比 {feats['hf_ratio']:.2f}，"
                f"频谱熵 {feats['entropy']:.2f}，振幅标准差 {feats['amp_std']:.3f}，"
                f"峰度 {feats['kurtosis']:.2f}，标签 {label}。"
            )
            by_label[label].append(
                {
                    "doc_id": f"vfdb-{name}-{start:07d}",
                    "text": text,
                    "images": [],
                    "tables": [],
                    "meta": {
                        "dataset": "vfdb",
                        "record": name,
                        "start_s": start / fs,
                        "end_s": end / fs,
                        "rhythm": rhythm,
                        "label": label,
                        "onset_distance_s": None if distance is None else distance / fs,
                        **feats,
                    },
                }
            )

    selected = []
    for label, docs in by_label.items():
        rng.shuffle(docs)
        selected.extend(docs[:per_class])
    selected.sort(key=lambda doc: doc["doc_id"])
    return selected, []


def build_ptbxl(ptbxl_root: Path, max_records: int = 300) -> list[dict]:
    metadata_path = ptbxl_root / "ptbxl_database.csv"
    if not metadata_path.exists():
        return []
    docs = []
    with metadata_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            codes = row.get("scp_codes") or ""
            if not codes:
                continue
            report = (row.get("report") or "").replace("\n", " ").strip()
            text = (
                f"PTB-XL 心电图记录 {row['ecg_id']}：患者 {row.get('age', '?')} 岁，"
                f"性别 {row.get('sex', '?')}，诊断代码 {codes}。"
            )
            if report:
                text += f"报告摘要：{report[:160]}。"
            docs.append(
                {
                    "doc_id": f"ptbxl-{row['ecg_id']}",
                    "text": text,
                    "images": [],
                    "tables": [],
                    "meta": {
                        "dataset": "ptbxl",
                        "ecg_id": row["ecg_id"],
                        "patient_id": row.get("patient_id", ""),
                        "age": row.get("age", ""),
                        "sex": row.get("sex", ""),
                        "scp_codes": codes,
                    },
                }
            )
            if len(docs) >= max_records:
                break
    return docs


def build_qa(docs: list[dict], limit: int = 60, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    qa = []
    vfdb_docs = [doc for doc in docs if doc["meta"].get("dataset") == "vfdb"]
    ptbxl_docs = [doc for doc in docs if doc["meta"].get("dataset") == "ptbxl"]
    rng.shuffle(vfdb_docs)
    rng.shuffle(ptbxl_docs)

    for doc in vfdb_docs[: limit // 2]:
        meta = doc["meta"]
        question = (
            f"VFDB 记录 {meta['record']} 在 {meta['start_s']:.1f}-{meta['end_s']:.1f} 秒"
            "窗口的标签是什么？"
        )
        answer = meta["label"]
        qa.append({"qid": f"q-{doc['doc_id']}", "question": question, "answer": answer,
                   "has_image": False, "regime": "k_only"})

    for doc in ptbxl_docs[: limit // 2]:
        meta = doc["meta"]
        codes = str(meta["scp_codes"])
        import re

        match = re.search(r"[A-Z][A-Z0-9_]+", codes)
        first_code = match.group(0) if match else codes
        qa.append(
            {
                "qid": f"q-{doc['doc_id']}",
                "question": f"PTB-XL 记录 {meta['ecg_id']} 的第一个诊断代码是什么？",
                "answer": first_code,
                "has_image": False,
                "regime": "k_only",
            }
        )
    return qa


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/public_pilot")
    parser.add_argument("--out-root", default="data/public_pilot")
    parser.add_argument("--per-class", type=int, default=120)
    parser.add_argument("--max-ptbxl", type=int, default=300)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    vfdb_docs, _ = build_vfdb(data_root / "vfdb", per_class=args.per_class)
    ptbxl_docs = build_ptbxl(data_root / "ptbxl", max_records=args.max_ptbxl)
    docs = vfdb_docs + ptbxl_docs
    qa = build_qa(docs)

    (out_root / "corpus.jsonl").write_text(
        "\n".join(json.dumps(doc, ensure_ascii=False) for doc in docs), encoding="utf-8"
    )
    (out_root / "qa.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in qa), encoding="utf-8"
    )
    features = [
        {"doc_id": doc["doc_id"], **{k: v for k, v in doc["meta"].items() if k != "dataset"}}
        for doc in vfdb_docs
    ]
    (out_root / "features.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in features), encoding="utf-8"
    )
    summary = {
        "vfdb_windows": len(vfdb_docs),
        "vfdb_label_counts": {
            label: sum(1 for doc in vfdb_docs if doc["meta"]["label"] == label)
            for label in ("pre", "va", "baseline")
        },
        "ptbxl_docs": len(ptbxl_docs),
        "qa": len(qa),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
