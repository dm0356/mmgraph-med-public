# Public ECG pilot (no hospital data)

Goal: validate the full pipeline — ECG feature extraction → text corpus →
hypergraph → memory retrieval — **without any patient data**, while the hospital
cohort awaits ethics approval.

## 1. Data

| Dataset | Access | Content |
| --- | --- | --- |
| [VFDB](https://physionet.org/content/vfdb/) | PhysioNet **open-access** (verified reachable from China) | 22 Holter records with rhythm-change annotations: `(N`, `(VT`, `(VFL`, `(VF`, `(VFIB)` |
| [PTB-XL](https://physionet.org/content/ptb-xl/) | PhysioNet open-access (metadata only in this pilot) | 21,799 12-lead ECGs with diagnostic codes and metadata |

> MIMIC-IV / eICU / HiRID are **credentialed** PhysioNet datasets and return
> HTTP 403 from mainland China; they are not used here.

## 2. Windowing and labels (VFDB)

* 5-second windows, 2.5-second stride, lead 0, 250 Hz.
* Rhythm timeline from `aux_note` (`N`, `VT`, `VFL`, `VF`, `VFIB`, `NOISE`, …).
* Labels:
  - `pre` — window ends within **60 s before** a VT/VF onset;
  - `va` — window overlaps a VT/VF/VFL/VFIB segment;
  - `baseline` — normal rhythm with no VT/VF onset within 120 s;
  - other windows are discarded.

## 3. Features

Per window: heart rate, RR standard deviation, RMSSD, PNN50, high-frequency
energy ratio (20–45 Hz), spectral entropy, amplitude standard deviation, kurtosis.

Each window becomes a Chinese text document, e.g.:

```text
VFDB 记录 424，窗口 650.0-655.0 秒：节律 N，心率 148 次/分，
RR 标准差 79 毫秒，RMSSD 94 毫秒，…，标签 baseline。
```

PTB-XL metadata rows become text documents with age/sex/diagnostic codes.

## 4. Commands (cluster)

```bash
# 1) download VFDB (+ optional PTB-XL waveforms)
PYTHONPATH=/path/to/pylibs python scripts/fetch_public_data.py \
    --data-root /path/to/public_pilot --ptbxl-signals 0

# 2) build corpus / QA / features
PYTHONPATH=/path/to/pylibs python scripts/build_public_pilot.py \
    --data-root /path/to/public_pilot \
    --out-root /path/to/public_pilot/pilot \
    --per-class 120 --max-ptbxl 300

# 3) build hypergraph with local BGE-M3 (server on :18000)
PYTHONPATH=src python scripts/build_hypergraph.py \
    --config configs/public_pilot_bge.toml \
    --data-root /path/to/public_pilot/pilot \
    --out /path/to/public_pilot/pilot/mmkg_public_pilot_bge_v2.json

# 4) retrieval evaluation
PYTHONPATH=src python scripts/eval_retrieval.py \
    --config configs/public_pilot_bge.toml \
    --store /path/to/public_pilot/pilot/mmkg_public_pilot_bge_v2.json \
    --data-root /path/to/public_pilot/pilot \
    --out /path/to/public_pilot/pilot/retrieval_eval_bge_v2.json
```

## 5. Current result (engineering validation only)

Corpus: **360 VFDB windows** (`pre`/`va`/`baseline` = 120 each) + **300 PTB-XL
documents** + **60 QA**.

| Config | answer-in-context recall |
| --- | --- |
| hashing embedder | 0.033 |
| **BGE-M3 embedder** | **0.933** (VFDB 0.867, PTB-XL 1.000) |

This is a retrieval smoke test, **not** an arrhythmia-prediction performance
metric. No AUROC/sensitivity/lead-time has been computed yet.

## 6. Bug found and fixed

The sentence splitter treated the decimal point as a sentence boundary, so
`650.0-655.0` was cut into `650.` and `0-655.` and one window became several
units. Fixed in `utils.split_sentences`: Chinese punctuation always splits;
English `. ! ?` split only when followed by whitespace. This also protects
clinical values such as `8.74 mmol/L`.

## 7. Next

- Add a real precursor baseline on VFDB: window-level classifier + AUROC, sensitivity at fixed lead time.
- Add CPSC2020 (PVC) and ACDC/CAMUS (cardiomyopathy/EF) branches.
- Replace the regex extractor with the local LLM extractor for richer entities.
- Keep public-data experiments separate from the hospital cohort in the paper.
