# Public cardiovascular datasets reachable from mainland China

This note is **specific to mainland-China network access** (verified 2026-11-13
from a Chinese university cluster) and to the project topic:

> Heart failure / cardiomyopathy · malignant ventricular arrhythmia ·
> multimodal precursor signals · RAG-LLM clinical warning.

> **No hospital data is needed for the work in this note.** These datasets are
> for method development while the clinical cohort awaits ethics approval.

## 1. The PhysioNet access rule you must know

PhysioNet announced (2025-07-15) that, under the U.S. DOJ Data Security Program
(Executive Order 14117), it **blocks controlled-access (credentialed) datasets**
for users connecting from IP addresses or affiliations in China (including Hong
Kong and Macau), Cuba, Iran, North Korea, Russia and Venezuela.

**What this means in practice (verified from a Chinese IP):**

| PhysioNet dataset type | Access from mainland China | Evidence |
| --- | --- | --- |
| **Open-access** datasets (no login) | ✅ still downloadable | `PTB-XL`, `MIT-BIH`, `VFDB`, `CUDB`, `SDDB` pages return 200 and files download |
| **Credentialed** datasets (CITI + DUA) | ❌ blocked | `MIMIC-IV` data file returns **HTTP 403** |

So the common claim "PhysioNet is inaccessible from China" is **half true**:
open data is fine; credentialed data (MIMIC-IV, MIMIC-IV-ECG, MIMIC-CXR, eICU,
HiRID, …) is not.

**Do not** ask an overseas collaborator to download MIMIC and send it to you:
the DUA forbids redistribution and the DOJ rule targets exactly that transfer.

## 2. Datasets that *are* reachable and relevant

### 2.1 ECG / ventricular arrhythmia (open PhysioNet, verified downloadable)

| Dataset | Content | Size | Why it matters |
| --- | --- | --- | --- |
| [MIT-BIH Arrhythmia](https://physionet.org/content/mitdb/) | 2-lead Holter + beat annotations | 48 × 30 min | PVC / VT episodes; classic benchmark |
| [MIT-BIH Malignant Ventricular Ectopy (VFDB)](https://physionet.org/content/vfdb/) | 2-lead Holter, VF/VT | 22 patients | **closest public data to malignant ventricular arrhythmia** |
| [CU Ventricular Tachyarrhythmia (CUDB)](https://physionet.org/content/cudb/) | 2-lead Holter, VT/VF | 35 records | VT/VF onset and precursors |
| [Sudden Cardiac Death Holter (SDDB)](https://physionet.org/content/sddb/) | 2-lead Holter, SCD | 23 patients | terminal arrhythmia trajectory |
| [PTB-XL](https://physionet.org/content/ptb-xl/) | 12-lead ECG + diagnostic statements + metadata | 21,799 ECGs / 18,869 patients | large multi-label ECG benchmark; metadata/report text useful for RAG |
| [Chapman-Shaoxing](https://physionet.org/content/ecg-arrhythmia/) | 12-lead ECG, 11 rhythms | 10,646 ECGs | Chinese hospital ECG |
| [Ningbo (Challenge 2020 bundle)](https://physionet.org/content/challenge-2020/) | 12-lead ECG | ~34k ECGs | Chinese hospital ECG |

### 2.2 ECG from Chinese challenge platforms (fast in China)

| Dataset | Content | Access |
| --- | --- | --- |
| [CPSC2018](http://2018.icbeb.org/) | 12-lead ECG, 9 classes (~6,877 ECGs) | ICBEB registration / Tianchi |
| [CPSC2020](http://2020.icbeb.org/) | 12-lead ECG, PVC/SPB detection | Tianchi |
| [CPSC2021](http://2021.icbeb.org/) | single-lead ECG, paroxysmal AF | Tianchi |

If the old ICBEB pages are down, search **"CPSC 2020 天池"** / **"CPSC 2018
天池"**; the competition data pages are usually still available after the
contest ends.

### 2.3 Echocardiography and heart failure

| Dataset | Content | Access from China |
| --- | --- | --- |
| [CAMUS](https://www.creatis.insa-lyon.fr/Challenge/camus/) | 2D echo, LV segmentation + EF, 500 patients | CREATIS, research registration — page reachable |
| [EchoNet-Dynamic](https://echonet.github.io/dynamic/) | apical-4-chamber echo videos + EF, 10,030 videos | Stanford AIMI registration/DUA — page reachable; registration may rely on Google services, test first |

### 2.4 Cardiomyopathy / cardiac MRI

| Dataset | Content | Access from China |
| --- | --- | --- |
| [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/) | cine CMR, DCM/HCM/ARV/MINP, 100 patients | CREATIS — page reachable, direct research download |
| [M&Ms](https://www.creatis.insa-lyon.fr/Challenge/mms/) | multi-centre cine CMR, 345 patients | CREATIS — **check current URL** (old path returned 404) |
| [EMIDEC](https://www.creatis.insa-lyon.fr/Challenge/EMIDEC/) | CMR myocarditis, 100 patients | CREATIS — **check current URL** |
| MM-WHS / MyoPS / LAScarQS | Chinese whole-heart / myocardial pathology / left-atrium challenges | Chinese challenge sites; small but relevant |

### 2.5 What is **not** usable from mainland China

| Dataset | Reason |
| --- | --- |
| MIMIC-IV, MIMIC-IV-ECG, MIMIC-CXR | credentialed PhysioNet dataset; file download returns **403** |
| eICU-CRD, HiRID | credentialed; blocked |
| UK Biobank | application/fee; not a 4-week option |
| CODE-15% (Brazil, Zenodo) | Zenodo reachability is unreliable from China; test before planning |
| AmsterdamUMCdb | application process, not a quick option; check current access policy |

## 3. Recommended accessible portfolio for this project

| Project component | Use these public data | Notes |
| --- | --- | --- |
| VT/VF precursor on ECG | **VFDB + CUDB + SDDB + MIT-BIH** | the most direct public analogue of malignant VA |
| General ECG representation | **PTB-XL + CPSC2020/2018 + Chapman/Ningbo** | Chinese + German populations; PTB-XL has report text for RAG |
| Cardiomyopathy imaging | **ACDC** (+ M&Ms/EMIDEC if URLs work) | DCM/HCM/ARV/MINP; CMR |
| HF / EF | **CAMUS** (+ EchoNet if registration works) | echo segmentation / EF |
| Multimodal EHR + ECG | **no public substitute for MIMIC** | use PTB-XL/Chapman metadata as a weak proxy; the real multimodal corpus must be your hospital cohort |

**Bottom line:** plan the public-data pilot around **ECG + CMR/echo**, and keep
the **EHR+ECG multimodal part** for your hospital data (or a European
collaboration that is not covered by the U.S. rule).

## 4. Access tips (China-specific)

- **Open PhysioNet files** download directly from `https://physionet.org/files/...`;
  use `wget -c` / aria2 and verify checksums.
- **CPSC / Tianchi**: register with a Chinese phone; download speed is good.
- **CREATIS (ACDC/CAMUS)**: research-use registration; pages reachable.
- **EchoNet-Dynamic**: registration may use Google Forms; if it fails, email the
  maintainers with an institutional address.
- **Zenodo / Kaggle / Figshare**: availability varies; test one small file
  before planning a large download.
- **Never use unofficial mirrors** of MIMIC or other credentialed data.

## 5. Suggested 4-week plan (no hospital data)

| Week | Work | Data |
| --- | --- | --- |
| 1 | VT/VF precursor feature extraction + label definition | VFDB, CUDB, SDDB, MIT-BIH, PTB-XL |
| 2 | Cardiomyopathy / EF image branch | ACDC, CAMUS |
| 3 | Chinese ECG generalisation | CPSC2020/2018, Chapman-Shaoxing, Ningbo |
| 4 | RAG corpus + synthetic multimodal pilot; align with hospital CRF | PTB-XL report text + own synthetic data |

## 6. Compliance reminders

- Each dataset has its own licence/DUA; **do not redistribute** raw data.
- Report dataset version, access date and any preprocessing in the paper.
- Keep public-data results separate from the hospital cohort.
- MIMIC/eICU are **not** available to you; do not attempt to obtain them via
  third parties.
