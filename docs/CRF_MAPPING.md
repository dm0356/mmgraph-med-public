# CRF 变量表与映射（MAARS 对齐草案 v0.1）

> 目标：让 **数据导出、报告抽取、风险模型、RAG 证据** 共用同一套变量定义。
> 文件：[`configs/crf_variables.json`](../configs/crf_variables.json)

## 1. 为什么需要 CRF

- 没有统一定义：HIS、LIS、PACS、超声/ECG 报告各说各话，变量名/单位/缺失规则不一致；
- 有 CRF：抽取、建模、统计、审稿都用同一张表，可版本化、可核对、可复现；
- CRF 是 **数据字典 + 采集表 + 映射规则** 三合一。

## 2. 当前规模（v0.1-draft，57 个变量）

| 类别 | 数量 | 说明 |
| --- | --- | --- |
| META | 3 | 患者/就诊编号、数据划分 |
| EHR | 25 | 人口学、病史、用药、检验 |
| CIR | 16 | 超声/CMR 报告变量 |
| ECG | 5 | 心率、QRS、QTc、节律、室早 |
| CMR | 1 | 是否有 LGE-CMR 原始图像 |
| OUTCOME | 7 | SCDA 事件、时间窗、随访 |

> 这是 MAARS 对齐的**草案**，需要心内科/统计老师最终确认后冻结。

## 3. 每个变量的字段

```json
{
  "name": "LVEF",
  "label_cn": "左室射血分数",
  "category": "CIR",
  "type": "numeric",
  "unit": "%",
  "range": [5, 90],
  "required": true,
  "source": "report",
  "maars_branch": "CIR",
  "schema_path": "echo[].lvef",
  "extraction_variable": "LVEF",
  "missing_rule": "optional"
}
```

| 字段 | 含义 |
| --- | --- |
| `source` | `structured`（HIS/LIS）、`report`（报告文本抽取）、`image`（影像文件） |
| `schema_path` | 对应 `CLINICAL_SCHEMA` 的字段 |
| `extraction_variable` | 对应 `clinical_features.VARIABLES` 的变量名（必须存在） |
| `maars_branch` | EHR / CIR / CMR，对应 MAARS 三个分支 |
| `missing_rule` | `required` / `optional` / `explicit_missing` / `conditional` |

## 4. 映射链

```text
HIS/LIS 结构化 ──► case JSON ──► CRF 行（demographics/labs/diagnoses/meds）
报告文本 ──► clinical_features 抽取 ──► CRF 行（LVEF、LGE、SAM…）
CMR 图像 ──► CLIP/Chinese-CLIP 向量 + LGE_image_available
事件标签 ──► labels.event/time ──► SCDA_event/event_type
```

关键映射示例：

| CRF 变量 | schema 字段 | 抽取变量 | MAARS 分支 |
| --- | --- | --- | --- |
| LVEF | `echo[].lvef` | `LVEF` | CIR |
| LGE_present | `cmr[].lge` | `LGE_present` | CMR |
| SAM | `echo[].sam` | `SAM` | CIR |
| potassium | `labs`（名称含"血钾"） | `potassium` | EHR |
| SCDA_event | `labels.event` | — | OUTCOME |

## 5. 导出

```bash
python scripts/export_crf.py \
    --crf configs/crf_variables.json \
    --out-dir data/crf \
    --cases data/clinical_demo/cases.jsonl
```

输出：

| 文件 | 用途 |
| --- | --- |
| `crf_variable_dictionary.csv` | 给临床团队/统计的变量字典（Excel 可打开） |
| `crf_form_template.csv` | 空白 CRF 采集表（一列一个变量） |
| `crf_mapping.json` | 机器可读映射 |
| `crf_demo_rows.csv` | 演示病例填好的 CRF 行（合成数据） |

## 6. 采集 SOP

1. **变量表定稿**：临床/统计确认变量、单位、缺失规则，版本号 +1；
2. **导出**：从 HIS/LIS/PACS 导出 case JSON，保留 `patient_id/encounter_id/window`；
3. **报告抽取**：对超声/ECG/CTA/CMR 报告做规则 + LLM 抽取，生成 `features.jsonl`；
4. **合并**：`case_to_crf_row(case, features)` 把结构化字段与抽取变量合并；
5. **缺失标注**：未查/未提及 = missing，不填 0；
6. **双人复核**：抽样核对，记录 reviewer/时间/修改原因；
7. **冻结**：CRF 版本、抽取版本、模型版本一起存档。

## 7. 数据治理

- 只处理**去标识化**数据，`deidentified=true`；
- 原始影像/报告不进入 Git；仓库只保留变量字典和合成示例；
- ICD-10 官方表为医院授权数据，不公开；
- CRF 变更需版本化，旧版本不可覆盖。

## 8. 局限与下一步

- v0.1 是草案，需临床确认（尤其 LGE 模式、NYHA、用药定义）；
- 部分变量（LGE_pattern、wall_motion、RV 功能）目前偏人工/报告描述，抽取覆盖有限；
- `schema_path` 是逻辑映射，真实 HIS 字段名需在数据接入时补齐；
- 下一步：用 CRF 宽表接 **Logistic/GBDT baseline**，再对照 MAARS 三分支多模态模型。

参见：[RISK_MODEL_ROADMAP.md](RISK_MODEL_ROADMAP.md)、[STRUCTURED_FEATURES.md](STRUCTURED_FEATURES.md)、[FEATURE_EXTRACTION.md](FEATURE_EXTRACTION.md)。
