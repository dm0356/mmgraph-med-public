# 结构化变量抽取原型（Feature Extraction）

对应文档：[STRUCTURED_FEATURES.md](STRUCTURED_FEATURES.md) 的工程实现。
当前版本是**规则 + 词典 + 本地 LLM 兜底 + 人工复核工具链**。

## 1. 组件

| 文件 | 作用 |
| --- | --- |
| `src/mmgraph_agent/eval/clinical_features.py` | 变量字典 + 规则抽取器 + 宽表输出 |
| `src/mmgraph_agent/eval/clinical_features_llm.py` | 本地 LLM 兜底 + 严格校验 + 规则/LLM 合并 |
| `src/mmgraph_agent/eval/clinical_review.py` | 复核表生成 + 字段级评分（状态准确率/F1/Kappa） |
| `scripts/make_report_demo.py` | 生成 5 份合成中文报告 |
| `scripts/extract_clinical_features.py` | CLI：报告 → `features.jsonl` + `features_wide.csv` |
| `scripts/make_review_sheet.py` | 生成人工复核 CSV |
| `scripts/score_extraction.py` | 复核完成后计算抽取质量指标 |
| `tests/test_clinical_features.py` / `test_feature_review.py` | 规则、单位、否定、LLM 合并、复核评分 |

## 2. 变量字典（MAARS 对齐的子集）

| 类别 | 变量 |
| --- | --- |
| 人口学/病史 | age、BMI、hypertension、diabetes、AF、NSVT、syncope、family_history_SCD、dyslipidemia |
| 超声/CMR | LVEF、IVS_thickness、LA_diameter、LVEDV、LVESV、LVOT_gradient、LGE_percent、LGE_present、SAM、thrombus、pericardial_effusion、mitral_regurgitation |
| 检验/ECG | potassium、NT_proBNP、eGFR、HbA1c、LDL、HR、QRS、QTc、PVC |

完整定义（别名、单位、合理范围）在 `clinical_features.py::VARIABLES`。

## 3. 抽取规则

1. **数值**：别名后 14 字内抓 `[<>]?数字` + 单位；
2. **单位归一**：cm→mm、m→mm、L→ml、ng/L↔pg/mL；
3. **范围校验**：超出合理范围标记 `out_of_range`（如 LVEF 120%）；
4. **否定/不确定**：按**标点分句**判断，`未见/无/阴性` → absent，`可疑/考虑/可能` → uncertain；
5. **缺失**：未提及 → `status="missing"`，**不填 0**；
6. **出处**：每条记录带 `time`、`source`、`snippet`、`confidence`。

## 4. 输出

长表 `features.jsonl`：

```json
{"variable":"LVEF","value":35.0,"unit":"%","status":"present","confidence":0.95,
 "snippet":"左室射血分数约35%","patient_id":"P0001",
 "time":"2026-01-26T10:00:00","source":"超声+CMR+检验","kind":"numeric"}
```

宽表 `features_wide.csv`：

```text
patient_id,age,BMI,...,LVEF,IVS_thickness,...,LGE_present,SAM,thrombus,...
P0001,...,35.0,12.0,...,True,,False,...
```

## 5. 演示结果（5 份合成报告）

规则 + LLM 兜底（`--use-llm`）：

| 状态 | 数量 |
| --- | --- |
| present | 27 |
| absent | 5 |
| uncertain | 1 |
| missing | 117 |

- 规则抽取 145 条，LLM 兜底补了 **5 条年龄**（规则不会从"78岁"直接抽 age）；
- LLM 没有被允许乱填：未提及的高血压/房颤保持 missing；
- 对"可疑SAM"这类不确定语义，保持 uncertain，不升级为 present。

示例：

- `左室射血分数约35%` → LVEF=35；
- `室间隔基底段厚度约1.2cm` → IVS=12mm（单位换算）；
- `LGE提示室间隔中层条片状强化` → LGE_present=true；
- `LGE未见明显强化` → LGE_present=false；
- `可疑SAM` → SAM=uncertain；
- `未见心尖部血栓` → thrombus=false；
- 未提及的 QTc → missing。

## 6. LLM 兜底的安全机制

1. 只对规则结果为 `missing`/`uncertain` 的变量调用 LLM（省 token）；
2. 变量列表分批（每批 12 个），避免长 JSON 被截断；
3. **每个 LLM 值必须被原文支持**：
   - 数值：原文中必须出现该数值，且证据片段包含变量别名；
   - 布尔：证据片段必须包含变量别名；
   - 证据片段必须能在原文中**逐字匹配**；
4. 证据含"可疑/考虑/可能"时，不允许判为 present；
5. 超出合理范围 → `out_of_range`；
6. 规则 present/absent 优先，LLM 只补空缺；冲突写入 `conflicts.json`。

## 7. 人工复核流程

```bash
# 1) 生成复核表（优先 uncertain / out_of_range / 低置信度）
python scripts/make_review_sheet.py \
    --features out/clinical_features/features.jsonl \
    --out out/clinical_features/review_sheet.csv --max 200

# 2) 人工填写 review_value / review_status / reviewer / comment

# 3) 计算字段级指标
python scripts/score_extraction.py \
    --features out/clinical_features/features.jsonl \
    --review out/clinical_features/review_sheet.csv \
    --out out/clinical_features/review_metrics.json
```

评分输出：状态准确率、present precision/recall/F1、数值 MAE、容忍误差内比例、Cohen's kappa、分变量指标。

## 8. 命令汇总

```bash
python scripts/make_report_demo.py

# 规则抽取
python scripts/extract_clinical_features.py \
    --reports data/clinical_demo/reports.jsonl \
    --out-dir out/clinical_features

# 规则 + 本地 LLM 兜底
python scripts/extract_clinical_features.py \
    --reports data/clinical_demo/reports.jsonl \
    --out-dir out/clinical_features_llm \
    --use-llm --config configs/cluster_local.toml

# 复核表 + 评分
python scripts/make_review_sheet.py \
    --features out/clinical_features_llm/features.jsonl \
    --out out/clinical_features_llm/review_sheet.csv --max 50
python scripts/score_extraction.py \
    --features out/clinical_features_llm/features.jsonl \
    --review out/clinical_features_llm/review_sheet.csv \
    --out out/clinical_features_llm/review_metrics.json
```

## 9. 局限与下一步

- **尚未做真实人工复核**：需要用合作医院报告抽样双人标注，得到字段级 P/R/F1 与 Kappa；
- LLM 证据校验偏保守，跨句/同义改写可能漏抽（可扩充别名与同义词表）；
- 变量字典是 MAARS 子集，需按最终 CRF 扩充；
- 抽取结果尚未自动写回 `CLINICAL_SCHEMA` 的 `echo/ecg/labs` 字段；
- 演示报告全部为合成数据，不是患者数据；
- 下一步：人工复核 50–100 份 → 变量表接风险模型 baseline（Logistic/GBDT）→ 再接 MAARS 式多模态融合。
