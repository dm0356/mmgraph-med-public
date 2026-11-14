# 风险分层与事件预测路线（MAARS 对照）

参考：Lai et al., **Multimodal AI to forecast arrhythmic death in hypertrophic
cardiomyopathy (MAARS)**, *Nature Cardiovascular Research* 2025;4:891–903.
DOI: 10.1038/s44161-025-00679-1

## 1. MAARS 做了什么

| 维度 | MAARS |
| --- | --- |
| 人群 | 肥厚型心肌病（HCM），内部 553 例 + 外部 286 例 |
| 终点 SCDA | 持续性 VT/VF 导致 ICD 放电/ATP，或院外心脏骤停 |
| 输入 | **EHR 结构化变量（40 个）** + **心脏影像报告变量（27 个，来自超声/CMR 报告）** + **LGE-CMR 原始图像** |
| 模型 | 三个单模态分支（EHR-FNN、CIR-FNN、LGE-CMR 3D-ViT）+ 多模态 bottleneck transformer 融合 |
| 性能 | 内部 AUROC 0.89（0.79–0.94），外部 0.81（0.69–0.93） |
| 对比 | 优于 ACC/AHA、ESC 指南与 HCM Risk-SCD 计算器 |
| 可解释 | 注意力/归因，分析各模态贡献 |

关键点：**报告文本本身就能提供 27 个结构化变量**，不需要对超声/ECG 视频做抽帧。

## 2. 当前 MMGraph-Med 架构的定位

当前仓库是**数据/证据层**，不是风险预测模型：

```text
多模态临床数据
  → 临床 schema（患者×时间窗 + 事件标签）
  → 文档化（诊断/检验/用药/ECG/超声/CMR/病程）
  → 超图（实体 + 证据单元 + ICD-10）
  → 嵌入（BGE-M3 文本 + CLIP/Chinese-CLIP 图像）
  → L1/L2 记忆 + 多跳检索
  → LLM/RAG 作答与解释
```

它解决的是：

1. **多模态数据统一存储**（可转成知识图谱/超图 JSON）；
2. **证据可追溯**（每条答案/预警能回到原文/影像）；
3. **本地化 RAG 数据源**（患者数据不出内网，供 LLM 检索）；
4. **可替换的组件**（换数据只改配置）。

它**不直接输出风险概率**——风险分层/事件预测需要额外的监督模型。

## 3. 与 MAARS 的映射

| MAARS 组件 | 当前仓库 | 状态 |
| --- | --- | --- |
| EHR 40 变量 | `CLINICAL_SCHEMA.md` 结构化字段（诊断/检验/用药/ demographics） | 🟡 schema 已定，变量表待映射 |
| 报告 27 变量 | 报告文本 + ICD/术语抽取；`case_to_documents` 生成证据 | 🟡 文本在，变量抽取待做 |
| LGE-CMR 3D-ViT | CLIP/Chinese-CLIP 图像编码（2D）；无 3D ViT | 🟡 2D 向量已通，3D 待接 |
| 多模态融合 | 尚未实现 | ❌ 需新增风险模型头 |
| 风险评分 | 尚未实现 | ❌ 需标签 + 监督训练 |
| 可解释/证据 | 超图 + RAG + 原文溯源 | ✅ 比 MAARS 更强 |
| 评测 | `clinical_metrics.py`（AUROC/AUPRC/校准/提前量） | ✅ 已就绪 |

## 4. 拿到合作医院数据后的最小落地顺序

### Step 1：标签与变量表（1–2 周）

- 终点定义与 MAARS 对齐：持续性 VT/VF → ICD 放电/ATP，或院外心脏骤停；
- 时间窗：如事件前 24h/48h；阴性需随访 ≥ 48h；
- 变量表：年龄、性别、BMI、NSVT、AF、晕厥/先兆晕厥、家族史、LGE%、LVEF、SAM、IVS 厚度、LA 直径、LV 容积等；
- 报告变量：从超声/ECG/CTA/CMR 报告中抽取（LLM + 规则 + 人工复核）。

### Step 2：基线风险模型（2–4 周）

先用可解释、易复现的模型建立 baseline：

- Logistic regression / Cox / GBDT（XGBoost/LightGBM）；
- 输入：EHR 变量 + 报告变量；
- 指标：AUROC、AUPRC、Brier、校准曲线、敏感度@固定特异度；
- 目的：确认变量与标签质量，再上深度多模态。

### Step 3：MAARS 式多模态模型

- 分支 A：EHR 变量 → FNN；
- 分支 B：报告变量 → FNN；
- 分支 C：LGE-CMR 图像 → 3D-ViT / 医学图像 encoder；
- 融合：bottleneck transformer / 简单拼接；
- 输出：SCDA 风险概率；
- 内部交叉验证 + 外部验证（不同院区/时间段）。

### Step 4：RAG/LLM 层接入

- 风险模型输出分数；
- RAG 检索该患者的证据（化验、报告、影像描述、指南）；
- LLM 生成"为什么高风险/建议"的解释；
- 医生在环，保留证据引用。

## 5. 当前仓库需要新增的模块

| 模块 | 说明 |
| --- | --- |
| `clinical_features.py` | 从结构化字段/报告文本抽取 MAARS 变量表 |
| `risk_model/` | Logistic/GBDT baseline + 多模态融合模型 |
| `eval_risk.py` | AUROC/AUPRC/校准/决策曲线 |
| `report_extraction.py` | 报告文本 → 27 个 CIR 变量（LLM + 规则） |
| CMR 3D 分支 | 体素/切片序列 → 3D encoder（后续） |

## 6. 结论

- **当前架构 = 数据/证据/RAG 层**：可以把多模态输入转成知识图谱/超图存储，并作为 LLM 的本地数据源；
- **风险分层/事件预测 = 额外的监督模型层**，需要标签、变量表、训练与验证；
- MAARS 证明了"EHR + 报告变量 + CMR 图像"的融合路线可行；我们的 RAG 层可以让它的输出**可解释、可溯源**；
- 两者结合才是完整课题：**预测模型负责"风险有多高"，RAG/LLM 负责"为什么、依据是什么、下一步做什么"**。
