# 多模态临床数据 → 风险分层/预测：完整流程与 baseline 骨架

面向本项目（EHR + 检查报告文本，暂不做原始图像）的方法学指南。
参考综述与报告规范列在文末。

## 0. 一句话总览

```text
临床问题
 → 伦理/数据治理
 → 队列与标签（谁、什么事件、什么时间窗）
 → 多模态数据接入与本地数据库
 → 去标识化/质控/CRF 特征表
 → 预处理（缺失/编码/标准化）
 → 患者级划分（训练/验证/测试）
 → baseline 模型 → 多模态模型
 → 评价（判别/校准/临床效用/外部验证）
 → 可解释性 → 前瞻性验证 → 部署监测
```

## 1. 先定义问题（PICO）

- **P**：目标人群（如心力衰竭/心肌病患者）；
- **I/E**：可用数据（EHR、超声、ECG、CTA、CMR 报告）；
- **O**：结局（恶性室性心律失常：持续性 VT/VF → ICD 放电/ATP 或院外心脏骤停）；
- **T**：预测时间窗（如事件前 24h/48h；阴性需随访 ≥ 48h）；
- **用途**：风险分层、预警、还是解释？决定阈值和评价指标。

## 2. 伦理与数据治理

- 伦理批件、知情豁免/知情同意、数据使用协议；
- 去标识化（姓名/住院号/日期泛化）、最小必要原则；
- 本地部署，原始数据不出院内网络；
- 数据版本、访问审计、模型版本可追溯。

## 3. 队列与标签（最容易出错）

- 纳入/排除标准；
- 事件裁定：两名医师复核、事件来源（ICD 程控/病历/死亡记录）；
- **索引时间点**（index time）与预测窗口；
- 阴性定义：随访足够长且无事件；
- 排除：事件发生在窗口内、随访不足、数据缺失严重；
- 类别不平衡：事件率低（MAARS 3%–9%），需要 focal loss/重采样/阈值选择。

## 4. 多模态数据接入与本地数据库

| 来源 | 内容 | 接入方式 |
| --- | --- | --- |
| HIS/EMR | 人口学、诊断、病史、用药 | 结构化表/接口 |
| LIS | 检验（血钾、NT-proBNP、eGFR、HbA1c、LDL） | 结构化表 |
| PACS/超声 | 报告文本（LVEF、LGE、SAM、瓣膜） | 报告文本 + 可选图像 |
| ECG | 报告文本/参数（HR、QRS、QTc、节律） | 文本/结构化 |
| 随访/器械 | ICD 程控、事件 | 登记表 |

关键：**统一患者 ID + 时间戳 + 检查类型**，保留来源与出处。

## 5. 去标识化、质控与 CRF 特征表

- 文本去标识化（姓名、日期、地址）；
- 单位归一（cm→mm、L/min→ml/s 等）；
- 缺失/否定/不确定语义；
- 规则 + LLM 抽取 + 人工复核；
- 输出 CRF 宽表（一行一个患者/时间窗，一列一个变量）；
- 每个变量带 `source/time/confidence`。

参见 [`CRF_MAPPING.md`](CRF_MAPPING.md)、[`FEATURE_EXTRACTION.md`](FEATURE_EXTRACTION.md)。

## 6. 预处理

- **缺失**：显式 missing；MICE/中位数插补 + 缺失指示变量；
- **类别**：one-hot / target encoding（注意泄漏）；
- **数值**：标准化/归一化（只在训练集 fit）；
- **文本**：分词、截断、embedding（Chinese-BERT / Qwen / TF-IDF）；
- **不平衡**：class weight / focal loss / 阈值调整。

> 所有预处理必须在训练集上拟合，再应用到验证/测试集。

## 7. 数据划分

- **患者级划分**（同一患者的所有时间窗只出现在一个集合）；
- 时间划分（更接近真实部署）；
- 训练/验证/测试 + 交叉验证；
- 外部验证（不同院区/时间段）；
- 防止泄漏：事件后变量、同一患者重复、未来信息。

## 8. Baseline 风险模型骨架是什么

**定义**：用最简单的监督模型，把 CRF 变量映射为事件概率，作为"地板"和 sanity check。

### 8.1 骨架组成

```text
data.py       CRF 宽表 + labels + split → X, y
preprocess.py 缺失/编码/标准化（fit on train）
models.py     LogisticRegression / GBDT / FNN
train.py      训练 + 交叉验证 + 早停
evaluate.py   AUROC/AUPRC/Brier/校准/sens@spec
explain.py    系数/OR + SHAP
report.py     表格 + 校准曲线 + 决策曲线
configs/risk_baseline.toml
```

### 8.2 最小可用版本

- 模型：L2 正则 Logistic Regression；
- 输入：EHR 变量 + 报告 CIR 变量；
- 输出：事件概率；
- 评价：AUROC、AUPRC、Brier、校准曲线、敏感度@90% 特异度；
- 对照：临床评分（ACC/AHA、ESC、HCM Risk-SCD）或单模态模型；
- 解释：OR + 95% CI、SHAP。

### 8.3 为什么先做 baseline

1. 验证标签/变量/流程是否可靠；
2. 给复杂模型一个可比的底线；
3. 发现数据泄漏、缺失、单位问题；
4. 论文/基金需要"从简到繁"的证据链。

## 9. 多模态模型与融合

### 9.1 融合策略

| 策略 | 做法 | 优点/缺点 |
| --- | --- | --- |
| Early fusion | 所有特征拼接 → 一个模型 | 简单；模态异质性难处理 |
| Late fusion | 每个模态一个模型 → 分数加权/堆叠 | 稳健；交互弱 |
| Intermediate fusion | 每个模态编码 → 融合层（MBT/attention） | 表达强；数据需求高 |
| Unified transformer | 多模态 token 一起进 transformer | 先进；训练难 |

MAARS 用 intermediate fusion（MBT），并证明优于 early fusion。

### 9.2 文本优先架构（本项目）

```text
Branch A: EHR 结构化变量 → FNN/GBDT
Branch B: 报告抽取的 CIR 变量 → FNN/GBDT
Branch C（可选）: 报告全文 → Chinese-BERT/LLM embedding → FNN
Fusion: concat → MLP / attention / MBT-lite
Output: SCDA 概率
```

不需要 LGE-CMR 3D-ViT；如果以后有原始 CMR，再加 Branch D。

## 10. 评价与临床验证

### 10.1 判别

- AUROC、AUPRC（不平衡时更重要）；
- 敏感度、特异度、PPV、NPV；
- 固定特异度下的敏感度（临床常用）。

### 10.2 校准

- Brier score；
- 校准曲线/ECE；
- 决策曲线分析（DCA）。

### 10.3 验证

- 内部交叉验证（患者级）；
- 外部验证（不同院区/时间段）；
- bootstrap 置信区间；
- 亚组公平性（性别、年龄）；
- 报告规范：**TRIPOD+AI**。

### 10.4 临床效用

- 预警提前量（lead time）；
- 每 1000 患者避免事件数/误报数；
- 医生工作流影响。

## 11. 可解释性

- 表格：系数/OR、SHAP（全局 + 个体）；
- 文本：attention/occlusion、关键句高亮；
- 图像（如有）：attention rollout/Grad-CAM；
- RAG/LLM：引用证据原文，给医生可核对的解释；
- 注意：可解释性 ≠ 因果性。

## 12. 部署与监测

- 静默模式（silent mode）先跑 3–6 个月；
- 前瞻性验证；
- 数据漂移/性能监测；
- 版本管理、模型卡、审计日志；
- 医生在环，避免自动化偏倚。

## 13. 常见坑

1. 事件后变量泄漏；
2. 同一患者跨训练/测试；
3. 用测试集调阈值；
4. 只看 AUROC 不看校准；
5. 缺失当 0；
6. 类别不平衡不处理；
7. 外部验证缺失；
8. 可解释性事后编故事。

## 14. 参考

- Acosta et al. **Multimodal biomedical AI**. *Nat Med* 2022.
  [nature.com/articles/s41591-022-01981-2](https://www.nature.com/articles/s41591-022-01981-2)
- **Developing clinical prediction models: a step-by-step guide**. *BMJ* 2024.
  [bmj.com/content/386/bmj-2023-078276.full](https://www.bmj.com/content/386/bmj-2023-078276.full)
- **TRIPOD+AI statement**. *BMJ* 2024;385:e078378.
  [doi.org/10.1136/bmj-2023-078378](https://doi.org/10.1136/bmj-2023-078378)
- EHR 结构化+文本融合系统综述.
  [sciencedirect.com/.../S1566253525000545](https://www.sciencedirect.com/science/article/abs/pii/S1566253525000545)
- EHR 多模态表示学习综述.
  [sciencedirect.com/.../S1566253525008723](https://www.sciencedirect.com/science/article/abs/pii/S1566253525008723)
- MAARS 复现审计见 [`MAARS_REPRODUCTION_AUDIT.md`](MAARS_REPRODUCTION_AUDIT.md)。
