# MAARS 算法复现审计（对照 Nature Cardiovascular Research 2025）

参考：Lai et al., **Multimodal AI to forecast arrhythmic death in hypertrophic
cardiomyopathy (MAARS)**, *Nat Cardiovasc Res* 2025;4:891–903.
DOI: [10.1038/s44161-025-00679-1](https://doi.org/10.1038/s44161-025-00679-1)

> 结论先行：**按"能训练出可比的 SCDA 风险模型"定义，当前复现度约 0%；按工程组件计约 10%**。
> 仓库目前是"数据/证据/RAG 层"，不是 MAARS 的预测模型实现。

## 1. MAARS Methods 拆解

### 1.1 数据与预处理

- 内部队列 JH-HCM：553 例；外部 SHVI-HCM：286 例；
- 终点 SCDA：持续性 VT/VF → ICD 放电/ATP，或院外心脏骤停；
- 67 个临床协变量：**40 个 EHR + 27 个 CIR（超声/CMR 报告变量）**；
- LGE-CMR：LV ROI 分割 → 左室以外置零 → 按血池中位像素强度归一化 → 重采样到 **96×96×20**（体素 4.0×4.0×6.3 mm）；
- EHR/CIR：缺失率 <40%，**MICE** 多重插补，z-score 标准化。

### 1.2 Modality-specific branch networks

| 分支 | 结构 |
| --- | --- |
| LGE-CMR | **3D-ViT**：3D 图像切 n 个不重叠 patch → 线性投影 E + CLS token + 位置编码 → 8 个 transformer block（8 heads，d=512）→ CLS 输出 |
| EHR | FNN：2 个隐藏层，latent 16 |
| CIR | FNN：1 个隐藏层，latent 16 |

### 1.3 Transformer-based multimodal neural network

- **MBT（multimodal bottleneck transformer）**：3 个 block；共享融合向量维度 8；
- 每个 block 把单模态向量与共享融合向量拼接，送入模态特定 transformer；
- 融合向量按平均更新；最后线性层 + sigmoid 输出 SCDA 风险概率。

### 1.4 训练

- 三个分支先独立预训练，再端到端联合训练；
- batch 64，最多 150 epochs，early stopping；
- Adam（β1=0.9，β2=0.999）；
- 学习率：LGE 1e-3、EHR/CIR 1e-2、fusion 3e-2；
- **balanced focal loss**（γ=2，α 按类别比例）。

### 1.5 Assessing model performance and clinical validation

- 内部 **5 折交叉验证**，按患者划分、按结局分层；外部独立验证；
- 指标：AUROC、balanced accuracy、AUPRC、Brier score；
- 阈值：最大化 Youden's J；与临床工具比较时，**匹配临床工具的敏感度**再比较特异度；
- 200 次 bootstrap 置信区间；亚组公平性（性别、年龄）。

### 1.6 Model interpretability

- EHR/CIR：**SHAP**（permutation），总体 mean |SHAP| 与 Pearson r 联合分析；
- LGE-CMR：**attention rollout**，把注意力回传到输入影像，生成热图；
- 论文还训练了简化协变量 FNN，验证可解释性与性能的折中。

## 2. 当前仓库复现状态

| MAARS 组件 | 当前状态 | 说明 |
| --- | --- | --- |
| 队列/终点标签 | ❌ 未开始 | 无合作医院数据；只有合成 demo |
| 40 EHR + 27 CIR 变量 | 🟡 草案 | CRF v0.1 共 57 变量，MAARS 对齐但未冻结 |
| 报告变量抽取 | 🟡 原型 | 规则 + LLM 兜底 + 人工复核工具，合成数据验证 |
| LGE-CMR ROI/归一化/重采样 | ❌ 未开始 | 无 3D 预处理 |
| 3D-ViT 分支 | ❌ 未开始 | 只有 2D CLIP/Chinese-CLIP 图像向量 |
| EHR FNN 分支 | ❌ 未开始 | 表格数据 schema 已具备 |
| CIR FNN 分支 | ❌ 未开始 | 变量抽取可供电，但模型未建 |
| MBT 多模态融合 | ❌ 未开始 | — |
| focal loss/训练流程 | ❌ 未开始 | 现有 GRPO 是 RAG 策略训练，与风险模型无关 |
| 5 折 CV/外部验证 | ❌ 未开始 | — |
| 评测指标 harness | ✅ 已实现 | AUROC/AUPRC/sens@spec/Brier/ECE/lead time/bootstrap |
| SHAP/attention rollout | ❌ 未开始 | RAG 证据溯源 ≠ 模型可解释性 |

**结论**：

- 若以"MAARS 算法复现"为标准：**约 0–10%**；
- 已有的是 **数据层、变量抽取层、评测层、RAG/证据层**；
- 不能对外声称"已复现 MAARS"。

## 3. 对你的文本优先方案意味着什么

你的数据是 **EHR + 超声/ECG/CTA/CMR 报告文本**，不做原始图像分析。因此：

- **可以跳过**：LGE-CMR ROI、3D-ViT、attention rollout、96×96×20 预处理；
- **可以直接复现**：EHR FNN + CIR FNN + 融合模块（MAARS 的 CIR 分支本来就是 27 个报告变量，不是原始图像）；
- **推荐替代**：
  - Branch A：EHR 结构化变量 → FNN/GBDT；
  - Branch B：报告抽取的 CIR 变量 → FNN/GBDT；
  - Branch C（可选）：报告全文的中文文本编码器（Chinese-BERT/LLM embedding）；
  - 融合：early concatenation baseline → late fusion → MBT-lite / attention fusion；
- **可解释性**：表格变量用 SHAP/系数；文本用 attention/occlusion；RAG 层给医生看证据。

> 这样是"MAARS-inspired text-first model"，不是 MAARS 复现。

## 4. 建议的复现路线（文本优先）

| 阶段 | 内容 | 产出 |
| --- | --- | --- |
| P0 | CRF 冻结、标签定义、患者级划分 | 数据字典 + 队列表 |
| P1 | Logistic/GBDT baseline | AUROC/AUPRC/校准基线 |
| P2 | EHR FNN + CIR FNN + early fusion | 两分支模型 |
| P3 | MBT-lite / attention fusion | 融合模型 |
| P4 | SHAP/文本归因 + RAG 解释 | 可解释报告 |
| P5 | 外部验证/前瞻性验证 | 临床验证 |

## 5. 参考

- MAARS 官方代码（论文声明）：`https://gitlab.com/natalia-trayanova/maars-public`（使用前核对许可证）；
- 论文 DOI 见文首；
- 临床预测模型方法学见 [MULTIMODAL_PIPELINE.md](MULTIMODAL_PIPELINE.md)。
