# 中文医学多模态：API / 模型选型建议

本工程有三类"模型组件"，可以分别用云端 API 或本地开源模型替换。

## 1. 组件划分

| 组件 | 作用 | 论文中的对应 |
|---|---|---|
| 文本抽取 | 从中文文本抽实体/关系 | MMGraph-Agent 的 spaCy NER / M³KG 的 extractor |
| 图像理解 | 描述医学图像、OCR、VQA | MMGraph-Agent 的 frozen CLIP（仅编码）|
| 向量嵌入 | 文本/图像检索 | frozen CLIP 的向量空间 |

## 2. 云端 API（快速起步）

| 组件 | 推荐 | 说明 |
|---|---|---|
| 文本抽取 | **DeepSeek `deepseek-chat`** | 中文强、便宜、OpenAI 兼容；已验证 |
| 文本抽取备选 | 百炼 `qwen-plus` / `qwen-max` | 与百炼生态一致 |
| 图像理解 | **百炼 `qwen3-vl-plus`** / `qwen3-vl-max` | 中文视觉理解强；`qwen3-vl-plus` 已验证 |
| 图像理解备选 | `qwen-vl-max-latest`、`qwen2.5-vl-72b-instruct` | 视账号开通情况 |
| 文本嵌入 | 百炼 `text-embedding-v3` | 中文/多语言 |
| 图文嵌入 | （云 API 无通用 CLIP） | 建议本地 Chinese-CLIP / BiomedCLIP |

> ⚠️ **DeepSeek 公开 API 目前是纯文本模型**，没有视觉能力；图像必须用 Qwen-VL。

## 3. 本地开源模型（L40S 上部署，隐私敏感数据首选）

| 组件 | 推荐 | 维度/说明 |
|---|---|---|
| 文本抽取 | Qwen2.5-7B/14B-Instruct（vLLM 部署） | 可 LoRA 微调 |
| 图像理解 | Qwen2.5-VL-7B/32B-Instruct | 通用医学 |
| 医学专用 VLM | HuatuoGPT-Vision-7B/34B、MedDr | 中文医学问答/描述 |
| 文本嵌入 | **BGE-M3** / bge-large-zh-v1.5 | 1024 维，中文强 |
| 图文嵌入 | **Chinese-CLIP**（OFA-Sys） | 512 维，中文图文 |
| 医学图文嵌入 | **BiomedCLIP**（microsoft） | 512 维，英文医学为主 |

## 4. 推荐组合（按阶段）

- **阶段 A — 现在（快速验证）**：DeepSeek-Chat（文本）+ Qwen3-VL-Plus（图像）；嵌入先用 BGE-M3 本地。
- **阶段 B — 数据敏感/上量**：L40S 上 vLLM 部署 Qwen2.5-7B + Qwen2.5-VL-7B；本地嵌入 BGE-M3 + Chinese-CLIP，数据不出院。
- **阶段 C — 论文级医学专精**：在脱敏数据上对 Qwen2.5-VL-7B 做 LoRA，或换 HuatuoGPT-Vision。

## 5. 映射到论文的 zero-API 路径

论文：spaCy（NER）+ frozen CLIP（图文编码）。中文医学版建议：

| 论文组件 | 中文医学替代 |
|---|---|
| spaCy NER | spaCy `zh_core_web_sm` + 医学词典；或 LLM 抽取（适配版） |
| frozen CLIP（文本） | BGE-M3 / bge-large-zh |
| frozen CLIP（图文） | Chinese-CLIP / BiomedCLIP |
| 图像解码（论文无） | Qwen-VL / HuatuoGPT-Vision（适配版才有） |

## 6. 合规

- 患者数据优先**本地部署**；外发只发**去标识化文本**，图像需脱敏。
- 记录每个 API 的**模型版本与调用时间**，便于复现与审计。
- 论文中明确写清用的是 "zero-API 忠实复现" 还是 "API 适配版"。

## 7. zero-API 在中文临床的可行性评估

| 组件 | zero-API 方案 | 中文临床可行性 | 结论 |
|---|---|---|---|
| 分句/分词 | spaCy `zh_core_web_sm` | ✅ 可用 | 保留 |
| 实体识别 | spaCy zh NER | ⚠️ 通用实体可用，医学实体覆盖差 | 需医学词典或微调 |
| 关系抽取 | spaCy 本身不做 | ❌ 需句法规则/远程监督 | 很难追上 LLM |
| 图文编码 | 冻结 CLIP | ❌ 通用 CLIP 对 X 光/CT/病理弱 | 需 BiomedCLIP/Chinese-CLIP，仍有限 |
| 构建成本 | 0 API | ✅ 最便宜 | 适合做基线/消融 |

**结论**：zero-API 适合"英文通用图文"的忠实复现；中文临床建议**以 API 适配版为主、
zero-API 作为对照基线**。模块 ②（L1/L2 记忆）与 ③（GRPO 策略）**与构建方式解耦**，
两种构建都能用，所以"适配构建 + 忠实复现记忆/策略"是可行组合。
