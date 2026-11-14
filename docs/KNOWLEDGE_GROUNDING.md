# 外部知识接地（Knowledge Grounding / Searcher）

M³KG-RAG 构建流程第 2 步的 "searcher"：为实体到外部知识源查**候选描述**，
再交给 selector/refiner 选择，减少 LLM 编造。

## 1. 支持的来源

| 来源 | 配置 | 说明 | 适用 |
|---|---|---|---|
| 本地医学词表 | `searcher = "local"` | TSV/JSONL 词表，完全离线 | **患者数据（推荐）** |
| 中文维基百科 | `searcher = "wikipedia"` | MediaWiki API 取页面摘要 | 通用概念 |
| 不启用 | `searcher = "none"` | 只用 LLM 描述 | 默认 |

> 外发到 Wikipedia 的**只有实体名称**，不会发送病历原文；但患者场景仍推荐本地词表。

## 2. 配置

本地词表（推荐）：

```toml
[grounding]
searcher = "local"
local_path = "data/glossary/medical_terms.sample.tsv"
max_candidates = 1
```

中文维基百科：

```toml
[grounding]
searcher = "wikipedia"
lang = "zh"
timeout = 10
max_candidates = 1
max_chars = 200
```

注意：这些阶段由 `[construction].enabled = true` 触发。

## 3. 词表格式

**TSV**（一行一个词条，制表符分隔）：

```
2型糖尿病	一种以胰岛素抵抗为特征的慢性代谢性疾病。
空腹血糖	空腹至少8小时后测得的静脉血糖浓度。
```

**JSONL**（支持别名）：

```json
{"term": "2型糖尿病", "description": "...", "aliases": ["T2DM", "二型糖尿病"]}
```

示例文件：[data/glossary/medical_terms.sample.tsv](../data/glossary/medical_terms.sample.tsv)（**合成示例，非真实临床数据**）。

## 4. 在构建流程中的位置

```
rewrite（上下文增强改写）
  → extract（抽取实体/关系）
  → normalize（规范名）
  → searcher（本地词表 / Wikipedia 查候选描述）   ← 本模块
  → LLM ground（模型生成描述）
  → refine（选择/改写最贴合的描述）
  → inspector（打分，低于阈值清空）
```

- `refinement = true`：外部候选与 LLM 描述一起交给 selector；
- `refinement = false`：优先采用外部候选，其次 LLM 描述。

## 5. 隐私与可靠性

- 患者数据：**优先本地词表**；Wikipedia 只发实体名。
- Wikipedia 医学内容质量参差，不能替代权威指南/教材；建议人工抽检。
- 未接入 SNOMED CT / ICD / UMLS 等**授权**知识库（需许可证），但接口是统一的，
  新增来源只需实现 `KnowledgeSearcher.search()`。

## 6. 代码

- `src/mmgraph_agent/hypergraph/knowledge_searcher.py`
  - `LocalGlossarySearcher`、`WikipediaSearcher`、`NullSearcher`、`get_searcher()`
- `src/mmgraph_agent/hypergraph/construction.py` → `ConstructionPipeline.enrich_entities()`
- 单测：`tests/test_grounding.py`

```bash
python -m unittest tests.test_grounding -v
```
