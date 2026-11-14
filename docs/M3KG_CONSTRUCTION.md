# M³KG-RAG 四个组件的实现对照

M³KG-RAG（CVPR 2026）的构建流程（论文 Figure 2）有四块内容，我们按"医学多模态"场景落地如下。

## 1. 多跳（Multi-hop）

- **论文**：多跳 MMKG，三元组通过共享实体形成多跳路径，检索时沿路径扩展。
- **我们的实现**：
  - `hypergraph/store.py` → `Store.neighbor_hyperedges(seed_ids, hops)`：从种子超边出发，沿**共享实体**走 `hops` 跳。
  - `memory/cache.py` → `TwoTierMemory.retrieve`：当 `[retrieval].hops > 0` 时，把多跳邻居加入候选，再按查询相似度重排。
- **开启**：`configs/*.toml` 里设
  ```toml
  [retrieval]
  hops = 1      # 0 = 关闭；1~2 常用
  ```
- **与《多跳检索》论文的关系**：那篇（NSCLQ）研究 KG 上**复杂逻辑查询（∧/∨/¬/∃）**的**核采样 + 剪枝**；我们目前只是**确定性共享实体扩展**，还没有实现它的自适应采样/剪枝——可作为后续优化。

## 2. 上下文增强的三元组抽取（Context-Enriched Triplet Extraction）

- **论文**：rewriter 先把语义 caption 改写成知识密集型文本，再 extractor 抽实体-关系三元组。
- **我们的实现**：
  - `hypergraph/construction.py` → `ConstructionPipeline.rewrite()`：调用 LLM 把文档改写成知识密集型中文描述；
  - `hypergraph/llm_extractor.py` → `LLMExtractor.extract_rich(text, context=...)`：抽取时把改写文本作为**消歧上下文**；
  - `hypergraph/builder.py`：`construction` 开启时，先对每篇文档 rewrite，再把上下文传给抽取器。
- **开启**：
  ```toml
  [construction]
  enabled = true
  rewriter = true
  ```

## 3. 知识接地（Knowledge Grounding）

- **论文**：normalizer（规范概念）+ searcher（Wikipedia/Wiktionary/爬虫）+ LLM callback（补描述）。
- **我们的实现**：
  - `ConstructionPipeline.enrich_entities()` 内的 **normalize** 阶段：LLM 把实体映射为规范名（去修饰、统一简称）；
  - **searcher** 阶段：`knowledge_searcher.py` 从**本地医学词表**或**中文维基百科**取候选描述（见 `docs/KNOWLEDGE_GROUNDING.md`）；
  - **ground** 阶段：LLM 基于上下文生成百科式描述；
  - **refine** 阶段把外部候选与 LLM 描述一起交给 selector 选择；
  - 输出写回 `Store.entities`：`name` 变规范名、`description` 为接地描述。
- **开启**：
  ```toml
  [construction]
  grounding = true

  [grounding]
  searcher = "local"        # 或 "wikipedia"
  local_path = "data/glossary/medical_terms.sample.tsv"
  ```

## 4. 上下文感知的描述精炼（Context-Aware Description Refinement）

- **论文**：selector（选候选）+ refiner（改写成原表述）+ inspector（自反思）。
- **我们的实现**：
  - **refine** 阶段：LLM 结合上下文选择/改写最贴切的描述；
  - **inspector** 阶段：LLM 给描述打分（0–10），低于阈值（6）的描述清空，回退到原文描述。
- **开启**：
  ```toml
  [construction]
  refinement = true
  inspector = false   # 额外一次 LLM 调用，默认关
  ```

## 5. 完整配置示例

```toml
[construction]
enabled = true
rewriter = true
grounding = true
refinement = true
inspector = false

[retrieval]
hops = 1
```

**成本**：每篇文档约 `1 次 rewrite + 每单元 1 次抽取 + 2~3 次文档级接地/精炼`；
抽取结果有磁盘缓存（`out/.extract_cache`），重复跑不重复计费。

## 6. 仍未实现（明确边界）

| 能力 | 状态 |
|---|---|
| 外部知识库 searcher（本地词表 / Wikipedia） | ✅ 已实现（见 `docs/KNOWLEDGE_GROUNDING.md`） |
| NSCLQ 式多跳采样/剪枝 | ❌ 待做 |
| 三元组级超边（当前是"单元级"超边） | ❌ 待做 |
| 多智能体（rewriter/extractor/normalizer/selector/refiner/inspector 各自独立进程） | ⚠️ 当前是单管线串行调用 |

## 7. 与"多跳检索.pdf"（NSCLQ）的区别

- **M³KG-RAG**：面向 **RAG** 的多模态知识图构建 + 检索增强；
- **NSCLQ（多跳检索.pdf）**：面向 **KG 逻辑查询**（FB15k 等）的**推理**，核心是核采样 + 组合优化；
- 两者都涉及"多跳"，但任务不同。NSCLQ 的自适应采样/剪枝思想可以借鉴到我们的多跳检索里，但它不是 RAG 的直接组件。
