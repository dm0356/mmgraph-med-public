# 医学术语词典接入（ICD-10）

## 1. 现状

目前 grounding 支持三种 searcher：

| searcher | 数据 | 说明 |
|---|---|---|
| `none` | 无 | 只靠 LLM 生成描述（可能编造） |
| `local` | 本地 TSV/JSONL 词表 | 院内术语表/自定义词典 |
| `wikipedia` | 中文维基 | 通用概念，医学准确性参差 |
| **`icd10`** | **ICD-10 中文疾病词表** | 规范疾病名 + 权威编码 |

## 2. 获取 ICD-10 中文数据

使用公开仓库 [`chaseliu/ICD-10-CN`](https://github.com/chaseliu/ICD-10-CN)：

```bash
bash scripts/fetch_icd10.sh
```

产物：`data/icd10/diseases.json`（树形 JSON：`value` 编码 / `label` 名称 / `children`）。

> 该文件**不入库**（`.gitignore` 已忽略 `data/icd10/`），使用前请核对该仓库 LICENSE 与数据来源合规性；
> 临床研究请优先使用**国家/医院正式发布的 ICD-10 版本**。

## 3. 配置

```toml
[construction]
enabled = true
grounding = true

[grounding]
searcher = "icd10"
icd10_path = "data/icd10/diseases.json"
max_candidates = 3
```

## 4. 它在流水线里做什么

1. **规范名（dictionary-first normalization）**：
   实体命中 ICD 名称/别名（如"冠心病"）→ canonical 名替换为 ICD 正式名
   （"冠状动脉粥样硬化性心脏病"）；
2. **权威描述（dictionary-first grounding）**：
   描述优先写成 `ICD-10 <编码> <名称>`，再与 LLM 描述一起交给 refiner；
   **词典命中时最终以词典为准**（覆盖 LLM 描述，减少编造）；
3. 未命中 ICD 的实体仍走 LLM 接地/精炼。

代码位置：
- `src/mmgraph_agent/hypergraph/knowledge_searcher.py` → `ICD10Searcher`
- `src/mmgraph_agent/hypergraph/construction.py` → `canonical()` / `preferred()` 钩子

## 5. 覆盖范围与局限

- ICD-10 主要覆盖**疾病诊断**；症状、检验、药品、手术操作不在 ICD-10 内。
- 需要补充：
  - **手术/操作**：ICD-9-CM-3；
  - **症状/体征**：SNOMED CT 中文子集、中文医学术语集；
  - **药品**：院内药品字典 / ATC；
  - **检验**：LOINC 中文。
- 缩写/别名覆盖有限，`ICD10Searcher.ALIASES` 可继续补充（冠心病、心梗、房颤…）。

## 7. 6 位临床版（国家临床版 / 医院编码库）

公开的 3 位大类码精度不足（例如 `I25` 无法区分 `I25.103`）。**6 位临床版由卫健委/医院发布，没有可自由再分发的公开版本**，因此采用"导入授权文件"的方式。

### 7.0 合作医院提供的授权文件

由合作医院提供（不随仓库分发）：

- 文件：`国家临床版2.0疾病诊断编码（ICD-10）.xlsx`
- 内容：**数万条**疾病诊断编码（6 位 + `x` 扩展码），表头为 `疾病诊断编码 / 疾病诊断名称`
- 导入命令（本仓库已验证）：

```bash
python scripts/import_icd10.py \
  --input "D:/path/to/institution-authorized-icd10.xlsx" \
  --out data/icd10/diseases.json
```

- 输出：`data/icd10/diseases.json`（数万条），已在本地/集群环境验证导入流程；
- **该文件属于医院授权资产**：仅内部使用，不提交 Git、不上传公网（`.gitignore` 已排除）。

### 7.1 导入医院授权文件

1. 向病案室/信息科索取本单位使用的 **国家临床版 ICD-10**（Excel 或 CSV，至少包含"编码、名称"两列）；
2. 转换：

```bash
python scripts/import_icd10.py \
  --input "国家临床版2.0疾病诊断编码.xlsx" \
  --out data/icd10/diseases.json
```

列名会自动识别（`编码/代码/code/疾病诊断编码`、`名称/诊断名称/name/疾病诊断名称`），也支持 `父级/上级` 列；
输出即 `ICD10Searcher` 可直接读取的 JSON。

3. 配置指向新文件并重跑建图：

```toml
[grounding]
searcher = "icd10"
icd10_path = "data/icd10/diseases.json"
```

### 7.2 演示数据（仅验证流程）

`data/icd10/diseases.sample6.json` 是一份**演示用**的小型 6 位心血管编码表
（冠心病 `I25.101`、NSTEMI `I21.401`、心力衰竭 `I50.900`、2 型糖尿病 `E11.900` 等）。
**编码可能随版本不同，仅用于验证流程，正式研究必须替换为医院授权版本。**

## 8. 验证

```bash
python -m unittest tests.test_grounding tests.test_icd10_import -v
```

在患者数据上可快速检查命中情况（示例）：

```bash
python - <<'EOF'
from mmgraph_agent.config import load_config
from mmgraph_agent.hypergraph.knowledge_searcher import get_searcher
cfg = load_config("configs/patient_construction.toml")
s = get_searcher(cfg)
for term in ["冠心病", "2型糖尿病", "心力衰竭", "心房颤动"]:
    print(term, "->", s.search(term), "| canonical:", s.canonical(term))
EOF
```
