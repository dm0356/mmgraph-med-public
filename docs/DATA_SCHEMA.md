# 临床多模态数据规范（JSONL）

本工程的输入是两个 JSONL 文件：`corpus.jsonl`（知识语料）和 `qa.jsonl`（评测问答）。
两者都是 **UTF-8 编码、一行一个 JSON 对象、无 BOM**。

> ⚠️ 合规提醒：患者数据外发给 DeepSeek / Qwen 等 API 之前，**文本必须去标识化**，图像建议先脱敏或只发本地模型。请遵守医院与伦理委员会的数据管理规定。

## 1. corpus.jsonl（知识语料）

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `doc_id` | string | ✅ | 文档唯一 ID，如 `case_0001` |
| `text` | string | ❌* | 正文，可含多句 |
| `images` | array | ❌ | 图像列表 |
| `tables` | array | ❌ | 表格列表（`text` 为拍平后的文本） |
| `diagrams` | array | ❌ | 图表列表（`text` 为说明文本） |

\* `text` / `images` / `tables` / `diagrams` **至少有一个非空**。

**images 元素**

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `image_id` | string | ✅ | 图像唯一 ID |
| `caption` | string | ❌ | 原始说明/标注 |
| `path` | string | ❌ | 图像文件路径（相对项目根目录）。**只有给了 `path` 才会走视觉模型解码**；否则用 caption。 |

**tables 元素**：`{"table_id": "...", "text": "..."}`
**diagrams 元素**：`{"diagram_id": "...", "text": "..."}`

示例：

```json
{"doc_id": "case_0001", "text": "患者反复胸闷三个月。心电图显示窦性心律。", "images": [{"image_id": "case_0001_img1", "caption": "心电图", "path": "data/cases/case_0001/ecg.png"}]}
```

## 2. qa.jsonl（评测问答）

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `qid` | string | ✅ | 唯一 ID |
| `question` | string | ✅ | 问题 |
| `answer` | string | ✅ | 参考答案 |
| `has_image` | bool | ❌ | 默认 `false`；为 `true` 时策略可能走图像检索 |
| `regime` | string | ❌ | `k_only` / `web_only` / `k_web`，便于分场景评测 |

示例：

```json
{"qid": "q1", "question": "心电图显示什么？", "answer": "窦性心律", "has_image": false, "regime": "k_only"}
```

## 3. 图像要求

- 格式：`jpg / jpeg / png / webp / bmp`
- 建议单张 ≤ 5 MB，长边 ≤ 2048
- **DICOM 不能直接作为 path**：先导出为常见图像格式，并**去除元数据**（患者信息、检查号、设备、时间）
- 图像放在 `path` 指定的位置，路径相对项目根目录

## 4. 去标识化清单（务必逐项检查）

**文本**
- 姓名、住院号/门诊号、身份证号、电话、详细地址
- 精确日期（可保留年份或相对时间，如"3 个月前"）
- 医院/科室具体名称（可泛化为"某三甲医院"）
- 罕见病 + 地点等可识别组合

**图像**
- 烧录文字、检查号、DICOM tag
- 面部、纹身、明显可识别特征（必要时打码/裁剪）
- 建议只保留检查部位区域

**流程**
1. 本地去标识化 → 2. 双人复核 → 3. 仅将去标识化文本发 API → 4. 原始数据不出院/不离开受控环境。

## 5. 评测分组（regime）

- `k_only`：只有私有知识库（离线超图）
- `web_only`：只有网络检索
- `k_web`：两者结合

训练时建议只用 in-domain 数据，另留一部分做 OOD（分布外）评测。

## 6. 数据校验

```bash
python scripts/validate_data.py --corpus data/smoke_zh/corpus.jsonl --qa data/smoke_zh/qa.jsonl
```

校验内容：JSON 合法性、`doc_id` 唯一、图像路径存在、空内容告警、QA 字段完整等。有 ERROR 时会非零退出，可直接用于 CI。
