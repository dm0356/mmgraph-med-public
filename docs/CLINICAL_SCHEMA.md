# 临床数据 Schema（患者 × 时间窗）

本文件定义**合作医院数据接入**时的统一格式。目标：伦理获批后，数据工程师/临床合作者按此格式导出，
代码只需改配置即可复用现有流水线。

## 1. 基本单元：一个"病例-时间窗"

```text
patient_id + window(start, end)  →  一条 case
```

一个患者可以有多个不重叠的时间窗（如每 24 小时一个），每个窗口独立打标签。

## 2. JSON 示例

```json
{
  "case_id": "P0001_W2026-01-26",
  "patient_id": "P0001",
  "encounter_id": "E0001",
  "window": {"start": "2026-01-26T00:00:00", "end": "2026-01-27T00:00:00", "hours": 24},
  "demographics": {"age": 78, "sex": "M"},
  "diagnoses": [
    {"code": "I25.103", "name": "冠状动脉粥样硬化性心脏病"},
    {"code": "I50.900", "name": "心力衰竭"}
  ],
  "labs": [
    {"time": "2026-01-26T06:00:00", "name": "血钾", "value": 3.4, "unit": "mmol/L"},
    {"time": "2026-01-26T06:00:00", "name": "NT-proBNP", "value": 2100, "unit": "pg/mL"}
  ],
  "medications": [
    {"name": "胺碘酮", "dose": "200mg", "route": "po", "start": "2026-01-26T08:00:00"}
  ],
  "ecg": [
    {"time": "2026-01-26T08:00:00", "path": "ecg/P0001_20260126.xml",
     "hr": 85, "qrs_ms": 110, "qtc_ms": 460, "rhythm": "窦性心律"}
  ],
  "echo": [
    {"time": "2026-01-26T10:00:00", "path": "echo/P0001_20260126.mp4",
     "lvef": 35, "notes": "左室扩大，室壁运动减弱"}
  ],
  "cmr": [],
  "notes": [
    {"time": "2026-01-26T09:00:00", "text": "患者诉心悸，无明显胸痛。"}
  ],
  "labels": {
    "event": "VT",
    "event_time": "2026-01-27T02:30:00",
    "event_source": "adjudicated",
    "outcome_horizon_hours": 48,
    "label_note": "事件经两名医师复核"
  },
  "split": "train",
  "deidentified": true
}
```

## 3. 字段说明

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `case_id` | ✅ | 全局唯一，建议 `patient_id + window` |
| `patient_id` | ✅ | 脱敏后的患者 ID |
| `window.start/end` | ✅ | ISO 8601；窗口长度建议 24h 或 48h |
| `demographics.age/sex` | ✅ | 年龄、性别（M/F） |
| `diagnoses` | ✅ | ICD-10 编码 + 名称（可为空数组） |
| `labs` | ⭕ | 检验：时间、名称、值、单位 |
| `medications` | ⭕ | 药品、剂量、途径、开始时间 |
| `ecg` | ⭕ | 心电图：时间、文件路径、心率/QRS/QTc/节律 |
| `echo` | ⭕ | 超声：时间、路径、LVEF、描述 |
| `cmr` | ⭕ | 心脏磁共振：时间、路径、LVEF、LGE 等 |
| `notes` | ⭕ | 病程/护理文本（已脱敏） |
| `labels.event` | ✅ | `none` / `VT` / `VF` / `SCD` / `ICD_shock` |
| `labels.event_time` | event≠none 时必填 | 事件发生时间 |
| `labels.outcome_horizon_hours` | ✅ | 预警提前量，默认 48 |
| `labels.event_source` | ⭕ | `adjudicated` / `ICD_code` / `device_log` |
| `split` | ⭕ | `train` / `val` / `test`（**按患者划分**） |
| `deidentified` | ✅ | 必须为 `true` 才会进入流水线 |

## 4. 标签口径（关键）

设窗口结束时间为 `t_end`，事件时间为 `t_event`，预警提前量为 `H` 小时：

| 条件 | 标签 | 说明 |
| --- | --- | --- |
| `0 ≤ t_event - t_end ≤ H` | **1（阳性，前驱窗）** | 窗口结束后 H 小时内发生事件 |
| `t_event - t_end > H` 或 `event = none` | **0（阴性）** | 随访至少 H 小时无事件 |
| `t_event < t_end` | **排除** | 事件已发生在窗口内，不是"前驱" |
| 随访不足 H 小时且无事件 | **排除** | 不能确定阴性 |

脚本实现：`src/mmgraph_agent/eval/clinical_schema.py::window_label`。

## 5. 从 case 生成 RAG 语料

`case_to_documents(case)` 会把每个病例拆成多条文档（诊断、检验、用药、ECG、超声、病程），
与现有 `corpus.jsonl` schema 完全兼容，可直接喂给 `scripts/build_hypergraph.py`。

## 6. 脱敏与合规

- 姓名、身份证、电话、地址、住院号等**不得出现**；
- 文本需经院内脱敏流程；
- 原始影像/波形不得离开院内网络；
- 编码表（ICD-10）属医院授权资产，不随仓库分发。

## 7. 校验

```bash
python scripts/validate_clinical_data.py --cases data/clinical_demo/cases.jsonl
```

输出每条 case 的字段错误；只有全部通过才进入建图。

## 8. 导出建议

- 每个患者一个 JSON 文件，或一个 JSONL；
- 时间统一 ISO 8601（带时区）；
- 缺失值用 `null`/空数组，不要用 `"NA"` 字符串；
- 数值统一单位（K: mmol/L，BNP: pg/mL，LVEF: %）。
