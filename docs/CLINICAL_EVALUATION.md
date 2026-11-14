# 临床评测方案（Clinical Evaluation Harness）

本文件定义**预警模型**的评测口径，供合作医院数据接入后直接使用。
当前阶段只用**合成/公开数据**验证脚本，不产生临床结论。

## 1. 输入格式

每个"病例-时间窗"一条预测：

```json
{
  "case_id": "P0001_W2026-01-26",
  "patient_id": "P0001",
  "score": 0.82,
  "label": 1,
  "alert_time": "2026-01-26T12:00:00",
  "event_time": "2026-01-27T02:30:00"
}
```

| 字段 | 说明 |
| --- | --- |
| `score` | 预警模型输出的风险概率（0–1） |
| `label` | 1 = 窗口结束后 H 小时内发生事件；0 = 未发生 |
| `alert_time` | 模型发出预警的时间（用于算提前量） |
| `event_time` | 实际事件时间（阴性可空） |

## 2. 指标定义

| 指标 | 含义 |
| --- | --- |
| **AUROC** | 区分阳性/阴性窗口的整体能力 |
| **AUPRC** | 类别不平衡下更敏感，事件率低时优先报告 |
| **Sensitivity @ Specificity** | 固定特异度（默认 90%）下的敏感度 |
| **PPV / NPV** | 预警阳性/阴性的预测值 |
| **Brier score** | 概率校准（越小越好） |
| **ECE** | 期望校准误差（分 10 箱） |
| **Lead time** | 真阳性预警的提前小时数（中位数/IQR/P90） |

脚本：`src/mmgraph_agent/eval/clinical_metrics.py`
入口：`scripts/eval_clinical.py`

## 3. 评测协议（关键）

1. **按患者划分**训练/验证/测试，禁止同一患者跨集合；
2. 报告**事件率、阳性数、样本量**；
3. 阈值必须在**训练集/验证集**上选，测试集只用一次；
4. 报告 **95% bootstrap CI**；
5. 缺失数据单独统计，不做插补；
6. 前瞻验证与回顾验证分开报告；
7. 预警提前量与临床可操作窗口对齐（如 24h/48h）。

## 4. 命令

```bash
# 合成数据自测（无需合作医院数据）
python scripts/eval_clinical.py --synthetic 400 --out out/clinical_eval.json

# 真实预测
python scripts/eval_clinical.py --predictions preds.jsonl --out out/clinical_eval.json
```

## 5. 输出示例（合成数据）

```json
{
 "n": 400,
 "positives": 80,
 "event_rate": 0.2,
 "auroc": 0.89,
 "auroc_ci95": [0.85, 0.93],
 "auprc": 0.71,
 "threshold_at_specificity": 0.45,
 "specificity": 0.9,
 "sensitivity": 0.78,
 "sensitivity_ci95": [0.68, 0.86],
 "ppv": 0.67,
 "npv": 0.94,
 "brier": 0.12,
 "ece": 0.05,
 "lead_time": {"n": 78, "median_hours": 11.2, "iqr_hours": 6.4, "p90_hours": 20.1}
}
```

> 以上数字为合成数据示例，**不可引用为临床性能**。

## 6. 报告模板

| 指标 | 回顾队列 | 前瞻队列 |
| --- | --- | --- |
| N（患者/窗口/阳性） | | |
| 事件率 | | |
| AUROC (95% CI) | | |
| AUPRC | | |
| Sens @ Spec=90% | | |
| PPV / NPV | | |
| Brier / ECE | | |
| Lead time 中位数 (IQR) | | |

## 7. 合规

- 预测结果文件含患者 ID，必须脱敏并限制访问；
- 评测脚本本身不联网；
- 公开数据与合作医院数据分开报告，不混合。
