import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "clinical_demo" / "reports.jsonl"

REPORTS = [
    {
        "patient_id": "P0001",
        "report_time": "2026-01-26T10:00:00",
        "source": "超声+CMR+检验",
        "text": (
            "患者男，78岁。超声心动图：左室射血分数约35%，室间隔基底段厚度约1.2cm，"
            "左房内径45mm。CMR：LGE提示室间隔中层条片状强化，未见心尖部血栓。"
            "心电图：窦性心律，QRS 110ms。检验：血钾3.4mmol/L，NT-proBNP 2100pg/mL，"
            "糖化血红蛋白8.74%。"
        ),
    },
    {
        "patient_id": "P0002",
        "report_time": "2026-02-10T10:00:00",
        "source": "超声+CMR",
        "text": (
            "患者女，49岁。LVEF 42%，IVSd 10mm，左室舒张末容积 180ml。"
            "LGE未见明显强化，可疑SAM。左房内径 45 mm，二尖瓣反流轻度。"
        ),
    },
    {
        "patient_id": "P0003",
        "report_time": "2026-03-05T10:00:00",
        "source": "超声",
        "text": (
            "患者男，66岁。射血分数55%，二尖瓣反流中度，未见心包积液。"
            "心电图：心房颤动，心率 96 次/分。"
        ),
    },
    {
        "patient_id": "P0004",
        "report_time": "2026-03-20T10:00:00",
        "source": "超声+CMR",
        "text": (
            "患者女，72岁。EF约60%，延迟强化阳性，累及下壁，SAM阴性。"
            "室间隔厚度 1.4 cm，左室射血分数 60%。"
        ),
    },
    {
        "patient_id": "P0005",
        "report_time": "2026-04-01T10:00:00",
        "source": "超声",
        "text": (
            "患者女，70岁。超声心动图：左心室射血分数 35%，"
            "左室流出道峰值压差 30mmHg，SAM阴性，左房内径 47mm。"
        ),
    },
]


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in REPORTS),
                   encoding="utf-8")
    print(f"wrote {len(REPORTS)} reports -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
