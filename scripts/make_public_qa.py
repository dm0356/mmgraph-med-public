"""Expand the public pilot into a GRPO/SFT QA set with group-aware splits.

Group keys:
  * VFDB  -> record id  (no record appears in both train and eval)
  * PTB-XL -> patient_id (no patient appears in both train and eval)

Each item also carries a ``category`` so evaluation can be reported per question
type instead of a single token-F1 number.

Usage:
    python scripts/make_public_qa.py --limit 12000 --eval-ratio 0.2
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def _vfdb_questions(meta: dict, text: str) -> list[tuple[str, str, str]]:
    record = meta.get("record", "?")
    start = meta.get("start_s", 0.0)
    end = meta.get("end_s", 0.0)
    prefix = f"VFDB 记录 {record} 在 {start:.1f}-{end:.1f} 秒窗口的"
    candidates = [
        (prefix + "标签是什么？", str(meta.get("label", "")), "label"),
        (prefix + "心率是多少？", f"{meta.get('hr', 0):.0f}次/分", "hr"),
        (prefix + "节律是什么？", str(meta.get("rhythm", "")), "rhythm"),
        (prefix + "RR 标准差是多少？", f"{meta.get('rr_sd', 0):.0f}毫秒", "rr_sd"),
        (prefix + "RMSSD 是多少？", f"{meta.get('rmssd', 0):.0f}毫秒", "rmssd"),
        (prefix + "PNN50 是多少？", f"{meta.get('pnn50', 0):.2f}", "pnn50"),
        (prefix + "高频能量比是多少？", f"{meta.get('hf_ratio', 0):.2f}", "hf_ratio"),
        (prefix + "频谱熵是多少？", f"{meta.get('entropy', 0):.2f}", "entropy"),
    ]
    return [(q, a, c) for q, a, c in candidates if a and a in text]


def _ptbxl_questions(meta: dict, text: str) -> list[tuple[str, str, str]]:
    ecg_id = meta.get("ecg_id", "?")
    codes = str(meta.get("scp_codes", ""))
    first_code = next(
        (token for token in codes.replace("'", " ").replace(":", " ").split() if token.isupper()),
        "",
    )
    candidates = [
        (f"PTB-XL 记录 {ecg_id} 的诊断代码是什么？", first_code, "ptbxl_code"),
        (f"PTB-XL 记录 {ecg_id} 的患者年龄是多少？", f"{meta.get('age', '')}岁", "age"),
        (f"PTB-XL 记录 {ecg_id} 的患者性别是什么？", str(meta.get("sex", "")), "sex"),
    ]
    return [(q, a, c) for q, a, c in candidates if a and a in text]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="/path/to/public_pilot/pilot_big/corpus.jsonl")
    parser.add_argument("--out-root", default="/path/to/public_pilot/pilot_big")
    parser.add_argument("--limit", type=int, default=12000)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.corpus).read_text(encoding="utf-8").splitlines() if line.strip()]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        meta = row.get("meta", {})
        text = row.get("text", "")
        if meta.get("dataset") == "vfdb":
            group = f"vfdb:{meta.get('record', '?')}"
            pairs = _vfdb_questions(meta, text)
        elif meta.get("dataset") == "ptbxl":
            group = f"ptbxl:{meta.get('patient_id') or meta.get('ecg_id')}"
            pairs = _ptbxl_questions(meta, text)
        else:
            continue
        for question, answer, category in pairs:
            grouped[group].append(
                {"question": question, "answer": answer, "category": category, "group": group}
            )

    groups = sorted(grouped)
    rng = random.Random(args.seed)
    rng.shuffle(groups)
    split = int(len(groups) * (1 - args.eval_ratio))
    train_groups, eval_groups = set(groups[:split]), set(groups[split:])

    train = [item for group in train_groups for item in grouped[group]]
    eval_items = [item for group in eval_groups for item in grouped[group]]
    rng.shuffle(train)
    rng.shuffle(eval_items)
    train = train[: args.limit]
    eval_items = eval_items[: max(200, args.limit // 5)]

    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)

    def write(path: Path, items: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for index, item in enumerate(items):
                handle.write(json.dumps(
                    {"qid": f"qa-{index}", "question": item["question"], "answer": item["answer"],
                     "category": item["category"], "group": item["group"],
                     "has_image": False, "regime": "k_only"},
                    ensure_ascii=False,
                ) + "\n")

    write(root / "qa_train.jsonl", train)
    write(root / "qa_eval.jsonl", eval_items)
    categories = defaultdict(int)
    for item in eval_items:
        categories[item["category"]] += 1
    print(json.dumps({
        "groups": len(groups),
        "train_groups": len(train_groups),
        "eval_groups": len(eval_groups),
        "train": len(train),
        "eval": len(eval_items),
        "eval_categories": dict(categories),
    }, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
