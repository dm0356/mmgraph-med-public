"""Import an official 6-digit ICD-10 clinical coding table into the JSON tree.

The national clinical version (国家临床版) is distributed by health authorities /
hospital medical-record departments and is **not** freely redistributable.
Put your authorized file here and convert it once:

    python scripts/import_icd10.py \
        --input "国家临床版2.0疾病诊断编码.xlsx" \
        --out data/icd10/diseases.json

Supported input: ``.csv`` / ``.tsv`` (UTF-8 or GBK) and ``.xlsx`` (first sheet).
Column names are auto-detected; the common Chinese headers work out of the box:

    编码 / 代码 / code / icd_code / 诊断编码 / 疾病编码
    名称 / 诊断名称 / name / icd_name / 疾病名称
    父级 / 上级 / parent / 上级编码   (optional; used to build the tree)

The output is the flat ``[{"value": "<code>", "label": "<name>"}, ...]`` form that
``ICD10Searcher`` already understands.
"""

import argparse
import csv
import json
from pathlib import Path

CODE_COLUMNS = (
    "编码", "代码", "code", "icd_code", "诊断编码", "疾病编码", "疾病诊断编码", "医保编码",
)
NAME_COLUMNS = (
    "名称", "诊断名称", "name", "icd_name", "疾病名称", "疾病诊断名称", "诊断名称（中文）",
)
PARENT_COLUMNS = ("父级", "上级", "parent", "上级编码", "父编码")


def _pick_column(header: list[str], candidates: tuple[str, ...]) -> int | None:
    normalized = {str(name).strip().lower(): index for index, name in enumerate(header)}
    for candidate in candidates:
        index = normalized.get(candidate.strip().lower())
        if index is not None:
            return index
    return None


def _read_csv(path: Path) -> list[list[str]]:
    for encoding in ("utf-8-sig", "gbk", "utf-8"):
        try:
            with path.open(encoding=encoding, newline="") as handle:
                return [row for row in csv.reader(handle) if any(cell.strip() for cell in row)]
        except UnicodeDecodeError:
            continue
    raise ValueError(f"cannot decode {path}; try converting it to UTF-8 CSV")


def _read_xlsx(path: Path) -> list[list[str]]:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise SystemExit("openpyxl is required for .xlsx; use CSV or install openpyxl") from exc
    workbook = load_workbook(path, read_only=True, data_only=True)
    sheet = workbook.worksheets[0]
    rows = []
    for row in sheet.iter_rows(values_only=True):
        cells = ["" if value is None else str(value).strip() for value in row]
        if any(cells):
            rows.append(cells)
    return rows


def load_rows(path: Path) -> list[list[str]]:
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        return _read_csv(path)
    if suffix in (".xlsx", ".xlsm"):
        return _read_xlsx(path)
    raise SystemExit(f"unsupported input format: {suffix} (use .csv/.tsv/.xlsx)")


def convert(rows: list[list[str]]) -> list[dict]:
    if not rows:
        raise SystemExit("empty input file")
    header = [str(cell).strip() for cell in rows[0]]
    code_index = _pick_column(header, CODE_COLUMNS)
    name_index = _pick_column(header, NAME_COLUMNS)
    if code_index is None or name_index is None:
        raise SystemExit(
            "cannot find code/name columns.\n"
            f"header was: {header}\n"
            f"expected code columns like {CODE_COLUMNS} and name columns like {NAME_COLUMNS}"
        )
    parent_index = _pick_column(header, PARENT_COLUMNS)

    entries: list[dict] = []
    seen: set[str] = set()
    for row in rows[1:]:
        if len(row) <= max(code_index, name_index):
            continue
        code = str(row[code_index]).strip()
        label = str(row[name_index]).strip()
        if not code or not label or code in seen:
            continue
        seen.add(code)
        entry = {"value": code, "label": label}
        if parent_index is not None and len(row) > parent_index:
            parent = str(row[parent_index]).strip()
            if parent:
                entry["parent"] = parent
        entries.append(entry)
    if not entries:
        raise SystemExit("no valid rows found")
    return entries


def write_tree(entries: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="official CSV/TSV/XLSX coding table")
    parser.add_argument("--out", default="data/icd10/diseases.json")
    args = parser.parse_args()

    source = Path(args.input)
    if not source.exists():
        raise SystemExit(f"input not found: {source}")
    rows = load_rows(source)
    entries = convert(rows)
    out = Path(args.out)
    write_tree(entries, out)
    leaves = sum(1 for entry in entries if "parent" not in entry)
    print(f"imported {len(entries)} ICD entries ({leaves} top-level) -> {out}")
    print("next: set [grounding] searcher = \"icd10\" and icd10_path = this file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
