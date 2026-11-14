"""Fetch open ECG datasets for the public pilot.

Verified reachable from a mainland-China IP (2026-11-13):
  * PhysioNet **open-access** datasets (no credentialing) are downloadable;
  * PhysioNet **credentialed** datasets (MIMIC/eICU/HiRID) return HTTP 403.

Usage (cluster, isolated libs):
    PYTHONPATH=/path/to/pylibs python scripts/fetch_public_data.py \
        --data-root data/public_pilot --ptbxl-signals 80
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

PTBXL_BASE = "https://physionet.org/files/ptb-xl/1.0.3"
PTBXL_METADATA = ["ptbxl_database.csv", "scp_statements.csv"]


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return
    request = Request(url, headers={"User-Agent": "mmgraph-med-public-pilot/0.1"})
    with urlopen(request, timeout=120) as response, target.open("wb") as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
    print(f"downloaded {url} -> {target} ({target.stat().st_size} bytes)", flush=True)


def fetch_vfdb(data_root: Path) -> None:
    sys.path.insert(0, "/path/to/pylibs")
    import wfdb  # noqa: PLC0415

    root = data_root / "vfdb"
    root.mkdir(parents=True, exist_ok=True)
    print("fetching VFDB (open PhysioNet) ...", flush=True)
    wfdb.dl_database("vfdb", dl_dir=str(root))
    print(f"VFDB files: {len(list(root.glob('*')))}", flush=True)


def fetch_ptbxl(data_root: Path, signals: int) -> None:
    root = data_root / "ptbxl"
    root.mkdir(parents=True, exist_ok=True)
    for name in PTBXL_METADATA:
        _download(f"{PTBXL_BASE}/{name}", root / name)

    metadata = root / "ptbxl_database.csv"
    rows = list(csv.DictReader(metadata.open(encoding="utf-8")))
    if not signals:
        print(f"PTB-XL metadata rows: {len(rows)}", flush=True)
        return

    selected = [row for row in rows if row.get("scp_codes") and row.get("filename_lr")][: signals]
    for row in selected:
        record = row["filename_lr"]
        for suffix in (".hea", ".dat"):
            _download(f"{PTBXL_BASE}/{record}{suffix}", root / f"{record}{suffix}")
    (root / "selected.json").write_text(
        json.dumps(
            [
                {
                    "ecg_id": row["ecg_id"],
                    "filename_lr": row["filename_lr"],
                    "scp_codes": row["scp_codes"],
                    "age": row.get("age", ""),
                    "sex": row.get("sex", ""),
                }
                for row in selected
            ],
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"PTB-XL selected records: {len(selected)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/public_pilot")
    parser.add_argument("--datasets", default="vfdb,ptbxl")
    parser.add_argument("--ptbxl-signals", type=int, default=0,
                        help="number of PTB-XL waveform records to download (0 = metadata only)")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    datasets = {item.strip() for item in args.datasets.split(",") if item.strip()}
    if "vfdb" in datasets:
        fetch_vfdb(data_root)
    if "ptbxl" in datasets:
        fetch_ptbxl(data_root, args.ptbxl_signals)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
