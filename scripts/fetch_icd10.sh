#!/usr/bin/env bash
# Download the Chinese ICD-10 dictionary (tree JSON) into data/icd10/.
#
# Source: https://github.com/chaseliu/ICD-10-CN (check its LICENSE before
# redistributing; the file is not vendored in this repository).
#
# Usage: bash scripts/fetch_icd10.sh [dest-dir]
set -euo pipefail

DEST=${1:-data/icd10}
mkdir -p "$DEST"
URL="https://api.github.com/repos/chaseliu/ICD-10-CN/contents/diseases.json"

echo "downloading ICD-10 Chinese dictionary ..."
curl -4 -s -H "Accept: application/vnd.github.raw" "$URL" -o "$DEST/diseases.json"
ls -lh "$DEST/diseases.json"
echo "done: $DEST/diseases.json"
