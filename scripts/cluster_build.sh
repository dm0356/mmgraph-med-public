#!/usr/bin/env bash
set -u
cd /path/to/mmgraph-agent || exit 1
PYTHONPATH=src /path/to/conda/envs/m3kg/bin/python scripts/build_hypergraph.py \
  --config configs/cluster_local.toml \
  --data-root data/smoke_zh \
  --out /path/to/out/mmkg_local.json
