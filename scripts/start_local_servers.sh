#!/usr/bin/env bash
# Start the three local vLLM services in detached tmux sessions.
#
# Usage (on the cluster, inside the project directory):
#   bash scripts/start_local_servers.sh
#   bash scripts/check_local_servers.sh
#
# Override any setting via environment variables, e.g.:
#   LLM_GPU=4 EMBED_GPU=5 bash scripts/start_local_servers.sh
set -euo pipefail

LLM_MODEL=${LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}
VL_MODEL=${VL_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
EMBED_MODEL=${EMBED_MODEL:-BAAI/bge-m3}
LLM_PORT=${LLM_PORT:-18000}
VL_PORT=${VL_PORT:-18001}
EMBED_PORT=${EMBED_PORT:-18002}
LLM_GPU=${LLM_GPU:-0}
VL_GPU=${VL_GPU:-1}
EMBED_GPU=${EMBED_GPU:-2}
LOG_DIR=${LOG_DIR:-$HOME/vllm_logs}

mkdir -p "$LOG_DIR"

start_service() {
  local name="$1" command="$2"
  if tmux has-session -t "$name" 2>/dev/null; then
    echo "[skip] tmux session '$name' already exists"
    return
  fi
  tmux new-session -d -s "$name" "$command 2>&1 | tee '$LOG_DIR/$name.log'"
  echo "[start] $name -> $LOG_DIR/$name.log"
}

start_service mmkg_llm \
  "CUDA_VISIBLE_DEVICES=$LLM_GPU vllm serve $LLM_MODEL --port $LLM_PORT --max-model-len 32768"

start_service mmkg_vl \
  "CUDA_VISIBLE_DEVICES=$VL_GPU vllm serve $VL_MODEL --port $VL_PORT --limit-mm-per-prompt image=1"

start_service mmkg_embed \
  "CUDA_VISIBLE_DEVICES=$EMBED_GPU vllm serve $EMBED_MODEL --task embed --port $EMBED_PORT"

echo ""
echo "services: llm=:$LLM_PORT vl=:$VL_PORT embed=:$EMBED_PORT"
echo "logs    : $LOG_DIR"
echo "attach  : tmux attach -t mmkg_llm   (mmkg_vl / mmkg_embed)"
echo "check   : bash scripts/check_local_servers.sh"
echo ""
echo "Note: the first start downloads the models; this can take a long time."
