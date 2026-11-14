#!/usr/bin/env bash
# One-click cluster workspace: ensure the local model service is running,
# print a short status, then open an interactive shell with helper commands.
set -u

SERVER_SESSION=local_server
PORT=18000

echo "======================================"
echo "  MMGraph-Med cluster workspace"
echo "======================================"

if ! curl -s -m 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "[1/2] local model service is not running; starting it (first load ~2-3 min) ..."
  tmux kill-session -t "$SERVER_SESSION" 2>/dev/null || true
  tmux new-session -d -s "$SERVER_SESSION" "bash /path/to/start_server.sh > /path/to/server.log 2>&1"
  for _ in $(seq 1 60); do
    if curl -s -m 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      break
    fi
    sleep 5
  done
fi

if curl -s -m 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "[ok] model service running: Qwen2.5-7B + BGE-M3 (port ${PORT})"
else
  echo "[!] service failed to start; check: tail -50 /path/to/server.log"
fi

echo ""
echo "Commands you can type directly:"
echo "  build    rebuild the hypergraph from data/smoke_zh"
echo "  status   show tmux / model service / GPU status"
echo "  exit     leave the cluster"
echo ""

if [ "${MMGRAPH_NO_SHELL:-0}" = "1" ]; then
  exit 0
fi

exec bash --rcfile /path/to/mmgraph_bashrc -i
