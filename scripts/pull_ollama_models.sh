#!/usr/bin/env bash
# Pull the three models used by configs/ollama.toml.
#
# Run this in a SECOND cluster terminal (ollama serve must already be running).
#   bash scripts/pull_ollama_models.sh
set -euo pipefail

export PATH=$HOME/ollama/bin:$PATH

if ! command -v ollama >/dev/null 2>&1; then
  echo "[error] ollama not found. Run scripts/install_ollama.sh first."
  exit 1
fi

echo "pulling text model: qwen2.5:7b"
ollama pull qwen2.5:7b
echo "pulling vision model: qwen2.5vl:7b"
ollama pull qwen2.5vl:7b
echo "pulling embedding model: bge-m3"
ollama pull bge-m3

echo ""
echo "installed models:"
ollama list
echo ""
echo "check the OpenAI-compatible API:"
echo "  curl http://127.0.0.1:11434/v1/models"
