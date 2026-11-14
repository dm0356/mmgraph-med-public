#!/usr/bin/env bash
# Install Ollama into the user's home directory (no root required) on Linux.
#
# Usage (on the cluster):
#   bash scripts/install_ollama.sh
#
# Then add it to PATH (the script prints the line):
#   export PATH=$HOME/ollama/bin:$PATH
set -euo pipefail

DEST=${DEST:-$HOME/ollama}
mkdir -p "$DEST"
cd "$DEST"

if [ -x "$DEST/bin/ollama" ]; then
  echo "[ok] ollama already installed: $DEST/bin/ollama"
else
  echo "downloading ollama linux tarball ..."
  curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama.tgz
  tar -xzf ollama.tgz
fi

echo ""
echo "installed: $DEST/bin/ollama"
echo "add to PATH for this shell:"
echo "  export PATH=$DEST/bin:\$PATH"
echo "make it permanent:"
echo "  echo 'export PATH=$DEST/bin:\$PATH' >> ~/.bashrc && source ~/.bashrc"
