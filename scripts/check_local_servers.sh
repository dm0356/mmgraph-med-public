#!/usr/bin/env bash
# Check that the three local vLLM services answer on their OpenAI-compatible API.
set -u

check() {
  local name="$1" port="$2"
  local tmp
  tmp=$(mktemp)
  local code
  code=$(curl -s -o "$tmp" -w "%{http_code}" "http://127.0.0.1:${port}/v1/models" || echo "000")
  if [ "$code" = "200" ]; then
    echo "[OK]   ${name} (port ${port})"
    cat "$tmp"
    echo
  else
    echo "[FAIL] ${name} (port ${port}) http=${code}"
  fi
  rm -f "$tmp"
}

check "text-extractor " 18000
check "image-captioner" 18001
check "text-embedder  " 18002
