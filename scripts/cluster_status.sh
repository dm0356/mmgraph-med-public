#!/usr/bin/env bash
set -u
echo '== tmux =='
tmux ls 2>/dev/null || echo 'no session'
echo '== model service =='
curl -s -m 3 http://127.0.0.1:18000/v1/models || echo 'not running'
echo
echo '== GPU =='
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
