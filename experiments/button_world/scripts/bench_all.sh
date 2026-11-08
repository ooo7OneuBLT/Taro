#!/bin/bash
# 並列数 × P を1つずつ別プロセスで測る。GPU は 90% まで（約7.3GB）に制限し、1つ120秒で打ち切る。
unset WAYLAND_DISPLAY DISPLAY
cd /tmp
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
OUT=/mnt/c/claude/AI/button_world/out/bench.jsonl
mkdir -p /mnt/c/claude/AI/button_world/out
: > "$OUT"
if pgrep -f craftax_play/venv/bin/python > /dev/null; then
  echo "既に venv の python が動いているので始めない"
  exit 1
fi
nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader
for P in 3 5 7 10; do
  for N in 256 1024 4096 8192; do
    line=$(timeout 120 /home/syun5/craftax_play/venv/bin/python /mnt/c/claude/AI/button_world/scripts/bench.py --n_envs "$N" --P "$P" 2>/tmp/bw_err.txt | tail -n 1)
    if [ -z "$line" ]; then
      reason=$(grep -o -m1 -E "RESOURCE_EXHAUSTED|Out of memory|out of memory" /tmp/bw_err.txt)
      line="{\"n_envs\": $N, \"P\": $P, \"failed\": \"${reason:-timeout_or_error}\"}"
    fi
    echo "$line" | tee -a "$OUT"
  done
done
# 描画なしの参考（P は関係ない）
for N in 1024 8192; do
  timeout 120 /home/syun5/craftax_play/venv/bin/python /mnt/c/claude/AI/button_world/scripts/bench.py --n_envs "$N" --P 5 --no_render 2>/dev/null | tail -n 1 | tee -a "$OUT"
done
nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader
