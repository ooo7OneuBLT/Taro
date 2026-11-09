#!/bin/bash
# 見張りつきで python を1本走らせる。使い方: bash run_watched.sh <ログの置き場> <python の引数...>
# RAM 16GB 超か GPU 全体 7.6GB 超が 3回続けば止める（SIGINT → 30秒 → SIGTERM → 15秒 → SIGKILL）。
unset WAYLAND_DISPLAY DISPLAY
cd /tmp
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.65  # 画面表示などが約1.5〜1.9GB 使うので、全体で 7.6GB を超えないように
LOG="$1"; shift
mkdir -p "$(dirname "$LOG")"
if pgrep -f "^/home/syun5/craftax_play/venv/bin/python" > /dev/null; then  # 先頭が python のプロセスだけを数える（外側の bash -c の文字列に引っかからないように）
  echo "既に venv の python が動いているので始めない"; exit 1
fi
/home/syun5/craftax_play/venv/bin/python "$@" > "$LOG" 2>&1 &
PID=$!
over=0; peak_ram=0; peak_gpu=0
while kill -0 "$PID" 2>/dev/null; do
  sleep 5
  ram_kb=$(awk '/VmRSS/ {print $2}' /proc/$PID/status 2>/dev/null || echo 0)
  gpu_mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  ram_gb=$(( ram_kb / 1048576 ))
  [ "$ram_kb" -gt "$peak_ram" ] && peak_ram=$ram_kb
  [ "$gpu_mb" -gt "$peak_gpu" ] && peak_gpu=$gpu_mb
  if [ "$ram_kb" -gt 16777216 ] || [ "$gpu_mb" -gt 7782 ]; then over=$((over+1)); else over=0; fi
  if [ "$over" -ge 3 ]; then
    echo "見張り: 上限を超えたので止める ram_kb=$ram_kb gpu_mb=$gpu_mb" | tee -a "$LOG"
    kill -INT "$PID"; sleep 30; kill -TERM "$PID" 2>/dev/null; sleep 15; kill -KILL "$PID" 2>/dev/null
  fi
done
wait "$PID"; code=$?
echo "見張り: 終了 code=$code 最大RAM=$((peak_ram/1024))MB 最大GPU全体=${peak_gpu}MB" | tee -a "$LOG"
