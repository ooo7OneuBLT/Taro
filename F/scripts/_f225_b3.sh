#!/bin/sh
cd /c/claude/AI/Taro
PY=.venv/Scripts/python.exe
BASE="F/models/F2-6_P1_喃語900s_12ヶ月_seed10_2026-08-24.pt"
for R in 1 2 3 4; do
  echo "== r$R =="
  head -5 "F/logs/_f225_B3_r$R.txt" | xargs -P 5 -I{} sh -c '$0 run/main.py "{}" > /dev/null 2>&1; echo done: {}' "$PY"
  tail -5 "F/logs/_f225_B3_r$R.txt" | xargs -P 5 -I{} sh -c '$0 run/main.py "{}" > /dev/null 2>&1; echo done: {}' "$PY"
  if [ "$R" = "1" ]; then MB="$BASE"; else MB="F/models/F2-25_B3_merged_r$((R-1)).pt"; fi
  $PY F/scripts/f_merge_models_v2.py "F/models/F2-25_B3_merged_r$R.pt" --base "$MB" --plain \
    F/models/F2-25_B3_r${R}_わんわん.pt F/models/F2-25_B3_r${R}_にゃんにゃん.pt \
    F/models/F2-25_B3_r${R}_ぶーぶー.pt F/models/F2-25_B3_r${R}_でんしゃ.pt \
    F/models/F2-25_B3_r${R}_りんご.pt F/models/F2-25_B3_r${R}_ボール.pt \
    F/models/F2-25_B3_r${R}_くつ.pt F/models/F2-25_B3_r${R}_ばなな.pt \
    F/models/F2-25_B3_r${R}_コップ.pt F/models/F2-25_B3_r${R}_ぼうし.pt || exit 1
done
echo "F225_B3_DONE"
