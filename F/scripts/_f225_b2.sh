#!/bin/sh
cd /c/claude/AI/Taro
PY=.venv/Scripts/python.exe
head -5 F/logs/_f225_B2.txt | xargs -P 5 -I{} sh -c '$0 run/main.py "{}" > /dev/null 2>&1; echo done: {}' "$PY"
tail -5 F/logs/_f225_B2.txt | xargs -P 5 -I{} sh -c '$0 run/main.py "{}" > /dev/null 2>&1; echo done: {}' "$PY"
$PY F/scripts/f_merge_models_v2.py "F/models/F2-25_B2_merged_final.pt" \
  --base "F/models/F2-25_B2_merged_1周目.pt" \
  F/models/F2-25_B2_2周目_わんわん.pt F/models/F2-25_B2_2周目_にゃんにゃん.pt \
  F/models/F2-25_B2_2周目_ぶーぶー.pt F/models/F2-25_B2_2周目_でんしゃ.pt \
  F/models/F2-25_B2_2周目_りんご.pt F/models/F2-25_B2_2周目_ボール.pt \
  F/models/F2-25_B2_2周目_くつ.pt F/models/F2-25_B2_2周目_ばなな.pt \
  F/models/F2-25_B2_2周目_コップ.pt F/models/F2-25_B2_2周目_ぼうし.pt || exit 1
echo "F225_B2_DONE"
