#!/bin/sh
# F2-25 A/B の実行台本。A=鎖・B=5本ずつの波→混ぜる→2周目→混ぜる
cd /c/claude/AI/Taro
PY=.venv/Scripts/python.exe

echo "== B 1周目（5本×2波・並列） =="
grep "1周目" F/logs/_f225_B並列.txt | head -5 | xargs -P 5 -I{} sh -c '$0 run/main.py "{}" > /dev/null 2>&1; echo done: {}' "$PY"
grep "1周目" F/logs/_f225_B並列.txt | tail -5 | xargs -P 5 -I{} sh -c '$0 run/main.py "{}" > /dev/null 2>&1; echo done: {}' "$PY"
echo "== 混ぜる（1周目） =="
$PY F/scripts/f_merge_models.py "F/models/F2-25_B並列_merged_1周目.pt" \
  F/models/F2-25_B並列_1周目_わんわん.pt F/models/F2-25_B並列_1周目_にゃんにゃん.pt \
  F/models/F2-25_B並列_1周目_ぶーぶー.pt F/models/F2-25_B並列_1周目_でんしゃ.pt \
  F/models/F2-25_B並列_1周目_りんご.pt F/models/F2-25_B並列_1周目_ボール.pt \
  F/models/F2-25_B並列_1周目_くつ.pt F/models/F2-25_B並列_1周目_ばなな.pt \
  F/models/F2-25_B並列_1周目_コップ.pt F/models/F2-25_B並列_1周目_ぼうし.pt || exit 1
echo "== B 2周目 =="
grep "2周目" F/logs/_f225_B並列.txt | head -5 | xargs -P 5 -I{} sh -c '$0 run/main.py "{}" > /dev/null 2>&1; echo done: {}' "$PY"
grep "2周目" F/logs/_f225_B並列.txt | tail -5 | xargs -P 5 -I{} sh -c '$0 run/main.py "{}" > /dev/null 2>&1; echo done: {}' "$PY"
echo "== 混ぜる（2周目＝最終） =="
$PY F/scripts/f_merge_models.py "F/models/F2-25_B並列_merged_final.pt" \
  F/models/F2-25_B並列_2周目_わんわん.pt F/models/F2-25_B並列_2周目_にゃんにゃん.pt \
  F/models/F2-25_B並列_2周目_ぶーぶー.pt F/models/F2-25_B並列_2周目_でんしゃ.pt \
  F/models/F2-25_B並列_2周目_りんご.pt F/models/F2-25_B並列_2周目_ボール.pt \
  F/models/F2-25_B並列_2周目_くつ.pt F/models/F2-25_B並列_2周目_ばなな.pt \
  F/models/F2-25_B並列_2周目_コップ.pt F/models/F2-25_B並列_2周目_ぼうし.pt || exit 1

echo "== A 逐次（20本の鎖） =="
while IFS= read -r f || [ -n "$f" ]; do
  $PY run/main.py "$f" > /dev/null 2>&1 && echo "done: $f" || echo "FAIL: $f"
done < F/logs/_f225_A逐次.txt
echo "F225_ALL_DONE"
