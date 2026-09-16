# -*- coding: utf-8 -*-
"""F2-41：猫を知らない太郎を喃語直後（F2-6）から9択で通しで育てる（ユーザー指示 2026-09-02）。

段取りはF2-40と同一（8ラウンド×3000歩・各語の個体1,2,3,6,7,8,9,10・連呼・短命海馬・
ラウンド別シード）。違いは①起点がF2-6（語の学習前）②猫（toy2）を世界から退場（9択）。
    .venv/Scripts/python.exe F/scripts/f_gen_f41.py
"""
import os, sys, io, json
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:\claude\AI\Taro")
DATE = "2026-09-02"
ROUNDS = 8
made = []
prev = "F/models/F2-6_P1_喃語900s_12ヶ月_seed10_2026-08-24.pt"
for r in range(1, ROUNDS + 1):
    sc = json.load(io.open("run/scenes/座位_12ヶ月_F2-40_10択_r%d_個体%s_%s.json" % (
        r, [1, 2, 3, 6, 7, 8, 9, 10][r - 1], DATE), encoding="utf-8"))
    sname = sc["name"].replace("F2-40_10択", "F2-41_9択猫なし")
    sc["name"] = sname
    sc["note"] = "%s F2-41 ラウンド%d。F2-40と同一の世界から猫（toy2）だけ退場した9択。" % (DATE, r)
    sc["world"]["toy2"]["enabled"] = False
    sc["world"]["parent_labeling"]["utterances"].pop("toy2", None)
    io.open("run/scenes/%s.json" % sname, "w", encoding="utf-8").write(json.dumps(sc, ensure_ascii=False, indent=1))
    ex = json.load(io.open("F/experiments/F2-40_r%d_%s.json" % (r, DATE), encoding="utf-8"))
    ex["name"] = "F2-41_r%d" % r
    ex["note"] = "%s F2-41 ラウンド%d/8。猫を知らない太郎をF2-6から9択で通し。" % (DATE, r)
    ex["scene"] = sname
    ex["taro"]["model"] = prev
    ex["taro"]["save"] = "F/models/F2-41_r%d_seed%d_%s.pt" % (r, 20 + r, DATE)
    ex["run"]["seed"] = 20 + r
    ex["run"]["csv"] = "F/logs/F2-41_9択通し/r%d/run.csv" % r
    ex["plugins"]["word_learning"]["events_out"] = "F/logs/F2-41_9択通し/r%d/発話イベント.csv" % r
    ex["plugins"]["model_snapshots"] = {"out_dir": "F/logs/F2-41_9択通し/r%d/snapshots" % r}
    p = "F/experiments/F2-41_r%d_%s.json" % (r, DATE)
    io.open(p, "w", encoding="utf-8").write(json.dumps(ex, ensure_ascii=False, indent=2))
    made.append(p); prev = ex["taro"]["save"]
    print("r%d: %s" % (r, sname))
os.makedirs("F/logs/F2-41_9択通し", exist_ok=True)
io.open("F/logs/_f241_chain.txt", "w", encoding="utf-8", newline="\n").write("\n".join(made) + "\n")
print("鎖:", len(made), "本")
