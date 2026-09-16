# -*- coding: utf-8 -*-
"""F2-25：学習の並列化のA/B検証の実験一式を作る。

  A 逐次   10語×2周＝20本を鎖で（従来方式）
  B 並列   周ごとに10語を同時に走らせ、f_merge_models.py で混ぜる（FedAvg型）
  総学習量は完全に同一（同じ実験設定・同じ600ステップ・違いは繋ぎ方だけ）

【逸脱ラベル】体が10個ある赤ちゃんは実在しない〔Tier3・実験インフラ〕。
学習則そのもの（勾配の平均≒ミニバッチ）は変えない。採用可否はこのA/Bで決める。

    .venv/Scripts/python.exe F/scripts/f_gen_parallel_ab.py
"""
import os
import sys
import io
import json

sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:\claude\AI\Taro")

DATE = "2026-08-31"
START = "F/models/F2-6_P1_喃語900s_12ヶ月_seed10_2026-08-24.pt"
STEPS = 600
WORDS = ["わんわん", "にゃんにゃん", "ぶーぶー", "でんしゃ", "りんご",
         "ボール", "くつ", "ばなな", "コップ", "ぼうし"]


def main():
    src = {}
    for l in io.open("F/logs/_f221_chain.txt", encoding="utf-8").read().splitlines():
        if not l.strip():
            continue
        parts = os.path.basename(l).split("_")     # F2-21_聞く学習_1周目_わんわん_...
        src[(parts[2], parts[3])] = l              # (周, 語) → 実験ファイル

    for arm in ("A逐次", "B並列"):
        made = []
        prev = START
        for rd in ("1周目", "2周目"):
            for w in WORDS:
                ex = json.load(io.open(src[(rd, w)], encoding="utf-8"))
                tag = "F2-25_%s" % arm
                ex["name"] = "%s_%s_%s" % (tag, rd, w)
                ex["note"] = ("2026-08-31 F2-25 並列化A/B（%s）。総学習量はA=B。" % arm)
                ex["run"]["steps"] = STEPS
                ex["run"]["checkpoint"] = STEPS
                if arm == "A逐次":
                    ex["taro"]["model"] = prev
                else:
                    # 並列：周の全語が同じ出発点（1周目=START、2周目=混ぜた結果）
                    ex["taro"]["model"] = START if rd == "1周目" \
                        else "F/models/F2-25_B並列_merged_1周目.pt"
                ex["taro"]["save"] = "F/models/%s_%s_%s.pt" % (tag, rd, w)
                ex["run"]["csv"] = "F/logs/F2-25_並列AB/%s/%s_%s/run.csv" % (arm, rd, w)
                p = "F/experiments/%s_%s_%s_%s.json" % (tag, rd, w, DATE)
                io.open(p, "w", encoding="utf-8").write(
                    json.dumps(ex, ensure_ascii=False, indent=2))
                made.append(p)
                prev = ex["taro"]["save"]
        io.open("F/logs/_f225_%s.txt" % arm, "w", encoding="utf-8",
                newline="\n").write("\n".join(made) + "\n")
        print(arm, len(made), "本")


if __name__ == "__main__":
    main()
