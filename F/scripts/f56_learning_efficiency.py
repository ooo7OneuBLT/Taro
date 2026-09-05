# -*- coding: utf-8 -*-
"""学習効率の測定（2026-09-03）：太郎は1語を何回聞いて覚えたか。

材料は F2-49c（赤染め修正後・列C）の鎖8本。各ラウンドの終わりに保存したモデルで
初見個体48枚（8語×2個体×3角度）の8択テストをした結果は既にある。
親の発話回数はログ（発話イベント.csv）から数える。

    .venv/Scripts/python.exe F/scripts/f56_learning_efficiency.py
出力: F/logs/F2-56_学習効率/図_1語あたりの提示回数と正答.png
"""
import os, sys, io, csv, collections
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:\claude\AI\Taro")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.abspath("F/scripts"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import f_gen_f49 as G

OUT = "F/logs/F2-56_学習効率"
# 初見個体48枚の正答数（研究日誌 F2-49c 列C）
SCORE = [32, 40, 47, 37, 38, 43, 44, 35]
LOGDIR = "F/logs/F2-49c_赤染め修正"


def main():
    os.makedirs(OUT, exist_ok=True)
    fp = "C:/Windows/Fonts/meiryo.ttc"
    font_manager.fontManager.addfont(fp)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
    SLOT_WORD = {k: w for k, (w, _, _) in zip(G.SLOT_KEYS, G.WORDS)}
    per_round, cum = [], []
    for r in range(1, 9):
        c = collections.Counter()
        with io.open("%s/r%d/発話イベント.csv" % (LOGDIR, r), encoding="utf-8") as f:
            for row in csv.DictReader(f):
                c[SLOT_WORD.get(row["target"], row["target"])] += 1
        per_round.append(c)
        cum.append(sum(c.values()) + (cum[-1] if cum else 0))
    n_word = len(SLOT_WORD)
    x = [c / n_word for c in cum]           # 1語あたりの提示回数（累計）
    y = SCORE
    fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.6))
    a = ax[0]
    a.plot(x, y, "o-", color="#2b6cb0", lw=2, ms=7, label="太郎（初見個体48枚・8択）")
    a.axhline(48 / 8, color="#999", ls=":", label="でたらめに答えた場合（6/48）")
    a.axhline(48, color="#2f855a", ls="--", lw=1, label="画像を読めるAI（48/48）")
    for xi, yi, r in zip(x, y, range(1, 9)):
        a.annotate("r%d" % r, (xi, yi), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    a.set_xlabel("1語あたり、親が言った回数（累計）"); a.set_ylabel("48枚中の正答数")
    a.set_ylim(0, 52); a.set_xlim(0, 125); a.grid(alpha=.3); a.legend(fontsize=9, loc="lower right")
    a.set_title("太郎が1語を何回聞いて覚えたか", fontsize=12)
    b = ax[1]
    words = list(per_round[0].keys())
    tot = collections.Counter()
    for c in per_round:
        tot.update(c)
    ws = [w for w, _ in tot.most_common()]
    b.barh(range(len(ws)), [tot[w] for w in ws], color="#4a7fb5")
    b.set_yticks(range(len(ws))); b.set_yticklabels(ws)
    b.invert_yaxis(); b.set_xlabel("8ラウンド合計で親が言った回数")
    b.set_title("語ごとの提示回数（合計 %d 回）" % sum(tot.values()), fontsize=12)
    b.grid(alpha=.3, axis="x")
    fig.tight_layout()
    p = OUT + "/図_1語あたりの提示回数と正答.png"
    fig.savefig(p, dpi=110); print("図:", p)
    print("1語あたりの累計提示回数:", [round(v) for v in x])
    print("正答数:", y)
    print("最高は r3（1語あたり %d 回）で %d/48 = %.0f%%" % (round(x[2]), y[2], 100 * y[2] / 48))


if __name__ == "__main__":
    main()
