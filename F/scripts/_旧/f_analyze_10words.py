# -*- coding: utf-8 -*-
"""F2-20（10語）テストの集計。混同行列と、そっくりペアの成績を図にする。

読むもの : F/logs/F2-20_10語_テスト/<個体>/発話イベント.csv
           （target_word=太郎が選んだ語。テストでは親は黙っているので、
             「正解」はフォルダ名の個体が属するカテゴリから決める）
出すもの : F/logs/F2-20_10語_テスト/図_混同行列.png と、要約の表（stdout）

    .venv/Scripts/python.exe F/scripts/f_analyze_10words.py
"""
import os
import sys
import io
import csv
import glob

sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:\claude\AI\Taro")

WORDS = ["わんわん", "にゃんにゃん", "ぶーぶー", "でんしゃ", "りんご",
         "ボール", "くつ", "ばなな", "コップ", "ぼうし"]
# 太郎の耳には長音「ー」が無く、「ぶーぶー」は「ぶうぶう」、「ボール」は「ボル」として
# 学習される（F2-19でも同様）。産出された内部表記を正解の表記に戻す対応表。
ALIAS = {"ぶうぶう": "ぶーぶー", "ボル": "ボール"}
CAT_OF = {"犬": "わんわん", "猫": "にゃんにゃん", "車": "ぶーぶー", "電車": "でんしゃ",
          "りんご": "りんご", "ボール": "ボール", "くつ": "くつ", "ばなな": "ばなな",
          "コップ": "コップ", "ぼうし": "ぼうし"}
PAIRS = [("わんわん", "にゃんにゃん", "犬-猫"),
         ("ぶーぶー", "でんしゃ", "車-電車"),
         ("りんご", "ボール", "りんご-ボール")]
DIR = "F/logs/F2-20_10語_テスト"


def cat_of_individual(name):
    # 「電車4」→でんしゃ（「車」より長い名前から先に照合する）
    for stem in sorted(CAT_OF, key=len, reverse=True):
        if name.startswith(stem):
            return CAT_OF[stem]
    raise RuntimeError("カテゴリ不明: " + name)


def main():
    conf = {w: {v: 0 for v in WORDS} for w in WORDS}   # 正解カテゴリ → 言った語
    rows = []
    for d in sorted(glob.glob(DIR + "/*/")):
        ind = os.path.basename(d.rstrip("/\\"))
        if ind.startswith("図"):
            continue
        truth = cat_of_individual(ind)
        f = os.path.join(d, "発話イベント.csv")
        if not os.path.exists(f):
            print("！発話イベントが無い:", ind)
            continue
        n = ok = 0
        sims = []
        with io.open(f, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                w = ALIAS.get(r["target_word"], r["target_word"])
                if w not in conf[truth]:
                    conf[truth][w] = 0
                conf[truth][w] += 1
                n += 1
                ok += int(w == truth)
                sims.append(float(r["sim"]))
        rows.append((ind, truth, n, ok, sum(sims) / max(len(sims), 1)))

    print("%-8s %-10s %6s %8s %8s" % ("個体", "正解", "発話数", "正解率", "確信度"))
    for ind, truth, n, ok, s in rows:
        print("%-8s %-10s %6d %7.0f%% %8.3f" % (ind, truth, n, 100.0 * ok / max(n, 1), s))
    tot = sum(r[2] for r in rows)
    hit = sum(r[3] for r in rows)
    print("%-8s %-10s %6d %7.1f%%" % ("合計", "", tot, 100.0 * hit / max(tot, 1)))

    print("\nそっくりペアの間違い方")
    for a, b, label in PAIRS:
        na = sum(conf[a].values())
        nb = sum(conf[b].values())
        print("  %-10s %s側: %s と誤答 %d/%d ／ %s側: %s と誤答 %d/%d"
              % (label, a, b, conf[a].get(b, 0), na, b, a, conf[b].get(a, 0), nb))

    # ---- 混同行列の図 ----
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, rcParams
    for c in ("Yu Gothic", "Meiryo", "MS Gothic"):
        if any(c in f.name for f in font_manager.fontManager.ttflist):
            rcParams["font.family"] = c
            break
    M = np.zeros((len(WORDS), len(WORDS)))
    for i, t in enumerate(WORDS):
        n = sum(conf[t].values())
        for j, w in enumerate(WORDS):
            M[i, j] = 100.0 * conf[t].get(w, 0) / max(n, 1)
    fig, ax = plt.subplots(figsize=(8.6, 7.6))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(WORDS)))
    ax.set_xticklabels(WORDS, rotation=45, ha="right")
    ax.set_yticks(range(len(WORDS)))
    ax.set_yticklabels(WORDS)
    ax.set_xlabel("太郎が言った語")
    ax.set_ylabel("見せた物（初見の個体）")
    for i in range(len(WORDS)):
        for j in range(len(WORDS)):
            if M[i, j] >= 1:
                ax.text(j, i, "%.0f" % M[i, j], ha="center", va="center",
                        color="white" if M[i, j] > 55 else "#123", fontsize=9)
    ax.set_title("F2-20：10語の混同行列（％・初見の個体4・5で言わせた）")
    fig.colorbar(im, shrink=.8)
    fig.tight_layout()
    out = DIR + "/図_混同行列.png"
    fig.savefig(out, dpi=110)
    print("\n図:", out)


if __name__ == "__main__":
    main()
