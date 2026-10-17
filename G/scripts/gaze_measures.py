"""段階6：太郎の「見方」を、赤ちゃん（Yu & Smith 2011）と WOLVES（Bhat ほか 2022）の表2と同じ物差しで測る。

入力：core_v2/tools/smith_yu.py --ticks の記録（<dir>/**/<mode>_seed<種>.json と _回ごと.csv）
数え方（記録の look 列＝その回に視線から10度以内で一番近い物。空＝どれも見ていない。0.1秒ごと）：
* 学ぶ画面（4秒）：見ていた時間（look が空でない回×0.1秒）・注視の回数（同じ物を続けて見たひとかたまりを1回）・
  1回の注視の長さ（ひとかたまりの長さの平均）
* テスト（8秒）：物を見ていた時間（正解＋別の物。smith_yu は最初の語から数えるので窓は7.5秒）・
  正解を見た割合・覚えた語の数（その語の2試行で、正解を見た合計＞別の物を見た合計なら「覚えた」）
* よく学ぶ太郎（S）とそうでない太郎（W）：覚えた語が6語中 STRONG_MIN（既定4）以上なら S。
  ［仮置き：Yu & Smith 2011 の分け方は原文で確かめていない。WOLVES の原文にも基準の数は書いていない］
人間と WOLVES の値は WOLVES 原文の表2（doc/文献調査/原文/交差状況学習の計算モデル_2026-10-17/wolves.txt）から写した。

使い方:  python G/scripts/gaze_measures.py G/logs/段階6_見方_2026-10-17 [--mode B] [--strong-min 4]
出力：<dir>/見方の物差し.json と 図_見方_赤ちゃんと比べる.png
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys

import numpy as np

TICK = 0.1
# WOLVES 表2（S＝よく学ぶ、W＝そうでない）。赤ちゃんの値は「範囲」（Smith & Yu 2008 と Yu & Smith 2011）
HUMAN = {
    "テスト：物を見ていた秒（8秒中）": (5.92, 6.10), "テスト：正解を見た割合": (0.54, 0.60),
    "テスト：覚えた語（6語中）": (3.5, 4.0), "よく学ぶ子の割合": (0.67, 0.67),
    "学ぶ画面：見ていた秒（4秒中）": (2.96, 3.07),
    "学ぶ画面：注視の回数 S": (2.75, 2.75), "学ぶ画面：注視の回数 W": (3.82, 3.82),
    "学ぶ画面：1回の注視の秒 S": (1.69, 1.69), "学ぶ画面：1回の注視の秒 W": (1.21, 1.21),
}
WOLVES = {"テスト：物を見ていた秒（8秒中）": 6.26, "テスト：正解を見た割合": 0.54, "テスト：覚えた語（6語中）": 4.0,
          "よく学ぶ子の割合": 0.74, "学ぶ画面：見ていた秒（4秒中）": 3.01}


def runs_of(seq):
    """[物の番号 or None] → [(物, 長さ)] 続けて同じ物を見たひとかたまり"""
    out, cur, n = [], None, 0
    for k in seq:
        if k is not None and k == cur:
            n += 1
        else:
            if cur is not None:
                out.append((cur, n))
            cur, n = k, (1 if k is not None else 0)
    if cur is not None:
        out.append((cur, n))
    return out


def one(json_path, strong_min):
    info = json.load(open(json_path, encoding="utf-8"))
    rows = list(csv.DictReader(open(json_path.replace(".json", "_回ごと.csv"), encoding="utf-8")))
    train = {}
    for r in rows:
        if r["phase"] == "train":
            train.setdefault(int(r["slide"]), []).append(int(r["look"]) if r["look"] != "" else None)
    look_s, nfix, fixlen = [], [], []
    for seq in train.values():
        rs = runs_of(seq)
        look_s.append(sum(n for _, n in rs) * TICK)
        nfix.append(len(rs))
        fixlen += [n * TICK for _, n in rs]
    per_word = {}
    for t in info["trials"]:
        w = per_word.setdefault(t["word"], [0, 0])
        w[0] += t["target"]; w[1] += t["distractor"]
    learned = sum(1 for a, b in per_word.values() if a > b)
    test_look = [(t["target"] + t["distractor"]) * TICK for t in info["trials"]]
    return {"seed": info["seed"], "share": info["target_share"], "learned": learned, "strong": learned >= strong_min,
            "train_look_s": float(np.mean(look_s)), "train_fix_n": float(np.mean(nfix)),
            "train_fix_s": float(np.mean(fixlen)) if fixlen else 0.0, "test_look_s": float(np.mean(test_look)),
            "n_learn": info["n_learn"]}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    d = argv[0]
    mode = argv[argv.index("--mode") + 1] if "--mode" in argv else "B"
    strong_min = int(argv[argv.index("--strong-min") + 1]) if "--strong-min" in argv else 4
    files = sorted(f for f in glob.glob(os.path.join(d, "**", f"{mode}_seed*.json"), recursive=True)
                   if os.path.exists(f.replace(".json", "_回ごと.csv")))
    per = [one(f, strong_min) for f in files]
    S = [p for p in per if p["strong"]]
    W = [p for p in per if not p["strong"]]
    m = lambda xs, k: round(float(np.mean([x[k] for x in xs])), 2) if xs else None   # noqa: E731
    taro = {"テスト：物を見ていた秒（8秒中）": m(per, "test_look_s"), "テスト：正解を見た割合": m(per, "share"),
            "テスト：覚えた語（6語中）": m(per, "learned"), "よく学ぶ子の割合": round(len(S) / len(per), 2) if per else None,
            "学ぶ画面：見ていた秒（4秒中）": m(per, "train_look_s"),
            "学ぶ画面：注視の回数 S": m(S, "train_fix_n"), "学ぶ画面：注視の回数 W": m(W, "train_fix_n"),
            "学ぶ画面：1回の注視の秒 S": m(S, "train_fix_s"), "学ぶ画面：1回の注視の秒 W": m(W, "train_fix_s")}
    out = {"n": len(per), "mode": mode, "strong_min": strong_min, "n_strong": len(S), "n_weak": len(W),
           "table": {k: {"太郎": taro[k], "赤ちゃん": HUMAN[k], "WOLVES": WOLVES.get(k)} for k in HUMAN},
           "per_taro": per}
    # 見方と学びの関係（太郎どうし）：学ぶ画面の注視の回数・長さと、テストで正解を見た割合の順位相関
    def spearman(a, b):
        if len(a) < 3:
            return None
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return round(float(np.corrcoef(ra, rb)[0, 1]), 3)
    out["関係"] = {"注視の回数と正解の割合": spearman([p["train_fix_n"] for p in per], [p["share"] for p in per]),
                 "1回の注視の長さと正解の割合": spearman([p["train_fix_s"] for p in per], [p["share"] for p in per])}
    print(f"  見方と学び（順位相関）：{out['関係']}")
    with open(os.path.join(d, "見方の物差し.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"太郎 {len(per)} 人（よく学ぶ S {len(S)}・そうでない W {len(W)}、S＝6語中{strong_min}語以上）")
    for k in HUMAN:
        h = HUMAN[k]
        hs = f"{h[0]}" if h[0] == h[1] else f"{h[0]}〜{h[1]}"
        print(f"  {k:22s} 太郎 {taro[k]}   赤ちゃん {hs}   WOLVES {WOLVES.get(k, '—')}")
    try:
        plot(out, d)
    except Exception as e:  # noqa: BLE001
        print("図は作れなかった:", e)
    return out


def plot(out, d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "sans-serif"]
    keys = list(out["table"])
    fig, axs = plt.subplots(3, 3, figsize=(12, 9))
    for ax, k in zip(axs.flat, keys):
        row = out["table"][k]
        h = row["赤ちゃん"]
        vals = [row["太郎"] or 0, (h[0] + h[1]) / 2, row["WOLVES"] or 0]
        bars = ax.bar(["太郎", "赤ちゃん", "WOLVES"], vals, color=["#2e86c1", "#e67e22", "#95a5a6"])
        if h[0] != h[1]:
            ax.errorbar([1], [(h[0] + h[1]) / 2], yerr=[[(h[1] - h[0]) / 2]], color="k", capsize=6)
        if row["WOLVES"] is None:
            bars[2].set_alpha(0.15)
        for b, v in zip(bars, vals):
            if v:
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(k, fontsize=10)
    fig.suptitle(f"太郎（{out['n']}人）の見方を、赤ちゃん（Smith & Yu 2008・Yu & Smith 2011）と WOLVES と比べる", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "図_見方_赤ちゃんと比べる.png"), dpi=110)


if __name__ == "__main__":
    main()
