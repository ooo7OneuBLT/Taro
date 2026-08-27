"""F2「見た物の名前を言う」本走行の結果を図にする（検証専用・実装本体には含めない）。

【入力】run/plugins/common/word_production.py が書き出す発話イベントCSV
  （F/logs/F2_初語検証/本走行300s/発話イベント_太郎.csv）。

【注意（司令塔からの指摘、2026-08-22）】vocal_tract.param_distanceの実測により、
  長音「ー」の調音パラメータが「あ」と完全に同一（(0,0,1,0)）と判明した。
  compute_imitation_reward の重み付き編集距離はこの2文字を区別できないため、
  「ぶーぶー」目標に対して「ぶあぶあ」が"完全一致"扱い（reward=1.0）になる一方、
  音声学的に正しい「ぶうぶう」はむしろ減点される（実測：あ↔ー距離0、う↔ー距離2）。
  ⇒ **「ぶーぶー」側のrewardは上達の証拠として使えない**。図1は「わんわん」側でのみ
  上達を判断できる形にする（ぶーぶー側は参考として薄く重ねるだけ）。
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo"]

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     os.pardir, os.pardir))

CSV_PATH = os.path.join(_ROOT, "E", "logs", "F2_初語検証", "本走行300s",
                        "発話イベント_太郎.csv")
OUT_DIR = os.path.join(_ROOT, "F", "logs", "F2_初語検証")


def load_rows():
    with open(CSV_PATH, encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def rolling_mean(xs, ys, window_n=5):
    """点数が少ないので、単純な移動平均（直近window_n点）で傾向を見る。"""
    out_x, out_y = [], []
    for i in range(len(xs)):
        lo = max(0, i - window_n + 1)
        out_x.append(xs[i])
        out_y.append(sum(ys[lo:i + 1]) / (i - lo + 1))
    return out_x, out_y


def fig1_time_trend(rows):
    """図1：時間軸で「言いたかった語」と「実際に出た音」の一致度(reward)を並べる。

    上達が見えるかを判断する主対象は「わんわん」（採点が壊れていない語）。
    「ぶーぶー」は参考として薄く重ねるだけ（長音「ー」＝「あ」の採点バグにより
    上達の証拠として使えないため、司令塔の指摘どおり主張には使わない）。
    """
    wan = [r for r in rows if r["target_word"] == "わんわん"]
    bu = [r for r in rows if r["target_word"] == "ぶーぶー"]

    fig, ax = plt.subplots(figsize=(10, 5))
    if bu:
        xs = [float(r["sim_sec"]) for r in bu]
        ys = [float(r["reward"]) for r in bu]
        ax.scatter(xs, ys, color="lightgray", marker="x", s=30,
                  label=f"ぶーぶー（n={len(bu)}、参考のみ：長音ー＝あの採点バグで"
                        f"上達の証拠にならない）")
    if wan:
        xs = [float(r["sim_sec"]) for r in wan]
        ys = [float(r["reward"]) for r in wan]
        ax.scatter(xs, ys, color="tab:blue", marker="o", s=45,
                  label=f"わんわん（n={len(wan)}、主判定）")
        rx, ry = rolling_mean(xs, ys, window_n=5)
        ax.plot(rx, ry, color="tab:blue", linewidth=2,
               label="わんわん・直近5発話の移動平均")
    ax.set_xlabel("sim_sec（シミュレーション時間 [秒]）")
    ax.set_ylabel("一致度 reward（言いたかった語 vs 実際に出た音）")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("F2 本走行300s：発話ごとの一致度の推移（主判定=わんわん）")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "図1_一致度の時間推移.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig2_word_by_toy(rows):
    """図2：見ていた物(toy)ごとに、選ばれた語(target_word=逆引きの結果)の内訳。

    「正しい語を選べているか」＝物→語の対応が合っているかを見る
    （設計がいう「逆向きテスト」＝いままでの語→物の逆）。
    """
    from collections import Counter
    by_toy = {"toy1": Counter(), "toy2": Counter()}
    for r in rows:
        toy = r["toy"]
        if toy in by_toy:
            by_toy[toy][r["target_word"]] += 1

    words = sorted({w for c in by_toy.values() for w in c})
    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(words))
    width = 0.35
    for i, (toy, label, color) in enumerate((
            ("toy1", "toy1（箱・正解=ぶーぶー）", "tab:orange"),
            ("toy2", "toy2（球・正解=わんわん）", "tab:blue"))):
        counts = [by_toy[toy].get(w, 0) for w in words]
        ax.bar([xi + (i - 0.5) * width for xi in x], counts, width=width,
              label=label, color=color)
    ax.set_xticks(list(x))
    ax.set_xticklabels(words)
    ax.set_ylabel("逆引きで選ばれた回数")
    ax.set_title("F2 本走行300s：見ていた物ごとに選ばれた語の内訳")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "図2_見ていた物ごとの語の内訳.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path, by_toy


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_rows()
    p1 = fig1_time_trend(rows)
    p2, by_toy = fig2_word_by_toy(rows)
    print(f"発話回数={len(rows)}")
    print(f"図1: {p1}")
    print(f"図2: {p2}")
    print("見ていた物ごとの語の内訳:", dict(by_toy))


if __name__ == "__main__":
    main()
