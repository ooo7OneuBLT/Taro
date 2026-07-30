"""「新生児期を経ると自己モデルが良くなる」を1枚の図にする。

【この実験、2026-07-29】
  A（継続）  新生児の体で12000回 → 4ヶ月の体へ載せ替えて6000回
  B（ゼロ）  最初から4ヶ月の体で6000回
  どちらも4ヶ月の体で評価。3シード（0/1/2）。

【結果】3シードすべてで A > B（+5.6〜+9.6%）。
＝発達の順序を守ることに、測定できる利点がある。

⚠️グラフは1ランのデータだけで描く方針だが、本図は**条件間の比較そのもの**が
主張なので、シードごとの点を全部出したうえで平均を示す（混ぜて平滑化しない）。

使い方:
    .venv/Scripts/python.exe E/scripts/e_selfmodel_transfer_figure.py
出力:
    E/logs/selfmodel/自己モデルの持ち越し.png
"""
import os
import re
import sys
import statistics as st

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))

import matplotlib                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt       # noqa: E402
import numpy as np                    # noqa: E402

# 日本語フォント（Windows）
for _f in ("Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP"):
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.family"] = _f
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(_ROOT, "E", "logs", "selfmodel")
# (ラベル, ログのパス) — シードごと
LOGS_A = {0: os.path.join(OUT, "train_A_seed0.log"),
          1: os.path.join(OUT, "train_A_seed1.log"),
          2: os.path.join(OUT, "train_A_seed2.log")}
LOGS_B = {0: os.path.join(OUT, "train_B_seed0.log"),
          1: os.path.join(OUT, "train_B_seed1.log"),
          2: os.path.join(OUT, "train_B_seed2.log")}
# seed0 はバックグラウンド実行だったので、CSVから拾う代替パスを持つ
FALLBACK = {"A0": None, "B0": None}

C_A, C_B = "#2b6cb0", "#c05621"       # 青＝継続 / 橙＝ゼロから


def load(path):
    """ログから (判断回数, margin, corr) の列を作る。"""
    rows = []
    if not path or not os.path.exists(path):
        return rows
    for l in open(path, encoding="utf-8", errors="replace"):
        m = re.search(r"classify=([\d.]+)%.*?margin=([+-][\d.]+)%.*?corr=([-\d.]+)"
                      r".*?persist=([\d.]+)%", l)
        if m:
            rows.append([float(x) for x in m.groups()])
    return rows


def tail_mean(rows, col=1, k=8):
    return st.mean([r[col] for r in rows[-k:]]) if rows else float("nan")


def main():
    A = {s: load(p) for s, p in LOGS_A.items()}
    B = {s: load(p) for s, p in LOGS_B.items()}
    A = {s: v for s, v in A.items() if v}
    B = {s: v for s, v in B.items() if v}
    if not A or not B:
        print("⚠️ログが見つからない。先に学習を回すこと")
        print(f"   探した場所: {OUT}")
        return 1

    fig = plt.figure(figsize=(13.5, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.0, 1.0], wspace=0.32)

    # ---- 左：学習曲線 ----
    ax = fig.add_subplot(gs[0])
    for s, rows in A.items():
        x = np.arange(len(rows)) * 250
        ax.plot(x, [r[1] for r in rows], color=C_A, alpha=0.75, lw=1.6)
    for s, rows in B.items():
        x = np.arange(len(rows)) * 250
        ax.plot(x, [r[1] for r in rows], color=C_B, alpha=0.75, lw=1.6)
    ax.plot([], [], color=C_A, lw=2.4, label="A 新生児の体で学んでから4ヶ月へ")
    ax.plot([], [], color=C_B, lw=2.4, label="B 最初から4ヶ月の体で学ぶ")
    ax.axhline(0, color="#999", lw=0.8, ls=":")
    ax.set_xlabel("4ヶ月の体での学習回数（判断）")
    ax.set_ylabel("自己モデルの質（margin, %）")
    ax.set_title("① 学習の経過（線は3シード分）", loc="left", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.25)

    # ---- 中：シードごとの対比 ----
    ax2 = fig.add_subplot(gs[1])
    seeds = sorted(set(A) & set(B))
    xs = np.arange(len(seeds))
    va = [tail_mean(A[s]) for s in seeds]
    vb = [tail_mean(B[s]) for s in seeds]
    ax2.bar(xs - 0.19, va, 0.36, color=C_A, label="A 継続")
    ax2.bar(xs + 0.19, vb, 0.36, color=C_B, label="B ゼロから")
    for i, (a, b) in enumerate(zip(va, vb)):
        ax2.text(i - 0.19, a + 0.5, f"{a:+.1f}", ha="center", fontsize=9, color=C_A)
        ax2.text(i + 0.19, b + 0.5, f"{b:+.1f}", ha="center", fontsize=9, color=C_B)
        ax2.annotate("", xy=(i + 0.19, b), xytext=(i - 0.19, a),
                     arrowprops=dict(arrowstyle="-", color="#666", lw=0.8, ls=":"))
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"シード{s}" for s in seeds])
    ax2.set_ylabel("margin（終盤8点の平均, %）")
    ax2.set_title("② 3シードすべてで A > B", loc="left", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.25)
    ax2.set_ylim(0, max(va + vb) * 1.28)

    # ---- 右：差 ----
    ax3 = fig.add_subplot(gs[2])
    d = [a - b for a, b in zip(va, vb)]
    ax3.bar(xs, d, 0.5, color="#38a169")
    for i, v in enumerate(d):
        ax3.text(i, v + 0.2, f"+{v:.1f}", ha="center", fontsize=10, color="#276749")
    ax3.axhline(st.mean(d), color="#276749", ls="--", lw=1.2,
                label=f"平均 +{st.mean(d):.1f}%")
    ax3.axhline(0, color="#999", lw=0.8)
    ax3.set_xticks(xs)
    ax3.set_xticklabels([f"シード{s}" for s in seeds])
    ax3.set_ylabel("A − B（%）")
    ax3.set_title("③ 新生児期を経る利得", loc="left", fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(axis="y", alpha=0.25)
    ax3.set_ylim(0, max(d) * 1.3)

    fig.suptitle("新生児の体で先に学ぶと、4ヶ月の体での自己モデルが良くなる"
                 "（乳児シミュレータ MIMo・3シード）", fontsize=13, y=0.99)
    fig.text(0.008, 0.015,
             "margin＝自分の行動と他人の行動で予測させたときの予測誤差の差。"
             "大きいほど「自分の体を分かっている」。0%＝区別できていない／"
             "どちらも4ヶ月の体で6000回学習・同一シードで対比",
             fontsize=8, color="#555")
    path = os.path.join(OUT, "自己モデルの持ち越し.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"→ {os.path.relpath(path, _ROOT)}")

    print("\n" + "=" * 62)
    print(f"{'':>8}{'A 継続':>11}{'B ゼロから':>13}{'差':>9}")
    for s, a, b in zip(seeds, va, vb):
        print(f"シード{s:<3}{a:>+10.1f}%{b:>+12.1f}%{a-b:>+8.1f}")
    print("-" * 62)
    print(f"{'平均':>8}{st.mean(va):>+10.1f}%{st.mean(vb):>+12.1f}%{st.mean(d):>+8.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
