"""「Aが上」に見えたのは天井の差だった — を1枚の図にする。

【なぜ要るか、2026-07-29】ユーザーの指摘：
> 今のところB'でそれっぽく見えたのは、学習量が足りなくてまだ天井に
> 行ってないのに、Aの方が確立できているって判断してたってこと？

実際にログを見ると、その通りだった。さらに悪いことに：
  A  4ヶ月の体に載せ替えた時点で +26.5%。6000回学習しても +24.6%（微減）
  B  6000回で +13.7%、18000回で +22.4%。まだ上昇中

＝ 6000回で切って比べるのは「山頂にいる人」と「登り始めた人」を比べていた。

注意：前の図（自己モデルの持ち越し.png）は横軸を 250回刻みで決め打ちしていたため、
  B18k（30分＝18000回）を6000回として描いていた。横軸のバグ。
  ここでは log の life=Nmin から実際の学習回数を復元する（回数 = 分 × 600）。

使い方:
    .venv/Scripts/python.exe E/scripts/e_selfmodel_ceiling_figure.py
出力:
    E/logs/selfmodel/天井の差.png
"""

# 注意：古い方式（2026-07-30 に整理）。新しい実験は `run/main.py` を通す。
#   【経緯】目標Eの実験スクリプトが118本あり、うち66本が**独立に環境を組み立てていた**。
#     そのため「学習は関節モード（90関節を独立に駆動＝逸脱リスト 逸脱5）、
#     測定とViewerは筋肉モード（拮抗筋2本/関節）」という**別の体で動く**事故が起きた
#     （ユーザーの目視「視線誘導反射の実験の時とは動きが全然違う」で発覚。
#      実測で動きが人間の新生児の約3.3倍速かった）。
#   【設計と移行計画】`E/docs/実行基盤_設計.md`
#   注意：このファイルは**記録として残す**（削除しない方針）。
#     中の測り方は再利用できるので、プラグインへ移すときの元にする。
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

for _f in ("Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP"):
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.family"] = _f
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(_ROOT, "E", "logs", "selfmodel")
STEPS_PER_MIN = 600            # life=1min あたりの判断回数
C_A, C_B, C_B6 = "#2b6cb0", "#c05621", "#dd9a6a"


def curve(path):
    """(4ヶ月の体での学習回数, margin) の列。life=Nmin から回数を復元する。"""
    rows = []
    if not os.path.exists(path):
        return rows
    for l in open(path, encoding="utf-8", errors="replace"):
        m = re.search(r"life=(\d+)min.*?margin=([+-][\d.]+)%", l)
        if m:
            rows.append((int(m.group(1)) * STEPS_PER_MIN, float(m.group(2))))
    return rows


def main():
    seeds = [0, 1, 2]
    A = {s: curve(os.path.join(OUT, f"train_A_seed{s}.log")) for s in seeds}
    B18 = {s: curve(os.path.join(OUT, f"train_B18k_seed{s}.log")) for s in seeds}
    B6 = {s: curve(os.path.join(OUT, f"train_B_seed{s}.log")) for s in seeds}
    A = {s: v for s, v in A.items() if v}
    B18 = {s: v for s, v in B18.items() if len(v) >= 20}   # 実行中は除く
    B6 = {s: v for s, v in B6.items() if v}
    if not A or not B18:
        print("注意ログが足りない")
        return 1

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.2),
                                  gridspec_kw={"width_ratios": [1.55, 1.0],
                                               "wspace": 0.26})

    # ---- 左：学習曲線を同じ横軸（4ヶ月の体での学習回数）で重ねる ----
    for s, r in B18.items():
        ax.plot([x for x, _ in r], [y for _, y in r], color=C_B, alpha=0.8, lw=1.7)
    for s, r in A.items():
        ax.plot([x for x, _ in r], [y for _, y in r], color=C_A, alpha=0.8, lw=1.7)
    ax.plot([], [], color=C_A, lw=2.4, label="A 新生児の体で12000回 学んだ後")
    ax.plot([], [], color=C_B, lw=2.4, label="B 最初から4ヶ月の体")
    ax.axvline(6000, color="#c53030", ls="--", lw=1.3)
    ax.annotate("ここで切って比べていた\n（Bはまだ登っている途中）",
                xy=(6000, 4.0), xytext=(7600, 2.0), fontsize=9.5, color="#c53030",
                arrowprops=dict(arrowstyle="->", color="#c53030", lw=1.1))
    a0 = st.mean([r[0][1] for r in A.values()])
    ax.annotate(f"載せ替えた時点で {a0:+.1f}%\n（4ヶ月の体は1回も経験していない）",
                xy=(0, a0), xytext=(2100, a0 + 5.2), fontsize=9.5, color=C_A,
                arrowprops=dict(arrowstyle="->", color=C_A, lw=1.1))
    ax.axhline(0, color="#999", lw=0.8, ls=":")
    ax.set_xlabel("4ヶ月の体での学習回数（判断）")
    ax.set_ylabel("自己モデルの質（margin, %）")
    ax.set_title("① 同じ横軸で重ねると、比べていたものが見える",
                 loc="left", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.25)
    ax.set_ylim(-4, 34)

    # ---- 右：Aは伸びていない / Bは伸び続けている ----
    def tail(rows, k=6):
        return st.mean([y for _, y in rows[-k:]])

    def head(rows, k=2):
        return st.mean([y for _, y in rows[:k]])

    def at(rows, x):
        c = [y for xx, y in rows if xx <= x]
        return st.mean(c[-3:]) if c else float("nan")

    ga = [tail(r) - head(r) for r in A.values()]
    gb = [tail(r) - at(r, 6000) for r in B18.values()]
    xs = np.arange(2)
    vals = [st.mean(ga), st.mean(gb)]
    bars = ax2.bar(xs, vals, 0.45, color=[C_A, C_B])
    for i, (v, pts) in enumerate(zip(vals, [ga, gb])):
        ax2.text(i, v + (0.6 if v >= 0 else -1.4), f"{v:+.1f}",
                 ha="center", fontsize=12,
                 color="#276749" if v > 0 else "#c53030")
        ax2.scatter([i] * len(pts), pts, color="#222", s=18, zorder=3, alpha=0.7)
    ax2.axhline(0, color="#666", lw=1.0)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(["A\n4ヶ月で6000回\n学習した分の伸び",
                         "B\n6000→18000回\nの伸び"], fontsize=9.5)
    ax2.set_ylabel("margin の変化（%ポイント）")
    ax2.set_title("② Aはもう伸びない／Bはまだ伸びる", loc="left", fontsize=11)
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle("「新生児期を経た方が上」に見えたのは、"
                 "止まった側と登っている側を同じ回数で切っていたから",
                 fontsize=12.5, y=0.985)
    fig.text(0.008, 0.012,
             "margin＝自分の行動と他人の行動で予測させたときの予測誤差の差。"
             "点は各シード／横軸は life分×600 で復元（B18kは30分＝18000回）",
             fontsize=8, color="#555")
    path = os.path.join(OUT, "天井の差.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"→ {os.path.relpath(path, _ROOT)}")

    print("\n" + "=" * 66)
    print(f"{'':<24}{'開始':>9}{'6000回':>10}{'18000回':>10}")
    for s in sorted(A):
        r = A[s]
        print(f"A seed{s} 新生児期あり{head(r):>+11.1f}{tail(r):>+10.1f}{'(終)':>10}")
    for s in sorted(B18):
        r = B18[s]
        print(f"B seed{s} ゼロから  {head(r):>+11.1f}{at(r,6000):>+10.1f}"
              f"{tail(r):>+10.1f}")
    print("-" * 66)
    print(f"A の4ヶ月6000回での伸び   {st.mean(ga):+.1f} ポイント（伸びていない）")
    print(f"B の6000→18000回での伸び  {st.mean(gb):+.1f} ポイント（まだ伸びる）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
