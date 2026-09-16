# -*- coding: utf-8 -*-
"""「腕が開いてしまう」のは筋力の問題ではなかった、という発見を図にする。

根拠：
  太郎側  scratchpad/limit_and_force.py の実測（安静時の角度・端での張り付き）
  人間側  Maekawa & Ochiai (1975) の筋電図による実測（抄録のみ確認＝Tier2）

出力  E/docs/figures/四肢の筋力/筋力ではなく受動的な性質.png
"""
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch  # noqa: E402

OUTDIR = r"C:\claude\AI\Taro\E\docs\figures\四肢の筋力"
OUT = os.path.join(OUTDIR, "筋力ではなく受動的な性質.png")

INK, INK2, MUTED, GRID = "#1a1a1a", "#4a4a4a", "#8a8a8a", "#e3e3e3"
BODY, TARO, HUMAN, ACCENT = "#c9d3dc", "#c2410c", "#0f766e", "#b45309"

fig = plt.figure(figsize=(13.6, 6.9))
gs = fig.add_gridspec(1, 2, wspace=0.10, left=0.035, right=0.975,
                      top=0.74, bottom=0.20)


def baby(ax, deg, col, muscle_on):
    """仰向けの赤ちゃん。degは肩の外転角度。muscle_onで筋活動の表示を変える。"""
    import numpy as np
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.5, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((-0.30, -0.72), 0.60, 1.10,
                                boxstyle="round,pad=0.06", facecolor=BODY,
                                edgecolor="none", zorder=2))
    ax.add_patch(Circle((0, 0.68), 0.29, facecolor=BODY, edgecolor="none",
                        zorder=3))
    for sx in (-0.16, 0.16):
        ax.plot([sx, sx], [-0.70, -1.24], color=BODY, lw=11,
                solid_capstyle="round", zorder=1)
    ax.plot([-0.28, -0.42], [0.30, -0.32], color=BODY, lw=11,
            solid_capstyle="round", zorder=1)

    sh = np.array([0.28, 0.30])
    d = np.array([np.cos(np.deg2rad(deg - 90)), np.sin(np.deg2rad(deg - 90))])
    if muscle_on:                                  # 太郎：まっすぐ開く
        hand = sh + d * 0.84
        ax.plot([sh[0], hand[0]], [sh[1], hand[1]], color=col, lw=11,
                solid_capstyle="round", zorder=4)
    else:                                          # 人間：肘を曲げて引き寄せる
        elbow = sh + d * 0.42
        hand = elbow + np.array([-0.30, 0.16])
        ax.plot([sh[0], elbow[0]], [sh[1], elbow[1]], color=col, lw=11,
                solid_capstyle="round", zorder=4)
        ax.plot([elbow[0], hand[0]], [elbow[1], hand[1]], color=col, lw=11,
                solid_capstyle="round", zorder=4)
    ax.add_patch(Circle(hand, 0.095, facecolor=col, edgecolor="white",
                        linewidth=2.0, zorder=5))
    return sh


# ── 左：太郎 ────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
sh = baby(ax, 80, TARO, muscle_on=True)
ax.text(0, 1.22, "太郎", fontsize=19, color=TARO, ha="center",
        fontweight="bold")
ax.text(0, -1.40,
        "筋力を10倍にしたら、何もしていなくても\n"
        "腕が80度開いた状態になった（実測）",
        fontsize=11, color=INK2, ha="center", va="top")
# 筋肉が引っぱっている印
ax.annotate("", xy=(sh[0] + 0.40, sh[1] + 0.30), xytext=(sh[0], sh[1]),
            arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2.6,
                            mutation_scale=19), zorder=6)
ax.text(sh[0] + 0.50, sh[1] + 0.40, "筋肉が\n引っぱっている", fontsize=10.5,
        color=ACCENT, ha="left", va="center", fontweight="bold")

# ── 右：人間の新生児 ─────────────────────────────
ax = fig.add_subplot(gs[0, 1])
sh = baby(ax, 22, HUMAN, muscle_on=False)
ax.text(0, 1.22, "人間の新生児", fontsize=19, color=HUMAN, ha="center",
        fontweight="bold")
ax.text(0, -1.40,
        "腕を曲げて体に引き寄せているのに、\n"
        "筋肉の電気的な活動は「低い」（筋電図の実測）",
        fontsize=11, color=INK2, ha="center", va="top")
ax.text(sh[0] + 0.46, sh[1] + 0.34,
        "筋肉は力を出していない\n子宮で丸まっていた名残の\n"
        "「関節の硬さ」で保たれている",
        fontsize=10.5, color=HUMAN, ha="left", va="center", fontweight="bold")

fig.suptitle("同じ「腕の姿勢」でも、その成り立ちが正反対だった",
             fontsize=15.5, color=INK, x=0.035, ha="left", y=0.955)
fig.text(0.035, 0.885,
         "太郎の腕が開くのは筋力を上げたから。人間の腕が曲がるのは筋力ではなく"
         "関節そのものの受動的な性質による。"
         "つまり筋力をいくら調整しても、この差は埋まらない",
         fontsize=11.5, color=INK2, ha="left")
fig.text(0.035, 0.045,
         "人間側の根拠：Maekawa & Ochiai (1975) 生後48時間以内の新生児の筋電図実測。"
         "見た目の屈曲は強いが筋活動はむしろ低く、著者は子宮内姿勢に由来する"
         "受動的な拘縮によるものと結論している［抄録のみ確認＝Tier2］",
         fontsize=9.4, color=MUTED, ha="left")

os.makedirs(OUTDIR, exist_ok=True)
fig.savefig(OUT, dpi=140, facecolor="white", bbox_inches="tight")
print(OUT)
