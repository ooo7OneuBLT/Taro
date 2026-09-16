# -*- coding: utf-8 -*-
"""「肩183度」がどの姿勢を指すのかを、寝ている赤ちゃんの絵で示す。

角度は実測（scratchpad/shoulder_axis.py）で確かめた手の位置に対応する。
仰向けなので、上から見下ろした図になる。

出力  E/docs/figures/肩の可動域/肩183度とはどの姿勢か.png
"""
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Wedge   # noqa: E402

OUTDIR = r"C:\claude\AI\Taro\E\docs\figures\肩の可動域"
OUT = os.path.join(OUTDIR, "肩183度とはどの姿勢か.png")

INK, INK2, MUTED, GRID = "#1a1a1a", "#4a4a4a", "#8a8a8a", "#e3e3e3"
BODY, ARM_OK, ARM_NG = "#c9d3dc", "#0f766e", "#c2410c"

# 角度ごとの説明（角度, 見出し, 説明, 人間の範囲内か）
POSES = [(0, "0度", "腕を体の横に\nぴったり下ろす", True),
         (45, "45度", "少し開く\n人間の自発運動の平均", True),
         (83, "83度", "ほぼ真横\n人間の自発運動の最大", True),
         (110, "110度", "太郎が外転筋の\n最大指令で届く角度", False),
         (183, "183度", "頭の上へ万歳\n可動域の端", False)]

fig = plt.figure(figsize=(14.2, 6.4))
gs = fig.add_gridspec(1, 5, wspace=0.06, left=0.02, right=0.985,
                      top=0.78, bottom=0.04)


def draw_baby(ax, deg, ok):
    """仰向けの赤ちゃんを上から見た図。右腕だけ角度をつける。"""
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.55, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # 胴体・頭・脚
    ax.add_patch(FancyBboxPatch((-0.30, -0.75), 0.60, 1.15,
                                boxstyle="round,pad=0.06", facecolor=BODY,
                                edgecolor="none", zorder=2))
    ax.add_patch(Circle((0, 0.72), 0.30, facecolor=BODY, edgecolor="none",
                        zorder=3))
    for sx in (-0.16, 0.16):
        ax.plot([sx, sx], [-0.72, -1.30], color=BODY, lw=11,
                solid_capstyle="round", zorder=1)
    # 左腕（動かさない。体側）
    ax.plot([-0.28, -0.42], [0.32, -0.34], color=BODY, lw=11,
            solid_capstyle="round", zorder=1)

    # 右腕：肩を支点に、0度＝足側、90度＝真横、180度＝頭側
    sh = np.array([0.28, 0.32])
    L1, L2 = 0.44, 0.40                     # 上腕・前腕
    th = np.deg2rad(deg - 90)               # 0度で真下（足側）を向くように
    d = np.array([np.sin(th + np.pi / 2), -np.cos(th + np.pi / 2)])
    d = np.array([np.cos(np.deg2rad(deg - 90)), np.sin(np.deg2rad(deg - 90))])
    elbow = sh + d * L1
    hand = elbow + d * L2
    col = ARM_OK if ok else ARM_NG
    ax.plot([sh[0], hand[0]], [sh[1], hand[1]], color=col, lw=11,
            solid_capstyle="round", zorder=4)
    ax.add_patch(Circle(hand, 0.10, facecolor=col, edgecolor="white",
                        linewidth=2.0, zorder=5))

    # 角度の目印（肩を中心とした扇）
    ax.add_patch(Wedge(sh, 0.30, -90, deg - 90, facecolor=col, alpha=0.16,
                       edgecolor="none", zorder=3))
    ax.plot([sh[0], sh[0]], [sh[1], sh[1] - 0.34], color=MUTED, lw=1.1,
            ls=":", zorder=3)


for i, (deg, title, note, ok) in enumerate(POSES):
    ax = fig.add_subplot(gs[0, i])
    draw_baby(ax, deg, ok)
    col = ARM_OK if ok else ARM_NG
    ax.text(0, 1.40, title, fontsize=17, color=col, ha="center",
            va="bottom", fontweight="bold")
    ax.text(0, -1.44, note, fontsize=10.3, color=INK2, ha="center", va="top")

fig.text(0.02, 0.955,
         "「肩183度」とは、右の肩の1つの関節の角度のこと。"
         "仰向けなので、頭の上へ万歳した姿勢を指す",
         fontsize=14, color=INK, ha="left")
fig.text(0.02, 0.885,
         "可動域を超えているのではない。可動域の端がちょうど183度で、"
         "そこに到達して張り付いていた（緑＝人間が実際に使う範囲、橙＝それを超える）",
         fontsize=10.8, color=INK2, ha="left")
fig.text(0.02, 0.022, "点線は0度の向き（腕を足のほうへ下ろした状態）。"
         "扇はそこから何度開いたかを表す。左腕は比較のため動かしていない",
         fontsize=9.3, color=MUTED, ha="left")

os.makedirs(OUTDIR, exist_ok=True)
fig.savefig(OUT, dpi=140, facecolor="white", bbox_inches="tight")
print(OUT)
