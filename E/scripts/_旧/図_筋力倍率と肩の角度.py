# -*- coding: utf-8 -*-
"""筋力の倍率を変えると、肩の角度がどうなるかを図にする。

実測（scratchpad/limit_and_force.py）：
  各倍率で環境を作り、肩の外転筋にだけ最大指令(1.0)を10秒間出し続けた。
  「安静時」はリセット直後（指令を出す前）の角度。

出力  E/docs/figures/肩の可動域/筋力倍率と肩の角度.png
"""
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

OUTDIR = r"C:\claude\AI\Taro\E\docs\figures\肩の可動域"
OUT = os.path.join(OUTDIR, "筋力倍率と肩の角度.png")

INK, INK2, MUTED, GRID = "#1a1a1a", "#4a4a4a", "#8a8a8a", "#e3e3e3"
REST, MAXC, HUMAN, NOW = "#7c9cbf", "#c2410c", "#0f766e", "#c2410c"

# 倍率, 安静時, 最大指令10秒後
DATA = [(1, 20.5, 26.2), (3, 23.4, 39.4), (5, 31.4, 60.4), (10, 79.7, 110.3)]
H_MEAN_LO, H_MEAN_HI, H_MAX = 35, 47, 83     # 人間の自発運動（Flores-Santy 2025）

fig = plt.figure(figsize=(13.8, 5.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.26,
                      left=0.055, right=0.98, top=0.79, bottom=0.13)


def style(ax):
    ax.set_facecolor("white")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9.5, length=3)
    ax.grid(color=GRID, lw=0.9)
    ax.set_axisbelow(True)


# ── 左：倍率ごとの安静時と最大 ──────────────────────
ax = fig.add_subplot(gs[0, 0])
style(ax)
xs = np.arange(len(DATA))
w = 0.36
rest = [d[1] for d in DATA]
mx = [d[2] for d in DATA]

ax.axhspan(H_MEAN_LO, H_MEAN_HI, color=HUMAN, alpha=0.13, zorder=1)
ax.axhline(H_MAX, color=HUMAN, lw=1.6, ls="--", zorder=2)
ax.set_xlim(-0.62, 4.30)
ax.text(3.62, H_MAX, "人間の最大\n83度", fontsize=9.5, color=HUMAN,
        ha="left", va="center")
ax.text(3.62, (H_MEAN_LO + H_MEAN_HI) / 2, "人間の平均\n35〜47度", fontsize=9.5,
        color=HUMAN, ha="left", va="center")

b1 = ax.bar(xs - w / 2, rest, w, color=REST, zorder=3, label="何もしていないとき")
b2 = ax.bar(xs + w / 2, mx, w, color=MAXC, zorder=3,
            label="外転筋に最大の指令を10秒")
for rect, v in list(zip(b1, rest)) + list(zip(b2, mx)):
    ax.text(rect.get_x() + rect.get_width() / 2, v + 2.5, f"{v:.0f}度",
            ha="center", fontsize=10, color=INK2, fontweight="bold")
ax.set_xticks(xs)
ax.set_xticklabels([f"{d[0]}倍" for d in DATA], fontsize=11, color=INK2)
ax.set_xlabel("四肢の筋力の倍率（1倍＝MIMoの既定値）", fontsize=10.5, color=INK2)
ax.set_ylabel("肩の外転角度［度］", fontsize=10.5, color=INK2)
ax.set_ylim(0, 130)
ax.set_title("① 筋力を上げると開く。ただし最大指令でも110度まで",
             fontsize=12.5, color=INK, loc="left", pad=9)
ax.legend(fontsize=10, frameon=False, loc="upper left")
ax.annotate("10倍だと、何もしなくても\n既に80度開いている",
            xy=(3 - w / 2, 79.7), xytext=(-116, 22), textcoords="offset points",
            fontsize=9.8, color=NOW, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=NOW, lw=1.4))

# ── 右：可動域の制限のかかり方 ───────────────────────
ax = fig.add_subplot(gs[0, 1])
style(ax)
x = np.linspace(150, 190, 400)
LIM = 183.0

# 太郎（MuJoCo）：width=0.001rad≒0.06度 手前から立ち上がる
w_taro = np.rad2deg(0.001)
taro = 1.0 / (1.0 + np.exp(-(x - LIM) / (w_taro / 4)))
# 人間：靭帯や関節包がじわじわ効く（模式。実測ではない）
w_human = 25.0
human = 1.0 / (1.0 + np.exp(-(x - LIM) / (w_human / 6)))

ax.plot(x, human * 100, color=HUMAN, lw=2.6, label="人間（模式図）", zorder=3)
ax.plot(x, taro * 100, color=NOW, lw=2.6, label="太郎（実測値から）", zorder=4)
ax.axvline(LIM, color=MUTED, lw=1.2, ls=":", zorder=2)
ax.text(LIM - 1.2, 103, "可動域の端 183度", fontsize=9.5, color=MUTED,
        ha="right", va="top")
ax.set_xlabel("肩の外転角度［度］", fontsize=10.5, color=INK2)
ax.set_ylabel("関節が押し返す力（端での力を100とした割合）", fontsize=10.5,
              color=INK2)
ax.set_ylim(-4, 112)
ax.set_title("② 制限のかかり方が違う", fontsize=12.5, color=INK, loc="left", pad=9)
ax.legend(fontsize=10, frameon=False, loc="center left")
ax.text(0.035, 0.44,
        "太郎は限界を超えるまで抵抗ゼロ。\n"
        "限界の手前では何も起きない\n"
        f"（{w_taro:.2f}度という値は「超えた後の\n"
        "柔らかさ」で、手前には効かない）\n\n"
        "人間は靭帯や関節包が\n手前からじわじわ効く\n"
        "（このカーブは模式。実測ではない）",
        transform=ax.transAxes, fontsize=9.3, color=INK2, va="top")

fig.suptitle("筋力10倍は「動く量」だけでなく「何もしていないときの姿勢」まで変えていた",
             fontsize=13.5, color=INK, x=0.055, ha="left", y=0.935)
os.makedirs(OUTDIR, exist_ok=True)
fig.savefig(OUT, dpi=140, facecolor="white", bbox_inches="tight")
print(OUT)
