# -*- coding: utf-8 -*-
"""肩の関節角度が、実際にどの姿勢を指すのかを図にする。

MuJoCoの関節角度と解剖学の外転角度が一致しているかを確かめるため、
関節を実際に動かして手の位置を計算した結果を描く（scratchpad/shoulder_axis.py の実測値）。

出力  E/docs/figures/肩の可動域/肩の角度が何を意味するか.png
"""
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

OUTDIR = r"C:\claude\AI\Taro\E\docs\figures\肩の可動域"
OUT = os.path.join(OUTDIR, "肩の角度が何を意味するか.png")

INK, INK2, MUTED, GRID = "#1a1a1a", "#4a4a4a", "#8a8a8a", "#e3e3e3"
HUMAN, OVER, BODY = "#0f766e", "#c2410c", "#b8b8b8"

# 実測値：体幹の座標系で見た右手の位置（左右, 上下）と、頭との距離
DATA = [(-84, +0.097, +0.008, 11.9),
        (-45, +0.033, -0.081, 16.2),
        (0,   -0.089, -0.108, 20.6),
        (45,  -0.195, -0.041, 22.8),
        (90,  -0.222, +0.081, 22.2),
        (135, -0.154, +0.187, 19.0),
        (183, -0.024, +0.212, 13.8)]
HEAD_Z = 0.077          # 体幹から見た頭の位置（上下）

fig = plt.figure(figsize=(13.5, 5.8))
gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.24,
                      left=0.05, right=0.975, top=0.80, bottom=0.12)


def style(ax):
    ax.set_facecolor("white")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9.5, length=3)
    ax.grid(color=GRID, lw=0.9)
    ax.set_axisbelow(True)


# ── 左：手がどこへ動くか（体を正面から見た図）────────────────
ax = fig.add_subplot(gs[0, 0])
style(ax)
# 体の目印
ax.plot([0, 0], [-0.16, 0.13], color=BODY, lw=14, solid_capstyle="round",
        zorder=1, alpha=0.55)
ax.scatter([0], [HEAD_Z], s=1300, color=BODY, zorder=2, alpha=0.55)
ax.text(0, HEAD_Z, "頭", fontsize=11, color="#5a5a5a", ha="center",
        va="center", zorder=3, fontweight="bold")
ax.text(-0.028, 0.020, "体幹", fontsize=10, color="#5a5a5a", ha="center",
        va="center", zorder=3)

ys = np.array([d[1] for d in DATA])
zs = np.array([d[2] for d in DATA])
ax.plot(ys, zs, color=MUTED, lw=1.4, ls="--", zorder=3)
for deg, y, z, dist in DATA:
    over = deg > 83                       # 人間の自発運動の実測上限
    c = OVER if over else HUMAN
    ax.scatter([y], [z], s=130, color=c, edgecolor="white", linewidth=1.9,
               zorder=4)
    off = (10, 6) if deg not in (0, 183) else ((10, -14) if deg == 0 else (-4, 10))
    ax.annotate(f"{deg}度", (y, z), textcoords="offset points", xytext=off,
                fontsize=10, color=c, fontweight="bold", zorder=5)
ax.set_xlabel("体の左右方向 ← 右　　　左 →　［m］", fontsize=10, color=INK2)
ax.set_ylabel("体の上下方向　足側 ← ／ → 頭側　［m］", fontsize=10, color=INK2)
ax.set_title("① 関節の角度と、実際の腕の位置", fontsize=12.5, color=INK,
             loc="left", pad=9)
ax.invert_xaxis()      # 右手を右に描く
ax.set_aspect("equal")
ax.text(0.98, 0.04,
        "0度＝腕を体側へ／90度＝真横へ／183度＝万歳\n"
        "解剖学の外転角度の定義と一致していた",
        transform=ax.transAxes, fontsize=9.5, color=INK2, va="bottom",
        ha="right")

# ── 右：人間の実測と重ねる ─────────────────────────
ax = fig.add_subplot(gs[0, 1])
style(ax)
bars = [("人間の可動域\n（関節が動く限界）", 180, HUMAN, "受動的に動かした限界"),
        ("太郎の可動域\n（シーンの設定）", 183, HUMAN, "MIMoの既定値をそのまま"),
        ("人間の自発運動\n（実際に使う範囲）", 83, HUMAN, "最大。平均は35〜47度"),
        ("太郎の自発運動\n（実際に出た値）", 183, OVER, "可動域の端に張り付く")]
ypos = np.arange(len(bars))[::-1]
for y, (lab, v, c, note) in zip(ypos, bars):
    ax.barh(y, v, height=0.52, color=c, zorder=3,
            alpha=1.0 if c == OVER else 0.85)
    ax.text(v + 4, y, f"{v}度", va="center", fontsize=11, color=INK2,
            fontweight="bold")
    ax.text(6, y, note, va="center", fontsize=9, zorder=4,
            color=("white" if v > 90 else "white"))
ax.axvline(83, color=OVER, lw=1.5, ls="--", zorder=5)
ax.text(88, 2.62, "人間が自発運動で\n使う上限 83度", fontsize=9.5, color=OVER,
        va="center")
ax.set_yticks(ypos)
ax.set_yticklabels([b[0] for b in bars], fontsize=10, color=INK2)
ax.set_xlabel("肩の外転角度［度］", fontsize=10, color=INK2)
ax.set_xlim(0, 218)
ax.set_title("② 可動域は合っている。使い方が違う", fontsize=12.5, color=INK,
             loc="left", pad=9)

fig.suptitle("肩の角度の定義は人間と一致していた。ずれているのは「可動域の広さ」ではなく"
             "「自発運動でどこまで使うか」",
             fontsize=13.5, color=INK, x=0.05, ha="left", y=0.945)
os.makedirs(OUTDIR, exist_ok=True)
fig.savefig(OUT, dpi=140, facecolor="white", bbox_inches="tight")
print(OUT)
