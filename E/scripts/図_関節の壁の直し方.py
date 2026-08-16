# -*- coding: utf-8 -*-
"""関節が可動域の端で「壁」になる問題と、その直し方を説明する図。

MuJoCoの関節制限には2つの別のパラメータがある。
  jnt_margin   限界の「手前どこから」効き始めるか   ← 太郎は 0.0（＝手前では何も起きない）
  jnt_solimp   限界を「超えた後」の反発の柔らかさ   ← 今回いじったのはこちら

出力  E/docs/figures/肩の可動域/関節の壁の直し方.png
"""
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

OUTDIR = r"C:\claude\AI\Taro\E\docs\figures\肩の可動域"
OUT = os.path.join(OUTDIR, "関節の壁の直し方.png")

INK, INK2, MUTED, GRID = "#1a1a1a", "#4a4a4a", "#8a8a8a", "#e3e3e3"
NG, MID, OK, HUMAN = "#c2410c", "#b45309", "#0f766e", "#0f766e"
LIM = 183.0

fig = plt.figure(figsize=(14.6, 5.4))
gs = fig.add_gridspec(1, 3, wspace=0.20, left=0.045, right=0.985,
                      top=0.72, bottom=0.15)

PANELS = [
    ("① もともとの太郎", NG,
     "限界まで何の抵抗もない。\n超えた瞬間に急に止まる",
     "腕は全速力で壁にぶつかり、\nそこに張り付いて動かなくなる"),
    ("② solimp を変えた（今回やったこと）", MID,
     "壁にクッションを貼ったが、\n当たるまでは相変わらず全速力",
     "止まる場所が183度から188度へ\nずれただけ。張り付きは残った"),
    ("③ margin を入れる（いま追加中）", OK,
     "限界の手前から抵抗が始まる。\n近づくほど強くなる",
     "人間の靭帯や関節包と同じ効き方。\n近づくにつれ自然に減速する"),
]


def style(ax):
    ax.set_facecolor("white")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9.5, length=3)
    ax.grid(color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    ax.set_xlim(160, 197)
    ax.set_ylim(-6, 118)


x = np.linspace(160, 197, 900)

for i, (title, col, how, result) in enumerate(PANELS):
    ax = fig.add_subplot(gs[0, i])
    style(ax)

    # 参考：人間の効き方（全パネルに薄く重ねる）
    human = 100.0 / (1.0 + np.exp(-(x - LIM) / 4.0))
    ax.plot(x, human, color=HUMAN, lw=1.6, ls=":", alpha=0.55, zorder=2,
            label="人間（模式）")

    if i == 0:
        y = np.where(x < LIM, 0.0, 100.0)
    elif i == 1:
        # 超えた後だけ少し柔らかい。手前は変わらずゼロ
        y = np.where(x < LIM, 0.0, 100.0 * (1 - np.exp(-(x - LIM) / 1.6)))
    else:
        # 限界の手前 margin の距離から立ち上がる
        MARGIN = 10.0
        y = np.clip((x - (LIM - MARGIN)) / MARGIN, 0, None) ** 2 * 100.0
        y = np.where(x < LIM - MARGIN, 0.0, np.minimum(y, 100.0))
        ax.axvspan(LIM - MARGIN, LIM, color=OK, alpha=0.10, zorder=1)
        ax.annotate("", xy=(LIM, 108), xytext=(LIM - MARGIN, 108),
                    arrowprops=dict(arrowstyle="<->", color=OK, lw=1.5))
        ax.text(LIM - MARGIN / 2, 112, "margin", fontsize=10, color=OK,
                ha="center", fontweight="bold")

    ax.plot(x, y, color=col, lw=3.0, zorder=4, label="太郎")
    ax.axvline(LIM, color=MUTED, lw=1.2, ls="--", zorder=3)
    ax.text(LIM + 0.7, -3, "可動域の\n限界 183度", fontsize=9, color=MUTED,
            va="bottom")

    ax.set_title(title, fontsize=12.5, color=col, loc="left", pad=10,
                 fontweight="bold")
    ax.set_xlabel("肩の外転角度［度］", fontsize=10, color=INK2)
    if i == 0:
        ax.set_ylabel("関節が押し返す力", fontsize=10.5, color=INK2)
        ax.legend(fontsize=9.5, frameon=False, loc="upper left")
    ax.text(0.035, -0.30, how, transform=ax.transAxes, fontsize=10.3,
            color=col, va="top", fontweight="bold")
    ax.text(0.035, -0.475, result, transform=ax.transAxes, fontsize=9.8,
            color=INK2, va="top")

fig.suptitle("関節の「壁」を直すには、いじる場所が2つある",
             fontsize=15, color=INK, x=0.045, ha="left", y=0.945)
fig.text(0.045, 0.855,
         "solimp ＝ 限界を「超えた後」の柔らかさ　／　"
         "margin ＝ 限界の「手前どこから」効き始めるか。"
         "太郎は margin が 0 だったので、手前では何も起きていなかった",
         fontsize=11.3, color=INK2, ha="left")

os.makedirs(OUTDIR, exist_ok=True)
fig.savefig(OUT, dpi=140, facecolor="white", bbox_inches="tight")
print(OUT)
