# -*- coding: utf-8 -*-
"""口元に報酬をつけると、頭に触れたうち口元の割合が上がるか。

2つの系統（ランダム側／学習進度側）で独立に確かめる。
同じ乱数の種で対にしてあるので、対応ありの検定を使う。

出力  E/docs/figures/自己接触立ち上がり_2026-08-13/口元報酬の効果_7シード.png
"""
import csv
import itertools
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

_R = r"C:\claude\AI\Taro"
D = os.path.join(_R, "E", "logs", "自己接触立ち上がり_2026-08-13")
OUTDIR = os.path.join(_R, "E", "docs", "figures", "自己接触立ち上がり_2026-08-13")
OUT = os.path.join(OUTDIR, "口元報酬の効果_7シード.png")

INK, INK2, MUTED, GRID = "#1a1a1a", "#4a4a4a", "#8a8a8a", "#e3e3e3"
COFF, CON = "#6b7280", "#c2410c"        # 報酬なし / 口元に報酬
TEAL = "#0f766e"

PAIRS = [("① 反射と振動子だけで動かす（内発的な報酬なし）",
          "条件1_純粋ランダム", "条件2_純粋ランダム_口元ON"),
         ("② 学習進度を報酬にして動かす",
          "条件3_進度あり", "条件4_進度あり_口元ON")]


def series(folder, seed):
    p = os.path.join(D, f"{folder}_seed{seed}", "selftouch_pace_trace.csv")
    m = h = 0.0
    sh = []
    for r in csv.DictReader(open(p, encoding="utf-8")):
        try:
            m += float(r["mouth_rising"])
            h += float(r["head_rising"])
            if float(r["step"]) >= 5000:
                sh.append(float(r["shoulder_abduction_deg"]))
        except (KeyError, ValueError, TypeError):
            continue
    return 100.0 * m / max(h, 1), float(np.mean(sh))


def wilcoxon_exact(d):
    """対応ありの符号順位検定（片側・全通り数え上げ）。"""
    rk = np.empty(len(d))
    rk[np.argsort(np.abs(d))] = np.arange(1, len(d) + 1)
    w = rk[d > 0].sum()
    hit = sum(1 for s in itertools.product([1, -1], repeat=len(d))
              if sum(r for r, g in zip(rk, s) if g > 0) >= w)
    return hit / 2 ** len(d)


fig = plt.figure(figsize=(14.5, 5.6))
gs = fig.add_gridspec(1, 3, wspace=0.30, left=0.05, right=0.985,
                      top=0.775, bottom=0.115)


def style(ax):
    ax.set_facecolor("white")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9.5, length=3)
    ax.grid(color=GRID, lw=0.9)
    ax.set_axisbelow(True)


store = []
for col, (title, foff, fon) in enumerate(PAIRS):
    ax = fig.add_subplot(gs[0, col])
    style(ax)
    off = np.array([series(foff, s)[0] for s in range(7)])
    on = np.array([series(fon, s)[0] for s in range(7)])
    store.append((off, on,
                  np.array([series(foff, s)[1] for s in range(7)]),
                  np.array([series(fon, s)[1] for s in range(7)])))
    d = on - off
    p = wilcoxon_exact(d)

    ax.axhline(16, color=TEAL, lw=1.4, ls="--", zorder=1)
    ax.text(2.42, 17.6, "口元の面積比 16%", fontsize=9, color=TEAL, ha="right")
    for s in range(7):
        up = on[s] > off[s]
        ax.plot([1, 2], [off[s], on[s]], color=(CON if up else MUTED),
                lw=1.8, alpha=0.6 if up else 0.32, zorder=2)
    # ラベルは下から順に最小間隔を空けて置き直す（重なり防止）
    order = np.argsort(on)
    ypos, prev = {}, -1e9
    for s in order:
        ypos[s] = prev = max(on[s], prev + 2.6)
    for s in order:
        ax.plot([2.03, 2.10], [on[s], ypos[s]], color=GRID, lw=0.9, zorder=1)
        ax.text(2.13, ypos[s], f"種{s}", fontsize=8.5, color=MUTED, va="center")
    ax.scatter([1] * 7, off, s=95, color=COFF, edgecolor="white",
               linewidth=1.8, zorder=3)
    ax.scatter([2] * 7, on, s=95, color=CON, edgecolor="white",
               linewidth=1.8, zorder=3)
    for xx, vals, c in ((1, off, COFF), (2, on, CON)):
        med = float(np.median(vals))
        ax.plot([xx - 0.17, xx + 0.17], [med, med], color=c, lw=3.2, zorder=4)
        ax.text(xx - 0.21, med, f"{med:.0f}%", fontsize=10.5, color=c,
                ha="right", va="center", fontweight="bold")
    ax.set_xlim(0.62, 2.46)
    ax.set_ylim(12, 95)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["口元に報酬なし", "口元に報酬 +0.2"],
                       fontsize=10.5, color=INK2)
    if col == 0:
        ax.set_ylabel("頭に触れたうち口元だった割合 [%]", fontsize=10.5, color=INK2)
    ax.set_title(title, fontsize=12, color=INK, loc="left", pad=9)
    n_up = int((d > 0).sum())
    ax.text(0.035, 0.945, f"7種中{n_up}種で上昇、中央値 {np.median(d):+.0f}ポイント",
            transform=ax.transAxes, fontsize=10.5, color=INK2)
    ax.text(0.035, 0.895, f"対応ありの検定  p = {p:.4f}",
            transform=ax.transAxes, fontsize=10,
            color=(CON if p < 0.05 else MUTED),
            fontweight=("bold" if p < 0.05 else "normal"))

# ── 右：姿勢では説明できないことを示す ─────────────────────
ax = fig.add_subplot(gs[0, 2])
style(ax)
labels = ["①反射と振動子", "②学習進度"]
xs = np.arange(2)
d_frac = [float(np.median(on - off)) for off, on, _, _ in store]
d_sh = [float(np.mean(shon - shoff)) for _, _, shoff, shon in store]
bars = ax.bar(xs, d_sh, 0.42, color="#7c9cbf", zorder=3)
ax.axhline(0, color=INK2, lw=1.3, zorder=4)
for rect, v in zip(bars, d_sh):
    ax.text(rect.get_x() + rect.get_width() / 2, v + (0.6 if v >= 0 else -0.6),
            f"{v:+.1f}度", ha="center", va="bottom" if v >= 0 else "top",
            fontsize=11.5, color=INK2, fontweight="bold")
ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=10.5, color=INK2)
ax.set_ylabel("肩の外転角度の変化［度］", fontsize=10.5, color=INK2)
ax.set_ylim(-11, 26)
ax.set_title("③ 姿勢のせいではない", fontsize=12, color=INK, loc="left", pad=9)
# 口元の割合は単位が違うので棒にせず、数字で並べる
ax.text(0.035, 0.96, "口元の割合はどちらも上がった", transform=ax.transAxes,
        fontsize=10.5, color=CON, va="top", fontweight="bold")
for i, v in enumerate(d_frac):
    ax.text(0.055, 0.895 - i * 0.062, f"{labels[i]}   {v:+.1f}ポイント",
            transform=ax.transAxes, fontsize=9.5, color=CON, va="top")
ax.text(0.035, 0.035,
        "けれど腕の開き方は①で開き②で閉じた。\n向きが揃わないので、姿勢では説明できない",
        transform=ax.transAxes, fontsize=9.5, color=MUTED, va="bottom")

fig.suptitle("口元に報酬をつけると、頭に触れたうち口元を触る割合が上がる"
             "（7種ずつ、2つの系統で独立に確認）",
             fontsize=13.5, color=INK, x=0.05, ha="left", y=0.945)
os.makedirs(OUTDIR, exist_ok=True)
fig.savefig(OUT, dpi=140, facecolor="white", bbox_inches="tight")
print(OUT)
