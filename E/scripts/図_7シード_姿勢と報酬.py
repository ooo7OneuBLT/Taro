# -*- coding: utf-8 -*-
"""7シードで見た「姿勢」「立ち上がり」「口元の狙い」の3つ。"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

D = r"C:\claude\AI\Taro\E\logs\自己接触立ち上がり_2026-08-13"
OUTDIR = r"C:\claude\AI\Taro\E\docs\figures\自己接触立ち上がり_2026-08-13"

INK, INK2, MUTED = "#1a1a1a", "#4a4a4a", "#8a8a8a"
C1, C2 = "#6b7280", "#c2410c"          # 報酬なし / 口元報酬あり
GRID = "#e3e3e3"
COLS = ["step", "mouth_rising", "head_rising",
        "dist_hand_head_cm", "shoulder_abduction_deg"]


def load(f):
    p = os.path.join(D, f, "selftouch_pace_trace.csv")
    if not os.path.exists(p):
        return None
    o = {c: [] for c in COLS}
    with open(p, encoding="utf-8") as fp:
        for r in csv.DictReader(fp):
            try:
                v = [float(r[c]) for c in COLS]
            except (KeyError, ValueError, TypeError):
                continue
            for c, x in zip(COLS, v):
                o[c].append(x)
    return {c: np.asarray(v) for c, v in o.items()} if o["step"] else None


def stats(folder):
    out = []
    for s in range(7):
        d = load(f"{folder}_seed{s}")
        if d is None:
            continue
        t = d["step"] >= 5000
        h = ~t
        m1, m2 = d["mouth_rising"][h].sum(), d["mouth_rising"][t].sum()
        out.append(dict(seed=s, m1=float(m1), m2=float(m2),
                        ratio=float(m2 / m1) if m1 else np.nan,
                        sh2=float(d["shoulder_abduction_deg"][t].mean()),
                        frac=100.0 * d["mouth_rising"].sum()
                        / max(d["head_rising"].sum(), 1)))
    return out


a = stats("条件1_純粋ランダム")            # 口元報酬なし
b = stats("条件2_純粋ランダム_口元ON")     # 口元報酬あり
allr = a + b

fig = plt.figure(figsize=(14.5, 5.4))
gs = fig.add_gridspec(1, 3, wspace=0.32, left=0.055, right=0.985,
                      top=0.80, bottom=0.13)


def style(ax):
    ax.set_facecolor("white")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9.5, length=3)
    ax.grid(color=GRID, lw=0.9)
    ax.set_axisbelow(True)


# ── 左：姿勢と接触回数（強い関係）──────────────────────────
ax = fig.add_subplot(gs[0, 0])
style(ax)
for rows, col, lab in ((a, C1, "口元の報酬なし"), (b, C2, "口元の報酬あり")):
    ax.scatter([r["sh2"] for r in rows], [r["m2"] for r in rows],
               s=95, color=col, edgecolor="white", linewidth=1.8,
               label=lab, zorder=3)
x = np.array([r["sh2"] for r in allr])
y = np.array([r["m2"] for r in allr])
m = x < 100                                  # 突出1本を除いた集団
k = np.polyfit(x[m], y[m], 1)
xs = np.linspace(x[m].min() - 2, x[m].max() + 2, 50)
ax.plot(xs, np.polyval(k, xs), color=MUTED, lw=1.6, ls="--", zorder=2)
r_all = np.corrcoef(x, y)[0, 1]
r_wo = np.corrcoef(x[m], y[m])[0, 1]
ax.annotate("この1本だけ突出\n（条件2 seed0）",
            xy=(x[y.argmax()], y.max()), xytext=(-40, -62),
            textcoords="offset points", fontsize=9, color=INK2,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.0))
ax.set_xlabel("肩の外転角度（後半5000tickの平均）[度]", fontsize=10, color=INK2)
ax.set_ylabel("後半に口元へ触れた回数", fontsize=10, color=INK2)
ax.set_title("① 腕が上がっているほど、口元によく当たる",
             fontsize=12.5, color=INK, loc="left", pad=9)
ax.text(0.30, 0.30, f"r = {r_wo:+.2f}",
        transform=ax.transAxes, fontsize=11, color=INK2)
ax.text(0.30, 0.235,
        f"突出1本を除いた13本。破線もその13本\n14本すべてなら r = {r_all:+.2f}",
        transform=ax.transAxes, fontsize=9, color=MUTED, va="top")
ax.legend(fontsize=9.5, frameon=False, loc="lower right")

# ── 中：姿勢と「伸び」（関係が無い）────────────────────────
ax = fig.add_subplot(gs[0, 1])
style(ax)
ax.axhline(1.0, color=MUTED, lw=1.2, ls=":", zorder=1)
ax.text(x.max(), 1.03, "後半＝前半", fontsize=9, color=MUTED, ha="right")
for rows, col, lab in ((a, C1, "口元の報酬なし"), (b, C2, "口元の報酬あり")):
    ax.scatter([r["sh2"] for r in rows], [r["ratio"] for r in rows],
               s=95, color=col, edgecolor="white", linewidth=1.8, zorder=3)
yr = np.array([r["ratio"] for r in allr])
r2 = np.corrcoef(x, yr)[0, 1]
ax.set_xlabel("肩の外転角度（後半5000tickの平均）[度]", fontsize=10, color=INK2)
ax.set_ylabel("後半 ÷ 前半（1より上なら増えた）", fontsize=10, color=INK2)
ax.set_title("② でも「増えたかどうか」とは無関係",
             fontsize=12.5, color=INK, loc="left", pad=9)
ax.text(0.03, 0.86, f"r = {r2:+.2f}", transform=ax.transAxes,
        fontsize=11, color=INK2)
ax.text(0.03, 0.80, "姿勢が良くても、後半に伸びるとは限らない",
        transform=ax.transAxes, fontsize=9, color=MUTED)

# ── 右：口元の割合（同じ種でペア比較）──────────────────────
ax = fig.add_subplot(gs[0, 2])
style(ax)
ax.axhline(16, color="#0f766e", lw=1.4, ls="--", zorder=1)
ax.text(2.46, 17.5, "口元の面積比 16%", fontsize=9, color="#0f766e", ha="right")
fa = {r["seed"]: r["frac"] for r in a}
fb = {r["seed"]: r["frac"] for r in b}
seeds = sorted(set(fa) & set(fb))
for s in seeds:
    up = fb[s] > fa[s]
    ax.plot([1, 2], [fa[s], fb[s]], color=(C2 if up else MUTED),
            lw=1.7, alpha=0.55 if up else 0.35, zorder=2)
# ラベルが重ならないよう、下から順に最小間隔を空けて置き直す
_lab = sorted(seeds, key=lambda s: fb[s])
_ypos, _prev = {}, -1e9
for s in _lab:
    yv = max(fb[s], _prev + 2.4)
    _ypos[s], _prev = yv, yv
for s in _lab:
    ax.plot([2.03, 2.10], [fb[s], _ypos[s]], color=GRID, lw=0.9, zorder=1)
    ax.text(2.13, _ypos[s], f"種{s}", fontsize=8.5, color=MUTED, va="center")
ax.scatter([1] * len(fa), list(fa.values()), s=95, color=C1,
           edgecolor="white", linewidth=1.8, zorder=3)
ax.scatter([2] * len(fb), list(fb.values()), s=95, color=C2,
           edgecolor="white", linewidth=1.8, zorder=3)
for xx, vals, col in ((1, list(fa.values()), C1), (2, list(fb.values()), C2)):
    med = float(np.median(vals))
    ax.plot([xx - 0.17, xx + 0.17], [med, med], color=col, lw=3.0, zorder=4)
    ax.text(xx - 0.21, med, f"{med:.0f}%", fontsize=10, color=col,
            ha="right", va="center", fontweight="bold")
ax.set_xlim(0.62, 2.5)
ax.set_xticks([1, 2])
ax.set_xticklabels(["報酬なし", "口元に報酬"], fontsize=10.5, color=INK2)
ax.set_ylabel("頭に触れたうち口元だった割合 [%]", fontsize=10, color=INK2)
ax.set_title("③ 報酬をつけると口元の割合が上がる",
             fontsize=12.5, color=INK, loc="left", pad=9)
ax.text(0.03, 0.93, "7種中5種で上昇、中央値 +11ポイント",
        transform=ax.transAxes, fontsize=10, color=INK2)
ax.text(0.03, 0.875, "同じ種の対どうしで検定すると p = 0.078（5%に届かない）",
        transform=ax.transAxes, fontsize=8.8, color=MUTED)

fig.suptitle("同じ乱数の種で7本ずつ。「姿勢が良いと点火する」という見立ては当たらず、"
             "効いていたのは報酬のほうだった",
             fontsize=13.5, color=INK, x=0.055, ha="left", y=0.955)
os.makedirs(OUTDIR, exist_ok=True)
out = os.path.join(OUTDIR, "7シード_姿勢と報酬のどちらが効いたか.png")
fig.savefig(out, dpi=140, facecolor="white", bbox_inches="tight")
print(out)
