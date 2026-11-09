"""バニラモデルの学習の曲線を1枚の図にする。
python plot_vanilla.py <metrics.csv> <出力 png> [手書きのプレイヤーの results.json] [その条件の名前（例 end_episode・ボタン3個）] [前回の metrics.csv]
前回の metrics.csv を渡すと、成功率と当たりの割合（2〜4回目の平均）に前回の線を灰色で重ねる。"""
import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

fm.fontManager.addfont("/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf")
plt.rcParams["font.family"] = "IPAPGothic"

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
x = np.array([float(r["steps"]) for r in rows]) / 1e6


def col(k):
    return np.array([float(r[k]) for r in rows])


def smooth(y, w=15):
    if len(y) < w:
        return y
    k = np.ones(w) / w
    return np.convolve(np.pad(y, (w // 2, w - 1 - w // 2), mode="edge"), k, mode="valid")


VIEWS = tuple((k, ja) for k, ja in (("top", "上から"), ("side", "横から")) if any(float(r.get(f"{k}_n", 1)) > 0 for r in rows))
COL = {"top": "#2a78d6", "side": "#eb6834"}
ref = None
if len(sys.argv) > 4:
    import json

    ref = json.load(open(sys.argv[3], encoding="utf-8"))["結果"]
    cond = sys.argv[4]
has_share = "top_goodshare_ep1" in rows[0]
prev = None
if len(sys.argv) > 5:
    prev_rows = list(csv.DictReader(open(sys.argv[5], encoding="utf-8")))
    prev = (np.array([float(r["steps"]) for r in prev_rows]) / 1e6, prev_rows)


def pcol(k):
    return np.array([float(r[k]) for r in prev[1]])
fig, axes = plt.subplots(1, 4 if has_share else 3, figsize=(22 if has_share else 17, 4.8), dpi=120)

ax = axes[0]
for k, ja in VIEWS:
    ax.plot(x, smooth(col(f"{k}_success")), color=COL[k], lw=2, label=ja)
ax.axhline(0.031, color="#888", ls="--", lw=1, label="でたらめ（3.1%）")
if prev:
    ax.plot(prev[0], smooth(pcol("top_success")), color="#9a9a9a", lw=1.2, label="前回・上から")
    ax.plot(prev[0], smooth(pcol("side_success")), color="#9a9a9a", lw=1.2, ls="--", label="前回・横から")
ax.set_title("成功率（練習用の絵柄）")
ax.set_xlabel("学習の歩数（百万）")
ax.set_ylim(0, 1)
ax.legend(frameon=False)
ax.grid(alpha=0.25)

ax = axes[1]
styles = ["-", "--", "-.", ":"]
for k, ja in VIEWS:
    for e in range(4):
        ax.plot(x, smooth(col(f"{k}_success_ep{e + 1}")), color=COL[k], ls=styles[e], lw=1.6, label=f"{ja} {e + 1}回目")
if ref:
    for k, ja in VIEWS:
        vja = "上から" if k == "top" else "横から"
        for pl in ("覚える人", "総当たりの人"):
            for v in [xx["成功率"] for xx in ref[f"{cond}・{vja}"][pl]["回ごと"]]:
                ax.axhline(v, color=COL[k], ls=":", lw=0.8, alpha=0.6)
ax.set_title("ゲームの中の回ごとの成功率\n（探し方を覚えると2回目以降が上がる。点線は手書きの覚える人・総当たりの人）")
ax.set_xlabel("学習の歩数（百万）")
ax.set_ylim(0, 1)
ax.legend(frameon=False, fontsize=8, ncol=2)
ax.grid(alpha=0.25)

ax = axes[2]
names = (("good", "効くボタン"), ("bad", "効かないボタン"), ("deco", "飾り"), ("none", "なし"))
cols = ["#1baf7a", "#e87ba4", "#eda100", "#888"]
for (k2, ja2), c in zip(names, cols):
    for k, ja in VIEWS:
        ax.plot(x, smooth(col(f"{k}_first_{k2}")), color=c, lw=2 if k == "top" else 1.5, ls="-" if k == "top" else "--", label=f"{ja2}（{ja}）")
    if "top_nopress" in rows[0] and k2 == "none":
        for k, ja in VIEWS:
            ax.plot(x, smooth(col(f"{k}_nopress")), color="#333", lw=2, ls=":", label=f"押さずに終わった回（{ja}）")
ax.set_title("最初に踏んだ物の割合")
ax.set_xlabel("学習の歩数（百万）")
ax.set_ylim(0, 1)
ax.legend(frameon=False, fontsize=8, ncol=2)
ax.grid(alpha=0.25)

if has_share:
    ax = axes[3]
    for k, ja in VIEWS:
        for e in range(4):
            ax.plot(x, smooth(col(f"{k}_goodshare_ep{e + 1}")), color=COL[k], ls=styles[e], lw=1.6, label=f"{ja} {e + 1}回目")
        if ref:
            vja = "上から" if k == "top" else "横から"
            for pl in ("覚える人", "総当たりの人"):
                vals = [xx["最初に押したボタンが当たりの割合"] for xx in ref[f"{cond}・{vja}"][pl]["回ごと"]]
                for v in vals:
                    ax.axhline(v, color=COL[k], ls=":", lw=0.8, alpha=0.6)
    if prev and "top_goodshare_ep2" in prev[1][0]:
        for k, ls in (("top", "-"), ("side", "--")):
            avg = (pcol(f"{k}_goodshare_ep2") + pcol(f"{k}_goodshare_ep3") + pcol(f"{k}_goodshare_ep4")) / 3
            ax.plot(prev[0], smooth(avg), color="#9a9a9a", lw=1.2, ls=ls, label=f"前回・{'上から' if k == 'top' else '横から'}（2〜4回目の平均）")
    ax.set_title("最初に押したボタンが当たりの割合（回ごと）\n（理想: 1回目33%→2回目50%以上→ほぼ100%。点線は手書き）")
    ax.set_xlabel("学習の歩数（百万）")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(alpha=0.25)

fig.suptitle("工夫なしのモデル（PPO＋CNN＋GRU、前の行動・報酬つき）の学習。線は15更新の移動平均", fontsize=12)
fig.tight_layout()
fig.savefig(sys.argv[2])
print("saved", sys.argv[2])
