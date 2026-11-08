"""バニラモデルの学習の曲線を1枚の図にする。python plot_vanilla.py <metrics.csv> <出力 png>"""
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


VIEWS = (("top", "上から"), ("side", "横から"))
COL = {"top": "#2a78d6", "side": "#eb6834"}
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), dpi=120)

ax = axes[0]
for k, ja in VIEWS:
    ax.plot(x, smooth(col(f"{k}_success")), color=COL[k], lw=2, label=ja)
ax.axhline(0.031, color="#888", ls="--", lw=1, label="でたらめ（3.1%）")
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
ax.set_title("ゲームの中の回ごとの成功率\n（探し方を覚えると2回目以降が上がる）")
ax.set_xlabel("学習の歩数（百万）")
ax.set_ylim(0, 1)
ax.legend(frameon=False, fontsize=8, ncol=2)
ax.grid(alpha=0.25)

ax = axes[2]
names = (("good", "効くボタン"), ("bad", "効かないボタン"), ("deco", "飾り"), ("none", "なし"))
cols = ["#1baf7a", "#e87ba4", "#eda100", "#888"]
for (k2, ja2), c in zip(names, cols):
    ax.plot(x, smooth(col(f"top_first_{k2}")), color=c, lw=2, label=f"{ja2}（上から）")
    ax.plot(x, smooth(col(f"side_first_{k2}")), color=c, lw=1.5, ls="--", label=f"{ja2}（横から）")
ax.set_title("最初に踏んだ物の割合")
ax.set_xlabel("学習の歩数（百万）")
ax.set_ylim(0, 1)
ax.legend(frameon=False, fontsize=8, ncol=2)
ax.grid(alpha=0.25)

fig.suptitle("工夫なしのモデル（PPO＋CNN＋GRU、前の行動・報酬つき）の学習。線は15更新の移動平均", fontsize=12)
fig.tight_layout()
fig.savefig(sys.argv[2])
print("saved", sys.argv[2])
