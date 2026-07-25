"""fidgety移行の結果を1枚のグラフで可視化する。"""
import os
import numpy as np
import matplotlib.pyplot as plt

# シード平均（e_fidgety_transition.py の 2026-07-26 実測値）
DATA = {
    "shoulder": {"age": [0.0, 1.5, 3.0, 5.0],
                 "rom":  [145.1, 15.4, 130.8, 114.4],
                 "beta": [1.48, 0.79, 1.41, 1.43],
                 "jerk": [45942, 74385, 56912, 60883]},
    "elbow":    {"age": [0.0, 1.5, 3.0, 5.0],
                 "rom":  [122.6, 41.2, 105.4, 83.3],
                 "beta": [1.13, 1.09, 1.09, 1.03],
                 "jerk": [47873, 70213, 42130, 42491]},
    "hip":      {"age": [0.0, 1.5, 3.0, 5.0],
                 "rom":  [20.1, 5.7, 7.2, 5.3],
                 "beta": [1.03, 0.79, 0.73, 0.76],
                 "jerk": [21018, 24128, 29930, 25207]},
    "knee":     {"age": [0.0, 1.5, 3.0, 5.0],
                 "rom":  [6.5, 1.6, 1.6, 0.1],
                 "beta": [0.35, 0.21, 0.15, 0.08],
                 "jerk": [17673, 10812, 10986, 962]},
}

COLORS = {"shoulder": "#3B82F6", "elbow": "#10B981",
          "hip": "#F59E0B", "knee": "#EF4444"}

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
metrics = [("rom",  "ROM (deg)   expected: DECREASE",  "log"),
           ("beta", "beta (spectral index)  expected: INCREASE (0.7->0.9)", "linear"),
           ("jerk", "jerk (deg/s^3)  expected: INCREASE", "log")]

for ax, (key, title, yscale) in zip(axes, metrics):
    for jt, d in DATA.items():
        ax.plot(d["age"], d[key], "o-", color=COLORS[jt], label=jt, linewidth=2, markersize=8)
    ax.set_xlabel("age (months)")
    ax.set_title(title)
    if yscale == "log":
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0.0, 1.5, 3.0, 5.0])

# 参考線: βの文献値
axes[1].axhline(0.686, color="gray", linestyle="--", alpha=0.6, label="Lopez 8w=0.686")
axes[1].axhline(0.877, color="gray", linestyle=":",  alpha=0.6, label="Lopez 30w=0.877")

axes[0].legend(loc="upper right", fontsize=9)
axes[1].legend(loc="upper right", fontsize=8)

fig.suptitle("writhing->fidgety transition (post muscle-fix, 2026-07-26)  "
             "3 seeds mean, no learning", fontsize=11)
fig.tight_layout()

out = os.path.join(os.path.dirname(__file__), os.pardir,
                   "docs", "figures", "fidgety_transition_20260726.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved:", os.path.abspath(out))
