# -*- coding: utf-8 -*-
"""目立ちの地図を、録画した視界で試す（太郎は走らせない）。

使い方: python F/scripts/f96_salience_probe.py <視界動画.mp4> <出力PNG>
`動画_視界.mp4` は「三人称(560px) ＋ 左目の実入力(448px) ＋ 字幕」の合成なので、
左目の部分だけを切り出して地図に通す。
"""
import sys, time
import numpy as np, cv2
sys.path.insert(0, "taro_core/src")
sys.stdout.reconfigure(encoding="utf-8")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic"]
from brain.midbrain.salience_map import SalienceMap

vid, out = sys.argv[1], sys.argv[2]
cap = cv2.VideoCapture(vid); frames = []
while True:
    ok, f = cap.read()
    if not ok: break
    frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)[0:448, 560:560 + 448])
cap.release()
print("読めたコマ数", len(frames))

sm = SalienceMap(cell=28)
# 全コマを順に通して時間を測る（動きの特徴を効かせるため連続で回す）
t0 = time.perf_counter(); results = []
for f in frames:
    results.append(sm.update(cv2.resize(f, (224, 224))))
dt = (time.perf_counter() - t0) / len(frames) * 1000
print("1コマあたり %.1f ms" % dt)

picks = [int(len(frames) * r) for r in (0.09, 0.20, 0.36, 0.55, 0.66, 0.78)]
fig, axes = plt.subplots(2, len(picks), figsize=(3.9 * len(picks), 8.8))
for k, i in enumerate(picks):
    img = cv2.resize(frames[i], (224, 224)); res = results[i]
    axes[0, k].imshow(img); axes[0, k].set_title("コマ %d" % i, fontsize=13)
    axes[0, k].plot(*res["peak"], marker="+", ms=22, mew=3.5, color="#1d3557")
    axes[0, k].axis("off")
    axes[1, k].imshow(res["salience"], cmap="YlOrRd", vmin=0, vmax=1)
    axes[1, k].plot(res["peak_cell"][0], res["peak_cell"][1], marker="+",
                    ms=20, mew=3.5, color="#1d3557")
    axes[1, k].set_xticks([]); axes[1, k].set_yticks([])
    # いちばん高い升で、どの特徴がいちばん効いたか（升の値そのもので比べる）
    c, r = res["peak_cell"]
    contrib = sorted(((float(v[r, c]), kk) for kk, v in res["channels"].items()),
                     reverse=True)
    axes[1, k].set_title("1位の升で効いた特徴：%s (%.2f)" % (contrib[0][1], contrib[0][0]),
                         fontsize=11)
axes[0, 0].set_ylabel("左目の視界", fontsize=14)
axes[1, 0].set_ylabel("目立ちの地図", fontsize=14)
fig.suptitle("目立ちの地図（28×28升目）　1コマ %.1f ms" % dt, fontsize=17, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(out, dpi=115, bbox_inches="tight")
print("saved", out)
