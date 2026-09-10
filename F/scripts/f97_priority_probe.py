# -*- coding: utf-8 -*-
"""場所の優先度地図（復帰抑制つき）を、録画した視界で試す（太郎は走らせない）。

使い方: python F/scripts/f97_priority_probe.py <視界動画.mp4> <出力PNG>
見る点：①注意が同じ升に貼り付かないか ②抑制が薄れて戻ってこられるか
        ③目が動いたとき、抑制の地図が一緒にずれるか
"""
import sys, time
import numpy as np, cv2
sys.path.insert(0, "taro_core/src")
sys.stdout.reconfigure(encoding="utf-8")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic"]
from brain.midbrain.salience_map import SalienceMap
from brain.cerebral_cortex.parietal_lobe.spatial_priority_map import SpatialPriorityMap

vid, out = sys.argv[1], sys.argv[2]
cap = cv2.VideoCapture(vid); frames = []
while True:
    ok, f = cap.read()
    if not ok: break
    frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)[0:448, 560:560 + 448])
cap.release()

sm = SalienceMap(cell=28)
pm = SpatialPriorityMap(cell=28, img_size=224.0)
t0 = time.perf_counter(); recs = []
for f in frames:
    img = cv2.resize(f, (224, 224))
    s = sm.update(img)
    p = pm.update(s["salience"], dt=0.1)
    recs.append((img, s, p))
dt = (time.perf_counter() - t0) / len(frames) * 1000

wins = [p["winner_cell"] for _, _, p in recs]
stay = sum(1 for i in range(1, len(wins)) if wins[i] == wins[i - 1]) / (len(wins) - 1)
uniq = len(set(wins))
switch = sum(1 for _, _, p in recs if p["switched"])
print("1コマ %.1f ms／同じ升に留まった割合 %.0f%%／選ばれた升の種類 %d／切り替え %d回"
      % (dt, 100 * stay, uniq, switch))

# 抑制が効いているか：抑制なしとの比較
pm2 = SpatialPriorityMap(cell=28, img_size=224.0, ior_gain=0.0)
sm2 = SalienceMap(cell=28)
w2 = []
for f in frames:
    s = sm2.update(cv2.resize(f, (224, 224)))
    w2.append(pm2.update(s["salience"], dt=0.1)["winner_cell"])
stay2 = sum(1 for i in range(1, len(w2)) if w2[i] == w2[i - 1]) / (len(w2) - 1)
print("抑制なしのとき：同じ升に留まった割合 %.0f%%／升の種類 %d" % (100 * stay2, len(set(w2))))

# 目が動いたときに抑制がずれるか（合成の確認）
pm3 = SpatialPriorityMap(cell=28, img_size=224.0)
flat = np.zeros((28, 28), dtype=np.float32); flat[14, 14] = 1.0
pm3.update(flat, dt=0.1)                       # (14,14) が勝って抑制が乗る
before = int(np.argmax(pm3.ior) % 28)
pm3.update(np.zeros((28, 28), dtype=np.float32), shift_px=(32.0, 0.0), dt=0.0)
after = int(np.argmax(pm3.ior) % 28)
print("目のずれ 32px（=4升）で、抑制の山が列 %d → %d へ移った（期待 +4）" % (before, after))

picks = [int(len(recs) * r) for r in (0.20, 0.205, 0.21, 0.215, 0.36, 0.55)]
fig, axes = plt.subplots(3, len(picks), figsize=(3.7 * len(picks), 11.4))
for k, i in enumerate(picks):
    img, s, p = recs[i]
    axes[0, k].imshow(img); axes[0, k].axis("off")
    axes[0, k].plot(*p["winner_px"], marker="+", ms=20, mew=3.5, color="#1d3557")
    axes[0, k].set_title("コマ %d" % i, fontsize=12)
    axes[1, k].imshow(s["salience"], cmap="YlOrRd", vmin=0, vmax=1)
    axes[1, k].set_xticks([]); axes[1, k].set_yticks([])
    axes[2, k].imshow(p["ior"], cmap="Blues", vmin=0, vmax=1.2)
    axes[2, k].plot(p["winner_cell"][0], p["winner_cell"][1], marker="+",
                    ms=18, mew=3.0, color="#c1121f")
    axes[2, k].set_xticks([]); axes[2, k].set_yticks([])
axes[0, 0].set_ylabel("視界", fontsize=13)
axes[1, 0].set_ylabel("目立ち", fontsize=13)
axes[2, 0].set_ylabel("直前に見た場所（抑制）", fontsize=13)
fig.suptitle("場所の優先度地図：復帰抑制で注意が同じ場所に貼り付かない（連続する4コマ＋2コマ）",
             fontsize=16, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(out, dpi=112, bbox_inches="tight")
print("saved", out)
