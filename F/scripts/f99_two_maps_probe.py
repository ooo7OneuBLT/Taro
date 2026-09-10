# -*- coding: utf-8 -*-
"""今の太郎が持っている「2枚の地図」を、同じ録画コマで並べて見せる（2026-09-10）。

  地図A：視線誘導反射が自前で作る動きの地図（orienting.py の motion_map）
  地図B：目立ちの地図（salience_map.py。明るさ・色・向き・動きの4枚を足したもの）

太郎は走らせない。録画した視界（左目の実入力）を両方に同じ順で通すだけ。
使い方: python F/scripts/f99_two_maps_probe.py <視界動画.mp4> <出力PNG>
"""
import sys, os
import numpy as np, cv2
sys.path.insert(0, "taro_core/src")
sys.path.insert(0, "taro_core/src/brain")
sys.path.insert(0, "taro_core/src/senses")
sys.stdout.reconfigure(encoding="utf-8")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic"]
import mujoco
from brain.midbrain.salience_map import SalienceMap
from brain.midbrain.orienting import OrientingReflexV2

vid, out = sys.argv[1], sys.argv[2]
cap = cv2.VideoCapture(vid); frames = []
while True:
    ok, f = cap.read()
    if not ok: break
    frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)[0:448, 560:560 + 448])
cap.release()
print("読めたコマ数", len(frames))

model = mujoco.MjModel.from_xml_path("MIMo/mimoEnv/assets/mimo/MIMo_modelv2.xml")
orn = OrientingReflexV2(model, data=None)      # 目の速さは読めないので抑制は無し
sm = SalienceMap(cell=28)

recs = []
for f in frames:
    img = cv2.resize(f, (224, 224))
    orn.update(img)
    s = sm.update(img)
    recs.append((img,
                 None if orn.motion_map is None else orn.motion_map.copy(),
                 (orn.h_dir, orn.v_dir, orn.strength),
                 s))

ok_idx = [i for i, r in enumerate(recs) if r[1] is not None]
picks = [ok_idx[int(len(ok_idx) * r)] for r in (0.10, 0.25, 0.42, 0.60, 0.80)]

fig, axes = plt.subplots(4, len(picks), figsize=(3.5 * len(picks), 14.2))
for k, i in enumerate(picks):
    img, mm, (h, v, st), s = recs[i]
    axes[0, k].imshow(img); axes[0, k].axis("off")
    axes[0, k].set_title("コマ %d" % i, fontsize=12)
    # 反射が指す向き（[-1,1]→画素）
    axes[0, k].plot(112 + h * 112, 112 - v * 112, marker="x", ms=18, mew=3.5, color="#e07b00")
    axes[0, k].plot(*s["peak"], marker="+", ms=20, mew=3.5, color="#1d3557")

    m = cv2.resize(mm.astype(np.float32), (28, 28))
    axes[1, k].imshow(m / max(m.max(), 1e-6), cmap="Oranges", vmin=0, vmax=1)
    axes[1, k].set_xticks([]); axes[1, k].set_yticks([])
    axes[1, k].set_title("撃つ強さ %.3f" % st, fontsize=10)

    ch = s["channels"]["動き"]
    axes[2, k].imshow(ch / max(ch.max(), 1e-6), cmap="Oranges", vmin=0, vmax=1)
    axes[2, k].set_xticks([]); axes[2, k].set_yticks([])

    axes[3, k].imshow(s["salience"], cmap="YlOrRd", vmin=0, vmax=1)
    axes[3, k].plot(s["peak_cell"][0], s["peak_cell"][1], marker="+", ms=18, mew=3.5, color="#1d3557")
    axes[3, k].set_xticks([]); axes[3, k].set_yticks([])

for r, lab in enumerate(["左目の視界\n×=反射の向き　+=地図の1位",
                         "地図A：反射が自前で\n作る動きの地図",
                         "地図Bの動きの面\n（4枚のうち1枚）",
                         "地図B：目立ちの地図\n（4枚の合計）"]):
    axes[r, 0].set_ylabel(lab, fontsize=12)
    axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])

fig.suptitle("同じ画像から、太郎は動きを2回計算している", fontsize=18, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.965]); fig.savefig(out, dpi=112, bbox_inches="tight")
print("saved", out)

# 2枚がどれだけ違う場所を指すか
d = []
for i in ok_idx:
    _, _, (h, v, st), s = recs[i]
    px, py = s["peak"]
    d.append(np.hypot((112 + h * 112) - px, (112 - v * 112) - py))
print("反射の向きと目立ちの1位の距離：中央値 %.1f px（画像は224px）" % np.median(d))
print("64px（=画像の1/3）より離れていたコマの割合 %.0f%%"
      % (100 * np.mean(np.array(d) > 64)))
