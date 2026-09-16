# -*- coding: utf-8 -*-
"""目がどう動いたかを、時間の線と視界のコマで見せる（2026-09-10）。
使い方: python F/scripts/f101_gaze_trace_fig.py <走行フォルダ> <出力PNG>
"""
import sys, os, csv
import numpy as np, cv2
sys.stdout.reconfigure(encoding="utf-8")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic"]

d, out = sys.argv[1], sys.argv[2]
rows = [r for r in csv.DictReader(open(f"{d}/物体ファイル.csv", encoding="utf-8"))
        if r.get("ec_eye_h") not in (None, "")]
seen, tr = set(), []
for r in rows:                     # 1歩1点にする
    s = int(r["step"])
    if s in seen: continue
    seen.add(s)
    tr.append((s, float(r["ec_eye_h"]), float(r["ec_eye_v"])))
tr = np.array(tr)

cap = cv2.VideoCapture(f"{d}/動画_視界.mp4"); frames = []
while True:
    ok, f = cap.read()
    if not ok: break
    frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)[0:448, 560:560 + 448])
cap.release()

att = {int(r["step"]): r for r in csv.DictReader(open(f"{d}/注意.csv", encoding="utf-8"))}
picks = [int(tr[int(len(tr) * p), 0]) for p in (0.12, 0.30, 0.48, 0.66, 0.84)]

fig = plt.figure(figsize=(17.5, 8.6))
ax = fig.add_axes([0.055, 0.60, 0.915, 0.33])
ax.plot(tr[:, 0], tr[:, 1], lw=1.4, color="#1d3557", label="左右（右が＋）")
ax.plot(tr[:, 0], tr[:, 2], lw=1.4, color="#c1121f", label="上下（上が＋）")
ax.axhline(0, color="#999", lw=0.8)
for p in picks:
    ax.axvline(p, color="#888", lw=1.0, ls=":")
ax.set_xlabel("歩", fontsize=11); ax.set_ylabel("眼球の角度［度］", fontsize=12)
ax.set_title("太郎の目がどこを向いていたか（可動域は左右±45度・上下 -47〜+33度）",
             fontsize=15, weight="bold")
ax.legend(fontsize=10, loc="upper right"); ax.grid(alpha=0.25)

for k, s in enumerate(picks):
    a = fig.add_axes([0.055 + k * 0.187, 0.06, 0.168, 0.44])
    i = min(max(s - 1, 0), len(frames) - 1)
    a.imshow(cv2.resize(frames[i], (224, 224))); a.axis("off")
    row = att.get(s)
    j = np.argmin(np.abs(tr[:, 0] - s))
    ttl = "歩 %d　目 (%+.0f, %+.0f)度" % (s, tr[j, 1], tr[j, 2])
    if row:
        ttl += "\n注意した物：%s" % ("見えている" if row["visible"] == "True" else "見えていない")
    a.set_title(ttl, fontsize=10)
fig.savefig(out, dpi=105, bbox_inches="tight")
print("saved", out, "／ 左右の振れ幅 %.1f度・上下 %.1f度"
      % (np.ptp(tr[:, 1]), np.ptp(tr[:, 2])))
