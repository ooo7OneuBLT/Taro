# -*- coding: utf-8 -*-
"""見る側の入力の総点検を1枚の絵にする（2026-09-10）。
使い方: python F/scripts/f103_input_audit_fig.py <走行フォルダ> <出力PNG>
"""
import sys, csv
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic"]

d, out = sys.argv[1], sys.argv[2]
seen, R = set(), []
for r in csv.DictReader(open(f"{d}/物体ファイル.csv", encoding="utf-8")):
    s = int(r["step"])
    if s in seen or r.get("ec_dx_pred") in (None, ""): continue
    seen.add(s); R.append(r)
g = lambda k: np.array([float(x[k]) for x in R])
step = g("step") if "step" in R[0] else np.arange(len(R))
step = np.array([float(x["step"]) for x in R])
pred = np.hypot(g("ec_dx_pred"), g("ec_dy_pred"))
used = np.hypot(g("ec_dx_used"), g("ec_dy_used"))
sa = g("sacc_n"); fired = np.concatenate([[False], np.diff(sa) > 0])

fig, ax = plt.subplots(2, 1, figsize=(15, 7.4), sharex=True)
ax[0].plot(step, pred, lw=1.3, color="#c1121f", label="これから動く分（地図をずらす量）")
ax[0].plot(step, used, lw=1.3, color="#1d3557", label="もう動いた分")
ax[0].axhline(224, color="#888", ls="--", lw=1.0)
ax[0].text(step[3], 232, "画像の幅 224px", fontsize=9, color="#666")
ax[0].axhline(8, color="#2a9d8f", ls=":", lw=1.2)
ax[0].text(step[3], 12, "地図1升ぶん 8px", fontsize=9, color="#2a9d8f")
ax[0].set_ylabel("ずらす量［画素］", fontsize=12)
ax[0].legend(fontsize=10, loc="upper right"); ax[0].grid(alpha=0.25)
ax[0].set_title("遠心性コピーが毎コマ地図をずらしている量", fontsize=15, weight="bold")

nf = ~fired
ax[1].scatter(step[fired], pred[fired], s=16, color="#c1121f", label="サッケードを撃ったコマ")
ax[1].scatter(step[nf], pred[nf], s=26, facecolor="none", edgecolor="#1d3557", lw=1.3,
              label="撃っていないコマ（本来ここは 0 のはず）")
ax[1].axhline(np.median(pred[nf]), color="#1d3557", ls="--", lw=1.2)
ax[1].text(step[3], np.median(pred[nf]) + 6,
           "撃っていないコマの中央値 %.0f px（地図 %.0f 升ぶん）"
           % (np.median(pred[nf]), np.median(pred[nf]) / 8),
           fontsize=10, color="#1d3557")
ax[1].set_xlabel("歩", fontsize=12); ax[1].set_ylabel("ずらす量［画素］", fontsize=12)
ax[1].legend(fontsize=10, loc="upper right"); ax[1].grid(alpha=0.25)
fig.tight_layout(); fig.savefig(out, dpi=110, bbox_inches="tight")
print("saved", out)
print("撃っていないコマ %d 件 中央値 %.1f px ／ 撃ったコマ %d 件 中央値 %.1f px"
      % (nf.sum(), np.median(pred[nf]), fired.sum(), np.median(pred[fired])))
