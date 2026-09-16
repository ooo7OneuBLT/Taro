# -*- coding: utf-8 -*-
"""2つの動きの地図を「同じ段階」で比べる（2026-09-10）。

【なぜ作ったか】f99 の図では、反射の動き地図（競合をかける前）と、目立ちの地図の
動きの面（中心周辺差＋反復競合をかけた後）を並べてしまっていた。段階が違うので
見た目が違うのは当たり前で、どちらが良いかの根拠にならない。

そこで4通りを作って比べる：
  A生 反射の動き検出（1・3コマ前との比較＋自己運動の割引 OMS）
  B生 今の地図の動き（1コマ前との差だけ）
  A競 A生を、目立ちの地図と同じ中心周辺差＋反復競合に通したもの
  B競 B生を同じく通したもの（＝いまの「動き」の面そのもの）

【外から動きを差し込む方法】SalienceMap は中で motion = |今 - 前| を計算する。
そこで呼ぶ直前に _prev_gray に「今 - 入れたい動き」を書いておくと、中の引き算が
入れたい動きをそのまま返す。本体のコードは1行も変えない（決める前なので）。

使い方: python F/scripts/f100_motion_fair_probe.py <視界動画.mp4> <出力PNG>
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

model = mujoco.MjModel.from_xml_path("MIMo/mimoEnv/assets/mimo/MIMo_modelv2.xml")
orn = OrientingReflexV2(model, data=None)
sm_b = SalienceMap(cell=28)      # そのまま（1コマ差）
sm_a = SalienceMap(cell=28)      # 反射の動きを差し込む

recs = []
for f in frames:
    img = cv2.resize(f, (224, 224))
    orn.update(img)
    ma = None if orn.motion_map is None else orn.motion_map.astype(np.float32)
    rb = sm_b.update(img)
    if ma is not None:
        m = ma / max(float(ma.max()), 1e-6)
        inten = sm_a._features(img.astype(np.float32))[0]
        sm_a._prev_gray = inten - m          # 中の引き算が m を返すように仕込む
        ra = sm_a.update(img)
        sm_a._prev_gray = inten.copy()       # 次のコマのために本来の値へ戻す
    else:
        ra = None
    recs.append((img, ma, ra, rb))

ok_idx = [i for i, r in enumerate(recs) if r[1] is not None]
picks = [ok_idx[int(len(ok_idx) * r)] for r in (0.10, 0.25, 0.42, 0.60, 0.80)]

rows = ["左目の視界",
        "A生：反射の動き検出\n（1・3コマ前＋自己運動の割引）",
        "B生：今の地図の動き\n（1コマ前との差だけ）",
        "A競：Aを同じ競合に通す",
        "B競：Bを同じ競合に通す\n（＝いまの動きの面）"]
fig, axes = plt.subplots(5, len(picks), figsize=(3.5 * len(picks), 17.5))
for k, i in enumerate(picks):
    img, ma, ra, rb = recs[i]
    axes[0, k].imshow(img); axes[0, k].axis("off"); axes[0, k].set_title("コマ %d" % i, fontsize=12)
    raw_a = cv2.resize(ma, (28, 28)); raw_a = raw_a / max(raw_a.max(), 1e-6)
    prev = sm_b  # 参照だけ
    # B生を作り直す（保存していないので、1コマ前との差をここで再計算）
    g_now = SalienceMap(cell=28)._features(img.astype(np.float32))[0]
    g_prev = SalienceMap(cell=28)._features(recs[i - 1][0].astype(np.float32))[0]
    raw_b = cv2.resize(np.abs(g_now - g_prev), (28, 28)); raw_b = raw_b / max(raw_b.max(), 1e-6)
    for r, m, cm in ((1, raw_a, "Oranges"), (2, raw_b, "Oranges"),
                     (3, ra["channels"]["動き"], "Oranges"),
                     (4, rb["channels"]["動き"], "Oranges")):
        mm = m / max(float(m.max()), 1e-6)
        axes[r, k].imshow(mm, cmap=cm, vmin=0, vmax=1)
        axes[r, k].set_xticks([]); axes[r, k].set_yticks([])
for r, lab in enumerate(rows):
    axes[r, 0].set_ylabel(lab, fontsize=11)
    axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
fig.suptitle("2つの動きの地図を、同じ段階で比べる", fontsize=18, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.966]); fig.savefig(out, dpi=110, bbox_inches="tight")
print("saved", out)

# ---- どれだけ同じか、数で出す ------------------------------------------------
def corr(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 1e-12 else np.nan

cr_raw, cr_comp, pk_same, pk_dist = [], [], [], []
for i in ok_idx[1:]:
    img, ma, ra, rb = recs[i]
    a_raw = cv2.resize(ma, (28, 28))
    g_now = sm_b._features(img.astype(np.float32))[0]
    g_prev = sm_b._features(recs[i - 1][0].astype(np.float32))[0]
    b_raw = cv2.resize(np.abs(g_now - g_prev), (28, 28))
    cr_raw.append(corr(a_raw, b_raw))
    ca, cb = ra["channels"]["動き"], rb["channels"]["動き"]
    cr_comp.append(corr(ca, cb))
    pa = np.unravel_index(int(np.argmax(ca)), ca.shape)
    pb = np.unravel_index(int(np.argmax(cb)), cb.shape)
    pk_same.append(pa == pb)
    pk_dist.append(float(np.hypot(pa[0] - pb[0], pa[1] - pb[1])))
print("生の動き地図どうしの相関  中央値 %.3f" % np.nanmedian(cr_raw))
print("競合後どうしの相関        中央値 %.3f" % np.nanmedian(cr_comp))
print("いちばん高い升が一致      %.0f%%（ずれの中央値 %.1f升）"
      % (100 * np.mean(pk_same), np.median(pk_dist)))

# 反射が撃つ向きは「動き」だけ、優先度地図は4枚の合計。どこで食い違うか。
d_mot, d_tot = [], []
for i in ok_idx[1:]:
    img, ma, ra, rb = recs[i]
    cb = rb["channels"]["動き"]
    r, c = np.unravel_index(int(np.argmax(cb)), cb.shape)
    mot_px = ((c + 0.5) * 8.0, (r + 0.5) * 8.0)
    tot_px = rb["peak"]
    d_mot.append(np.hypot(mot_px[0] - tot_px[0], mot_px[1] - tot_px[1]))
print("『動き』の1位と『4枚の合計』の1位の距離 中央値 %.1f px（224px中）"
      % np.median(d_mot))
