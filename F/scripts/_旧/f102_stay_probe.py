# -*- coding: utf-8 -*-
"""注意が「留まれない」のはどこのせいかを切り分ける（太郎は走らせない）。
使い方: python F/scripts/f102_stay_probe.py <視界動画.mp4>
"""
import sys
import numpy as np, cv2
sys.path.insert(0, "taro_core/src")
sys.stdout.reconfigure(encoding="utf-8")
from brain.midbrain.salience_map import SalienceMap
from brain.cerebral_cortex.parietal_lobe.spatial_priority_map import SpatialPriorityMap

cap = cv2.VideoCapture(sys.argv[1]); frames = []
while True:
    ok, f = cap.read()
    if not ok: break
    frames.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)[0:448, 560:560+448], (224, 224)))
cap.release()
print("コマ数", len(frames))

def stay(seq):
    return 100.0 * np.mean([seq[i] == seq[i-1] for i in range(1, len(seq))])

# ① 目立ちの地図そのもの（毎コマ一番高い升）
sm = SalienceMap(cell=28); peaks = []
for f in frames:
    peaks.append(sm.update(f)["peak_cell"])
print("① 目立ちの地図の1位が前コマと同じ升        %.0f%%（種類 %d）"
      % (stay(peaks), len(set(peaks))))

# ②③ 優先度地図（溜め＋復帰抑制）の勝者
for gain in (1.0, 0.0):
    s2 = SalienceMap(cell=28)
    pm = SpatialPriorityMap(cell=28, img_size=224.0, ior_gain=gain)
    wins = []
    for f in frames:
        wins.append(pm.update(s2.update(f)["salience"], dt=0.1)["winner_cell"])
    lab = "復帰抑制あり" if gain else "復帰抑制なし"
    print("%s 優先度地図の勝者が前コマと同じ升（%s）  %.0f%%（種類 %d）"
          % ("②" if gain else "③", lab, stay(wins), len(set(wins))))

# ④ 溜めの時定数を3倍遅くしたら（＝次へ移る準備を遅くしたら）
s3 = SalienceMap(cell=28)
pm3 = SpatialPriorityMap(cell=28, img_size=224.0, ior_gain=0.0, acc_tau_s=0.11/0.3)
w3 = []
for f in frames:
    w3.append(pm3.update(s3.update(f)["salience"], dt=0.1)["winner_cell"])
print("④ 抑制なし＋準備を0.3倍の速さに               %.0f%%（種類 %d）"
      % (stay(w3), len(set(w3))))

# ⑤ 「目を動かした分だけ地図をずらす」を毎コマ入れたらどうなるか。
#    今の太郎は、まだ動いていない分（目標までの残り）を**毎コマ**ずらしている。
#    人間はサッケード1回につき1回しかずらさない。過剰にずらすとどうなるかを見る。
for amp in (8.0, 24.0, 48.0):
    s4 = SalienceMap(cell=28)
    pm4 = SpatialPriorityMap(cell=28, img_size=224.0, ior_gain=1.0)
    w4 = []
    rng = np.random.default_rng(0)
    for f in frames:
        sh = tuple(rng.normal(0, amp, 2))
        w4.append(pm4.update(s4.update(f)["salience"], shift_px=sh, dt=0.1)["winner_cell"])
    print("⑤ 毎コマ ±%.0fpx ずらす（過剰なずらし直し）      %.0f%%（種類 %d）"
          % (amp, stay(w4), len(set(w4))))
