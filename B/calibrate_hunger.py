"""
空腹速度の較正 — blood_vessel.consumption_rate を振って「1日あたりの食事回数」を実測し、
人間（生後6〜12ヶ月：おおよそ1日6〜10回）に合う値を探す。

背景：現行の既定 consumption_rate=0.0001 だと実測で約27回/日と、人間の3〜4倍速く
空腹になっていた。これを人間の「2〜3時間おき」に合わせるのが目的。

使い方: python calibrate_hunger.py [日数]   （既定14日）
        run_simulation_b を短期間だけ回し、食事回数/日・泣き/日・平均血糖を出す。
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
torch.set_num_threads(2)

import environment.core_b as cb

DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 14
SECONDS = DAYS * 86400

# 試す消費速度（1日6〜10回を挟むように）。基準0.0001＝約27回/日。
RATES = [0.0001, 0.00004, 0.00003, 0.000022]

_Orig = cb.BloodVessel


def patched_bloodvessel(rate):
    class Cal(_Orig):
        def __init__(self, initial_glucose=0.5, consumption_rate=0.0001):
            super().__init__(initial_glucose=initial_glucose, consumption_rate=rate)
    return Cal


print(f"[較正] {DAYS}日分を各消費速度で実測（人間の目安：6〜10回/日）\n")
print(f"{'consumption_rate':>18} | {'食事/日':>7} | {'泣き/日':>7} | {'要求語/日':>9}")
print("-" * 52)

from environment.parent_sim_b import run_simulation_b

for rate in RATES:
    cb.BloodVessel = patched_bloodvessel(rate)
    r = run_simulation_b(max_sim_seconds=SECONDS, verbose=False,
                         run_name=f"calib_{rate}")
    feeds_per_day = r["feed_count"] / DAYS
    cries_per_day = r["cry_count"] / DAYS
    req_per_day = r["request_count"] / DAYS
    mark = "  ★人間域" if 6 <= feeds_per_day <= 10 else ""
    print(f"{rate:>18.7f} | {feeds_per_day:>7.1f} | {cries_per_day:>7.1f} | "
          f"{req_per_day:>9.1f}{mark}")

print("\n[完了] 人間域(6〜10回/日)に入った rate を core_b.py の既定に採用する。")
