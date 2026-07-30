"""神経ノイズの大きさを決めるためのスイープ。

【何を確かめるか】ノイズは「対称な2標的で決着を生む」ために入れるが、
入れすぎれば単一標的の精度を壊す。両立する範囲を実測で探す。

  条件A（単一標的）      ： ノイズを入れても正しい方向を指し続けるか
  条件B（近い2標的 10/30度）： 中間を指し続けるか（文献: 30度までは平均化）
  条件C（遠い2標的 45/55度）： どちらかに決着するか（文献: 45度超で二峰）
                              かつ左右がおおむね拮抗するか

【文献の境界】Van der Stigchel & de Vries (2013) Vision Research
  ~30度まで 単峰（平均化）／~35度 混在／45度超 二峰（選択）
"""

# ⚠️★古い方式（2026-07-30 に整理）。新しい実験は `run/main.py` を通す。
#   【経緯】目標Eの実験スクリプトが118本あり、うち66本が**独立に環境を組み立てていた**。
#     そのため「学習は関節モード（90関節を独立に駆動＝逸脱リスト 逸脱5）、
#     測定とViewerは筋肉モード（拮抗筋2本/関節）」という**別の体で動く**事故が起きた
#     （ユーザーの目視「視線誘導反射の実験の時とは動きが全然違う」で発覚。
#      実測で動きが人間の新生児の約3.3倍速かった）。
#   【設計と移行計画】`E/docs/実行基盤_設計.md`
#   ⚠️このファイルは**記録として残す**（削除しない方針）。
#     中の測り方は再利用できるので、プラグインへ移すときの元にする。
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from e_orienting_v2 import OrientingReflexV2
from e_orient_v2_test import DummyModel, blank, square, RES

N_SEEDS = 20
NOISE_LEVELS = [0.0, 0.02, 0.05, 0.10, 0.20]


def osc_single(cx, cy=64, n=30, amp=4, size=15):
    frames = []
    for i in range(n):
        dx = amp if i % 2 == 0 else -amp
        frames.append(square(blank(), cx + dx, cy, size))
    return frames


def osc_pair(sep_deg, n=30, amp=4, size=13):
    sep_px = int(round(sep_deg * RES / 60.0))
    cx1 = int(round(63.5 - sep_px / 2))
    cx2 = int(round(63.5 + sep_px / 2))
    frames = []
    for i in range(n):
        dx = amp if i % 2 == 0 else -amp
        frames.append(square(square(blank(), cx1 + dx, cy := 64, size), cx2 + dx, cy, size))
    return frames


def run(frames, noise, seed):
    rf = OrientingReflexV2(DummyModel(), noise=noise, seed=seed)
    for f in frames:
        rf.update(f)
    return rf.h_dir


def main():
    conds = [
        ("A. 単一標的 右",   osc_single(96),  "expect |h| 大きく正"),
        ("A. 単一標的 左",   osc_single(32),  "expect |h| 大きく負"),
        ("B. 2標的 10度",    osc_pair(10),    "expect 中間 |h|<0.15"),
        ("B. 2標的 30度",    osc_pair(30),    "expect 中間 |h|<0.15"),
        ("C. 2標的 45度",    osc_pair(45),    "expect 決着 |h|>0.15"),
        ("C. 2標的 55度",    osc_pair(55),    "expect 決着 |h|>0.15"),
    ]

    print("=== 神経ノイズのスイープ（20シード）===")
    print("視野角60度 / 128画素。文献: ~30度=平均化, 45度超=選択")
    print("Kim & Basso 2010: 上丘の Fano factor 1.44、ばらつきが選択を予測\n")

    header = f"{'条件':<18}" + "".join(f"{f'noise={n}':>16}" for n in NOISE_LEVELS)
    print(header)
    print("-" * len(header))

    summary = {}
    for label, frames, note in conds:
        row = f"{label:<18}"
        for noise in NOISE_LEVELS:
            hs = [run(frames, noise, seed) for seed in range(N_SEEDS)]
            hs = np.array(hs)
            mean_abs = np.abs(hs).mean()
            decided = (np.abs(hs) > 0.15).mean()      # 決着した割合
            row += f"{mean_abs:>7.3f}/{decided*100:>5.0f}%  "
            summary[(label, noise)] = (mean_abs, decided, hs)
        print(row + f"  {note}")

    print("\n  （表の見方）平均|h| / 決着率%")
    print("  決着率 = |h_dir| > 0.15 になったシードの割合")

    # 遠い2標的で左右が拮抗しているか
    print("\n=== 遠い2標的の左右バランス（決着したシードのうち右を選んだ割合）===")
    for label in ["C. 2標的 45度", "C. 2標的 55度"]:
        line = f"  {label:<16}"
        for noise in NOISE_LEVELS:
            _, _, hs = summary[(label, noise)]
            decided = hs[np.abs(hs) > 0.15]
            if len(decided) == 0:
                line += f"{'---':>10}"
            else:
                right_frac = (decided > 0).mean()
                line += f"{right_frac*100:>8.0f}% "
        print(line)

    # 単一標的の精度がノイズで劣化していないか
    print("\n=== 単一標的の方向の精度（ノイズなしとの差）===")
    for label in ["A. 単一標的 右", "A. 単一標的 左"]:
        base = summary[(label, 0.0)][2].mean()
        line = f"  {label:<16} base={base:+.3f}  "
        for noise in NOISE_LEVELS[1:]:
            m = summary[(label, noise)][2].mean()
            line += f"n={noise}: {m:+.3f}({m-base:+.3f})  "
        print(line)


if __name__ == "__main__":
    main()
