"""★複数シードの結果をまとめて、改善が「ばらつきに埋もれていないか」を判定する。

【なぜ要るか、2026-07-28】2026-07-27 に「高速化で悪化した（0.154→0.490）」と判断したが、
実際は神経ノイズのシードが毎回変わっていただけで、同じ条件3回で
0.319 / 0.215 / 0.189 とばらついていた（ばらつきの幅0.13）。
★今日の改善幅（1シードで 0.175→0.076＝0.10）は**このばらつきより小さい可能性がある**。
＝1シードの結果では、改善が本物かどうか言えない。

【判定の考え方】
  条件ごとに複数シードの平均と標準偏差を出し、
    反射ON の平均 + 標準偏差 < 反射OFF の平均 - 標準偏差
  なら「ばらつきを考えても優位」と言える（厳しめの基準）。
  ⚠️シード3つでは統計的な検定に足りない。あくまで**目安**。

使い方:
    .venv/Scripts/python.exe E/scripts/e_orient_multiseed_summary.py
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
import os
import sys
import json
import glob

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
IN_DIR = os.path.join(_ROOT, "E", "logs", "orient_converge")

import numpy as np  # noqa: E402

# 視野の半角[度]。無次元の「ずれ」を度に直して人間の文献と比べられるようにする。
HALF_FOV_DEG = 30.0


def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(IN_DIR, "result_*.json"))):
        with open(p, encoding="utf-8") as fp:
            rows.extend(json.load(fp))
    if not rows:
        print("結果がありません。先に e_orient_converge_test.py を回してください")
        return

    seeds = sorted({r["seed"] for r in rows})
    ages = sorted({r["age"] for r in rows})
    holds = sorted({r["head_hold"] for r in rows})
    print("=" * 84)
    print(" 複数シードのまとめ（視線誘導反射）")
    print("=" * 84)
    print(f"  シード {seeds}   体年齢 {ages}ヶ月   頭を抑える {holds}")
    print(f"  1条件あたり {len(seeds)}回   ★シード3つでは検定に足りない＝目安として読む")

    def pick(axis, off, orient):
        return [r["half_mean"] for r in rows
                if r["axis"] == axis and abs(r["offset"] - off) < 1e-9
                and r["orient"] == orient and not np.isnan(r["half_mean"])]

    print("\n" + "=" * 84)
    print(" 条件ごとの平均±標準偏差（後半の|ずれ|平均。小さいほど中心で捉えている）")
    print("=" * 84)
    print(f"{'軸':>4}{'位置':>9}{'反射ONの平均':>16}{'反射OFFの平均':>16}"
          f"{'差':>9}{'判定':>22}")
    verdicts = []
    for axis in ("h", "v"):
        offs = sorted({r["offset"] for r in rows if r["axis"] == axis})
        for off in offs:
            on = pick(axis, off, True)
            of = pick(axis, off, False)
            if not on or not of:
                continue
            mon, son = float(np.mean(on)), float(np.std(on))
            mof, sof = float(np.mean(of)), float(np.std(of))
            # 厳しめの判定：ばらつきを考えても重ならないか
            sep = (mon + son) < (mof - sof)
            weak = (mon < mof) and not sep
            v = "★優位（重ならない）" if sep else ("優位だが重なる" if weak else "優位でない")
            verdicts.append(sep)
            print(f"{('水平' if axis=='h' else '垂直'):>4}{off*100:>+8.1f}cm"
                  f"{mon:>10.3f}±{son:<5.3f}{mof:>10.3f}±{sof:<5.3f}"
                  f"{mof-mon:>9.3f}{v:>22}")

    # 全体
    print("\n" + "=" * 84)
    print(" 全体")
    print("=" * 84)
    all_on = [r["half_mean"] for r in rows if r["orient"] and not np.isnan(r["half_mean"])]
    all_of = [r["half_mean"] for r in rows if not r["orient"] and not np.isnan(r["half_mean"])]
    print(f"  反射ON   平均 {np.mean(all_on):.3f} ± {np.std(all_on):.3f}"
          f"   （度に直すと {np.mean(all_on)*HALF_FOV_DEG:.1f}度）  n={len(all_on)}")
    print(f"  反射OFF  平均 {np.mean(all_of):.3f} ± {np.std(all_of):.3f}"
          f"   （度に直すと {np.mean(all_of)*HALF_FOV_DEG:.1f}度）  n={len(all_of)}")
    print(f"  ★条件ごとに『ばらつきを考えても優位』なのは "
          f"{sum(verdicts)}/{len(verdicts)} 条件")

    # シードごとのばらつき（同じ条件が何回ぶれるか）
    print("\n" + "=" * 84)
    print(" シードによるばらつき（反射ONのみ・条件ごと）")
    print("=" * 84)
    print(f"{'軸':>4}{'位置':>9}" + "".join(f"{'seed'+str(s):>10}" for s in seeds)
          + f"{'幅':>9}")
    spreads = []
    for axis in ("h", "v"):
        offs = sorted({r["offset"] for r in rows if r["axis"] == axis})
        for off in offs:
            vals = []
            for s in seeds:
                v = [r["half_mean"] for r in rows
                     if r["axis"] == axis and abs(r["offset"] - off) < 1e-9
                     and r["orient"] and r["seed"] == s]
                vals.append(v[0] if v else float("nan"))
            good = [v for v in vals if not np.isnan(v)]
            sp = (max(good) - min(good)) if len(good) > 1 else float("nan")
            spreads.append(sp)
            print(f"{('水平' if axis=='h' else '垂直'):>4}{off*100:>+8.1f}cm"
                  + "".join(f"{v:>10.3f}" for v in vals) + f"{sp:>9.3f}")
    print(f"\n  ばらつきの幅の平均 {np.nanmean(spreads):.3f}"
          f"（度に直すと {np.nanmean(spreads)*HALF_FOV_DEG:.1f}度）")
    print("  ★これより小さい差は『シードを変えただけで出る差』＝意味がない")

    print("\n" + "=" * 84)
    print(" 読み方")
    print("=" * 84)
    print("  ・『優位（重ならない）』が全条件なら、反射の効果はばらつきより大きい")
    print("  ・『優位だが重なる』が多いなら、シードを増やさないと結論が出せない")
    print("  ・2026-07-27 の1シード測定と比べるときは、この『ばらつきの幅』を必ず見る")


if __name__ == "__main__":
    main()
