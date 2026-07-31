"""体型の係数を「人間の新生児の実測値」に合わせて探す（項目9.4）。

【なぜ要るか】体型の係数（脚の長さ・太さ等）は 2026-07-21 に**目視だけで**決められ、
質量も絶対サイズも一度も測られていなかった（実測すると身長35.6cm・体重1.44kg・
頭が体重の49.5%＝人間の新生児 49.9cm/3.5kg/25% と大きく食い違う）。
[[feedback-watch-dont-just-measure]] の裏返しで、**目視だけでも騙される**。

【探し方】1パラメータずつ応答を測って内挿する（総当たりは組合せ爆発するため）。
  段階1: leg を振って下肢長 19.6cm に合わせる
  段階2: arm を振って上肢長 21.0cm に合わせる
  段階3: 太さ（leg_thick/arm_thick/trunk_width）を共通倍率で振って体重 3.50kg に合わせる
  段階4: 最終確認（身長・頭の質量比もここで見る）
注意：各段階は独立でない（太さを変えると体重が変わり、頭の質量比も動く）ので、
段階3のあと段階1〜2をもう一度確認する。

使い方: python E/scripts/e_body_fit.py
"""

# 注意：古い方式（2026-07-30 に整理）。新しい実験は `run/main.py` を通す。
#   【経緯】目標Eの実験スクリプトが118本あり、うち66本が**独立に環境を組み立てていた**。
#     そのため「学習は関節モード（90関節を独立に駆動＝逸脱リスト 逸脱5）、
#     測定とViewerは筋肉モード（拮抗筋2本/関節）」という**別の体で動く**事故が起きた
#     （ユーザーの目視「視線誘導反射の実験の時とは動きが全然違う」で発覚。
#      実測で動きが人間の新生児の約3.3倍速かった）。
#   【設計と移行計画】`E/docs/実行基盤_設計.md`
#   注意：このファイルは**記録として残す**（削除しない方針）。
#     中の測り方は再利用できるので、プラグインへ移すときの元にする。
import os
import sys
import warnings

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import e_body_measure as M  # noqa: E402

# 人間の新生児（満期産）の実測値。出典は e_body_measure.HUMAN と同じ。
TARGET = {"height_cm": 49.9, "mass_kg": 3.50, "head_frac": 0.25,
          "arm_cm": 20.96, "leg_cm": 19.60}

THICK_KEYS = ("leg_thick", "arm_thick", "trunk_width", "foot_width")


def base():
    return dict(M.NEWBORN_SHAPE_DEFAULTS)


def with_(scales, **kw):
    s = dict(scales)
    s.update(kw)
    return s


def row(label, r):
    print(f"{label:<24}{r['height_cm']:>8.1f}{r['mass_kg']:>9.3f}"
          f"{r['head_frac']*100:>8.1f}{r['arm_cm']:>8.1f}{r['leg_cm']:>8.1f}"
          f"{r['arm_leg_ratio']:>9.3f}", flush=True)


def header():
    print(f"{'条件':<24}{'身長cm':>8}{'体重kg':>9}{'頭の%':>8}"
          f"{'上肢cm':>8}{'下肢cm':>8}{'上肢/下肢':>9}")
    print("-" * 78)


def interp(pairs, target):
    """(入力, 出力) の列から、出力=target になる入力を線形内挿する。"""
    pairs = sorted(pairs)
    for (x0, y0), (x1, y1) in zip(pairs, pairs[1:]):
        if (y0 - target) * (y1 - target) <= 0 and y1 != y0:
            return x0 + (target - y0) * (x1 - x0) / (y1 - y0)
    # 範囲外：端の2点で外挿する
    (x0, y0), (x1, y1) = pairs[0], pairs[-1]
    if y1 == y0:
        return x1
    return x0 + (target - y0) * (x1 - x0) / (y1 - y0)


def sweep(scales, key, values, out_key, label):
    print(f"\n── {label}: {key} を振る ──", flush=True)
    header()
    pairs = []
    for v in values:
        if key == "thick":
            s = with_(scales, **{k: v for k in THICK_KEYS})
        else:
            s = with_(scales, **{key: v})
        r = M.measure(s)
        row(f"{key}={v:.2f}", r)
        pairs.append((v, r[out_key]))
    best = interp(pairs, TARGET[out_key])
    print(f"  → {out_key}={TARGET[out_key]} になるのは {key} = {best:.3f}", flush=True)
    return best


def main():
    print("体型の係数を人間の新生児に合わせる（現在のコードで実測）\n")
    print("目標: 身長49.9cm / 体重3.50kg / 頭25.0% / 上肢21.0cm / 下肢19.6cm\n")

    s = base()
    print("── 出発点 ──")
    header()
    row("v2（今の既定）", M.measure(s))

    # 段階1: 脚の長さ
    leg = sweep(s, "leg", [0.7, 1.0, 1.3, 1.6], "leg_cm", "段階1 脚の長さ")
    s = with_(s, leg=round(leg, 3))

    # 段階2: 腕の長さ
    arm = sweep(s, "arm", [0.7, 1.0, 1.3, 1.6], "arm_cm", "段階2 腕の長さ")
    s = with_(s, arm=round(arm, 3))

    # 段階3: 太さ（体重で合わせる）
    thick = sweep(s, "thick", [0.8, 1.1, 1.4, 1.7], "mass_kg", "段階3 太さ→体重")
    s = with_(s, **{k: round(thick, 3) for k in THICK_KEYS})

    # 段階4: 太さを変えたので長さをもう一度確認して微調整
    print("\n── 段階4 太さ変更後に長さを再確認 ──", flush=True)
    header()
    r = M.measure(s)
    row("段階3までの係数", r)
    if abs(r["leg_cm"] - TARGET["leg_cm"]) > 0.5:
        leg = sweep(s, "leg", [leg * 0.85, leg, leg * 1.15], "leg_cm", "段階4a 脚を再調整")
        s = with_(s, leg=round(leg, 3))
    if abs(r["arm_cm"] - TARGET["arm_cm"]) > 0.5:
        arm = sweep(s, "arm", [arm * 0.85, arm, arm * 1.15], "arm_cm", "段階4b 腕を再調整")
        s = with_(s, arm=round(arm, 3))

    print("\n=== 結果 ===")
    header()
    row("v2（今の既定）", M.measure(base()))
    r = M.measure(s)
    row("探索の結果", r)
    print(f"{'人間の新生児':<24}{TARGET['height_cm']:>8.1f}{TARGET['mass_kg']:>9.3f}"
          f"{TARGET['head_frac']*100:>8.1f}{TARGET['arm_cm']:>8.1f}"
          f"{TARGET['leg_cm']:>8.1f}{TARGET['arm_cm']/TARGET['leg_cm']:>9.3f}")

    print("\n係数（環境変数で再現できる形）:")
    envmap = {"leg": "E_LEG_SCALE", "leg_thick": "E_LEG_THICK", "arm": "E_ARM_SCALE",
              "arm_thick": "E_ARM_THICK", "trunk_len": "E_TRUNK_LEN",
              "trunk_width": "E_TRUNK_WIDTH", "foot": "E_FOOT_SCALE",
              "foot_width": "E_FOOT_WIDTH", "hand": "E_HAND_SCALE"}
    parts = [f"{envmap[k]}={v}" for k, v in s.items()
             if k in envmap and abs(v - M.NEWBORN_SHAPE_DEFAULTS[k]) > 1e-9]
    print("  " + " ".join(parts) if parts else "  （v2と同じ）")


if __name__ == "__main__":
    main()
