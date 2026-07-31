"""ステップ4（階段状サッケード）の単体テスト。

【何を確かめるか】出力が「毎tick連続」ではなく「間欠的な跳躍の連続」になっているか。
新生児の定位は smooth pursuit ではなくサッケードの階段状の連続
（Aslin & Salapatek 1975 / Kremenitzer 1979）。

環境は通さない。人工画像で右に振動する対象を見せ続け、
apply() の出力を時系列で記録して波形を見る。
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
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt

from e_orienting_v2 import (OrientingReflexV2, SACCADE_LATENCY, SACCADE_DURATION,
                            NECK_GAIN, EYE_GAIN)
from e_orient_v2_test import blank, square, RES

DT = 0.01          # 物理1step
VISION_HZ = 10     # 視覚が更新される頻度
SIM_SEC = 2.0


class ActModel:
    """首と目のアクチュエータを持つ最小モデル。"""
    NAMES = ["act:head_swivel", "act:head_tilt",
             "act:left_eye_horizontal", "act:right_eye_horizontal",
             "act:left_eye_vertical", "act:right_eye_vertical"]
    nu = len(NAMES)
    class _A:
        def __init__(self, n): self.name = n
    def actuator(self, i):
        return ActModel._A(ActModel.NAMES[i])


def main():
    reflex = OrientingReflexV2(ActModel(), dt=DT, seed=0)
    n_steps = int(SIM_SEC / DT)
    vision_every = int((1.0 / VISION_HZ) / DT)

    swivel, eye_h, times, sacc_marks = [], [], [], []
    frame_i = 0
    for step in range(n_steps):
        # 視覚は 10Hz で更新（右で振動する対象）
        if step % vision_every == 0:
            dx = 4 if frame_i % 2 == 0 else -4
            reflex.update(square(blank(), 96 + dx, 64, 15))
            frame_i += 1
        before = reflex.n_saccades
        action = np.zeros(ActModel.nu, dtype=float)
        out = reflex.apply(action)
        if reflex.n_saccades > before:
            sacc_marks.append(step * DT)
        swivel.append(out[0])       # act:head_swivel
        eye_h.append(out[2])        # act:left_eye_horizontal
        times.append(step * DT)

    swivel, eye_h, times = np.array(swivel), np.array(eye_h), np.array(times)

    # --- 判定 ---
    print("=== ステップ4（階段状サッケード）の判定 ===")
    print(f"  シミュレーション {SIM_SEC}秒、視覚 {VISION_HZ}Hz、物理 {1/DT:.0f}Hz")
    print(f"  設定: 潜時 {SACCADE_LATENCY}s / 持続 {SACCADE_DURATION}s "
          f"/ 首ゲイン {NECK_GAIN} / 目ゲイン {EYE_GAIN}\n")

    checks = []

    # 1. サッケードが複数回撃たれているか
    n = reflex.n_saccades
    exp_n = int(SIM_SEC / SACCADE_LATENCY)
    ok1 = abs(n - exp_n) <= 1
    checks.append(("サッケードの発火回数", ok1,
                   f"{n}回（潜時から予想 {exp_n}回）"))

    # 2. 出力が間欠か（ゼロの時間があるか）
    duty = float((np.abs(swivel) > 1e-9).mean())
    exp_duty = SACCADE_DURATION / SACCADE_LATENCY
    ok2 = abs(duty - exp_duty) < 0.1
    checks.append(("出力の間欠性（動いている時間の割合）", ok2,
                   f"{duty*100:.0f}%（設計値 {exp_duty*100:.0f}% = 持続÷潜時）"))

    # 3. 発火間隔が潜時どおりか
    if len(sacc_marks) >= 2:
        gaps = np.diff(sacc_marks)
        ok3 = np.allclose(gaps, SACCADE_LATENCY, atol=0.02)
        checks.append(("発火間隔", ok3,
                       f"平均 {gaps.mean():.3f}s（設定 {SACCADE_LATENCY}s）"))
    else:
        checks.append(("発火間隔", False, "発火が2回未満"))

    # 4. 方向が正しいか（右の対象 → 正の出力）
    peak = swivel[np.abs(swivel).argmax()]
    ok4 = peak > 0
    checks.append(("方向（右の対象 → 正の出力）", ok4, f"ピーク {peak:+.3f}"))

    # 5. 首と目の比がゲイン比どおりか
    ratio = np.abs(eye_h).max() / max(np.abs(swivel).max(), 1e-9)
    exp_ratio = EYE_GAIN / NECK_GAIN
    ok5 = abs(ratio - exp_ratio) < 0.05
    checks.append(("目と首の出力比", ok5,
                   f"{ratio:.3f}（設計値 {exp_ratio:.3f} = 目ゲイン÷首ゲイン）"))

    for name, ok, detail in checks:
        print(f"  [{'OK ' if ok else 'NG!'}] {name}: {detail}")
    passed = sum(1 for _, ok, _ in checks if ok)
    print(f"\n  通過 {passed}/{len(checks)}")

    # --- 図 ---
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    axes[0].plot(times, swivel, lw=1.4, color="#1f77b4")
    for t in sacc_marks:
        axes[0].axvline(t, color="red", alpha=0.25, lw=0.8)
    axes[0].set_ylabel("neck (head_swivel)")
    axes[0].set_title("Step 4: staircase saccade  (red = saccade fired)", fontsize=11)
    axes[0].grid(alpha=0.3)

    axes[1].plot(times, eye_h, lw=1.4, color="#2ca02c")
    for t in sacc_marks:
        axes[1].axvline(t, color="red", alpha=0.25, lw=0.8)
    axes[1].set_ylabel("eye (horizontal)")
    axes[1].set_xlabel("time (s)")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), os.pardir, "docs", "figures",
                       "orient_v2_step4_20260726.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=115, bbox_inches="tight")
    print("\nsaved:", os.path.abspath(out))
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
