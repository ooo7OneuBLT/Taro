"""e_vergence.py の視差測定・制御則の単体テスト（実走行なし・数秒で終わる）。

置き場所：E/scripts/（設計に「単体テスト」とのみ指定があり、書き捨てでなく本実装と
セットで残す価値があると判断したため taro_core/tests ではなくここに置いた。
run/main.py の入口は通らない＝実験ファイルではない）。

実行： C:\\claude\\AI\\Taro\\.venv\\Scripts\\python.exe E\\scripts\\test_e_vergence.py
"""
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e_vergence import (_measure_shift_px, _grayscale, VergenceReflex,
                        PERIPHERAL_FOVY_DEG)

W = 128
H = 32
FOV = PERIPHERAL_FOVY_DEG   # 60度
DEG_PER_PX = FOV / W        # 設計の 60/128


def _make_pair(true_d, width=W, height=H, seed=0):
    """既知のずれ true_d を持つ左右画像ペアを作る。

    _measure_shift_px の定義（d>=0: left[:, d:] と right[:, :w-d] を比べる）に
    正確に合わせて構成する：
        d>=0 のとき  right[:, :w-d] = left[:, d:]
        d<0  のとき  right[:, -d:]  = left[:, :w+d]
    残りの列はランダムなノイズで埋める（一致度を薄めない程度に、比較対象外の
    領域に置く）。
    """
    rng = np.random.default_rng(seed)
    left = rng.integers(0, 256, size=(height, width), dtype=np.int64).astype(np.float64)
    right = rng.integers(0, 256, size=(height, width), dtype=np.int64).astype(np.float64)
    if true_d >= 0:
        right[:, :width - true_d] = left[:, true_d:]
    else:
        right[:, -true_d:] = left[:, :width + true_d]
    # (H, W, 3) の疑似カラー画像にする（update()はRGBを受け取る前提）
    left_rgb = np.stack([left] * 3, axis=-1)
    right_rgb = np.stack([right] * 3, axis=-1)
    return left_rgb, right_rgb


def test_measure_shift_known_positive():
    left, right = _make_pair(true_d=20)
    d = _measure_shift_px(_grayscale(left), _grayscale(right), max_shift=70)
    assert d == 20, f"期待20画素、実際{d}画素"
    print(f"OK: 既知のずれ+20画素 → 検出 {d}画素")


def test_measure_shift_known_negative():
    left, right = _make_pair(true_d=-15)
    d = _measure_shift_px(_grayscale(left), _grayscale(right), max_shift=70)
    assert d == -15, f"期待-15画素、実際{d}画素"
    print(f"OK: 既知のずれ-15画素 → 検出 {d}画素")


def test_measure_shift_zero():
    left, right = _make_pair(true_d=0)
    d = _measure_shift_px(_grayscale(left), _grayscale(right), max_shift=70)
    assert d == 0, f"期待0画素、実際{d}画素"
    print(f"OK: ずれ無し → 検出 {d}画素")


def test_disparity_deg_conversion():
    """measure_disparity_deg() が画素→角度変換（設計④）を正しく行うか。"""
    class _FakeModel:
        nu = 0
        def actuator(self, i):
            raise IndexError
    # __init__ は model.nu を使うだけで actuator ループは nu=0 なら空になる。
    refl = VergenceReflex.__new__(VergenceReflex)
    refl.max_shift_px = 70
    left, right = _make_pair(true_d=20)
    deg = VergenceReflex.measure_disparity_deg(refl, left, right)
    expected = 20 * DEG_PER_PX
    assert abs(deg - expected) < 1e-9, f"期待{expected:.4f}度、実際{deg:.4f}度"
    print(f"OK: +20画素 → {deg:.4f}度（期待 {expected:.4f}度 = 20*60/128）")


def test_control_law_deadzone_and_gain():
    """update()の積分則：不感帯以下では動かず、以上では符号どおりに積分されるか。

    VergenceReflex.__init__ は model/data の実オブジェクトを要求するため、ここでは
    制御則だけを直接 update() 越しに検証する（アクチュエータ探索は model.nu=0 で
    空になるので apply() 側の実機能は使わない＝視差測定と積分則のみのテスト）。
    """
    class _FakeModel:
        nu = 0
    refl = VergenceReflex(_FakeModel(), data=None)

    # 不感帯以下（0.5度未満相当）：真ん中付近のずれ 0 画素 → 0度 → 動かない
    left0, right0 = _make_pair(true_d=0)
    for _ in range(refl._n_delay + 2):
        refl.update(left0, right0)
    assert refl.vergence_deg == 0.0, f"不感帯以下で動いた: {refl.vergence_deg}"
    print(f"OK: ずれ0 → vergence_deg={refl.vergence_deg}（不感帯内で不動）")

    # 不感帯を超えるずれ（+20画素 ≒ 9.375度）：正方向に積分されるはず
    refl.reset()
    left1, right1 = _make_pair(true_d=20)
    n_calls = refl._n_delay + 5
    for _ in range(n_calls):
        refl.update(left1, right1)
    # 【2026-08-28】視差→目標の符号を実測で確定した（DISPARITY_SIGN=-1）。
    #   「寄り目」は関節角の和のマイナス方向なので、正のずれには負方向へ積む。
    assert refl.vergence_deg < 0.0, f"正のずれなのに負方向へ動かない: {refl.vergence_deg}"
    # 速度上限20度/秒・dt=0.1秒なので1回の積分は最大2度。
    # 積分が起きる呼び出し回数は「バッファが埋まった回」から最後まで
    #   （n_calls - (n_delay - 1) 回）。それを超えたら速度上限違反。
    n_integrations = n_calls - (refl._n_delay - 1)
    assert refl.vergence_deg <= n_integrations * 2.0 + 1e-6, \
        f"速度上限を超えている: {refl.vergence_deg}"
    print(f"OK: ずれ+20画素 → vergence_deg={refl.vergence_deg:.3f}度（正方向・上限内）")

    # 範囲 ±35度でクリップされるか（大きなずれを長時間与える）
    refl.reset()
    for _ in range(400):
        refl.update(left1, right1)
    assert abs(refl.vergence_deg) <= 35.0 + 1e-6, f"範囲±35度を超えている: {refl.vergence_deg}"
    print(f"OK: 長時間の大きなずれ → vergence_deg={refl.vergence_deg:.3f}度（±35度でクリップ）")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n全{len(tests)}件のテストに合格")
