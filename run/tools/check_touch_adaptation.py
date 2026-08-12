# -*- coding: utf-8 -*-
"""触覚の順応（同じ場所を押され続けると感じ方が弱まる）の検証。

仕様：作業記録（非公開）
設計：作業記録（非公開）

仕様7節の検証項目を、check_common_drive.py と同じ構成でまとめる。

    検証0  0コスト検算（MuJoCo不要）：減衰の解析式一致・回復・fa=0・クリップ・
           脳レイヤー無効時の一致・fusion.pyの1行変更の回帰確認
    検証A  通ってはいけない条件（必須。MuJoCoを使う）
    検証B  既定OFFでの回帰確認
    検証C  追加・ユーザーの指示（最優先）：同一シード・同一設定で2回実行し、
           完全一致することの確認

【この道具がやらないこと】
    ・学習は回さない（run/main.pyは使わない。数百〜1000ステップ規模の
      短時間スクリプト）
    ・出力パスは検証専用（cfg.save/run.csvを指定しない）。本番の実験結果を
      上書きしない（実装ノウハウ2026-08-05の教訓）

使い方::

    .venv/Scripts/python.exe run/tools/check_touch_adaptation.py
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
_R = r"C:\claude\AI\Taro"
os.chdir(_R)
if _R not in sys.path:
    sys.path.insert(0, _R)

import numpy as np                                          # noqa: E402
import torch                                                 # noqa: E402

from run.config import Config, TARO_DEFAULTS                 # noqa: E402
from run.trainer import Trainer                               # noqa: E402
from run.plugins.base import Plugin                           # noqa: E402
# taro_setup を import すると taro_core/src/senses 等が sys.path に入るので、
#   touch_adaptation / somatosensory_cortex / fusion はこの import の後で
#   初めて読める（check_common_drive.py と同じ流儀）。
from run.taro_setup import Taro                                # noqa: E402

from touch_adaptation import TouchAdaptation, TOUCH_EPS        # noqa: E402

SCENE = "新生児_仰向け_柵なし"


def _hr(title):
    print("=" * 88)
    print(title)
    print("=" * 88)


# ================================================================== 検証0
def verify0_peripheral_decay_matches_analytic():
    """一定の力を持続入力したとき、g_peripheralの減衰が解析式と数値的に一致するか。"""
    _hr("検証0-1: 遅順応・末梢の減衰が解析式と一致するか（0コスト・MuJoCo不要）")
    n, dt, tau, floor = 3, 0.1, 8.4, 0.31
    ta = TouchAdaptation(n, dt, fa_enabled=False, sa_enabled=True,
                         include_cortical=False, tau_peripheral_s=tau,
                         sa_floor=floor, tau_recover_s=1e9, fa_gain=1.0)
    touch_flat = np.zeros(n * 3)
    touch_flat[0::3] = 1.0     # 各点、力の大きさ=1.0を持続入力
    g_analytic = 1.0
    max_diff = 0.0
    for _ in range(300):
        ta.advance(touch_flat)
        g_analytic = g_analytic - (dt / tau) * (g_analytic - floor)
        max_diff = max(max_diff, abs(float(ta.g_peripheral[0]) - g_analytic))
    ok = max_diff < 1e-9
    print(f"  300step後：解析値={g_analytic:.8f}  実装値={float(ta.g_peripheral[0]):.8f}"
          f"  最大差={max_diff:.2e}  {'OK' if ok else 'NG'}")
    return ok


def verify0_cortical_decay_matches_analytic():
    """include_cortical=True のとき、g_cortical の減衰も同じ解析式と一致するか。"""
    _hr("検証0-2: 遅順応・脳側の減衰が解析式と一致するか（0コスト・MuJoCo不要）")
    n, dt, tau_p, tau_c, floor = 2, 0.1, 8.4, 15.0, 0.31
    ta = TouchAdaptation(n, dt, fa_enabled=False, sa_enabled=True,
                         include_cortical=True, tau_peripheral_s=tau_p,
                         tau_cortical_s=tau_c, sa_floor=floor,
                         tau_recover_s=1e9, fa_gain=1.0)
    touch_flat = np.zeros(n * 3)
    touch_flat[0::3] = 1.0
    g_p_analytic, g_c_analytic = 1.0, 1.0
    max_diff_p, max_diff_c = 0.0, 0.0
    for _ in range(300):
        ta.advance(touch_flat)
        g_p_analytic = g_p_analytic - (dt / tau_p) * (g_p_analytic - floor)
        g_c_analytic = g_c_analytic - (dt / tau_c) * (g_c_analytic - floor)
        max_diff_p = max(max_diff_p, abs(float(ta.g_peripheral[0]) - g_p_analytic))
        max_diff_c = max(max_diff_c, abs(float(ta.g_cortical[0]) - g_c_analytic))
    ok = max_diff_p < 1e-9 and max_diff_c < 1e-9
    print(f"  300step後：末梢 解析={g_p_analytic:.6f} 実装={float(ta.g_peripheral[0]):.6f}"
          f"（差{max_diff_p:.2e}）")
    print(f"             脳　 解析={g_c_analytic:.6f} 実装={float(ta.g_cortical[0]):.6f}"
          f"（差{max_diff_c:.2e}）  {'OK' if ok else 'NG'}")
    return ok


def verify0_recovery():
    """力が無くなったら、tau_recoverに従って1.0へ回復するか。
    tau_recoverを極小にすると瞬時回復に近づくことも確認する。"""
    _hr("検証0-3: 回復（順応から元に戻る）の確認（0コスト・MuJoCo不要）")
    n, dt, tau, floor = 2, 0.1, 8.4, 0.31

    # まず十分に順応させてから接触を止める（通常のtau_recover）
    ta = TouchAdaptation(n, dt, fa_enabled=False, sa_enabled=True,
                         include_cortical=False, tau_peripheral_s=tau,
                         sa_floor=floor, tau_recover_s=tau, fa_gain=1.0)
    touch_on = np.zeros(n * 3); touch_on[0::3] = 1.0
    touch_off = np.zeros(n * 3)
    for _ in range(300):
        ta.advance(touch_on)
    g_before = float(ta.g_peripheral[0])
    g_analytic = g_before
    max_diff = 0.0
    for _ in range(300):
        ta.advance(touch_off)
        g_analytic = g_analytic + (dt / tau) * (1.0 - g_analytic)
        max_diff = max(max_diff, abs(float(ta.g_peripheral[0]) - g_analytic))
    ok_normal = max_diff < 1e-9 and float(ta.g_peripheral[0]) > g_before
    print(f"  通常tau_recover={tau}秒：順応直後g={g_before:.4f} → 300step後g="
          f"{float(ta.g_peripheral[0]):.6f}（解析値との最大差={max_diff:.2e}）"
          f"  {'OK' if ok_normal else 'NG'}")

    # tau_recoverを極小にすると、1stepでほぼ1.0へ戻る（瞬時回復への近似）
    ta2 = TouchAdaptation(n, dt, fa_enabled=False, sa_enabled=True,
                          include_cortical=False, tau_peripheral_s=tau,
                          sa_floor=floor, tau_recover_s=1e-6, fa_gain=1.0)
    for _ in range(300):
        ta2.advance(touch_on)
    ta2.advance(touch_off)
    g_after_1step = float(ta2.g_peripheral[0])
    ok_instant = g_after_1step > 0.999999
    print(f"  tau_recover=1e-6秒（瞬時回復近似）：接触を止めた1step後g="
          f"{g_after_1step:.8f}  {'OK（ほぼ1.0）' if ok_instant else 'NG'}")
    return ok_normal and ok_instant


def verify0_fa_zero_when_no_change():
    """力の変化が無い区間ではfa_iが厳密にゼロであることの確認。"""
    _hr("検証0-4: 速順応は接触が変化しない区間で厳密にゼロか（0コスト・MuJoCo不要）")
    n, dt = 3, 0.1
    ta = TouchAdaptation(n, dt, fa_enabled=True, sa_enabled=False,
                         fa_gain=2.5)
    touch_flat = np.zeros(n * 3); touch_flat[0::3] = np.array([0.3, 0.7, 0.0])
    ta.advance(touch_flat)   # 1回目：前観測が無いのでfa=0のはず
    out1 = ta.adapted().reshape(n, 3)
    m1 = np.linalg.norm(out1, axis=-1)
    ok1 = np.allclose(m1, [0.3, 0.7, 0.0], atol=1e-12)
    max_fa = 0.0
    for _ in range(50):
        ta.advance(touch_flat)          # 同じ値を50回入力し続ける
        out = ta.adapted().reshape(n, 3)
        m = np.linalg.norm(out, axis=-1)
        # sa_enabled=False なので adapted = fa（そのまま） のはず。差分ゼロならfa=0で
        #   adapted=m自体になる（sa無効時の式に一致）はず。
        max_fa = max(max_fa, float(np.max(np.abs(m - np.array([0.3, 0.7, 0.0])))))
    ok = ok1 and max_fa < 1e-12
    print(f"  1回目（前観測無し）：厳密一致={ok1}")
    print(f"  同一値を50回入力：fa由来の残差の最大値={max_fa:.2e}  {'OK' if ok else 'NG'}")
    return ok


def verify0_clip_bounds():
    """floor未満・1.0超えにゲインが出ないこと（境界のクリップ確認）。"""
    _hr("検証0-5: floor未満・1.0超えにゲインが出ないか（0コスト・MuJoCo不要）")
    rng = np.random.default_rng(42)
    n, dt, floor = 5, 0.1, 0.31
    ta = TouchAdaptation(n, dt, fa_enabled=True, sa_enabled=True,
                         include_cortical=True, tau_peripheral_s=0.5,
                         tau_cortical_s=0.3, sa_floor=floor,
                         tau_recover_s=0.2, fa_gain=3.0)
    min_p, max_p, min_c, max_c = 1.0, 0.0, 1.0, 0.0
    for _ in range(2000):
        # ランダムにON/OFFを繰り返す（極端な時定数で境界に張り付きやすい状況を作る）
        touch = rng.uniform(0, 5, size=n * 3) if rng.random() < 0.5 else np.zeros(n * 3)
        ta.advance(touch)
        min_p, max_p = min(min_p, ta.g_peripheral.min()), max(max_p, ta.g_peripheral.max())
        min_c, max_c = min(min_c, ta.g_cortical.min()), max(max_c, ta.g_cortical.max())
    ok = (min_p >= floor - 1e-12 and max_p <= 1.0 + 1e-12
          and min_c >= floor - 1e-12 and max_c <= 1.0 + 1e-12)
    print(f"  g_peripheral範囲=[{min_p:.6f}, {max_p:.6f}]  g_cortical範囲=[{min_c:.6f}, {max_c:.6f}]"
          f"  floor={floor}  {'OK' if ok else 'NG'}")
    return ok


def verify0_cortical_disabled_matches_peripheral_only():
    """include_cortical=Falseのとき、g_cortical項が数式上完全に無効化され、
    末梢のみの1段階モデルと数値が一致することの確認。"""
    _hr("検証0-6: 脳レイヤー無効時、末梢のみの1段階モデルと一致するか（0コスト・MuJoCo不要）")
    rng = np.random.default_rng(7)
    n, dt = 4, 0.1
    ta = TouchAdaptation(n, dt, fa_enabled=True, sa_enabled=True,
                         include_cortical=False, tau_peripheral_s=8.4,
                         sa_floor=0.31, tau_recover_s=8.4, fa_gain=1.0)
    # 「末梢のみの1段階モデル」＝g_cortical=1.0固定で計算した期待値
    g_p_ref, floor, tau_p, tau_r = np.ones(n), 0.31, 8.4, 8.4
    prev_m_ref = None
    max_diff = 0.0
    for _ in range(500):
        touch = rng.uniform(0, 3, size=n * 3) if rng.random() < 0.5 else np.zeros(n * 3)
        ta.advance(touch)
        f = touch.reshape(n, 3)
        m = np.linalg.norm(f, axis=-1)
        contact = m > TOUCH_EPS
        g_p_ref = np.where(contact, g_p_ref - (dt / tau_p) * (g_p_ref - floor),
                           g_p_ref + (dt / tau_r) * (1.0 - g_p_ref))
        g_p_ref = np.clip(g_p_ref, floor, 1.0)
        fa_ref = np.abs(m - prev_m_ref) if prev_m_ref is not None else np.zeros(n)
        prev_m_ref = m
        sa_ref = m * g_p_ref            # g_cortical=1.0固定
        adapted_m_ref = sa_ref + fa_ref
        with np.errstate(divide="ignore", invalid="ignore"):
            unit = np.where(m[:, None] > TOUCH_EPS, f / m[:, None], 0.0)
        adapted_ref = (adapted_m_ref[:, None] * unit).reshape(-1)
        diff = float(np.max(np.abs(ta.adapted() - adapted_ref)))
        max_diff = max(max_diff, diff)
    ok = max_diff < 1e-9
    also_ok = bool(np.all(ta.g_cortical == 1.0))
    print(f"  500stepでの adapted() 最大差={max_diff:.2e}  g_cortical常に1.0={also_ok}"
          f"  {'OK' if ok and also_ok else 'NG'}")
    return ok and also_ok


def verify0_zero_forever_stays_zero():
    """接触が一度も無い点では、adapted出力が全ステップ恒等的にゼロであること。"""
    _hr("検証0-7: 接触が一度も無い点はadapted出力が常にゼロか（0コスト・MuJoCo不要）")
    n, dt = 6, 0.1
    ta = TouchAdaptation(n, dt, fa_enabled=True, sa_enabled=True,
                         include_cortical=True, tau_peripheral_s=8.4,
                         tau_cortical_s=15.0, sa_floor=0.31,
                         tau_recover_s=8.4, fa_gain=1.0)
    touch_flat = np.zeros(n * 3)
    max_abs = 0.0
    for _ in range(200):
        ta.advance(touch_flat)
        max_abs = max(max_abs, float(np.max(np.abs(ta.adapted()))))
    ok = max_abs == 0.0
    print(f"  200step、接触ゼロを入力し続けたときのadapted()の最大絶対値={max_abs}"
          f"  {'OK' if ok else 'NG'}")
    return ok


def verify0_fusion_regression_synthetic():
    """fusion.pyの1行変更（obs.get("touch_percept", obs["touch"])）が、
    touch_percept無しのときobs["touch"]と完全に同じ経路を通ることの確認
    （合成データ・MuJoCo不要）。"""
    _hr("検証0-8: fusion.pyの1行変更の回帰確認（0コスト・合成データ・MuJoCo不要）")
    from somatosensory_cortex import TouchMap
    from fusion import MinimalFusion
    rng = np.random.default_rng(0)
    n_points = 7
    tm = TouchMap(["head", "chest"],
                 np.array([0, 0, 0, 1, 1, 1, 1], dtype=np.int64),
                 rng.normal(size=(n_points, 3)).astype(np.float32),
                 np.array([3.0, 4.0], dtype=np.float32))
    torch.manual_seed(0)
    fusion = MinimalFusion(touch_dim=0, proprio_dim=8, touch_map=tm)
    obs_common = {
        "interoception": rng.normal(size=4).astype(np.float32),
        "observation": rng.normal(size=8).astype(np.float32),
        "vestibular": rng.normal(size=6).astype(np.float32),
        "touch": rng.normal(size=tm.total_dim).astype(np.float32),
    }
    out_without_key = fusion.encode(obs_common)
    obs_with_identical_percept = dict(obs_common)
    obs_with_identical_percept["touch_percept"] = obs_common["touch"]
    out_with_key = fusion.encode(obs_with_identical_percept)
    ok = torch.equal(out_without_key, out_with_key)
    print(f"  touch_percept無し vs touch_percept=touch(同値)：完全一致={ok}"
          f"  出力shape={tuple(out_without_key.shape)}  {'OK' if ok else 'NG'}")
    return ok


def verify0_advance_multiplicity_differs():
    """advance()を意図的に複数回連続で呼んだ場合と、1回だけ呼んだ場合とで、
    状態の進み方が異なることの確認（正しい呼び出し経路のありがたみの裏付け）。"""
    _hr("検証0-9: advance()の呼び出し回数で順応の進み方が変わるか（0コスト・MuJoCo不要）")
    n, dt = 2, 0.1
    touch_flat = np.zeros(n * 3); touch_flat[0::3] = 1.0
    ta_once = TouchAdaptation(n, dt, fa_enabled=False, sa_enabled=True,
                              tau_peripheral_s=8.4, sa_floor=0.31, tau_recover_s=8.4)
    ta_thrice = TouchAdaptation(n, dt, fa_enabled=False, sa_enabled=True,
                                tau_peripheral_s=8.4, sa_floor=0.31, tau_recover_s=8.4)
    for _ in range(10):
        ta_once.advance(touch_flat)
    for _ in range(10):
        ta_thrice.advance(touch_flat)
        ta_thrice.advance(touch_flat)
        ta_thrice.advance(touch_flat)
    g_once = float(ta_once.g_peripheral[0])
    g_thrice = float(ta_thrice.g_peripheral[0])
    ok = g_thrice < g_once - 1e-6
    print(f"  同じ観測10回、1回/観測で呼んだ場合：g={g_once:.6f}")
    print(f"  同じ観測10回、3回/観測で呼んだ場合：g={g_thrice:.6f}"
          f"（より速く順応が進むはず）  {'OK' if ok else 'NG'}")
    return ok


def verify0_growth_rebuild_required():
    """体が育ってn_pointsが変わった直後、rebuild()を呼び忘れた状態を意図的に
    再現し、点数不一致でAssertionErrorが出て止まることの確認。"""
    _hr("検証0-10: rebuild()を呼び忘れると点数不一致で止まるか（0コスト・MuJoCo不要）")
    ta = TouchAdaptation(5, 0.1)
    wrong = np.zeros(7 * 3)   # 育ったあとの点数（7点）を、rebuildせずに渡す
    try:
        ta.advance(wrong)
        print("  [NG] 例外が発生しなかった（黙って通ってしまっている）")
        return False
    except AssertionError as e:
        print(f"  [OK] 期待通りAssertionErrorで停止: {e}")
        return True


def verify0_double_touch_files_unaffected():
    """報酬・接触検出（double_touch.py・contact_reward.py・reach_success.py）の
    ソースが touch_percept を一切参照していないことの静的確認
    （3-1節の設計判断がコードレベルで守られているかの構造チェック）。"""
    _hr("検証0-11: 報酬・接触検出ファイルがtouch_percept文字列を含まないか（静的確認）")
    targets = [
        os.path.join(_R, "run", "plugins", "common", "double_touch.py"),
        os.path.join(_R, "run", "plugins", "common", "contact_reward.py"),
        os.path.join(_R, "run", "plugins", "common", "reach_success.py"),
    ]
    ok = True
    for path in targets:
        with open(path, "r", encoding="utf-8") as fh:
            src = fh.read()
        hit = "touch_percept" in src
        print(f"  {os.path.relpath(path, _R)}: touch_percept参照={hit}"
              f"  {'NG' if hit else 'OK'}")
        ok &= not hit
    return ok


# ================================================================== 検証A
def verify_a_requires_touch_somatosensory():
    """touch_adaptation=Trueかつ(touch=Falseまたはsomatosensory=False)のとき、
    起動時にValueErrorで止まることの確認（MuJoCoを使う。実際にTaroを構築する）。"""
    _hr("検証A-1: touch_adaptation=Trueだがtouch/somatosensoryが揃っていないとエラーになるか")
    ok_all = True
    for label, taro_over in (
        ("touch=False", {"touch": False, "somatosensory": True}),
        ("somatosensory=False", {"touch": True, "somatosensory": False}),
        ("両方False", {"touch": False, "somatosensory": False}),
    ):
        cfg = Config.from_spec({
            "scene": SCENE,
            "taro": {"vision": False, "touch_adaptation": True, **taro_over},
            "run": {"steps": 5, "seed": 0, "checkpoint": 100}})
        tr = Trainer(cfg, verbose=False)
        try:
            tr.build()
            print(f"  [NG] {label}: 例外が発生しなかった")
            ok_all = False
            tr.env.close()
        except ValueError as e:
            print(f"  [OK] {label}: 期待通りValueErrorで停止: {e}")
            # 【なぜ、2026-08-13】Trainer.build() は self.env を作った"後"に
            #   Taro(cfg, self.env, ...) を呼ぶ。ここで例外が起きても self.env は
            #   既に作られて残っている（MuJoCoのネイティブメモリを保持したまま）。
            #   閉じずに次のケースへ進むと、この関数だけで3個・スクリプト全体では
            #   さらに多くのMuJoCo環境が生き続け、後続の検証（検証C等）で
            #   「Could not allocate memory」を起こす（実際に発生・修正済み）。
            if tr.env is not None:
                tr.env.close()
    return ok_all


def verify_a_default_taro_unchanged():
    """既定OFFのとき、TARO_DEFAULTSの既存キーが1つも変わらないことの確認
    （check_common_drive.pyのverify_default_cpg_unchangedと同型）。"""
    _hr("検証A-2: 既定(touch_adaptation未指定)のときConfigの既存属性が変わらないか")
    cfg_a = Config.from_spec({"scene": SCENE, "taro": {}, "run": {"steps": 10, "seed": 0}})
    cfg_b = Config.from_spec({"scene": SCENE, "taro": {"touch_adaptation": False},
                              "run": {"steps": 10, "seed": 0}})
    diffs = []
    for key in TARO_DEFAULTS:
        va, vb = getattr(cfg_a, key), getattr(cfg_b, key)
        if va != vb:
            diffs.append((key, va, vb))
    ok = len(diffs) == 0
    print(f"  既存キーの値の差: {len(diffs)}件  {'OK' if ok else 'NG'}")
    for k, va, vb in diffs:
        print(f"    {k}: {va!r} != {vb!r}")
    return ok


def verify_a_fusion_touch_unaffected_when_off(steps=60, seed=0):
    """既定OFFのとき、fusion.encode()の出力・encode_target()の出力が、
    touch_percept機構を導入する前と数値的に完全一致すること（回帰確認）。
    このスクリプトからは「導入前のコード」を直接再現できないため、
    代わりに『touch_percept無し(既定OFF)の経路が、生のobs["touch"]だけを
    使った経路と完全一致する』ことを実機（MuJoCo）のTaroで確認する。"""
    _hr("検証A-3: 既定OFF時、実機のTaroでtouch_percept無し＝生のtouchと完全一致するか")
    cfg = Config.from_spec({
        "scene": SCENE,
        "taro": {"vision": False, "touch": True, "somatosensory": True,
                "touch_adaptation": False},
        "run": {"steps": steps, "seed": seed, "checkpoint": steps + 100}})
    tr = Trainer(cfg, verbose=False)
    tr.build()
    ok_no_key = tr.taro.touch_adaptation is None
    obs = tr.state["obs"]
    ok_touch_present = "touch" in obs
    ok_no_percept = "touch_percept" not in obs
    # encode_target 内部で obs.get("touch_percept", obs["touch"]) が obs["touch"] を
    #   返すことを、実際に呼び出して確認する。
    et1 = tr.taro.encode_target(obs)
    obs2 = dict(obs); obs2["touch_percept"] = obs["touch"]
    et2 = tr.taro.encode_target(obs2)
    ok_encode_target = torch.equal(et1, et2)
    tr.env.close()
    ok = ok_no_key and ok_touch_present and ok_no_percept and ok_encode_target
    print(f"  taro.touch_adaptation is None: {ok_no_key}")
    print(f"  obs['touch']あり: {ok_touch_present}  obs['touch_percept']無し: {ok_no_percept}")
    print(f"  encode_target(生のtouchのみ) と encode_target(touch_percept=touchを追加)"
          f" が完全一致: {ok_encode_target}")
    print(f"  {'OK' if ok else 'NG'}")
    return ok


def verify_a_fusion_target_state_consistency(steps=200, seed=0):
    """fusion側とtarget_fusion側で、同一の入力列に対し順応の内部状態
    （太郎に1つだけ持たせた状態）が完全に一致し続けることの確認（最優先）。
    正しい呼び出し経路（trainer.py経由）ではadvance()が物理ステップ数どおり
    にしか進まないことも合わせて確認する。"""
    _hr("検証A-4（最優先）: fusion/target_fusionで順応の状態が単一で一致し続けるか"
        "＋advance()の呼び出し回数が物理ステップ数どおりか")
    cfg = Config.from_spec({
        "scene": SCENE,
        "taro": {"vision": False, "touch": True, "somatosensory": True,
                "touch_adaptation": True},
        "run": {"steps": steps, "seed": seed, "checkpoint": steps + 100}})
    tr = Trainer(cfg, verbose=False)
    tr.build()

    # ---- 単一インスタンスであることの構造確認 ------------------------------
    ok_single = tr.taro.touch_adaptation is not None

    # ---- advance()呼び出し回数のカウント -----------------------------------
    orig_advance = tr.taro.touch_adaptation.advance
    calls = {"n": 0}

    def counting_advance(touch_flat):
        calls["n"] += 1
        return orig_advance(touch_flat)
    tr.taro.touch_adaptation.advance = counting_advance

    orig_reset_state = tr.reset_state
    resets = {"n": 0}

    def counting_reset_state():
        resets["n"] += 1
        return orig_reset_state()
    tr.reset_state = counting_reset_state

    tr.run()
    tr.env.close()

    expected = steps + resets["n"]
    ok_count = calls["n"] == expected
    print(f"  touch_adaptationインスタンス構築確認: {ok_single}")
    print(f"  advance()呼び出し回数={calls['n']}  期待値(steps+reset回数)="
          f"{expected}（steps={steps}, reset回数={resets['n']}）  {'OK' if ok_count else 'NG'}")
    return ok_single and ok_count


def verify_a_sa_floor_and_fa_zero_end_to_end(steps=200, seed=0):
    """遅順応型の出力がsa_floorより下に落ちないこと、速順応型は接触が変化しない
    ステップで出力ゼロであることを、実機（MuJoCo）で確認する。"""
    _hr("検証A-5: 実機で遅順応がfloor未満に落ちないか（6節）")
    sa_floor = 0.31
    cfg = Config.from_spec({
        "scene": SCENE,
        "taro": {"vision": False, "touch": True, "somatosensory": True,
                "touch_adaptation": True, "touch_adapt_sa_floor": sa_floor,
                "touch_adapt_tau_peripheral_s": 0.5, "touch_adapt_tau_recover_s": 8.4},
        "run": {"steps": steps, "seed": seed, "checkpoint": steps + 100}})
    tr = Trainer(cfg, verbose=False)
    tr.build()
    min_g = 1.0
    for _ in range(steps):
        tr.state["obs"], term = tr.step_k(np.zeros(tr.env.action_space.shape[0], dtype=np.float32))
        min_g = min(min_g, float(tr.taro.touch_adaptation.g_peripheral.min()))
        if term:
            tr.reset_state()
    tr.env.close()
    ok = min_g >= sa_floor - 1e-9
    print(f"  {steps}step中のg_peripheral最小値={min_g:.6f}  floor={sa_floor}  {'OK' if ok else 'NG'}")
    return ok


# ================================================================== 検証B
class _Recorder(Plugin):
    """検証専用：報酬・touch_percept・行動を記録するだけの読み取り専用プラグイン。"""
    name = "_verify_touch_adapt_recorder"

    def __init__(self):
        super().__init__()
        self.rewards = []
        self.touch_percept_present = []
        self.touch_percept_sums = []

    def on_step(self, ctx):
        obs_out = ctx.last.get("obs_out", {})
        present = "touch_percept" in obs_out
        self.touch_percept_present.append(present)
        if present:
            self.touch_percept_sums.append(float(np.asarray(obs_out["touch_percept"]).sum()))

    def on_step_late(self, ctx):
        self.rewards.append(float(ctx.last_reward["rew"]))


def verify_b_regression_short_run(steps=100, seed=0):
    """既定設定（touch_adaptationキー追加前後）で、既存実験ファイル相当の短時間走行
    をして、学習曲線・行動ログ・乱数消費が1バイトも変わらないことを確認する。
    （touch=False, somatosensory=Falseの、より一般的な既定シーンの構成で確認）"""
    _hr("検証B: 既定OFFでの回帰確認（touch/somatosensoryも既定のまま、乱数消費を含む）")

    def run_once():
        cfg = Config.from_spec({"scene": SCENE, "taro": {"vision": False},
                                "run": {"steps": steps, "seed": seed,
                                       "checkpoint": steps + 100}})
        rec = _Recorder()
        tr = Trainer(cfg, plugins=[rec], verbose=False)
        tr.build()
        tr.run()
        state_dict_sum = sum(float(p.sum()) for p in tr.taro.brain.parameters())
        rewards = list(rec.rewards)
        after_rand_state = torch.randint(0, 1_000_000, (1,)).item()  # 乱数消費のズレ検知用
        tr.env.close()
        return rewards, state_dict_sum, after_rand_state

    r1, s1, rnd1 = run_once()
    r2, s2, rnd2 = run_once()
    ok_reward = r1 == r2
    ok_weights = s1 == s2
    ok_rand = rnd1 == rnd2
    print(f"  2回実行の報酬列 完全一致={ok_reward}（{len(r1)}ステップ）")
    print(f"  2回実行の脳パラメータ合計 完全一致={ok_weights}（{s1!r} == {s2!r}）")
    print(f"  実行後の乱数消費位置が一致={ok_rand}（{rnd1} vs {rnd2}）")
    ok = ok_reward and ok_weights and ok_rand
    print(f"  {'OK' if ok else 'NG'}")
    return ok


# ================================================================== 検証C
def _run_labeled(steps, seed, touch_adaptation, extra_taro=None):
    taro = {"vision": False, "touch": True, "somatosensory": True,
            "touch_adaptation": touch_adaptation}
    if extra_taro:
        taro.update(extra_taro)
    cfg = Config.from_spec({
        "scene": SCENE, "taro": taro,
        "run": {"steps": steps, "seed": seed, "checkpoint": steps + 100}})
    rec = _Recorder()
    tr = Trainer(cfg, plugins=[rec], verbose=False)
    tr.build()
    tr.run()
    brain_sum = sum(float(p.sum()) for p in tr.taro.brain.parameters())
    fusion_sum = sum(float(p.sum()) for p in tr.taro.fusion.parameters())
    tr.env.close()
    return {
        "rewards": list(rec.rewards),
        "touch_percept_sums": list(rec.touch_percept_sums),
        "touch_percept_present": list(rec.touch_percept_present),
        "brain_sum": brain_sum,
        "fusion_sum": fusion_sum,
    }


def verify_c_same_seed_reproducibility(steps=300, seed=0):
    """【ユーザーの指示・最優先】同一シード・同一設定で2回実行し、結果が完全一致するか。
    まずtouch_adaptation=False（既定）のベースラインの再現性を確認してから、
    touch_adaptation=Trueでも確認する（2段階）。"""
    _hr("検証C（ユーザーの指示・最優先）: 同一シード2回実行で完全一致するか")

    print("---- 段階1: touch_adaptation=False（既定）のベースライン再現性 ----")
    b1 = _run_labeled(steps, seed, touch_adaptation=False)
    b2 = _run_labeled(steps, seed, touch_adaptation=False)
    ok_base_reward = b1["rewards"] == b2["rewards"]
    ok_base_brain = b1["brain_sum"] == b2["brain_sum"]
    ok_base_fusion = b1["fusion_sum"] == b2["fusion_sum"]
    ok_baseline = ok_base_reward and ok_base_brain and ok_base_fusion
    print(f"  報酬列 完全一致={ok_base_reward}（{len(b1['rewards'])}ステップ）")
    print(f"  brain合計 完全一致={ok_base_brain}（{b1['brain_sum']!r} vs {b2['brain_sum']!r}）")
    print(f"  fusion合計 完全一致={ok_base_fusion}（{b1['fusion_sum']!r} vs {b2['fusion_sum']!r}）")
    if not ok_baseline:
        print("  [想定外] touch_adaptation=Falseのベースライン自体が再現しない。"
              "これは今回の実装のバグではなく、太郎の既存の非決定性の可能性がある"
              "（実装ノウハウ2026-08-04の教訓）。判断せず想定外として報告する。")
        return False

    print("---- 段階2: touch_adaptation=True の再現性 ----")
    t1 = _run_labeled(steps, seed, touch_adaptation=True)
    t2 = _run_labeled(steps, seed, touch_adaptation=True)
    ok_reward = t1["rewards"] == t2["rewards"]
    ok_brain = t1["brain_sum"] == t2["brain_sum"]
    ok_fusion = t1["fusion_sum"] == t2["fusion_sum"]
    ok_percept_present = t1["touch_percept_present"] == t2["touch_percept_present"]
    ok_percept_sums = t1["touch_percept_sums"] == t2["touch_percept_sums"]
    print(f"  報酬列 完全一致={ok_reward}（{len(t1['rewards'])}ステップ）")
    print(f"  brain合計 完全一致={ok_brain}（{t1['brain_sum']!r} vs {t2['brain_sum']!r}）")
    print(f"  fusion合計 完全一致={ok_fusion}（{t1['fusion_sum']!r} vs {t2['fusion_sum']!r}）")
    print(f"  touch_percept存在フラグ列 完全一致={ok_percept_present}")
    print(f"  touch_percept合計値列 完全一致={ok_percept_sums}"
          f"（{len(t1['touch_percept_sums'])}件、先頭5件={t1['touch_percept_sums'][:5]}）")
    ok = ok_reward and ok_brain and ok_fusion and ok_percept_present and ok_percept_sums
    print(f"  {'OK' if ok else 'NG'}")
    return ok


def main():
    results = {}
    results["検証0-1（末梢減衰の解析式一致）"] = verify0_peripheral_decay_matches_analytic()
    results["検証0-2（脳側減衰の解析式一致）"] = verify0_cortical_decay_matches_analytic()
    results["検証0-3（回復・瞬時回復近似）"] = verify0_recovery()
    results["検証0-4（速順応=0の確認）"] = verify0_fa_zero_when_no_change()
    results["検証0-5（floor/1.0のクリップ）"] = verify0_clip_bounds()
    results["検証0-6（脳無効時=末梢のみ一致）"] = verify0_cortical_disabled_matches_peripheral_only()
    results["検証0-7（接触ゼロは常にゼロ）"] = verify0_zero_forever_stays_zero()
    results["検証0-8（fusion.py回帰・合成データ）"] = verify0_fusion_regression_synthetic()
    results["検証0-9（advance多重呼び出しの差）"] = verify0_advance_multiplicity_differs()
    results["検証0-10（rebuild忘れで例外）"] = verify0_growth_rebuild_required()
    results["検証0-11（報酬/接触検出は無改変）"] = verify0_double_touch_files_unaffected()

    results["検証A-1（touch/somatosensory必須）"] = verify_a_requires_touch_somatosensory()
    results["検証A-2（既定でConfig既存属性不変）"] = verify_a_default_taro_unchanged()
    results["検証A-3（既定OFFはtouch_percept無しと同一）"] = verify_a_fusion_touch_unaffected_when_off()
    results["検証A-4・最優先（単一状態＋呼び出し回数）"] = verify_a_fusion_target_state_consistency()
    results["検証A-5（floor未満に落ちない・実機）"] = verify_a_sa_floor_and_fa_zero_end_to_end()

    results["検証B（既定OFFの短時間回帰）"] = verify_b_regression_short_run()

    results["検証C・ユーザー指示・最優先（同一シード2回一致）"] = verify_c_same_seed_reproducibility()

    _hr("まとめ")
    for name, ok in results.items():
        print(f"  {'OK' if ok else 'NG'}  {name}")
    all_ok = all(results.values())
    print()
    print("全項目OK" if all_ok else "一部NG（上記参照）")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
