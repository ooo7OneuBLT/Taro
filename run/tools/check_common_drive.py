# -*- coding: utf-8 -*-
"""新しい駆動モジュール（伸張反射＋揺らぐ振動子の共通駆動）の検証。

仕様：作業記録（非公開）
設計：作業記録（非公開）

仕様「検証（依頼書②の順番を厳守）」の1〜4を、この1本のスクリプトにまとめる。

    検証1  0コスト・振動子の合成の検算（MuJoCo不要）
    検証2  回帰確認：rho=0のとき共通成分が出力に一切影響しないこと（MuJoCo不要）
    検証3  静的な運動学スイープ：moment_1は常に正・moment_2は常に負であることの
           確認（write_joint_commandの罠を踏んでいないことの根拠、仕様C節）＋
           qposスイープでlce_1/lce_2が単調であることの確認
    検証4  通ってはいけない条件：共通駆動の振幅を実質ゼロにした状態で伸張反射だけを
           動かすと、外力が無い限り関節が平衡点で静止し続けること
    追加   spinal_drive_mode=reflex_common を actuation=joint で指定したとき、
           またnoise=coloredと同時指定したとき、起動時にValueErrorで止まること
           （落とし穴チェックリスト項86「通ってはいけない条件で確かめる」）

【この道具がやらないこと】
    ・学習は回さない（run/main.pyは使わない。数秒〜十数秒で終わる単体スクリプト）
    ・感度分析（rho・grouping・振動子パラメータを振って達成条件を見る）はしない
      （測定の担当。仕様「やらないこと」）

使い方::

    .venv/Scripts/python.exe run/tools/check_common_drive.py
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
_R = r"C:\claude\AI\Taro"
os.chdir(_R)
if _R not in sys.path:
    sys.path.insert(0, _R)

import numpy as np                                          # noqa: E402

from run.config import Config                                # noqa: E402
from run.plugins.common import scene as scene_mod            # noqa: E402
# taro_setup を import すると taro_core/src/brain 等が sys.path に入るので、
#   spinal_cord パッケージはこの import の後で初めて読める（cpg.pyと同じ流儀）。
from run.taro_setup import Taro, _reflex_common_joint_indices  # noqa: E402
from run.trainer import Trainer                               # noqa: E402

from spinal_cord.common_drive import RhythmicCommonDriveGroup  # noqa: E402

SCENE = "新生児_仰向け_柵なし"


def _hr(title):
    print("=" * 88)
    print(title)
    print("=" * 88)


# ---------------------------------------------------------------- 検証1
def verify1_oscillator_mix():
    """独立なWanderingOscillator2本を7-2節の式で混ぜたとき、
    a) 混ぜた後の信号の分散が、混ぜる前（独立成分単体）の分散とほぼ一致すること
    b) 十分長い時間平均をとったときの時間平均相関が、rhoに近い値になること
    を確認する（実装前に済ませる0コスト診断）。
    """
    _hr("検証1: 振動子の合成の検算（0コスト・MuJoCo不要）")
    dt = 0.01
    T = 400000           # 4000秒ぶん（振動子の周期0.5Hz=2秒の何百倍。
                          # 中間のrho値は相関のサンプリング誤差が大きく、
                          # 40000stepでは許容差0.1を僅かに超えることがあった
                          # ため、サンプル数を10倍にして安定させた）
    osc_kwargs = dict(f0=0.5, A0=1.0, tau_f=2.0, tau_A=2.0)

    # ---- a) 分散の保存 -----------------------------------------------------
    # 「混ぜる前」の基準＝独立な1本のWanderingOscillator単体の分散。
    solo = RhythmicCommonDriveGroup([0], 0.0, osc_kwargs, seed=101)
    xs_solo = np.array([solo.step(dt)[0] for _ in range(T)])
    var_solo = float(np.var(xs_solo))

    ok_a = True
    for rho in (0.0, 0.3, 0.6, 0.9, 1.0):
        group = RhythmicCommonDriveGroup([0, 1], rho, osc_kwargs, seed=202)
        xs = np.array([list(group.step(dt).values()) for _ in range(T)])
        var_mixed = float(np.var(xs[:, 0]))
        rel_err = abs(var_mixed - var_solo) / var_solo
        ok = rel_err < 0.15
        ok_a &= ok
        print(f"  rho={rho:.1f}  混合後分散={var_mixed:.4f}  基準分散={var_solo:.4f}"
              f"  相対誤差={rel_err*100:.1f}%  {'OK' if ok else 'NG'}")

    # ---- b) 時間平均相関がrhoに近い ------------------------------------------
    ok_b = True
    print("  ---")
    for rho in (0.0, 0.3, 0.6, 0.9, 1.0):
        group = RhythmicCommonDriveGroup([0, 1], rho, osc_kwargs, seed=303)
        xs = np.array([list(group.step(dt).values()) for _ in range(T)])
        corr = float(np.corrcoef(xs[:, 0], xs[:, 1])[0, 1])
        ok = abs(corr - rho) < 0.1
        ok_b &= ok
        print(f"  rho={rho:.1f}  時間平均相関={corr:.3f}  |差|={abs(corr-rho):.3f}"
              f"  {'OK' if ok else 'NG'}")
    print("  注意：振動子はノイズと違い瞬時の相関は位相依存で振れるため、"
          "時間平均で確認している（瞬時値では確認しない）。")
    return ok_a and ok_b


# ---------------------------------------------------------------- 検証2
def verify2_rho_zero_no_effect():
    """rho=0のとき、RhythmicCommonDriveGroup経由の各関節の出力が、共通成分の
    WanderingOscillatorの値をどう変えても一切変化しないこと（sqrt(rho)=0で
    完全に無効化されていることをコードレベルで確認する）。"""
    _hr("検証2: 回帰確認（rho=0で共通成分が出力に一切影響しないこと。MuJoCo不要）")
    dt = 0.01
    kwargs = dict(f0=0.5, A0=1.0, tau_f=2.0, tau_A=2.0)
    group_a = RhythmicCommonDriveGroup([0, 1, 2], 0.0, kwargs, seed=7)
    group_b = RhythmicCommonDriveGroup([0, 1, 2], 0.0, kwargs, seed=7)
    # bだけ、共通成分の出力を極端な値へ強制的に差し替える（step()を丸ごと置換）。
    group_b.common.step = lambda _dt: 1.0e12
    n_checked, n_mismatch = 0, 0
    for _ in range(3000):
        oa = group_a.step(dt)
        ob = group_b.step(dt)
        for j in oa:
            n_checked += 1
            if abs(oa[j] - ob[j]) > 1e-9:
                n_mismatch += 1
    ok = n_mismatch == 0
    print(f"  共通成分を1e12に固定しても出力が変わった箇所: {n_mismatch}/{n_checked}"
          f"  {'OK' if ok else 'NG'}")
    return ok


# ---------------------------------------------------------------- 検証3
def verify3_moment_arm_signs_and_sweep(scene_name=SCENE):
    """moment_1は常に正・moment_2は常に負であることを実機の値で確認する
    （write_joint_commandの罠を構造的に踏まない根拠、仕様C節）。
    あわせて、moment arm経由の変換（線形式）でqposを可動域いっぱいスイープしたとき、
    lce_1・lce_2が単調に変化することを確認する。

    注意：moment_1・moment_2はMuscleModel構築時に確定する**固定の線形係数**
    （lce_1 = (qpos-qpos_spring)*moment_1 + lce_1_ref）であり、qposに対して
    常に線形＝単調である（傾きが定数のため、原理的に非単調にはなり得ない）。
    2026-08-11のCPGシナジー符号バグ修正で問題になった「静的スイープでの非単調」
    （肩水平・肩外転内転・手首）は、**生のqposと生理学的な屈曲/伸展方向の対応づけ**
    に関するものであり、moment_1/moment_2経由の変換方式ではその対応づけ自体が
    不要（4節）なため、この種の非単調性は原理的に起こり得ない。これはこの検証で
    "単調だった"という結果が出ても、それは**この変換方式の構造上保証されている
    ことの確認**であって、CPGの静的スイープが直面したのと同じ種類の不確実性を
    解消したわけではないことを、ここに明記しておく（仕様の要求どおり）。
    """
    _hr("検証3: moment_1/moment_2の符号確認 ＋ qposスイープでの単調性確認")
    cfg = Config.from_spec({"scene": scene_name, "taro": {"vision": False},
                            "run": {"steps": 10, "seed": 0}})
    env, sc, hands = scene_mod.build(cfg.scene, taro=dict(cfg._taro), seed=0,
                                     verbose=False, hybrid=True)
    env.reset(seed=0)
    u = env.unwrapped
    am = u.actuation_model
    m1 = np.asarray(am.moment_1, dtype=np.float64)
    m2 = np.asarray(am.moment_2, dtype=np.float64)
    n = len(m1)
    n_pos1 = int(np.sum(m1 > 0))
    n_neg2 = int(np.sum(m2 < 0))
    print(f"  moment_1 > 0 : {n_pos1}/{n}（min={m1.min():.4f}, max={m1.max():.4f}）")
    print(f"  moment_2 < 0 : {n_neg2}/{n}（min={m2.min():.4f}, max={m2.max():.4f}）")
    ok_sign = (n_pos1 == n) and (n_neg2 == n)

    model = u.model
    jnt_ids = np.asarray(am.mimo_actuated_joints, dtype=int)
    n_mono = 0
    for i in range(n):
        jid = int(jnt_ids[i])
        lo, hi = float(model.jnt_range[jid, 0]), float(model.jnt_range[jid, 1])
        qpos_spring = float(model.qpos_spring[int(model.jnt_qposadr[jid])])
        qs = np.linspace(lo, hi, 7)
        lce1 = (qs - qpos_spring) * m1[i] + float(am.lce_1_ref[i])
        lce2 = (qs - qpos_spring) * m2[i] + float(am.lce_2_ref[i])
        mono1 = bool(np.all(np.diff(lce1) > 0) or np.all(np.diff(lce1) < 0))
        mono2 = bool(np.all(np.diff(lce2) > 0) or np.all(np.diff(lce2) < 0))
        if mono1 and mono2:
            n_mono += 1
    ok_mono = n_mono == n
    print(f"  qposスイープ(7点、線形式)で単調だった関節: {n_mono}/{n}"
          f"  {'OK' if ok_mono else 'NG'}")
    env.close()
    return ok_sign and ok_mono


# ---------------------------------------------------------------- 検証4
def verify4_no_drift_without_oscillator(scene_name=SCENE, n_ticks=300,
                                        threshold_deg=3.0):
    """共通駆動（振動子）の振幅を実質ゼロにした状態（A0=A_min=A_max=0）で伸張反射
    だけを動かすと、外力が無い限り関節が平衡点で静止し続けることを確認する
    （反射だけでは自発運動が生まれない、という設計の前提。案①3節の実証）。

    注意：設計11節の文面は「rho=0で伸張反射だけを動かす」だが、rho=0でも
    独立成分の振動子自体は動き続けるため、文字通りrho=0にするだけでは関節は
    静止しない（独立成分が基準長を動かし続けるため）。この検証の趣旨（反射単独
    では自発運動を生まないことの実証）に忠実に、振動子の振幅をゼロにする条件で
    検証する（仕様の指定通り）。

    判定：前半n_ticks//2で初期姿勢からの収束（settle）を許し、**後半**の
    ドリフト（後半の開始時点 vs 終了時点の関節角の差）が小さいことを見る
    （t=0からの単純比較だと、反射がL0(0)へ収束する過程の動きまで
    「静止していない」と誤判定するため）。
    """
    _hr("検証4: 通ってはいけない条件（振動子の振幅ゼロ＋伸張反射のみ→静止するか）")
    osc_params = {"f0": 0.5, "A0": 0.0, "A_min": 0.0, "A_max": 0.0,
                  "tau_f": 2.0, "tau_A": 2.0, "amp": 1.0}
    cfg = Config.from_spec({
        "scene": scene_name,
        "taro": {"vision": False, "spinal_drive_mode": "reflex_common",
                 "common_drive_rho": 0.0, "common_drive_grouping": "none",
                 "common_drive_osc_params": osc_params},
        "run": {"steps": 10, "seed": 0}})
    tr = Trainer(cfg, verbose=False)
    tr.build()
    env = tr.env
    u = env.unwrapped
    am = u.actuation_model

    idx_by_limb = _reflex_common_joint_indices(env)
    all_idx = sorted({i for v in idx_by_limb.values() for i in v})
    jnt_ids = np.asarray(am.mimo_actuated_joints, dtype=int)
    qpos_adr = [int(u.model.jnt_qposadr[int(jnt_ids[i])]) for i in all_idx]

    a_zero = np.zeros(am.n_actuators * 2, dtype=np.float32)
    n_half = n_ticks // 2
    q_start = u.data.qpos[qpos_adr].copy()
    for _ in range(n_half):
        o, term = tr.step_k(a_zero)
        if term:
            tr.reset_state()
    q_mid = u.data.qpos[qpos_adr].copy()
    for _ in range(n_ticks - n_half):
        o, term = tr.step_k(a_zero)
        if term:
            tr.reset_state()
    q_end = u.data.qpos[qpos_adr].copy()

    drift_settle_deg = np.degrees(np.abs(q_mid - q_start))
    drift_static_deg = np.degrees(np.abs(q_end - q_mid))
    print(f"  対象関節数={len(all_idx)}（両腕・両脚）")
    print(f"  前半{n_half}tickの動き（収束過程・許容）："
          f"最大{drift_settle_deg.max():.3f}度 中央値{np.median(drift_settle_deg):.3f}度")
    print(f"  後半{n_ticks-n_half}tickのドリフト（静止しているはず）："
          f"最大{drift_static_deg.max():.3f}度 中央値{np.median(drift_static_deg):.3f}度")
    ok = bool(drift_static_deg.max() < threshold_deg)
    print(f"  しきい値={threshold_deg}度  判定={'OK（静止）' if ok else 'NG（動き続けている）'}")
    env.close()
    return ok


# ---------------------------------------------------------------- 追加：通ってはいけない条件
def verify_muscle_required():
    """spinal_drive_mode=reflex_common を actuation=joint で指定したら
    起動時にValueErrorで止まることを確認する（設計7-6節「追加の制約」）。"""
    _hr("追加検証: actuation=joint で reflex_common を指定するとエラーになるか")
    try:
        Config.from_spec({"scene": SCENE,
                          "taro": {"actuation": "joint", "spinal_drive_mode": "reflex_common"},
                          "run": {"steps": 10}})
        print("  [NG] 例外が発生しなかった（黙って通ってしまっている）")
        return False
    except ValueError as e:
        print(f"  [OK] 期待通りValueErrorで停止: {e}")
        return True


def verify_noise_conflict_rejected():
    """noise=colored と spinal_drive_mode=reflex_common の同時指定でエラーになるか。"""
    _hr("追加検証: noise=colored と reflex_common の同時指定でエラーになるか")
    try:
        Config.from_spec({"scene": SCENE,
                          "taro": {"noise": "colored", "spinal_drive_mode": "reflex_common"},
                          "run": {"steps": 10}})
        print("  [NG] 例外が発生しなかった")
        return False
    except ValueError as e:
        print(f"  [OK] 期待通りValueErrorで停止: {e}")
        return True


def verify_default_cpg_unchanged():
    """spinal_drive_mode既定("cpg")のとき、新キー追加そのものが既存キーの読み込みに
    影響しないことを確認する（Configオブジェクトの属性比較。2026-08-07のノウハウ）。"""
    _hr("追加検証: 既定(cpg)のときConfigの既存属性が変わらないか")
    cfg_a = Config.from_spec({"scene": SCENE, "taro": {}, "run": {"steps": 10, "seed": 0}})
    cfg_b = Config.from_spec({"scene": SCENE, "taro": {"spinal_drive_mode": "cpg"},
                              "run": {"steps": 10, "seed": 0}})
    diffs = []
    from run.config import TARO_DEFAULTS
    for key in TARO_DEFAULTS:
        va, vb = getattr(cfg_a, key), getattr(cfg_b, key)
        if va != vb:
            diffs.append((key, va, vb))
    ok = len(diffs) == 0
    print(f"  既存キーの値の差: {len(diffs)}件  {'OK' if ok else 'NG'}")
    for k, va, vb in diffs:
        print(f"    {k}: {va!r} != {vb!r}")
    return ok


def main():
    results = {}
    results["検証1（振動子の合成）"] = verify1_oscillator_mix()
    results["検証2（rho=0の回帰確認）"] = verify2_rho_zero_no_effect()
    results["検証3（moment_1/2の符号・単調性）"] = verify3_moment_arm_signs_and_sweep()
    results["検証4（振幅ゼロで静止するか）"] = verify4_no_drift_without_oscillator()
    results["追加（muscle必須チェック）"] = verify_muscle_required()
    results["追加（noise=colored同時指定の拒否）"] = verify_noise_conflict_rejected()
    results["追加（既定cpgでConfigの既存属性不変）"] = verify_default_cpg_unchanged()

    _hr("まとめ")
    for name, ok in results.items():
        print(f"  {'OK' if ok else 'NG'}  {name}")
    all_ok = all(results.values())
    print()
    print("全項目OK" if all_ok else "一部NG（上記参照）")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
