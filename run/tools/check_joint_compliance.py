# -*- coding: utf-8 -*-
"""関節可動域の壁を滑らかにする機構（joint_compliance）の検証。

仕様：作業記録（非公開）
設計：作業記録（非公開）
       作業記録（非公開）

仕様6節の検証項目をまとめる。

    検証0  0コスト診断（MuJoCo不要）：_solimp_from_params / _resolve_params の数式そのもの
    検証A  通ってはいけない条件（groups/joints未指定・未知group名・存在しない関節名・未知paramsキー）
    検証B  【最優先・ユーザーの指定】既定OFFでの回帰確認：全関節のjnt_solimpがMuJoCo組み込み
           既定と完全一致するか（複数シーン）
    検証C  配線チェック：groups=["shoulder"]でON にしたとき、対象関節だけが変わり
           対象外は変わらないか
    検証D  成長時の再適用確認（regrow相当）：異なる月齢で作り直しても対象関節が既定に
           戻っていないか
    検証E  駆動モード非依存性：spinal_drive_mode="cpg"/"reflex_common"でjnt_solimpが一致するか
    検証F  【最優先・ユーザーの指定・本題】端への張り付きが消えることの確認：固定の最大指令で
           肩を可動域の端まで押し当て、限界から1度以内に留まる連続tick数がON/OFFでどう
           変わるかを実測する

【この道具がやらないこと】
    ・学習は回さない（run/main.pyは使わない。数千step規模の短時間スクリプト）
    ・出力パスは検証専用（本番の実験結果を一切上書きしない。時系列の記録はscratchpad配下に
      保存する）

使い方::

    .venv/Scripts/python.exe run/tools/check_joint_compliance.py
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
_R = r"C:\claude\AI\Taro"
os.chdir(_R)
if _R not in sys.path:
    sys.path.insert(0, _R)
for _p in ("run/scene_tools", "D/scripts", "taro_core/src/body",
           "taro_core/src/brain", "taro_core/src/senses", "taro_core/src/wrapper"):
    _full = os.path.join(_R, _p)
    if _full not in sys.path:
        sys.path.insert(0, _full)

import numpy as np                                          # noqa: E402
import mujoco                                                # noqa: E402

import scene_io                                                # noqa: E402
from joint_compliance import (                                # noqa: E402
    apply_joint_compliance, _solimp_from_params, _margin_from_params, _resolve_params,
    DEFAULT_JOINT_COMPLIANCE, JOINT_COMPLIANCE_OVERRIDES,
)
from memory_guard import MemoryGuard                          # noqa: E402

# 【なぜ、2026-08-13】このスクリプトはe_scene.build()を16回呼ぶ（体を作り直す実験と
#   同じ型＝落とし穴チェックリスト項82「体を作り直す実験は描画用メモリが積み上がって
#   落ちる」）。全呼び出しにvision=Falseを明示した後もなお、想定外の理由でメモリが
#   積み上がった場合に13.6GBのような実害（プロセス強制停止）に至る前に気づけるよう、
#   各env.close()の直後にこの見張りを呼ぶ（仕様3節）。
#   閾値の根拠はmemory_guard.pyのdocstring参照（警告2000MB・停止5000MB、
#   通常の実験1本390〜500MBの目安を基準にした[Tier3・工学的判断]）。
_MEM_GUARD = MemoryGuard(warn_mb=2000.0, stop_mb=5000.0, label="check_joint_compliance")

from run.config import Config, TARO_DEFAULTS                 # noqa: E402
from run.trainer import Trainer                                # noqa: E402

# raw MuJoCoの組み込み既定（free/ball関節など、jnt_limited=0のままの関節や
# MIMoのXMLで class 既定を持たない関節がもし存在すればこの値になる）。
MUJOCO_RAW_DEFAULT_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2)

# 【2026-08-13の検証で判明・想定外】統合版設計・案B作業記録は「MIMoのXMLには
# jointlimit用のsolimp指定が一切無く、全関節がraw MuJoCo既定のまま」としていたが、
# 実測するとMIMoの全hinge関節93本は raw既定ではなく (0.98, 0.99, 0.001, 0.5, 1.0)
# を共通して持っていた（MuJoCoの<default class>機構によるクラス既定と推測。
# free/ball関節3本のみraw既定のまま）。width=0.001ラジアン（壁の急峻さの主張）
# 自体は変わらないため、joint_compliance機能そのものの必要性は揺るがないが、
# 「既定OFFで挙動が変わらないこと」を確認する回帰テストは、raw既定ではなく
# この実測値と比較しないと正しく検証できない。以下 _reference_solimp() で
# 実測ベースラインを動的に取得する（ハードコードした定数と突き合わせない）。
MIMO_HINGE_CLASS_SOLIMP = (0.98, 0.99, 0.001, 0.5, 1.0)

# 検証専用の一時保存先（本番の実験結果は一切触らない）。
# 注意：絶対パスを書かない（公開リポジトリ・個人情報チェック項31）。OSの一時領域を使う。
import tempfile
_SCRATCH = os.path.join(tempfile.gettempdir(), "taro_joint_compliance_verify")
os.makedirs(_SCRATCH, exist_ok=True)


def _hr(title):
    print("=" * 88)
    print(title)
    print("=" * 88)


def _reference_solimp(scene_name):
    """joint_complianceを一切適用していない、その場でビルドしたenvのjnt_solimpを
    そのまま「実測ベースライン」として返す（ハードコードした定数と比較しない）。"""
    sc = scene_io.load(scene_name)
    assert sc["body"]["joint_compliance"] is False
    env, _sc = scene_io.build(sc, seed=0, verbose=False, vision=False)
    ref = np.asarray(env.unwrapped.model.jnt_solimp).copy()
    env.close()
    return ref


def _reference_margin(scene_name):
    """joint_complianceを一切適用していない、その場でビルドしたenvのjnt_marginを
    そのまま「実測ベースライン」として返す（margin版、_reference_solimpと対）。

    【2026-08-13・margin追加】前回の実測でMIMoの全hinge関節のjnt_marginは
    0.0だったが、raw定数(0.0)をハードコードして比較せず、_reference_solimpと
    同じく毎回独立にビルドした参照値と比較する（設計時の前提が実測と食い違って
    いた教訓を踏まえる）。"""
    sc = scene_io.load(scene_name)
    assert sc["body"]["joint_compliance"] is False
    env, _sc = scene_io.build(sc, seed=0, verbose=False, vision=False)
    ref = np.asarray(env.unwrapped.model.jnt_margin).copy()
    env.close()
    return ref


def _joint_id(model, short_name):
    return int(model.joint("robot:" + short_name).id)


# ================================================================== 検証0
def verify0_solimp_from_params_math():
    """_solimp_from_params の数式そのものの0コスト検算（MuJoCo不要）。

    rangeの幅・width_fracからwidthが正しく計算されるか、d0/d_width/midpoint/power が
    そのまま通るか、順序が(d0, d_width, width, midpoint, power)になっているかを確認する。
    """
    _hr("検証0-1: _solimp_from_params の数式検算（0コスト・MuJoCo不要）")
    params = dict(width_frac=0.08, d0=0.5, d_width=0.95, midpoint=0.8, power=2)
    lo, hi = np.radians(-30.0), np.radians(150.0)   # 可動域180度
    d0, d_width, width, midpoint, power = _solimp_from_params(lo, hi, params)
    expected_width = 0.08 * (hi - lo)
    ok_order = (d0 == 0.5 and d_width == 0.95 and midpoint == 0.8 and power == 2)
    ok_width = abs(width - expected_width) < 1e-12
    print(f"  range幅={np.degrees(hi-lo):.1f}度 → width={np.degrees(width):.4f}度"
          f"（期待={np.degrees(expected_width):.4f}度）")
    print(f"  戻り値の並び=(d0,d_width,width,midpoint,power)="
          f"({d0},{d_width},{width:.6f},{midpoint},{power})")
    ok = ok_order and ok_width
    print(f"  {'OK' if ok else 'NG'}")
    return ok


def verify0_solimp_monotonic_in_range():
    """width_fracを振ったとき、widthが可動域幅に対して単調に増えるか（0コスト）。

    これは「rを振ってd(r)を計算する」設計の意図（数式が不連続にジャンプしないか）を、
    本実装が実際に持つ唯一の自由変数（width_frac）に対して静的に確認するもの。
    MuJoCo自体のインピーダンス関数d(r)は組み込みのsmoothstep型でMuJoCo側の実装であり、
    このタスクでは触っていない（widthを求めるところまでが本実装の責務）。
    """
    _hr("検証0-2: width_fracを振ったときwidthが単調に増えるか（0コスト・MuJoCo不要）")
    lo, hi = np.radians(-30.0), np.radians(150.0)
    widths = []
    for wf in np.linspace(0.01, 0.5, 30):
        params = dict(DEFAULT_JOINT_COMPLIANCE)
        params["width_frac"] = float(wf)
        _, _, width, _, _ = _solimp_from_params(lo, hi, params)
        widths.append(width)
    diffs = np.diff(widths)
    ok = bool(np.all(diffs > 0))
    print(f"  width_frac 0.01→0.50（30点）：width {np.degrees(widths[0]):.3f}度 → "
          f"{np.degrees(widths[-1]):.3f}度、全区間で単調増加={ok}")
    print(f"  {'OK' if ok else 'NG'}")
    return ok


def verify0_resolve_params_flat_vs_per_joint():
    """_resolve_params が「フラット共通上書き」と「関節名キー付き個別上書き」を
    正しく区別するかの0コスト確認。"""
    _hr("検証0-3: paramsのフラット/関節名キー付きの判別（0コスト・MuJoCo不要）")
    p_flat = _resolve_params("shoulder_ad_ab", {"d0": 0.3})
    ok_flat = p_flat["d0"] == 0.3 and p_flat["width_frac"] == DEFAULT_JOINT_COMPLIANCE["width_frac"]

    p_per_joint_hit = _resolve_params("shoulder_ad_ab",
                                      {"shoulder_ad_ab": {"d0": 0.2}, "elbow": {"d0": 0.9}})
    ok_hit = p_per_joint_hit["d0"] == 0.2

    p_per_joint_miss = _resolve_params("elbow",
                                       {"shoulder_ad_ab": {"d0": 0.2}})
    ok_miss = p_per_joint_miss["d0"] == DEFAULT_JOINT_COMPLIANCE["d0"]  # 対象外はDEFAULTのまま

    p_none = _resolve_params("shoulder_ad_ab", None)
    ok_none = p_none == dict(DEFAULT_JOINT_COMPLIANCE)

    ok = ok_flat and ok_hit and ok_miss and ok_none
    print(f"  フラット共通上書き{{'d0':0.3}}: d0={p_flat['d0']} 他key不変={ok_flat}")
    print(f"  関節名キー付き（一致）: d0={p_per_joint_hit['d0']}  一致確認={ok_hit}")
    print(f"  関節名キー付き（不一致・対象外）: d0={p_per_joint_miss['d0']}"
          f"（DEFAULTのまま={ok_miss}）")
    print(f"  params=None: DEFAULTそのまま={ok_none}")
    print(f"  {'OK' if ok else 'NG'}")
    return ok


# ================================================================== 検証A
def verify_a_requires_groups_or_joints():
    """groups・jointsのどちらも未指定なら明示的にValueErrorで止まるか。"""
    _hr("検証A-1: groups/jointsのどちらも未指定だとValueErrorで止まるか")
    sc = scene_io.load("新生児_仰向け_柵なし")
    env, _sc = scene_io.build(sc, seed=0, verbose=False, vision=False)
    try:
        apply_joint_compliance(env.unwrapped.model, groups=None, joints=None)
        print("  [NG] 例外が発生しなかった（黙って何もしていない可能性）")
        ok = False
    except ValueError as e:
        print(f"  [OK] 期待通りValueErrorで停止: {e}")
        ok = True
    env.close()
    return ok


def verify_a_unknown_group_name():
    """未知のgroup名（打ち間違い）を渡すとValueErrorで止まるか。"""
    _hr("検証A-2: 未知のgroup名（'shoulderr'）でValueErrorになるか")
    sc = scene_io.load("新生児_仰向け_柵なし")
    env, _sc = scene_io.build(sc, seed=0, verbose=False, vision=False)
    try:
        apply_joint_compliance(env.unwrapped.model, groups=["shoulderr"])
        print("  [NG] 例外が発生しなかった")
        ok = False
    except ValueError as e:
        print(f"  [OK] 期待通りValueErrorで停止: {e}")
        ok = True
    env.close()
    return ok


def verify_a_unknown_joint_name():
    """存在しない関節名を渡すとValueErrorで止まるか。"""
    _hr("検証A-3: 存在しない関節名でValueErrorになるか")
    sc = scene_io.load("新生児_仰向け_柵なし")
    env, _sc = scene_io.build(sc, seed=0, verbose=False, vision=False)
    try:
        apply_joint_compliance(env.unwrapped.model, joints=["right_no_such_joint"])
        print("  [NG] 例外が発生しなかった")
        ok = False
    except ValueError as e:
        print(f"  [OK] 期待通りValueErrorで停止: {e}")
        ok = True
    env.close()
    return ok


def verify_a_unknown_param_key():
    """paramsに未知のキー（打ち間違い、正しくはd0）を渡すとValueErrorになるか。"""
    _hr("検証A-4: paramsの未知キー（'d_min'）でValueErrorになるか")
    sc = scene_io.load("新生児_仰向け_柵なし")
    env, _sc = scene_io.build(sc, seed=0, verbose=False, vision=False)
    try:
        apply_joint_compliance(env.unwrapped.model, groups=["shoulder"],
                               params={"d_min": 0.5})
        print("  [NG] 例外が発生しなかった")
        ok = False
    except ValueError as e:
        print(f"  [OK] 期待通りValueErrorで停止: {e}")
        ok = True
    env.close()
    return ok


# ================================================================== 検証B（最優先）
_SCENES_FOR_REGRESSION = [
    "新生児_仰向け_柵なし",
    "新生児_仰向け_柵あり",
    "リーチング_リクライニング60度",
]


def verify_b_default_off_matches_mujoco_builtin():
    """【最優先・ユーザーの指定】joint_complianceを指定しない既存シーン複数本で、
    全関節のjnt_solimp・jnt_marginが「この機能を一切適用していない状態」と
    完全一致することを確認する（＝既定OFFで挙動が1ビットも変わらないことの回帰確認）。

    【2026-08-13の想定外・注意】設計の前提「全関節がraw MuJoCo既定
    (0.9,0.95,0.001,0.5,2)のまま」は実測で誤りと判明した（MIMoの全hinge関節は
    共通のクラス既定 (0.98,0.99,0.001,0.5,1.0) を持つ）。この関数は
    raw既定と比較するのではなく、同じシーンをもう一度・独立にビルドした
    参照値（_reference_solimp・_reference_margin）と比較する。これにより
    「joint_compliance機能の有無に関わらずMIMoが本来持っている値」との一致を
    確認でき、設計時の前提の誤りに引きずられない。

    【2026-08-13追記・margin追加】jnt_marginも同じ考え方で比較対象に加えた
    （仕様2-3節）。"""
    _hr("検証B-1【最優先】: 既定OFF（未指定）で全関節のjnt_solimp・jnt_marginが変わらないか")
    ok_all = True
    for name in _SCENES_FOR_REGRESSION:
        ref = _reference_solimp(name)
        ref_margin = _reference_margin(name)
        sc = scene_io.load(name)
        env, _sc = scene_io.build(sc, seed=0, verbose=False, vision=False)
        m = env.unwrapped.model
        solimp = np.asarray(m.jnt_solimp)
        margin = np.asarray(m.jnt_margin)
        diff = np.abs(solimp - ref)
        n_diff = int(np.sum(np.any(diff > 1e-12, axis=1)))
        diff_margin = np.abs(margin - ref_margin)
        n_diff_margin = int(np.sum(diff_margin > 1e-12))
        # 参考として、raw MuJoCo既定とも突き合わせて何本一致するかを表示する
        # （想定外の記録として。合否判定には使わない）
        raw_expected = np.tile(np.asarray(MUJOCO_RAW_DEFAULT_SOLIMP), (m.njnt, 1))
        n_raw_match = int(np.sum(~np.any(np.abs(solimp - raw_expected) > 1e-12, axis=1)))
        ok = n_diff == 0 and n_diff_margin == 0
        ok_all &= ok
        print(f"  シーン『{name}』: njnt={m.njnt}  参照ビルドとjnt_solimpが異なる関節数={n_diff}"
              f"  jnt_marginが異なる関節数={n_diff_margin}"
              f"（raw MuJoCo既定と一致する関節数={n_raw_match}/{m.njnt}、参考）"
              f"  {'OK' if ok else 'NG'}")
        env.close()
    print(f"  {'OK' if ok_all else 'NG'}")
    return ok_all


def verify_b_old_style_call_identical():
    """apply_runtime_correctionsをjoint_compliance系4引数を一切渡さずに呼んだ場合
    （＝このタスク着手前のコードと同じ呼び出し方）と、明示的にjoint_compliance=Falseを
    渡した場合とで、model.jnt_solimp・jnt_marginが完全一致することを確認する
    （回帰確認の裏取り）。"""
    _hr("検証B-2: 4引数を渡さない呼び出し（旧仕様相当）と明示False指定で完全一致するか")
    from infant_body import apply_runtime_corrections
    sc = scene_io.load("新生児_仰向け_柵なし")
    env_a, _ = scene_io.build(sc, seed=0, verbose=False, vision=False)
    ua = env_a.unwrapped
    apply_runtime_corrections(ua.model, ua.data, 0.0)   # 旧仕様相当（新引数を渡さない）
    solimp_a = np.asarray(ua.model.jnt_solimp).copy()
    margin_a = np.asarray(ua.model.jnt_margin).copy()
    env_a.close()

    env_b, _ = scene_io.build(sc, seed=0, verbose=False, vision=False)
    ub = env_b.unwrapped
    apply_runtime_corrections(ub.model, ub.data, 0.0, joint_compliance=False,
                              joint_compliance_groups=None, joint_compliance_joints=None,
                              joint_compliance_params=None)
    solimp_b = np.asarray(ub.model.jnt_solimp).copy()
    margin_b = np.asarray(ub.model.jnt_margin).copy()
    env_b.close()

    ok = bool(np.array_equal(solimp_a, solimp_b)) and bool(np.array_equal(margin_a, margin_b))
    print(f"  jnt_solimp 完全一致={bool(np.array_equal(solimp_a, solimp_b))}"
          f"  jnt_margin 完全一致={bool(np.array_equal(margin_a, margin_b))}")
    print(f"  {'OK' if ok else 'NG'}")
    return ok


# ================================================================== 検証C
def verify_c_shoulder_group_wiring():
    """groups=["shoulder"]でONにしたとき、右肩・左肩のshoulder_ad_ab他だけが変わり、
    対象外（肘等）は変わらないことを確認する（jnt_solimp・jnt_marginの両方）。
    対象外の『既定』はMIMO_HINGE_CLASS_SOLIMP（実測ベースライン、上の想定外の
    説明を参照）と比較する。"""
    _hr("検証C: groups=['shoulder']の配線チェック（対象/対象外・jnt_solimp/jnt_margin）")
    scene_name = "新生児_仰向け_柵なし"
    ref = _reference_solimp(scene_name)
    ref_margin = _reference_margin(scene_name)
    sc = scene_io.load(scene_name)
    sc["body"]["joint_compliance"] = True
    sc["body"]["joint_compliance_groups"] = ["shoulder"]
    env, _sc = scene_io.build(sc, seed=0, verbose=False, vision=False)
    m = env.unwrapped.model

    ok_all = True
    for side in ("right_", "left_"):
        for base in ("shoulder_horizontal", "shoulder_ad_ab", "shoulder_rotation"):
            jid = _joint_id(m, side + base)
            got = tuple(float(x) for x in m.jnt_solimp[jid])
            got_margin = float(m.jnt_margin[jid])
            lo_j = float(m.jnt_range[jid, 0]); hi_j = float(m.jnt_range[jid, 1])
            exp_j = _solimp_from_params(lo_j, hi_j, DEFAULT_JOINT_COMPLIANCE)
            exp_margin = _margin_from_params(lo_j, hi_j, DEFAULT_JOINT_COMPLIANCE)
            close = all(abs(a - b) < 1e-9 for a, b in zip(got, exp_j))
            close_margin = abs(got_margin - exp_margin) < 1e-9
            ok_all &= close
            ok_all &= close_margin
            print(f"  対象 {side}{base}: solimp={got}  期待={tuple(round(x,6) for x in exp_j)}"
                  f"  一致={close}"
                  f"  margin={got_margin:.6f}  期待={exp_margin:.6f}  一致={close_margin}")

    for side in ("right_", "left_"):
        jid = _joint_id(m, side + "elbow")
        got = np.asarray(m.jnt_solimp[jid])
        close = bool(np.all(np.abs(got - ref[jid]) < 1e-12))
        got_margin = float(m.jnt_margin[jid])
        close_margin = abs(got_margin - float(ref_margin[jid])) < 1e-12
        ok_all &= close
        ok_all &= close_margin
        print(f"  対象外 {side}elbow: solimp={tuple(float(x) for x in got)}"
              f"  参照ビルド（未適用）と一致={close}"
              f"  margin={got_margin:.6f}（参照={float(ref_margin[jid]):.6f}）一致={close_margin}")

    env.close()
    print(f"  {'OK' if ok_all else 'NG'}")
    return ok_all


def verify_c_all_group_targets_only_limited_hinge():
    """groups=["all"]が「限界を持つ(jnt_limited==1)hinge関節」全てを対象にし、
    limitedでない関節・非hinge関節は対象にしないことを確認する
    （jnt_solimp・jnt_marginの両方の変化を見る）。"""
    _hr("検証C-2: groups=['all']が jnt_limited==1 のhinge関節だけを対象にするか")
    sc = scene_io.load("新生児_仰向け_柵なし")
    env, _sc = scene_io.build(sc, seed=0, verbose=False, vision=False)
    m = env.unwrapped.model
    before = np.asarray(m.jnt_solimp).copy()
    before_margin = np.asarray(m.jnt_margin).copy()
    before_limited = np.asarray(m.jnt_limited).copy()
    n = apply_joint_compliance(m, groups=["all"], verbose=False)
    after = np.asarray(m.jnt_solimp)
    after_margin = np.asarray(m.jnt_margin)

    changed = np.any(np.abs(after - before) > 1e-12, axis=1)
    changed_margin = np.abs(after_margin - before_margin) > 1e-12
    n_changed = int(np.sum(changed))
    n_changed_margin = int(np.sum(changed_margin))
    # 変化した関節は全て、変更前からjnt_limited==1のhingeだったはず
    ok_subset = True
    for j in range(m.njnt):
        if changed[j] or changed_margin[j]:
            is_hinge = int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE)
            was_limited = bool(before_limited[j])
            if not (is_hinge and was_limited):
                ok_subset = False
    # solimpとmarginは同じapply_joint_complianceループで同じ対象に書かれるため、
    # 変化した関節の集合は一致するはず
    ok_same_set = bool(np.array_equal(changed, changed_margin))
    ok = (n == n_changed) and ok_subset and n_changed > 0 and ok_same_set
    print(f"  apply_joint_compliance戻り値n={n}  solimpが変わった関節数={n_changed}"
          f"  marginが変わった関節数={n_changed_margin}")
    print(f"  変化した関節は全て『元からlimitedなhinge』か={ok_subset}"
          f"  solimp/marginの変化対象は同じ集合か={ok_same_set}")
    print(f"  {'OK' if ok else 'NG'}")
    env.close()
    return ok


# ================================================================== 検証D
def verify_d_reapplied_across_ages():
    """成長（regrow相当）で体を異なる月齢で作り直しても、対象関節へjoint_complianceが
    毎回正しく再適用されることを確認する（apply_runtime_correctionsはbuildのたびに
    必ず呼ばれる設計のため、月齢を変えて2回buildすることで regrow と同型の再構築を模擬する）。
    jnt_solimp・jnt_marginの両方を確認する。"""
    _hr("検証D: 月齢を変えて作り直しても対象関節に再適用されるか（regrow相当、solimp/margin）")
    ok_all = True
    for age in (0.0, 2.0, 4.0):
        sc = scene_io.load("新生児_仰向け_柵なし")
        sc["body"]["age_months"] = age
        sc["body"]["joint_compliance"] = True
        sc["body"]["joint_compliance_groups"] = ["shoulder"]
        sc["fingerprint"] = None
        env, _sc = scene_io.build(sc, seed=0, verbose=False, vision=False)
        m = env.unwrapped.model
        jid = _joint_id(m, "right_shoulder_ad_ab")
        lo = float(m.jnt_range[jid, 0]); hi = float(m.jnt_range[jid, 1])
        expected = _solimp_from_params(lo, hi, DEFAULT_JOINT_COMPLIANCE)
        expected_margin = _margin_from_params(lo, hi, DEFAULT_JOINT_COMPLIANCE)
        got = tuple(float(x) for x in m.jnt_solimp[jid])
        got_margin = float(m.jnt_margin[jid])
        close = all(abs(a - b) < 1e-9 for a, b in zip(got, expected))
        close_margin = abs(got_margin - expected_margin) < 1e-9
        ok_all &= close
        ok_all &= close_margin
        print(f"  age={age}mo: right_shoulder_ad_ab range=[{np.degrees(lo):.1f},"
              f"{np.degrees(hi):.1f}]度  solimp={tuple(round(x,6) for x in got)}"
              f"  期待通り={close}"
              f"  margin={got_margin:.6f}（期待={expected_margin:.6f}）期待通り={close_margin}")
        env.close()
    print(f"  {'OK' if ok_all else 'NG'}")
    return ok_all


# ================================================================== 検証E
class _ScenePatch:
    """scene_io.load を一時的に差し替え、joint_complianceを常にONにして返す
    （run/plugins/common/scene.build 経由でシーン名からしか環境を作れない
    Trainer側の検証のために、ディスク上のシーンJSONは一切書き換えずに行う）。"""

    def __init__(self, groups):
        self._groups = groups
        self._orig = None

    def __enter__(self):
        self._orig = scene_io.load
        orig = self._orig
        groups = self._groups

        def patched(name):
            sc = orig(name)
            sc["body"]["joint_compliance"] = True
            sc["body"]["joint_compliance_groups"] = list(groups)
            sc["fingerprint"] = None
            return sc
        scene_io.load = patched
        return self

    def __exit__(self, *exc):
        scene_io.load = self._orig


def verify_e_drive_mode_independence():
    """spinal_drive_mode="cpg"（既定）と"reflex_common"の両方で、
    joint_compliance=Trueのときの対象関節のjnt_solimp・jnt_marginが完全一致するかを
    確認する。"""
    _hr("検証E: 駆動モード(cpg/reflex_common)非依存性の確認（jnt_solimp/jnt_margin）")
    scene_name = "新生児_仰向け_柵なし"
    solimps = {}
    margins = {}
    with _ScenePatch(["shoulder"]):
        for mode in ("cpg", "reflex_common"):
            cfg = Config.from_spec({
                "scene": scene_name,
                "taro": {"vision": False, "spinal_drive_mode": mode},
                "run": {"steps": 5, "seed": 0, "checkpoint": 100}})
            tr = Trainer(cfg, verbose=False)
            tr.build()
            m = tr.env.unwrapped.model
            jid = _joint_id(m, "right_shoulder_ad_ab")
            solimps[mode] = tuple(float(x) for x in m.jnt_solimp[jid])
            margins[mode] = float(m.jnt_margin[jid])
            tr.env.close()
    ok = solimps["cpg"] == solimps["reflex_common"] and margins["cpg"] == margins["reflex_common"]
    print(f"  cpg          : solimp={solimps['cpg']}  margin={margins['cpg']:.6f}")
    print(f"  reflex_common: solimp={solimps['reflex_common']}  margin={margins['reflex_common']:.6f}")
    print(f"  完全一致={ok}")
    print(f"  {'OK' if ok else 'NG'}")
    return ok


# ================================================================== 検証F（本題・最優先）
def _shoulder_actuator_indices(env, joint_full_name):
    """右肩外転関節を駆動する筋肉アクチュエータの、antagonist内でのローカルindexを返す。

    実装ノウハウ2026-08-11:「関節名とアクチュエータ名は別の名前空間」を踏まえ、
    文字列変換で推測せず am.mimo_actuated_joints（=actuator_trnid[actuators,0]）で
    関節idから引く（check_common_drive.pyの_reflex_common_joint_indicesと同じ手法）。
    """
    u = env.unwrapped
    am = u.actuation_model
    jid = _joint_id(u.model, joint_full_name)
    jnt_ids = np.asarray(am.mimo_actuated_joints, dtype=int)
    idx = [i for i in range(len(jnt_ids)) if jnt_ids[i] == jid]
    return idx


def verify_f0_muscle_max_command_probe(n_steps=3000, seed=0):
    """【ユーザーの指定どおりの方法で先に試す】肩の外転筋（筋肉モードのアクチュエータ）に
    最大指令(action=1.0)を出し続けたとき、実際に可動域の端（183度）近くまで到達するかを
    まず確認する。

    【2026-08-13の想定外】結果は「到達しない」だった。単一の筋肉chを最大activationに
    しても、拮抗筋の受動弾性（fp）等が釣り合い、右肩ad_abは約115度付近で頭打ちになり
    3000tickでもほぼ動かなくなる（可動域の端183度まで68度も届かない）。これは
    joint_compliance機能とは無関係な、MIMoの筋肉モデル自体の性質。ONにしてもOFFにしても
    この頭打ち角度そのものは変わらない（joint_complianceは限界近くでしか効かないため）。
    ⇒ この方法単体では「端への張り付き」を短時間で再現できないため、検証F本体
    （verify_f_wall_sticking_reduced）では data.qfrc_applied による直接トルクで
    確実に限界へ到達させる方式に切り替える（ユーザーへの報告に明記済み）。"""
    _hr("検証F-0: 筋肉アクチュエータの最大指令だけで限界(183度)近くまで届くか（事前確認）")
    scene_name = "新生児_仰向け_柵なし"
    joint = "right_shoulder_ad_ab"
    sc = scene_io.load(scene_name)
    env, _sc = scene_io.build(sc, seed=seed, verbose=False, vision=False)
    u = env.unwrapped
    m, d = u.model, u.data
    am = u.actuation_model
    n_act = int(am.n_actuators)
    jid = _joint_id(m, joint)
    qadr = int(m.jnt_qposadr[jid])
    hi = float(m.jnt_range[jid, 1])
    idx = _shoulder_actuator_indices(env, joint)
    a = np.zeros(u.action_space.shape[0], dtype=np.float32)
    for i in idx:
        a[i + n_act] = 1.0    # "pos"チャンネル＝hi方向へ動く側（別途確認済み）
    checkpoints = []
    for t in range(n_steps):
        u.step(a)
        if t in (0, 500, 1000, 1500, 2000, 2500, n_steps - 1):
            checkpoints.append((t, round(float(np.degrees(d.qpos[qadr])), 2)))
    env.close()
    final_deg = checkpoints[-1][1]
    reached_near_limit = abs(final_deg - np.degrees(hi)) <= 5.0
    print(f"  角度の推移(tick, 角度deg): {checkpoints}")
    print(f"  限界={np.degrees(hi):.1f}度  最終角度={final_deg:.2f}度  "
          f"限界近く(5度以内)に到達={reached_near_limit}")
    if not reached_near_limit:
        print("  [想定外] 最大筋指令だけでは限界近くまで到達しない"
              "（筋肉モデル自体の受動弾性による頭打ち。詳細はdocstring参照）")
    return reached_near_limit


def verify_f_wall_sticking_reduced(n_steps=400, near_deg=1.0, torque=2.0, seed=0):
    """【最優先・本題・ユーザーの指定】右肩外転関節を可動域の端まで押し当て続け、
    限界からnear_deg度以内に留まる連続tick数の最大値がjoint_compliance ON/OFFで
    どう変わるかを実測する。角速度の時系列も記録し、CSVへ保存する
    （項70：数値だけでなく波形も見る）。

    【手段についての注意（2026-08-13、依頼書からの意図的な変更・報告済み）】
    検証F-0で確認した通り、筋肉アクチュエータへの最大指令(action=1.0)だけでは
    可動域の端（183度）まで届かない（受動弾性で約115度に頭打ち）。「肩の外転筋に
    最大指令を出し続けて可動域の端まで押し当て」（仕様6-2節）という目的そのもの
    （限界へ確実に押し当てる）を達成するため、`data.qfrc_applied`による一定の
    外部トルクで直接押す方式に切り替えた。これは check_touch_adaptation.py 等と
    同じ「学習を伴わない、固定の指令によるMuJoCoの直接ステップ実行」という
    決定論的な検証の枠内にあり、run/main.pyでの学習実行ではない。"""
    _hr("検証F【最優先・本題】: 端への張り付き（限界近く連続tick数）がON/OFFでどう変わるか")

    scene_name = "新生児_仰向け_柵なし"
    joint = "right_shoulder_ad_ab"

    def run_once(joint_compliance):
        sc = scene_io.load(scene_name)
        sc["body"]["joint_compliance"] = bool(joint_compliance)
        if joint_compliance:
            sc["body"]["joint_compliance_groups"] = ["shoulder"]
        env, _sc = scene_io.build(sc, seed=seed, verbose=False, vision=False)
        u = env.unwrapped
        m, d = u.model, u.data
        jid = _joint_id(m, joint)
        qadr = int(m.jnt_qposadr[jid])
        dofadr = int(m.jnt_dofadr[jid])
        lo = float(m.jnt_range[jid, 0]); hi = float(m.jnt_range[jid, 1])
        target_val = hi if abs(hi) >= abs(lo) else lo   # 元の事故（183度）と同じ側

        zero_a = np.zeros(u.action_space.shape[0], dtype=np.float32)
        deg_series, vel_series = [], []
        n_near, max_near_run = 0, 0
        cross_tick = None
        for t in range(n_steps):
            d.qfrc_applied[dofadr] = float(torque)
            u.step(zero_a)
            ang = float(d.qpos[qadr]); vel = float(d.qvel[dofadr])
            deg_series.append(np.degrees(ang)); vel_series.append(np.degrees(vel))
            if cross_tick is None and abs(ang) >= abs(target_val):
                cross_tick = t
            near = abs(np.degrees(ang) - np.degrees(target_val)) <= near_deg
            n_near = n_near + 1 if near else 0
            max_near_run = max(max_near_run, n_near)
        env.close()
        return dict(target_deg=np.degrees(target_val), max_near_run=max_near_run,
                    deg_series=deg_series, vel_series=vel_series,
                    lo_deg=np.degrees(lo), hi_deg=np.degrees(hi), cross_tick=cross_tick)

    off = run_once(False)
    on = run_once(True)

    print(f"  対象関節: {joint}  可動域=[{off['lo_deg']:.1f}, {off['hi_deg']:.1f}]度"
          f"  一定トルク={torque}N・m  目標端={off['target_deg']:.1f}度")
    print(f"  OFF（既定）: 限界から{near_deg}度以内に留まった最大連続tick数="
          f"{off['max_near_run']}  最終角度={off['deg_series'][-1]:.2f}度"
          f"  最初に限界に到達したtick={off['cross_tick']}")
    print(f"  ON （groups=['shoulder']）: 限界から{near_deg}度以内に留まった最大連続tick数="
          f"{on['max_near_run']}  最終角度={on['deg_series'][-1]:.2f}度"
          f"  最初に限界に到達したtick={on['cross_tick']}")

    ok_reduced = on["max_near_run"] < off["max_near_run"]
    print(f"  ONの方が『限界のちょうど±{near_deg}度』に留まる連続tick数が少ない={ok_reduced}")

    # ---- 衝突の瞬間（最初に限界へ到達したtick）前後の角速度を比較（波形、項70）------
    ct = off["cross_tick"]
    if ct is not None and ct >= 1:
        v_off_before = off["vel_series"][ct - 1]
        v_off_after = off["vel_series"][ct]
        v_on_before = on["vel_series"][ct - 1] if ct - 1 < len(on["vel_series"]) else None
        v_on_after = on["vel_series"][ct] if ct < len(on["vel_series"]) else None
        print(f"  衝突tick={ct}: OFF 速度 {v_off_before:.1f}→{v_off_after:.1f}度/tick"
              f"  ON 速度 {v_on_before:.1f}→{v_on_after:.1f}度/tick"
              "（衝突の瞬間の減速そのものはmargin=0のためON/OFFでほぼ同じになる想定。"
              "下の想定外の説明を参照）")
        print("  衝突後の推移（tick, OFF角度, ON角度）:")
        for k in range(max(0, ct - 1), min(n_steps, ct + 8)):
            print(f"    t={k:4d}  OFF={off['deg_series'][k]:8.3f}度"
                  f"  ON={on['deg_series'][k]:8.3f}度")

    # ---- 角速度・角度の時系列をCSVへ保存（波形を見る、項70）--------------------
    import csv
    csv_path = os.path.join(_SCRATCH, "verify_f_wall_sticking.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["tick", "off_angle_deg", "off_angvel_deg_per_tick",
                    "on_angle_deg", "on_angvel_deg_per_tick"])
        for i in range(n_steps):
            w.writerow([i, off["deg_series"][i], off["vel_series"][i],
                       on["deg_series"][i], on["vel_series"][i]])
    print(f"  時系列CSV保存先（検証専用、本番実験結果は上書きしていない）: {csv_path}")

    print("  [2026-08-13追記・margin追加後] 前回（marginを追加する前）は「衝突の瞬間の"
          "減速そのものはmargin=0のためON/OFFでほぼ同じになる想定」と書いていたが、"
          "marginをDEFAULT_JOINT_COMPLIANCEに組み込んだ後は前提が変わっている"
          "（既定でmargin_frac>0になった）。実際にON/OFFで差が出るかどうかは"
          "上に印字した『衝突tick前後の速度』を見て判断すること。より詳しい"
          "『限界に近づくにつれて速度がどう変わるか』は verify_f_margin_sweep の方で"
          "残り角度ごとに測っている（そちらが本題の実測）。")

    ok = ok_reduced
    print(f"  {'OK' if ok else 'NG'}")
    return ok


# ---- margin_frac のスイープ（仕様2-2節。ここが本題の実測） -------------------
def verify_f_margin_sweep(n_steps=400, near_deg=1.0, torque=2.0, seed=0):
    """【本題・ユーザーの指定】margin_fracの値を振り、限界に近づくときの速度の変化と
    最終静止角度（超過量）を実測する。

    測っているのは静止角度そのものではなく、限界に近づくときの速度の変化
    （margin無しなら直前までほぼ一定、marginが効いていれば手前から徐々に
    減っているはず）。verify_f_wall_sticking_reducedと同じ押し当て条件
    （data.qfrc_applied による一定トルク、同じtorque・seed・n_steps）を使う。

    margin候補はユーザーの例示（1度・5度・10度）に加え、width_frac=0.08が意味する
    width（この関節では約21度）・その2倍（約43度）も含める（実装担当判断、
    仕様2-2節1項）。理由：MuJoCoの仕様上marginがwidthより小さいと、限界到達時点
    でもまだ最大強度に達していないと考えられるため、widthとの大小関係で効き方が
    質的に変わるかを見るには、widthより小さい値と大きい値の両方が要る。widthは
    可動域幅に依存するため、この関数の中で実際に右肩の可動域を実測してから
    度数を決める（ハードコードしない）。"""
    _hr("検証F-margin【本題】: margin_fracを振ったときの、限界に近づく速度の変化と最終角度")

    scene_name = "新生児_仰向け_柵なし"
    joint = "right_shoulder_ad_ab"

    # 右肩の可動域を実測する（ハードコードしない）
    sc0 = scene_io.load(scene_name)
    env0, _sc0 = scene_io.build(sc0, seed=seed, verbose=False, vision=False)
    m0 = env0.unwrapped.model
    jid0 = _joint_id(m0, joint)
    lo0 = float(m0.jnt_range[jid0, 0]); hi0 = float(m0.jnt_range[jid0, 1])
    env0.close()
    range_span_deg = np.degrees(hi0 - lo0)
    width_deg = 0.08 * range_span_deg   # DEFAULT_JOINT_COMPLIANCE の width_frac と同じ割合

    remaining_thresholds = [30.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.057]
    candidates = [
        ("0度(margin無し=前回のバグ状態)", 0.0),
        ("1度", 1.0),
        ("5度", 5.0),
        ("10度", 10.0),
        (f"{width_deg:.1f}度(width相当)", width_deg),
        (f"{2 * width_deg:.1f}度(widthの2倍)", 2.0 * width_deg),
    ]
    print(f"  対象関節: {joint}  実測した可動域幅={range_span_deg:.1f}度"
          f"  width(width_frac=0.08時)={width_deg:.1f}度  一定トルク={torque}N・m"
          f"  n_steps={n_steps}")

    rows = {}
    for label, margin_deg in candidates:
        margin_frac = (margin_deg / range_span_deg) if range_span_deg > 0 else 0.0
        sc = scene_io.load(scene_name)
        sc["body"]["joint_compliance"] = True
        sc["body"]["joint_compliance_groups"] = ["shoulder"]
        sc["body"]["joint_compliance_params"] = {"margin_frac": float(margin_frac)}
        env, _sc = scene_io.build(sc, seed=seed, verbose=False, vision=False)
        u = env.unwrapped
        m, d = u.model, u.data
        jid = _joint_id(m, joint)
        qadr = int(m.jnt_qposadr[jid])
        dofadr = int(m.jnt_dofadr[jid])
        lo = float(m.jnt_range[jid, 0]); hi = float(m.jnt_range[jid, 1])
        target_val = hi if abs(hi) >= abs(lo) else lo
        target_deg = np.degrees(target_val)

        zero_a = np.zeros(u.action_space.shape[0], dtype=np.float32)
        n_near, max_near_run = 0, 0
        vel_at_remaining = {}
        remaining_recorded = set()
        last_deg = None
        for t in range(n_steps):
            d.qfrc_applied[dofadr] = float(torque)
            u.step(zero_a)
            ang_deg = float(np.degrees(d.qpos[qadr]))
            vel_deg = float(np.degrees(d.qvel[dofadr]))
            last_deg = ang_deg
            remaining = abs(target_deg - ang_deg)
            for th in remaining_thresholds:
                if th not in remaining_recorded and remaining <= th:
                    vel_at_remaining[th] = abs(vel_deg)
                    remaining_recorded.add(th)
            near = remaining <= near_deg
            n_near = n_near + 1 if near else 0
            max_near_run = max(max_near_run, n_near)
        env.close()

        overshoot = abs(last_deg) - abs(target_deg)
        rows[label] = dict(margin_frac=margin_frac, margin_deg=margin_deg,
                           vel_at_remaining=vel_at_remaining, max_near_run=max_near_run,
                           final_deg=last_deg, overshoot=overshoot, target_deg=target_deg)

    # ---- 表として出力（margin候補×指標のマトリクス）------------------------
    print()
    print("  margin候補 × 限界までの残り角度ごとの角速度(度/tick)：")
    header = "  " + "margin".ljust(28) + "".join(f"残{th:>7g}度".rjust(11) for th in remaining_thresholds)
    print(header)
    for label, _ in candidates:
        r = rows[label]
        cells = ""
        for th in remaining_thresholds:
            v = r["vel_at_remaining"].get(th)
            cells += (f"{v:10.2f} " if v is not None else f"{'到達せず':>10s} ")
        print("  " + label.ljust(28) + cells)

    print()
    print("  margin候補ごとの単調性判定（残り角度が小さくなるにつれ速度が単調に"
          "減少しているか）・張り付き・最終角度：")
    monotonic_flags = {}
    for label, _ in candidates:
        r = rows[label]
        vals = [r["vel_at_remaining"][th] for th in remaining_thresholds if th in r["vel_at_remaining"]]
        # 「残り角度が大きい方から小さい方へ順に見て、速度が単調に非増加か」
        is_monotonic = all(vals[i] >= vals[i + 1] - 1e-9 for i in range(len(vals) - 1)) if len(vals) >= 2 else False
        monotonic_flags[label] = is_monotonic
        print(f"  {label:<28s} margin_frac={r['margin_frac']:.4f}"
              f"  単調減少={is_monotonic}"
              f"  連続tick(限界±{near_deg}度以内)={r['max_near_run']}"
              f"  最終角度={r['final_deg']:.2f}度（目標{r['target_deg']:.1f}度、超過={r['overshoot']:.2f}度）")

    # ---- 判定：margin無し(0度)は単調でなく、widthおよびwidthの2倍は単調になっているか
    label_zero = candidates[0][0]
    label_width = candidates[4][0]
    label_2width = candidates[5][0]
    ok = (not monotonic_flags[label_zero]) and monotonic_flags[label_width] and monotonic_flags[label_2width]
    print()
    print(f"  判定：margin無し(0度)は単調減少でない={not monotonic_flags[label_zero]}"
          f"、width相当は単調減少={monotonic_flags[label_width]}"
          f"、widthの2倍は単調減少={monotonic_flags[label_2width]}")

    # ---- 「188.6度で新たに静止する現象」が解消したか（正直に書く）--------------
    zero_overshoot = rows[label_zero]["overshoot"]
    print()
    print(f"  参考（前回2026-08-13実測）：margin無しでの最終静止角度は約188.6度"
          f"（超過約5.6度）と報告されていた。")
    for label, _ in candidates:
        r = rows[label]
        print(f"    {label}: 最終角度={r['final_deg']:.2f}度  超過={r['overshoot']:.2f}度")
    max_overshoot = max(abs(rows[l]["overshoot"]) for l, _ in candidates)
    min_overshoot = min(abs(rows[l]["overshoot"]) for l, _ in candidates)
    resolved = max_overshoot < 1.0   # 全候補で超過が1度未満なら「解消した」とみなす
    print(f"  『188.6度で新たに静止する現象』は、今回の実測範囲では"
          f"{'解消した' if resolved else '解消していない'}"
          f"（各margin候補の超過量は{min_overshoot:.2f}〜{max_overshoot:.2f}度の範囲。"
          f"詳細・正直な結論は作業記録本文を参照）。")

    print(f"  {'OK' if ok else 'NG'}")
    return ok


def main():
    results = {}
    results["検証0-1（_solimp_from_params数式）"] = verify0_solimp_from_params_math()
    results["検証0-2（width_frac単調性）"] = verify0_solimp_monotonic_in_range()
    results["検証0-3（paramsフラット/関節名判別）"] = verify0_resolve_params_flat_vs_per_joint()

    results["検証A-1（groups/joints未指定）"] = verify_a_requires_groups_or_joints()
    results["検証A-2（未知group名）"] = verify_a_unknown_group_name()
    results["検証A-3（存在しない関節名）"] = verify_a_unknown_joint_name()
    results["検証A-4（未知paramsキー）"] = verify_a_unknown_param_key()

    results["検証B-1・最優先（既定OFF=MuJoCo既定と一致）"] = verify_b_default_off_matches_mujoco_builtin()
    results["検証B-2（4引数省略=明示False と完全一致）"] = verify_b_old_style_call_identical()

    results["検証C-1（shoulder配線チェック）"] = verify_c_shoulder_group_wiring()
    results["検証C-2（all=limited hingeのみ）"] = verify_c_all_group_targets_only_limited_hinge()

    results["検証D（月齢を変えて再構築しても再適用）"] = verify_d_reapplied_across_ages()

    results["検証E（駆動モード非依存性）"] = verify_e_drive_mode_independence()

    # 検証F-0は合否判定の対象外（想定外の記録・事前確認のための診断であり、
    #   「到達しない」という結果自体が想定どおりの発見。詳細はdocstring参照）
    verify_f0_muscle_max_command_probe()
    results["検証F・最優先・本題（張り付き緩和の実測）"] = verify_f_wall_sticking_reduced()
    results["検証F-margin・本題（margin_fracスイープ実測）"] = verify_f_margin_sweep()

    _hr("まとめ")
    for name, ok in results.items():
        print(f"  {'OK' if ok else 'NG'}  {name}")
    all_ok = all(results.values())
    print()
    print("全項目OK" if all_ok else "一部NG（上記参照）")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
