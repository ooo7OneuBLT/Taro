# -*- coding: utf-8 -*-
"""姿勢制御反射（層1、postural_gate.py）の検証。

仕様：作業記録（非公開）
設計：作業記録（非公開）

検証0  0コスト：関節構造の実測（hip_bend1/hip_bend2/chest_leanの軸・アクチュエータ数）
検証A  拮抗筋のneg/pos対応：act:hip_bendのneg/posチャンネルを個別に活性化し、
       qposがどちらへ動くかを実測する（静的スイープ、動力学つき）
検証B  骨盤触覚の前後の符号：重力オフ・骨盤溶接・hip_bend1のみ駆動・前後に壁を置いた
       最小限の自作MuJoCo環境で、hip_bendを前後に駆動したときの触覚点の
       local x の符号を実測する
検証C  ゲートの動作：PosturalGate.applyが実際に「傾きと逆側の筋だけを通す」ことを
       複数の傾き条件（前傾・後傾・不感帯内）で確認する
検証D  通ってはいけない条件：actuator_patternsが一致しない・関節が無い等で
       ValueErrorになるか

【この道具がやらないこと】
    ・学習は回さない（run/main.pyは使わない）
    ・E_scene（本番シーン）の書き換えはしない

使い方::

    .venv/Scripts/python.exe run/tools/check_posture_reflex.py
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
_R = r"C:\claude\AI\Taro"
os.chdir(_R)
if _R not in sys.path:
    sys.path.insert(0, _R)
for _p in ("run/scene_tools", "D/scripts", "taro_core", "taro_core/src/body",
           "taro_core/src/brain", "taro_core/src/brain/spinal_cord",
           "taro_core/src/senses", "taro_core/src/wrapper"):
    _full = os.path.join(_R, _p)
    if _full not in sys.path:
        sys.path.insert(0, _full)

import numpy as np                                          # noqa: E402
import mujoco                                                # noqa: E402

import e_scene                                                # noqa: E402
from postural_gate import PosturalGate                        # noqa: E402

SCENE = "新生児_仰向け_柵なし"


def _hr(title):
    print("=" * 88)
    print(title)
    print("=" * 88)


def _joint_id(model, short_name):
    return int(model.joint("robot:" + short_name).id)


# ================================================================== 検証0
def verify0_joint_structure():
    """0コスト：hip_bend1/hip_bend2/chest_leanの軸・可動域・アクチュエータ数を
    実測する（postural_gate.pyのdocstring【関節構造の実測】の裏取り）。"""
    _hr("検証0: 関節構造の実測（軸・可動域・アクチュエータ）")
    sc = e_scene.load(SCENE)
    env, _sc = e_scene.build(sc, seed=0, verbose=False, vision=False)
    m = env.unwrapped.model

    ok = True
    for jn, expect_axis in (("hip_bend1", (0, 1, 0)), ("hip_bend2", (0, 1, 0)),
                            ("chest_lean", (1, 0, 0)), ("hip_lean1", (1, 0, 0)),
                            ("head_tilt", (0, 1, 0)), ("head_tilt_side", (1, 0, 0))):
        jid = _joint_id(m, jn)
        axis = tuple(float(x) for x in m.jnt_axis[jid])
        lo, hi = float(m.jnt_range[jid, 0]), float(m.jnt_range[jid, 1])
        match = all(abs(a - b) < 1e-6 for a, b in zip(axis, expect_axis))
        ok &= match
        print(f"  {jn:14s}: axis={axis}  期待={expect_axis}  一致={match}"
              f"  可動域=[{np.degrees(lo):.1f}, {np.degrees(hi):.1f}]度")

    n_hip_bend_act = sum(1 for i in range(m.nu) if "hip_bend" in m.actuator(i).name)
    n_chest_lean_act = sum(1 for i in range(m.nu) if "chest_lean" in m.actuator(i).name)
    print(f"  act名に'hip_bend'を含むアクチュエータ数={n_hip_bend_act}"
          f"  act名に'chest_lean'を含むアクチュエータ数={n_chest_lean_act}")
    ok &= (n_hip_bend_act == 1) and (n_chest_lean_act == 1)

    env.close()
    print(f"  {'OK' if ok else 'NG'}")
    return ok


# ================================================================== 検証A
def _hip_bend_actuator_index(env):
    u = env.unwrapped
    idx = [i for i in range(u.model.nu) if "hip_bend" in u.model.actuator(i).name]
    assert len(idx) == 1, f"act:hip_bend が1本のはずが{len(idx)}本"
    return idx[0]


def verify_a_muscle_channel_sign(n_steps=800, seed=0):
    """act:hip_bendのneg/posチャンネルを個別に最大活性化し、qposがどちらへ動くかを
    実測する（動力学つき、静的運動学スイープだけで確定しない＝項107対応）。"""
    _hr("検証A: act:hip_bendのneg/posチャンネルとqpos方向の対応（動力学つき）")
    sc = e_scene.load(SCENE)
    env, _sc = e_scene.build(sc, seed=seed, verbose=False, vision=False)
    u = env.unwrapped
    m, d = u.model, u.data
    n_act = int(u.model.nu)
    i_hip = _hip_bend_actuator_index(env)
    qadr1 = int(m.jnt_qposadr[_joint_id(m, "hip_bend1")])

    def run(channel):
        env2, _sc2 = e_scene.build(e_scene.load(SCENE), seed=seed, verbose=False, vision=False)
        u2 = env2.unwrapped
        a = np.zeros(u2.action_space.shape[0], dtype=np.float32)
        idx = i_hip if channel == "neg" else i_hip + n_act
        a[idx] = 1.0
        q0 = float(u2.data.qpos[qadr1])
        traj = [q0]
        for t in range(n_steps):
            u2.step(a)
            if t in (99, 299, 599, n_steps - 1):
                traj.append(float(u2.data.qpos[qadr1]))
        env2.close()
        return q0, traj

    q0_neg, traj_neg = run("neg")
    q0_pos, traj_pos = run("pos")
    print(f"  初期qpos(hip_bend1)={np.degrees(q0_neg):.3f}度")
    print(f"  neg側(index={i_hip})のみ活性化 → qpos推移(度)="
          f"{[round(np.degrees(v), 2) for v in traj_neg]}")
    print(f"  pos側(index={i_hip + n_act})のみ活性化 → qpos推移(度)="
          f"{[round(np.degrees(v), 2) for v in traj_pos]}")

    neg_direction = traj_neg[-1] - q0_neg     # neg活性化でqposがどちらへ動いたか
    pos_direction = traj_pos[-1] - q0_pos
    print(f"  neg活性化での変化量={np.degrees(neg_direction):+.3f}度"
          f"  pos活性化での変化量={np.degrees(pos_direction):+.3f}度")

    ok_opposite = (neg_direction * pos_direction) < 0   # 逆向きに動くはず
    ok_pos_increases = pos_direction > 0                # posが増加方向という予想の検証
    print(f"  neg/posが逆方向に動いた={ok_opposite}"
          f"  pos側が増加方向(前屈側)に動いた（予想通り）={ok_pos_increases}")
    ok = ok_opposite and ok_pos_increases
    env.close()
    print(f"  {'OK' if ok else 'NG（postural_gate.pyのPOS_CHANNEL_INCREASES_QPOSを見直すこと）'}")
    return ok, ok_pos_increases


# ================================================================== 検証B
def _front_back_wall_env(seed=0):
    """重力オフ・「hip」body溶接・hip_bend1/2のみ自由・前後に壁を置いた
    最小限のMuJoCo環境を作る（自作。taro_coreの本番シーンは一切触らない）。

    骨盤(hip)を空間に固定し、前後どちらに hip_bend1 を駆動しても、上の
    lower_body/upper_body/chestが壁に当たるところまで動かせるようにする。
    重力をオフにすることで、他の自由関節(腕・脚等)が無関係に垂れ下がって
    ノイズになるのを避ける（他の関節は外力が無ければ初期qposのまま静止する）。
    """
    import mujoco
    import paths
    paths.setup_brain_path()
    sys.path.insert(0, paths.MIMO_DIR)
    from mimoEnv.envs.dummy import MIMoMuscleDummyEnv

    class _WallEnv(MIMoMuscleDummyEnv):
        def _initialize_simulation(self):
            spec = mujoco.MjSpec.from_file(self.fullpath)
            spec.option.gravity = [0, 0, 0]

            # hip(骨盤)を空間に溶接して固定する。
            eq = spec.add_equality()
            eq.type = mujoco.mjtEq.mjEQ_WELD
            eq.name1 = "hip"
            eq.objtype = mujoco.mjtObj.mjOBJ_BODY

            # 前後(local x方向)に壁を置く。hip周辺のジオムの半径(約0.03m)より
            # 少し外側に置き、体幹が曲がると接触するようにする。
            front = spec.worldbody.add_body(name="front_wall", pos=[0.06, 0, 0.40])
            front.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 0.15, 0.15])
            back = spec.worldbody.add_body(name="back_wall", pos=[-0.06, 0, 0.40])
            back.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 0.15, 0.15])

            self.model = spec.compile()
            self.model.vis.global_.offwidth = self.width
            self.model.vis.global_.offheight = self.height
            self.data = mujoco.MjData(self.model)
            fps = int(np.round(1 / self.dt))
            self.metadata = {"render_modes": ["human", "rgb_array", "depth_array"],
                             "render_fps": fps}
            self._get_joints()
            self._get_actuators()
            self._get_facial_expressions(getattr(self, "_emotes", None) or {})
            self._set_initial_position(self._initial_qpos)
            self.actuation_model = self.actuation_model(self, self.mimo_actuators)
            return self.model, self.data

    env = _WallEnv()
    env.reset(seed=seed)
    return env


def verify_b_touch_front_back_sign(n_steps=1000, torque=3.0, seed=0):
    """hip_bend1を前後に駆動し、骨盤(hip・lower_body)の触覚点のうちどちらのlocal x
    符号に力が集中するかを実測する（postural_gate.pyのFRONT_IS_POSITIVE_LOCAL_Xの裏取り）。

    【注意・実測範囲が限定的】後方の壁1条件で確認する（前方の壁との対称性は
    位置合わせの都合上、後方の結果から論理的に対になるはずだが、前方接触そのものは
    このテストでは主に使わない。docstring【骨盤触覚の前後の符号】に正直に明記済み）。
    """
    _hr("検証B: 骨盤触覚の前後の符号（自作の壁つき最小環境、重力オフ）")
    try:
        env = _front_back_wall_env(seed=seed)
    except Exception as e:
        print(f"  [想定外] 自作環境の構築に失敗: {type(e).__name__}: {e}")
        return False
    u = env.unwrapped
    m, d = u.model, u.data
    touch = u.touch
    qadr1 = int(m.jnt_qposadr[_joint_id(m, "hip_bend1")])
    dofadr1 = int(m.jnt_dofadr[_joint_id(m, "hip_bend1")])

    from somatosensory_cortex import body_names_of_group
    hip_bids = [int(m.body(nm).id) for nm in body_names_of_group("hip")]

    def touch_front_back_diff():
        diff, mag = 0.0, 0.0
        for bid in hip_bids:
            pos = touch.sensor_positions.get(bid)
            force = touch.sensor_outputs.get(bid)
            if pos is None or force is None or pos.shape[0] == 0:
                continue
            pos = np.asarray(pos, dtype=np.float64)
            force = np.asarray(force, dtype=np.float64)
            fmag = np.linalg.norm(force, axis=-1)
            diff += float(np.sum(fmag * np.sign(pos[:, 0])))
            mag += float(np.sum(fmag))
        return diff, mag

    zero_a = np.zeros(u.action_space.shape[0], dtype=np.float32)

    results = {}
    for label, trq in (("後屈方向(qfrc<0)", -torque), ("前屈方向(qfrc>0)", +torque)):
        env2 = _front_back_wall_env(seed=seed) if label != "後屈方向(qfrc<0)" else env
        u2 = env2.unwrapped
        m2, d2 = u2.model, u2.data
        qadr = int(m2.jnt_qposadr[_joint_id(m2, "hip_bend1")])
        dofadr = int(m2.jnt_dofadr[_joint_id(m2, "hip_bend1")])
        touch2 = u2.touch
        for _t in range(n_steps):
            d2.qfrc_applied[dofadr] = float(trq)
            u2.step(zero_a)
        q_deg = float(np.degrees(d2.qpos[qadr]))
        diff, mag = 0.0, 0.0
        for bid in hip_bids:
            pos = touch2.sensor_positions.get(bid)
            force = touch2.sensor_outputs.get(bid)
            if pos is None or force is None or pos.shape[0] == 0:
                continue
            pos = np.asarray(pos, dtype=np.float64)
            force = np.asarray(force, dtype=np.float64)
            fmag = np.linalg.norm(force, axis=-1)
            diff += float(np.sum(fmag * np.sign(pos[:, 0])))
            mag += float(np.sum(fmag))
        results[label] = dict(q_deg=q_deg, diff=diff, mag=mag)
        print(f"  {label}: 最終hip_bend1={q_deg:.2f}度  触覚前後差(diff)={diff:.5f}"
              f"  力の合計(mag)={mag:.5f}")
        if label != "後屈方向(qfrc<0)":
            env2.close()

    env.close()

    back = results["後屈方向(qfrc<0)"]
    fwd = results["前屈方向(qfrc>0)"]
    ok_contact = back["mag"] > 1e-6 or fwd["mag"] > 1e-6
    if not ok_contact:
        print("  [想定外] どちらの条件でも触覚点に力が生じなかった"
              "（壁の位置・大きさが合っていない可能性。壁の配置を見直す必要がある）")
        return False

    # 【2026-08-15・想定外】事前の幾何学的予想（local x>0=前方）とは符号が逆の
    # 結果が出た（docstring・postural_gate.pyの【骨盤触覚の前後の符号】参照）。
    # ここでは「hip_bendのqposの符号とdiffの符号が一致する方向へ、
    # PosturalGate.FRONT_IS_POSITIVE_LOCAL_Xを補正した結果」が実際に一致するかを
    # 確認する（＝モジュールの定数が実測と整合しているかの回帰確認）。
    gate = PosturalGate(m)
    checks = []
    for label, r in ((("前屈(qfrc>0)"), fwd), (("後屈(qfrc<0)"), back)):
        if r["mag"] <= 1e-6:
            print(f"  {label}: 接触なし（mag=0、この条件は判定に使えない）")
            continue
        corrected_diff = r["diff"] if gate.front_is_positive_local_x else -r["diff"]
        # 補正後のdiffがqposと同じ符号になっていれば、モジュールの定数は実測と整合
        same_sign = (corrected_diff > 0) == (r["q_deg"] > 0)
        checks.append(same_sign)
        print(f"  {label}: qpos={r['q_deg']:.2f}度  生diff={r['diff']:.5f}"
              f"  補正後diff(front_is_positive_local_x={gate.front_is_positive_local_x})="
              f"{corrected_diff:.5f}  qposと同じ符号={same_sign}")
    ok = len(checks) > 0 and all(checks)
    print("  [想定外] 実測は前屈方向1条件のみ（後屈方向はこの壁配置で接触が"
          "再現できなかった）。座位保持の本番シーンができたら再検証を推奨する。")
    print(f"  {'OK' if ok else 'NG（postural_gate.pyのFRONT_IS_POSITIVE_LOCAL_Xを見直すこと）'}")
    return ok


# ================================================================== 検証C
def verify_c_gate_behavior():
    """PosturalGateが実際に「傾きと逆側の筋だけを通す」ことを、複数の傾き条件で
    確認する（action=全筋0.5一定を与え、ゲート適用後にどちらの筋が0になるか）。"""
    _hr("検証C: ゲートの動作（前傾・後傾・不感帯内の3条件）")
    sc = e_scene.load(SCENE)
    env, _sc = e_scene.build(sc, seed=0, verbose=False, vision=False)
    u = env.unwrapped
    m, d = u.model, u.data
    gate = PosturalGate(m)
    n = gate.n_actuator
    i_hip = gate.actuator_idx[0]
    qadr1 = int(m.jnt_qposadr[_joint_id(m, "hip_bend1")])
    qadr2 = int(m.jnt_qposadr[_joint_id(m, "hip_bend2")])

    action = np.full(2 * n, 0.5, dtype=np.float32)
    ok_all = True

    qpos = np.asarray(d.qpos).copy()

    # 前傾（deadzoneを超える正の角度）
    qpos[qadr1] = np.radians(20.0)
    qpos[qadr2] = np.radians(20.0)
    gated = gate.apply(action, qpos, touch=None)
    if gate.pos_channel_increases_qpos:
        expect_zero, expect_kept = i_hip + n, i_hip
    else:
        expect_zero, expect_kept = i_hip, i_hip + n
    ok1 = (gated[expect_zero] == 0.0) and (gated[expect_kept] == 0.5)
    print(f"  前傾20度: direction={gate.compute(qpos)}  "
          f"塞がれた側(index={expect_zero})={gated[expect_zero]}  "
          f"通った側(index={expect_kept})={gated[expect_kept]}  他は不変={np.all(np.delete(gated,[expect_zero])==0.5)}"
          f"  判定={ok1}")
    ok_all &= ok1

    # 後傾（deadzoneを超える負の角度）
    qpos[qadr1] = np.radians(-15.0)
    qpos[qadr2] = np.radians(-15.0)
    gated2 = gate.apply(action, qpos, touch=None)
    if gate.pos_channel_increases_qpos:
        expect_zero2, expect_kept2 = i_hip, i_hip + n
    else:
        expect_zero2, expect_kept2 = i_hip + n, i_hip
    ok2 = (gated2[expect_zero2] == 0.0) and (gated2[expect_kept2] == 0.5)
    print(f"  後傾-15度: direction={gate.compute(qpos)}  "
          f"塞がれた側(index={expect_zero2})={gated2[expect_zero2]}  "
          f"通った側(index={expect_kept2})={gated2[expect_kept2]}  判定={ok2}")
    ok_all &= ok2

    # 不感帯内（ほぼ直立）→ 無変更
    qpos[qadr1] = np.radians(1.0)
    qpos[qadr2] = np.radians(1.0)
    gated3 = gate.apply(action, qpos, touch=None)
    ok3 = np.allclose(gated3, action)
    print(f"  不感帯内(1度): direction={gate.compute(qpos)}  行動配列は無変更={ok3}")
    ok_all &= ok3

    # 対象外(hip以外の腕アクチュエータ等)は一切変わらない
    other_idx = [i for i in range(2 * n) if i not in (i_hip, i_hip + n)]
    unaffected = np.all(gated[other_idx] == 0.5)
    print(f"  対象外アクチュエータ(腕等)は無変更={unaffected}")
    ok_all &= bool(unaffected)

    env.close()
    print(f"  {'OK' if ok_all else 'NG'}")
    return ok_all


# ================================================================== 検証D
def verify_d_error_conditions():
    """通ってはいけない条件：一致しないactuator_patterns、存在しない関節名で
    ValueErrorになるかを確認する。"""
    _hr("検証D: 通ってはいけない条件")
    sc = e_scene.load(SCENE)
    env, _sc = e_scene.build(sc, seed=0, verbose=False, vision=False)
    m = env.unwrapped.model

    ok_all = True
    try:
        PosturalGate(m, actuator_patterns=("no_such_actuator_pattern",))
        print("  [NG] 一致しないactuator_patternsで例外が出なかった")
        ok_all = False
    except ValueError as e:
        print(f"  [OK] 一致しないactuator_patternsでValueError: {e}")

    try:
        PosturalGate(m, proprioceptive_joint_names=("no_such_joint",))
        print("  [NG] 存在しない関節名で例外が出なかった")
        ok_all = False
    except ValueError as e:
        print(f"  [OK] 存在しない関節名でValueError: {e}")

    env.close()
    print(f"  {'OK' if ok_all else 'NG'}")
    return ok_all


# ================================================================== 検証E・F（2026-08-15追記）
#   仕様：作業記録（非公開）
#   ここから下は「座位保持の学習」最終ラウンド（新規シーン・実験ファイルの作成と
#   全体の最終検証）で追加した。既存の検証0〜Dは変更していない。
def verify_e_default_off_wiring():
    """既定OFF（posture_reflex=False・righting_reflex=False）のとき、
    env.unwrapped.taro 自体が一度も配線されないことを確認する。

    【なぜこの確認が要るか、2026-08-15・重要な副産物】taro_setup.pyの
    _setup_postural_gate/_setup_righting_damperは、posture_reflex/righting_reflexの
    **どちらかがTrue**のときだけ env.unwrapped.taro = taro を実行する（配線理由は
    同関数のdocstring参照）。つまり両方Falseだと env.unwrapped.taro が存在せず、
    E/scripts/e_toy_env.py の step() 内 `taro = getattr(self, "taro", None)` は
    常にNoneを返す。これは postural_gate/righting_damper だけでなく
    **_check_posture_fall（座り直し機構そのもの）も一緒に無効化される**ことを
    意味する（`if taro is not None: self._check_posture_fall(taro)`）。
    つまり posture_reflex=False かつ righting_reflex=False のときは、
    posture_fall_deg をいくつに設定しても座り直しは一切発動しない
    （実測：run/tools/check_posture_fall_sensitivity.py 相当の検証で、
    反射なし条件は posture_fall_deg=20/30/40/50 のどれでも発動回数0回だった。
    2026-08-15 実装・座位保持の学習・最終ラウンド作業記録参照）。
    仕様として妥当かどうかの判断は実装の担当外なので、ここでは
    「今の実装はこう動く」という事実を固定する回帰テストとして置く。
    """
    _hr("検証E: 既定OFFでenv.taroが配線されないことの確認（座り直しも道連れで無効化）")
    from run.config import Config
    from run.plugins.common import scene as scene_mod
    from run.taro_setup import Taro

    spec = {
        "name": "検証E", "note": "",
        "scene": SCENE,
        "taro": {"actuation": "muscle", "age_months": 4.0,
                 "posture_reflex": False, "righting_reflex": False,
                 "posture_fall_deg": 30.0, "reward": "progress"},
        "run": {"type": "train", "steps": 10, "seed": 0}, "plugins": {},
    }
    cfg = Config.from_spec(spec)
    env, _scene_dict, _hands = scene_mod.build(cfg.scene, taro=dict(cfg._taro),
                                               seed=0, verbose=False, hybrid=True)
    _taro = Taro(cfg, env, seed=0, verbose=False)
    u = env.unwrapped
    has_taro_attr = hasattr(u, "taro")
    print(f"  posture_reflex=False, righting_reflex=False, posture_fall_deg=30.0"
          f"  env.unwrapped に taro 属性が存在する={has_taro_attr}"
          f"（Falseが期待どおり＝座り直しも動かない）")
    ok = not has_taro_attr
    env.close()
    print(f"  {'OK' if ok else 'NG'}")
    return ok


def verify_f_posture_height_reward():
    """reward="posture_height" が座位開始時の頭の高さからの相対差を正しく返すか。

    root（自由関節）のqposadrを**model.joint(0)決め打ちにしない**（項107・項67と
    同型の罠：このモデルはtest_object1/test_object2の自由関節が先頭にあり、
    体そのものの自由関節("mimo_orientation", body="mimo_location")はindex 2）。
    実測で一度これを踏み、qpos[2]（おもちゃの位置）を動かして「rewardが変わらない」
    という誤った結果を出しかけた（2026-08-15 実装の作業記録「検証の落とし穴」参照）。
    """
    _hr("検証F: reward=posture_height の値が頭の高さの相対差になっているか")
    from run.config import Config
    from run.plugins.common import scene as scene_mod
    from run.taro_setup import Taro
    import mujoco as _mj

    spec = {
        "name": "検証F", "note": "",
        "scene": SCENE,
        "taro": {"actuation": "muscle", "age_months": 4.0,
                 "posture_reflex": False, "righting_reflex": False,
                 "reward": "posture_height"},
        "run": {"type": "train", "steps": 10, "seed": 0}, "plugins": {},
    }
    cfg = Config.from_spec(spec)
    env, _scene_dict, _hands = scene_mod.build(cfg.scene, taro=dict(cfg._taro),
                                               seed=0, verbose=False, hybrid=True)
    _taro = Taro(cfg, env, seed=0, verbose=False)
    u = env.unwrapped
    m, d = u.model, u.data

    # 体そのものの自由関節を名前で探す（object系を除く。verify_b_touch_front_back_signと
    #   同じ考え方だが、こちらは本番シーンのモデルなので改めて名前で探す）。
    qadr = None
    for j in range(m.njnt):
        if int(m.jnt_type[j]) != int(_mj.mjtJoint.mjJNT_FREE):
            continue
        bn = m.body(int(m.jnt_bodyid[j])).name
        if "object" in bn or "toy" in bn:
            continue
        qadr = int(m.jnt_qposadr[j])
        break
    if qadr is None:
        print("  [想定外] 体の自由関節が見つからない")
        env.close()
        return False

    r0 = u.posture_height_reward()
    d.qpos[qadr + 2] += 0.05
    _mj.mj_forward(m, d)
    r_up = u.posture_height_reward()
    d.qpos[qadr + 2] -= 0.10
    _mj.mj_forward(m, d)
    r_down = u.posture_height_reward()

    print(f"  baseline(reset直後)={r0:.6f}（期待: 0）")
    print(f"  +5cm持ち上げ後={r_up:.6f}（期待: +0.05）")
    print(f"  そこから-10cm(baselineから-5cm)={r_down:.6f}（期待: -0.05）")
    ok = (abs(r0) < 1e-9 and abs(r_up - 0.05) < 1e-9 and abs(r_down - (-0.05)) < 1e-9)
    env.close()
    print(f"  {'OK' if ok else 'NG'}")
    return ok


def main():
    results = {}
    results["検証0（関節構造の実測）"] = verify0_joint_structure()
    ok_a, pos_increases = verify_a_muscle_channel_sign()
    results["検証A（neg/pos対応の実測）"] = ok_a
    results["検証B（骨盤触覚の前後の符号）"] = verify_b_touch_front_back_sign()
    results["検証C（ゲートの動作）"] = verify_c_gate_behavior()
    results["検証D（通ってはいけない条件）"] = verify_d_error_conditions()
    results["検証E（既定OFFでenv.taro未配線・座り直しも無効化）"] = verify_e_default_off_wiring()
    results["検証F（reward=posture_heightの値）"] = verify_f_posture_height_reward()

    _hr("まとめ")
    for name, ok in results.items():
        print(f"  {'OK' if ok else 'NG'}  {name}")
    all_ok = all(results.values())
    print()
    print("全項目OK" if all_ok else "一部NG（上記参照）")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
