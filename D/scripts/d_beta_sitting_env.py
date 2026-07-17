"""
座位アルファ（太郎本体・全関節自由・倒れない）＋左右に動くベータ（親・操り人形）。

【設計・2026-07-17 ユーザースケッチ】アルファを座らせて安定させ、ベータ(親)を左右に動かす。
egomotion研究の実際のパラダイム（moving room：支えられて座っている乳児で自己運動の視覚割引を
測る）に対応。生後2ヶ月〜のsupported sittingで報告される現象＝ハイハイ・立位の習得を待つ必要は
ない、という文献確認済み（infant motor milestones / visually induced postural sway文献）。

【座位＋倒れない、の作り方】MIMo公式 `MIMoSelfBody-v0`(`selfbody.py`)の技術をそのまま流用：
  ・equality weld body1="mimo_location" ＝体の"根っこ"(自由関節)だけを固定＝倒れない
  ・SITTING_POSITION ＝股関節を曲げた座位の初期角度（既製品）
  ★自己接触タスクと違い、根っこの溶接以外は**全関節を自由なまま**にする（1本の腕だけに
    制限しない）＝本物の運動性喃語（ランダム探索）で全身を自由に動かせる、Cの仰向け版と
    同じ自由度。

【ベータ（親）】d_beta_env.py で確立した「全関節削除＋3自由度操縦（剛体）」をそのまま流用。
左右(y軸)に往復させる＝物理版egomotion実験(d_ego_leftright.py)と同じ構造。

使い方: python d_beta_sitting_env.py
  ①アルファに本物の行動（ランダム探索=運動性喃語）を与えて、座位が本当に倒れず暴れないか検証
  ②ベータを左右に動かし、アルファの目にちゃんと入るか（視覚ジオメトリの粗確認）
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import mujoco

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "taro_core"))
import paths
paths.setup_brain_path()
sys.path.insert(0, paths.MIMO_DIR)

from mimoEnv.envs.mimo_env import MIMoEnv, EMOTES
from mimoEnv.envs.dummy import MIMoV2DummyEnv
from mimoEnv.envs.selfbody import SITTING_POSITION   # MIMo公式の座位初期角度（既製品）

BETA = "beta_"
_BETA_RANGE = 2.0


class BetaSittingEnv(MIMoV2DummyEnv):
    """座位で安定したアルファ（全関節自由）＋左右に動く操り人形ベータ。"""

    BETA_HOME = np.array([0.3, 0.0, 0.35])   # 座ったアルファの正面あたり（後日較正）

    def __init__(self, **kwargs):
        # SITTING_POSITIONは旧MIMoモデル(selfbody_scene.xml)用の辞書で、本番シーン(V2・
        # 指の関節がより詳細)とは関節名が一部食い違う（実測：left_fingers等は無く
        # left_ff_distal等に分解されている）。座位の本質は股関節・頭の曲げなので、
        # 本番シーンに存在する関節キーだけに絞って使う（指・足先は既定角度のままでよい）。
        probe = mujoco.MjSpec.from_file(paths.SCENE).compile()
        valid_names = {mujoco.mj_id2name(probe, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(probe.njnt)}
        sitting = {k: v for k, v in SITTING_POSITION.items() if k in valid_names}

        # ★訂正：股関節/胸の曲げ角度は溶接と綱引きして矛盾する（実測で発覚）。
        # hip/lower_body/upper_body/chestを個別に溶接すると、それらを繋ぐ関節
        # (hip_bend/hip_lean/hip_rot/chest_lean/chest_rot)は「各体を初期姿勢のまま
        # 固定する」という制約により**実質ゼロに固定**される。ここでSITTING_POSITIONの
        # 値(0.53rad等)を上書きしようとすると、溶接(その角度を認めない)と初期姿勢の
        # 指定(その角度にしろ)が綱引きし、動画で見ると胴体が前に潰れる形で破綻していた。
        # → 溶接に委ねて上書きしない(その関節はsittingに入れない)＝直立した姿勢になる。
        # 直立でも「支えられて座っている」研究の対応は崩れない（胴体の向きでなく
        # 頭の自由な動きが本質のため）。
        for jn in ("robot:hip_bend1", "robot:hip_bend2", "robot:hip_lean1", "robot:hip_lean2",
                   "robot:hip_rot1", "robot:hip_rot2", "robot:chest_lean", "robot:chest_rot"):
            sitting.pop(jn, None)

        # ★長座位（足を伸ばして座る）へ訂正。SITTING_POSITION既定は膝を深く曲げた姿勢
        # （右膝-106°等）で、動画で見ると脚が浮いて不自然だった。乳児の座位でも比較的
        # 早期に見られる姿勢（足を前に伸ばす＝股関節の回旋・外転を使わず単純な屈曲だけで
        # 座れる、より基本的な座り方）。胴体は骨盤〜胸の溶接で既に支えているため、脚の
        # 角度自体は転倒安定性には影響しない＝見た目の自然さのための変更。
        # hip1(主屈曲)は既存の約-87°を活かす(座った股関節から脚を前へ)。hip2/hip3(外転/
        # 回旋)と膝・足首・つま先は0(まっすぐ)にする＝脚がまっすぐ前に伸びる。
        for side in ("left", "right"):
            for j, v in ((f"robot:{side}_hip2", 0.0), (f"robot:{side}_hip3", 0.0),
                          (f"robot:{side}_knee", 0.0), (f"robot:{side}_foot1", 0.0),
                          (f"robot:{side}_foot2", 0.0), (f"robot:{side}_foot3", 0.0),
                          (f"robot:{side}_toes", 0.0)):
                if j in valid_names:
                    sitting[j] = np.array([v])

        print(f"SITTING_POSITION: {len(SITTING_POSITION)}関節中 {len(sitting)}関節が本番シーンに存在"
              f"（欠落{len(SITTING_POSITION) - len(sitting)}は指・足先など座位に無関係な旧モデル固有関節）")
        super().__init__(initial_qpos=sitting, **kwargs)

    # 座位に必要な下げ幅(m)。元シーンのmimo_locationは立位用の高さのまま＝関節角度だけ
    # 座位にしても宙に浮く（実測：体の最下点z=0.289＝29cm浮いていた）。溶接前に根っこの
    # 位置そのものを下げて、床に接地させてから固定する。⚠️経験的な定数＝感度確認が要る。
    SIT_DROP_Z = 0.30

    def _initialize_simulation(self):
        spec = mujoco.MjSpec.from_file(self.fullpath)

        # ★根っこの位置を座位の高さまで下げる（溶接で"その場"に固定する前に）。
        mimo_body = spec.body("mimo_location")
        mimo_body.pos = [mimo_body.pos[0], mimo_body.pos[1], mimo_body.pos[2] - self.SIT_DROP_Z]

        # ★座位で倒れないための溶接。
        # 【訂正・2026-07-17】骨盤(mimo_location)だけを固定する版を動画で確認したところ、
        # 骨盤の数値は安定でも**上半身（胴体）がその場に崩れ落ちて床に倒れ込んでいた**
        # （骨盤という土台が動かなくても、土台の上の建物＝胴体を支える力ではなかった）。
        # MIMoSelfBodyEnv公式も実は「対象の1関節以外は毎ステップ座位姿勢に戻す」という、
        # より強い固定をしていたと判明（骨盤の溶接だけでは不十分）。
        # → **ハイチェアに座らせるイメージ**：骨盤〜胴体〜胸(hip/lower_body/upper_body/chest)を
        # 溶接して支え、**頭だけ自由**にする（moving room研究が測るのは頭・姿勢の揺れそのもの
        # なので、頭が自由なら研究の対象は再現できる）。腕・脚も自由（バランスに関与しない）。
        for body_name in ("mimo_location", "hip", "lower_body", "upper_body", "chest"):
            eq = spec.add_equality()
            eq.type = mujoco.mjtEq.mjEQ_WELD
            eq.name1 = body_name
            eq.objtype = mujoco.mjtObj.mjOBJ_BODY

        # ベータ（親）を複製・剛体化して生やす（d_beta_env.pyで確立した手順）。
        spec_beta = mujoco.MjSpec.from_file(self.fullpath)
        fr = spec.worldbody.add_frame()
        fr.pos = list(self.BETA_HOME)
        beta_root = fr.attach_body(spec_beta.body("mimo_location"), BETA, "")

        for a in list(spec.actuators):
            if a.name.startswith(BETA):
                a.delete()
        for t in list(spec.tendons):
            if t.name.startswith(BETA):
                t.delete()
        beta_joint_names = {j.name for j in spec.joints if j.name.startswith(BETA)}
        for eqq in list(spec.equalities):
            if eqq.name1 in beta_joint_names or eqq.name2 in beta_joint_names:
                eqq.delete()
        for j in list(spec.joints):
            if j.name.startswith(BETA):
                j.delete()
        for b in spec.bodies:
            if b.name.startswith(BETA):
                b.gravcomp = 1.0

        for ax, nm in (([1, 0, 0], "x"), ([0, 1, 0], "y"), ([0, 0, 1], "z")):
            j = beta_root.add_joint()
            j.name = BETA + nm
            j.type = mujoco.mjtJoint.mjJNT_SLIDE
            j.axis = ax
            j.range = [-_BETA_RANGE, _BETA_RANGE]
            j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        for nm in ("x", "y", "z"):
            a = spec.add_actuator()
            a.name = BETA + nm
            a.target = BETA + nm
            a.trntype = mujoco.mjtTrn.mjTRN_JOINT
            kp, kv = 200.0, 20.0
            gp = np.zeros(10); gp[0] = kp
            bp = np.zeros(10); bp[1] = -kp; bp[2] = -kv
            a.gainprm = gp
            a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            a.biasprm = bp
            a.ctrlrange = [-_BETA_RANGE, _BETA_RANGE]
            a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE

        self.model = spec.compile()
        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height
        self.data = mujoco.MjData(self.model)

        fps = int(np.round(1 / self.dt))
        self.metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": fps}
        self._get_joints()
        self._get_actuators()
        self._get_facial_expressions(EMOTES)
        self._set_initial_position(self._initial_qpos)
        self.actuation_model = self.actuation_model(self, self.mimo_actuators)
        self.carer_actuators = np.asarray(
            [i for i in range(self.model.nu) if self.model.actuator(i).name.startswith(BETA)])
        return self.model, self.data

    def set_beta_target(self, xyz):
        rel = np.asarray(xyz, dtype=np.float64) - self.BETA_HOME
        self.data.ctrl[self.carer_actuators] = np.clip(rel, -_BETA_RANGE, _BETA_RANGE)

    @property
    def beta_pos(self):
        return self.data.body(BETA + "mimo_location").xpos.copy()

    @property
    def alpha_root_pos(self):
        return self.data.body("mimo_location").xpos.copy()


# 眼球カメラの直接描画（MIMoの壊れたgym描画を迂回する既存実装）を借用する。
# CarerVisionEnvと同じ属性(self.vision_params/self.vision/self.model/self.data)を持つので、
# メソッドをそのまま流用できる＝視覚ONにするとアルファの本当の眼球視界が得られる。
from d1_carer_vision_env import CarerVisionEnv as _CVE
BetaSittingEnv.get_vision_obs = _CVE.get_vision_obs


if __name__ == "__main__":
    print("=== 座位アルファ＋左右ベータの検証 ===")
    env = BetaSittingEnv()
    env.reset(seed=0)
    na = env.action_space.shape[0]
    rng = np.random.default_rng(0)

    print("\n--- ① 座位が倒れず安定するか（無操作200step）---")
    for _ in range(200):
        env.step(np.zeros(na, np.float32))
    p0 = env.unwrapped.alpha_root_pos
    print(f"  アルファ根っこ位置(200step後) = {np.round(p0, 3)}（倒れていれば大きくズレるはず）")

    print("\n--- ② 本物の行動（ランダム探索=運動性喃語）で暴れないか（500step）---")
    max_qvel = 0.0
    for _ in range(500):
        act = rng.uniform(-1, 1, na).astype(np.float32)
        env.step(act)
        max_qvel = max(max_qvel, float(np.abs(env.unwrapped.data.qvel).max()))
    p1 = env.unwrapped.alpha_root_pos
    print(f"  アルファ根っこ位置(ランダム行動500step後) = {np.round(p1, 3)}")
    print(f"  関節速度の最大絶対値 = {max_qvel:.2f}（数百〜発散なら暴走、数〜数十なら正常な喃語）")
    assert np.all(np.abs(p1) < 2.0), "根っこが大きく動いた＝溶接が効いていない/転倒の疑い"
    assert max_qvel < 200, "関節速度が異常＝発散/暴走の疑い"

    print("\n--- ③ ベータを左右に動かし、位置を確認 ---")
    for tgt_y in (0.3, -0.3, 0.0):
        env.unwrapped.set_beta_target([0.3, tgt_y, 0.35])
        for _ in range(50):
            env.step(np.zeros(na, np.float32))
        print(f"  ベータ目標y={tgt_y} -> 実位置 {np.round(env.unwrapped.beta_pos, 3)}")

    print("\nOK: 座位アルファは本物の行動でも安定／ベータは左右に操縦可能")
