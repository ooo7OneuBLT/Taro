"""
D1：仰向けの太郎を、**工学的に作った「養育者の手」**が触る環境。

【なぜ相手をMIMoの複製にしないか】2026-07-15・ユーザー指摘
太郎の流儀は「**太郎の中身は本能のみ**」＝制約は**太郎の内側**にかかっており、環境や親には
かかっていない。実際、目標Bの親（`parent_sim_b.py`）は**YAMLのスケジュールで動くただの
スクリプト**で、太郎の複製ではない。D1でベータをMIMoの複製にする必然性は、一度も検証して
いなかった。

工学的な手にする利点（すべて実測・実績にもとづく）：
  ・**物理が爆発しない**：MIMo2体を重ねて置いたら**3.3m吹き飛んだ**（ラグドール同士の
    めり込み解消）。剛体をスライド関節で動かすだけなら起きない。
  ・**本当に触れる**：太郎は握力の制御を持たない（到達力~50%）。手なら、そう作ればいい。
  ・**ほぼタダ**：MIMo2体は物理も触覚も2倍。手はカプセル1個。
  ・**正解ラベルが完全**：設計書が「相手がランダムだと目標推論が空回り」と問題視していた点。
    こちらが行動を定義するので、太郎が当てるべき正解に曖昧さがゼロになる。

【限界（正直に）】Dの本題「自分のモデルを他者に当てはめる」には、相手が自分とある程度
同型である必要がある。ただし**0か1かではなく程度**（人は猫の動きをある程度読めるが、猫を
読み違えることでも有名）。手の形なら太郎の手と対応するので、対応が無いわけではない。
＝**まず「触覚に他者が写るか」の門を通し、同型が要る段階（D2の目標推論）で複製に差し替える**。

【なぜ「抱っこ」でなく「触られる」か】
抱っこの本質は「支える」と「動く他者が触れ続ける」の2つ。**支えは床が既に解いている**
（仰向けの太郎は腰の高さ0.050mで20秒間まったく崩れない＝実測）。残る本質は後者だけ。
＝太郎に必要な能力はゼロ（到達力も姿勢制御も要らない）。

【養育者の手の行動】
  still : 何もしない（＝対照群。基準線は触覚の変動係数1.6%）
  stroke: 胸を一定速度で撫でる ← Ackerley et al. (2014) がC触覚線維の実験で使った条件
          （人肌の温度でゆっくり撫でる）に対応
  press : 一点を押し続ける
  rock  : 周期的に揺らす
  lift  : 腕を下から押し上げる
"""
import os
import sys
import numpy as np
import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "C"))
import paths
paths.setup_brain_path()
sys.path.insert(0, paths.MIMO_DIR)

from mimoEnv.envs.mimo_env import EMOTES
from mimoEnv.envs.dummy import MIMoV2DummyEnv
from mimo_lean import strip_textures

# 仰向けにする四元数（roll_over.py の supine と同じ式）
_SUPINE_QUAT = np.array([0, -0.7071068, 0, 0.7071068]) * np.array([1, -1, 1, 1])
CARER = "carer_"   # MIMoEnvの `robot:` / `act:` 判定から外れる接頭辞


class CarerEnv(MIMoV2DummyEnv):
    """仰向けの太郎＋世界に固定された「養育者の手」（3自由度のスライド関節）。

    手は太郎の関節にもアクチュエータにもならない（接頭辞が `robot:` / `act:` でないため、
    MIMoEnv が自動的に「自分の体」から除外する）。＝太郎の観測・行動空間は1体のときと同一。
    """

    def __init__(self, hand_size=0.035, settle_steps=100, jitter=0.01, **kwargs):
        self._hand_size = hand_size
        self._settle_steps = settle_steps
        self._jitter = jitter
        super().__init__(**kwargs)
        m = self.model
        m.body("hip").pos = [0, 0, 0.2]
        m.body("hip").quat = _SUPINE_QUAT.copy()
        for _ in range(self._settle_steps):
            mujoco.mj_step(self.model, self.data)
        self.init_position = self.data.qpos.copy()

    def _initialize_simulation(self):
        spec = mujoco.MjSpec.from_file(paths.SCENE)
        # --- 養育者の手を生やす（世界に直付け＝付け根は動かない＝姿勢制御を持つ大人の代理）---
        # 太郎（仰向け）は頭が+x側・腰が-x側に寝る（実測：腰[-0.036,0.003,0.05] 頭[0.263,-0.001,0.035]）。
        # 手の**基準位置は太郎の遥か上(z=0.50)**にする。ここを胸の高さにすると、太郎が
        # 落ち着く100stepの間ずっと手が邪魔をして、**太郎が手の上に乗ってしまう**
        # （実測：腰が 0.050m → 0.132m に浮いた）。実験時にctrlで降ろす。
        hand = spec.worldbody.add_body(name=CARER + "hand", pos=[0.10, 0.0, 0.50])
        # 【重要】重力補償。これが無いと手が落下して太郎を弾き飛ばす
        # （実測：太郎の腰が 0.050m → 0.303m に浮き、x が -0.64 までずれた）。
        # 物理的にも正しい：**養育者の腕は大人自身が支えている**ので、太郎の上に落ちてはこない。
        hand.gravcomp = 1.0
        for ax, nm in (([1, 0, 0], "x"), ([0, 1, 0], "y"), ([0, 0, 1], "z")):
            j = hand.add_joint()
            j.name = CARER + nm
            j.type = mujoco.mjtJoint.mjJNT_SLIDE
            j.axis = ax
            j.range = [-1.0, 1.0]
            j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        g = hand.add_geom()
        g.name = CARER + "palm"
        g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        g.size = [self._hand_size, self._hand_size * 1.2, 0]
        g.rgba = [0.9, 0.5, 0.4, 1.0]
        for nm in ("x", "y", "z"):
            a = spec.add_actuator()
            a.name = CARER + nm
            a.target = CARER + nm
            a.trntype = mujoco.mjtTrn.mjTRN_JOINT
            # 位置サーボ。MuJoCoの gainprm/biasprm は**10要素固定**（3要素で渡すと型エラー）。
            kp, kv = 200.0, 20.0
            gp = np.zeros(10); gp[0] = kp
            bp = np.zeros(10); bp[1] = -kp; bp[2] = -kv
            a.gainprm = gp
            a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            a.biasprm = bp
            a.ctrlrange = [-1.0, 1.0]
            a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE

        self.n_textures_stripped = strip_textures(spec) if self.vision_params is None else 0
        self.model = spec.compile()
        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height
        self.data = mujoco.MjData(self.model)

        fps = int(np.round(1 / self.dt))
        self.metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": fps}
        self._get_joints()      # `robot:` のみ → 太郎の関節だけ（手は入らない）
        self._get_actuators()   # `act:` のみ   → 太郎のアクチュエータだけ（手は入らない）
        self._get_facial_expressions(EMOTES)
        self._set_initial_position(self._initial_qpos)
        self.actuation_model = self.actuation_model(self, self.mimo_actuators)
        self.carer_actuators = np.asarray(
            [i for i in range(self.model.nu) if self.model.actuator(i).name.startswith(CARER)])
        self.carer_joints = np.asarray(
            [i for i in range(self.model.njnt) if self.model.joint(i).name.startswith(CARER)])
        return self.model, self.data

    def set_hand_target(self, xyz):
        """養育者の手を、この位置（絶対座標に近い指令）へ動かす。"""
        self.data.ctrl[self.carer_actuators] = np.asarray(xyz, dtype=np.float64)

    @property
    def hand_pos(self):
        return self.data.body(CARER + "hand").xpos.copy()

    def reset_model(self):
        self.set_state(self.init_qpos, self.init_qvel)
        qpos = self.init_position.copy()
        qpos[7:] += self.np_random.uniform(low=-self._jitter, high=self._jitter,
                                           size=len(qpos) - 7)
        self.set_state(qpos, np.zeros(self.data.qvel.shape))
        self._set_action(np.zeros(self.action_space.shape))
        mujoco.mj_step(self.model, self.data, nstep=self._settle_steps)
        return self._get_obs()
