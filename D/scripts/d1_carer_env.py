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
# 手の定位置（＝ctrl=0 のときに手がいる場所）。**太郎の落下開始高さ(z=0.61)より上**に置く。
# 【なぜ・2026-07-16 実測】`hip.pos`を書き換えても太郎は瞬間移動せず、**立位(腰z=0.61)から
# 重力で自由落下し、床に激突して仰向けに落ち着く**（settle_steps=100はこの落下〜着地が
# ちょうど収まる時間だった）。旧・定位置 z=0.50 は**この落下経路のど真ん中**で：
#   ・手をそのまま置く → 落ちてくる太郎を受け止める（実測：腰が0.050m→0.163m、着地未完了）
#   ・落下中だけ手を退避させる → 退避へ**動く手が太郎を殴る**（斜め上へ逃がすと横から薙いで
#     腰が水平に0.46mズレ、真上へ逃がすと下から突き上げて腰が0.61m→0.76mへ打ち上がった）
# ＝**手を動かして避ける**のが誤り。定位置を最初から落下経路の外(上)に置けば、退避処理も
# 二段階の落ち着かせも要らない。物理的にも自然：親は赤ちゃんを**寝かせてから**手を差し伸べる。
_HAND_HOME = np.array([0.10, 0.0, 1.40])
# スライド関節の可動域。定位置(z=1.40)から太郎(z≈0.05〜0.13)まで降ろせる幅が要る。
_HAND_RANGE = 2.0


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
        # 手は定位置(z=1.40＝落下経路の外)にいるので、太郎はここで邪魔されず自由落下し、
        # 手なし版とまったく同じように仰向けへ着地する。詳細は _HAND_HOME の説明。
        for _ in range(self._settle_steps):
            mujoco.mj_step(self.model, self.data)
        self.init_position = self.data.qpos.copy()

    def _initialize_simulation(self):
        # 【修正・2026-07-16】paths.SCENE(無調整の生データ)ではなく self.fullpath を読む。
        # MIMoEnv.__init__ は既定age=18でadjust_mimo_to_ageを実行し、年齢調整済みの一時シーン
        # (benchmarkv2_scene_temp.xml)をself.fullpathに格納する。paths.SCENEを直接読むと
        # この調整を素通りし、他の環境(仰向け単体など)と体のスケールが食い違う
        # （実測：左手先ジオメトリ 0.00392m(調整後) vs 0.005m(無調整)、触覚センサ数1202→1543の真因）。
        spec = mujoco.MjSpec.from_file(self.fullpath)
        # --- 養育者の手を生やす（世界に直付け＝付け根は動かない＝姿勢制御を持つ大人の代理）---
        # 太郎（仰向け）は頭が+x側・腰が-x側に寝る（実測：腰[0.005,0.004,0.050] 頭[0.303,-0.002,0.068]）。
        # 手の定位置は **太郎の落下開始高さ(z=0.61)より上**（_HAND_HOME の説明を参照）。
        # 実験時に set_hand_target で降ろす。
        hand = spec.worldbody.add_body(name=CARER + "hand", pos=list(_HAND_HOME))
        # 【重要】重力補償。これが無いと手が落下して太郎を弾き飛ばす
        # （実測：太郎の腰が 0.050m → 0.303m に浮き、x が -0.64 までずれた）。
        # 物理的にも正しい：**養育者の腕は大人自身が支えている**ので、太郎の上に落ちてはこない。
        hand.gravcomp = 1.0
        for ax, nm in (([1, 0, 0], "x"), ([0, 1, 0], "y"), ([0, 0, 1], "z")):
            j = hand.add_joint()
            j.name = CARER + nm
            j.type = mujoco.mjtJoint.mjJNT_SLIDE
            j.axis = ax
            j.range = [-_HAND_RANGE, _HAND_RANGE]
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
            a.ctrlrange = [-_HAND_RANGE, _HAND_RANGE]
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
        """養育者の手を、この**ワールド座標**へ動かす（位置サーボの目標を置くだけ）。

        中のスライド関節は定位置 `_HAND_HOME` からの**相対**で動くので、ここで引き算して
        吸収する。＝呼ぶ側は「太郎の胸は z=0.13 だからそこへ」と素直に書ける。
        （相対のまま渡す旧仕様は、定位置を動かした途端に意味が変わる＝バグの温床だった）
        """
        rel = np.asarray(xyz, dtype=np.float64) - _HAND_HOME
        self.data.ctrl[self.carer_actuators] = np.clip(rel, -_HAND_RANGE, _HAND_RANGE)

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
