"""
D1（視覚あり）：仰向けの太郎＋養育者の手を、太郎が**目で見られる**環境。

【なぜ視覚を足したか】2026-07-16
触覚だけで「相手の行動を知覚・先読み」する路線が、3日間かけて構造的な天井に当たった：
  ・他者は触覚の**3.9%**しかない（床82.5%・自己接触13.6%）＝信号が床に埋もれる
  ・姿勢→接触の予測は AUC 0.71 で頭打ち（学習AUC=1.0なのに検証0.71・容量を8倍にしても不変
    ＝エピソードをまたぐと安定しない関数＝決定論的だが実質学習不能）
  ・力の回帰・温度・自己/外部の分離（二重接触）… 打った手が全部0.7前後に収束
＝**触覚だけで姿勢から接触を予測するのは構造的に上限**。視覚は触覚と違い①他者が常に写る
②接触が要らない（離れていても見える＝先読みの余地）③画素は相手が動けば滑らかに変化する。

【★この環境の肝：視覚ONでもテクスチャを落とす】
`mimo_lean.strip_textures` は従来 `vision_params is None` のときだけ適用していた。理由は
「視覚ONで絵を消すと自分の体か相手の体かを見分けられなくなる」から。**だがこの構成では
養育者は赤いカプセル**で、落とすテクスチャ（顔の表情7種＋服の柄＝976MB）は他者識別に
一切無関係。＝**視覚ONのまま落とせる**。実測：視覚ON 64x64 で 2797MB → 232MB（12分の1）。
＝設計書が視覚を保留した最大の理由（1本2.8GBに戻り並列6本に落ちる）が消える。

【★養育者の色を、太郎自身と変える】
太郎の肌は肌色。養育者を肌色にすると「自分か相手か」が色で見分けにくい。赤に固定して
おく（`CARER_RGBA`）＝視覚で他者だと分かる最小の手がかり。テクスチャは落とすが、この
単色は builtin なので残る。

【継承】
`CarerEnv`（`d1_carer_env.py`）を継承し、`_initialize_simulation` だけ差し替える：
  ・手を生やす処理・重力補償・落下対策・アクチュエータは**親と完全に同じ**（重複を避ける）
  ・違いは strip_textures を「視覚の有無に関わらず」呼ぶ1点だけ
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
from mimo_lean import strip_textures
from d1_carer_env import CarerEnv, CARER, _HAND_HOME, _HAND_RANGE


def lean_vision_params(size=64):
    """他者（赤カプセル）を見分けるための最小の視覚パラメータ。

    既定の 256x256 は他者識別には過剰（実測：メモリは落とせば効かないが、CNNの計算は
    解像度に効く）。まず 64x64 で「相手が写る・行動が読める」を確かめ、足りなければ上げる。
    acuity/foveation は乳児の視覚acuityだが、この段階では交絡を避けてOFF（素の画素で測る）。
    """
    eye = {"width": size, "height": size, "fovy": 60, "acuity": False, "foveation": False}
    return {"eye_left": dict(eye), "eye_right": dict(eye)}


class CarerVisionEnv(CarerEnv):
    """`CarerEnv` と物理は同一で、**視覚ONでもテクスチャを落とす**版。

    親の `_initialize_simulation` は `strip_textures(spec) if self.vision_params is None else 0`
    ＝視覚ONだと落とさない。ここでは**視覚の有無に関わらず落とす**（養育者が赤カプセルなので
    顔・服の絵は不要）。それ以外（手を生やす・重力補償・アクチュエータ・落下対策）は親と同じ。
    """

    CARER_RGBA = (0.9, 0.2, 0.2, 1.0)   # 養育者は赤＝太郎(肌色)と色で見分けられる

    def _initialize_simulation(self):
        spec = mujoco.MjSpec.from_file(self.fullpath)   # 親と同じ＝年齢調整済みシーン
        hand = spec.worldbody.add_body(name=CARER + "hand", pos=list(_HAND_HOME))
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
        g.rgba = list(self.CARER_RGBA)
        for nm in ("x", "y", "z"):
            a = spec.add_actuator()
            a.name = CARER + nm
            a.target = CARER + nm
            a.trntype = mujoco.mjtTrn.mjTRN_JOINT
            kp, kv = 200.0, 20.0
            gp = np.zeros(10); gp[0] = kp
            bp = np.zeros(10); bp[1] = -kp; bp[2] = -kv
            a.gainprm = gp
            a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            a.biasprm = bp
            a.ctrlrange = [-_HAND_RANGE, _HAND_RANGE]
            a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE

        # ★唯一の違い：視覚ONでもテクスチャを落とす（顔・服の絵は赤カプセルの識別に無関係）
        self.n_textures_stripped = strip_textures(spec)
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
            [i for i in range(self.model.nu) if self.model.actuator(i).name.startswith(CARER)])
        self.carer_joints = np.asarray(
            [i for i in range(self.model.njnt) if self.model.joint(i).name.startswith(CARER)])
        return self.model, self.data
