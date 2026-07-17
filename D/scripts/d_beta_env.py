"""
2体環境（本番）：アルファ＝太郎本体（脳で自律・フル感覚）＋ベータ＝操り人形（見た目は太郎の
複製・中身は3自由度で操縦する剛体）。

【設計・2026-07-17 ユーザー提案】ベータの見た目だけ太郎と同じにし、中身は単純なプログラムで
操る。全関節を削除して1個の剛体にする（`d_beta_puppet.py`で単体検証済み）ことで、MIMo2体を
関節ごと重ねると3.3m吹き飛ぶ物理爆発を回避（危険なのは「関節が独立に動くラグドール2体」）。

【アルファはそのまま】接頭辞なし（robot:/act:）＝MIMoEnvが「自分の体」と認識＝フル感覚
(固有感覚621・前庭・触覚・視覚)・観測フォーマットは単体時と1バイトも変わらない＝Cの脳・
fusionがそのまま載る。ベータは接頭辞 beta_ ＝MIMoEnvの対象外＝「物理的に存在する他者」。

【継承】`CarerVisionEnv` を継承：
  ・`__init__`（仰向け＋落ち着かせ）と `get_vision_obs`（眼球カメラの直接描画）を流用
  ・`_initialize_simulation` だけ差し替え＝手の代わりにベータ人形を生やす

使い方: python d_beta_env.py   （2体が共存・アルファがフル感覚・ベータが操縦可能かを検証）
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import mujoco

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # 端末のcp932で日本語/記号が化けるのを防ぐ
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "taro_core"))
import paths
paths.setup_brain_path()
sys.path.insert(0, paths.MIMO_DIR)

from mimoEnv.envs.mimo_env import EMOTES
from mimo_lean import strip_textures
from d1_carer_vision_env import CarerVisionEnv

BETA = "beta_"
_BETA_RANGE = 2.0   # ベータ操縦スライド関節の可動域(m)


class BetaPuppetEnv(CarerVisionEnv):
    """仰向けの太郎（アルファ）＋見た目は複製・中身は剛体のベータ人形（3自由度で操縦）。"""

    # ベータの定位置（＝ctrl=0のときの位置）。仰向けアルファ(hip z≈0.05)の上方に置く
    # ＝親が寝た赤ちゃんの上に立つ／覗き込むイメージ。ここは視覚ジオメトリ調整で後日詰める。
    BETA_HOME = np.array([0.15, 0.0, 0.9])
    STRIP_BETA_TEXTURE = False   # 見た目を太郎の複製のまま残す（ユーザー方針）。Trueで単色化。

    def _initialize_simulation(self):
        # アルファ（接頭辞なし）は self.fullpath（年齢調整済みシーン）から。
        spec = mujoco.MjSpec.from_file(self.fullpath)
        # ベータ（複製）は別インスタンスから読んで attach（同じspecの自己attachはID不整合になる）。
        spec_beta = mujoco.MjSpec.from_file(self.fullpath)
        fr = spec.worldbody.add_frame()
        fr.pos = list(self.BETA_HOME)
        beta_root = fr.attach_body(spec_beta.body("mimo_location"), BETA, "")

        # --- ベータを「関節ゼロ＝1個の剛体」にする（d_beta_puppet.pyで確立した手順）---
        # 依存の逆順で消す：アクチュエータ→腱→等式拘束→関節（先に関節を消すと参照エラー）。
        for a in list(spec.actuators):
            if a.name.startswith(BETA):
                a.delete()
        for t in list(spec.tendons):
            if t.name.startswith(BETA):
                t.delete()
        beta_joint_names = {j.name for j in spec.joints if j.name.startswith(BETA)}
        for eq in list(spec.equalities):
            if eq.name1 in beta_joint_names or eq.name2 in beta_joint_names:
                eq.delete()
        for j in list(spec.joints):
            if j.name.startswith(BETA):
                j.delete()
        # 重力補償はベータ配下の全56体に（ルート1体だけでは残り55体分が沈む＝実測）。
        for b in spec.bodies:
            if b.name.startswith(BETA):
                b.gravcomp = 1.0

        # --- ベータを動かす3自由度スライド関節＋位置サーボ（d1_carer_env.pyの手と同じ）---
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

        if self.STRIP_BETA_TEXTURE:
            strip_textures(spec)

        self.model = spec.compile()
        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height
        self.data = mujoco.MjData(self.model)

        fps = int(np.round(1 / self.dt))
        self.metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": fps}
        self._get_joints()      # robot: のみ → アルファの関節だけ（beta_は入らない）
        self._get_actuators()   # act:   のみ → アルファのアクチュエータだけ
        self._get_facial_expressions(EMOTES)
        self._set_initial_position(self._initial_qpos)
        self.actuation_model = self.actuation_model(self, self.mimo_actuators)
        # 「carer_*」の名前を流用（親のset_hand_target等が使うため）＝ここではベータ操縦関節を指す。
        self.carer_actuators = np.asarray(
            [i for i in range(self.model.nu) if self.model.actuator(i).name.startswith(BETA)])
        self.carer_joints = np.asarray(
            [i for i in range(self.model.njnt) if self.model.joint(i).name.startswith(BETA)])
        return self.model, self.data

    def set_beta_target(self, xyz):
        """ベータを、このワールド座標へ動かす（定位置BETA_HOMEからの相対に直して位置サーボへ）。"""
        rel = np.asarray(xyz, dtype=np.float64) - self.BETA_HOME
        self.data.ctrl[self.carer_actuators] = np.clip(rel, -_BETA_RANGE, _BETA_RANGE)

    @property
    def beta_pos(self):
        return self.data.body(BETA + "mimo_location").xpos.copy()


if __name__ == "__main__":
    from d1_carer_vision_env import lean_vision_params

    print("=== 2体環境（アルファ太郎＋ベータ人形）の検証 ===")
    env = BetaPuppetEnv(vision_params=lean_vision_params(64))
    obs, _ = env.reset(seed=0)

    print("\n--- ① アルファはフル感覚か（観測フォーマットが単体と同じか）---")
    for k in ("observation", "vestibular", "touch"):
        if k in obs:
            print(f"  obs['{k}'] 次元 = {np.asarray(obs[k]).shape}")
    print(f"  固有感覚621・前庭6なら単体Cと同一フォーマット")

    print("\n--- ② ベータは操縦可能・物理は安定か ---")
    na = env.action_space.shape[0]
    print(f"  アルファの行動次元(beta除外) = {na}")
    for tgt in [(0.15, 0.2, 0.9), (0.15, -0.2, 0.9), (0.15, 0.0, 0.7)]:
        env.unwrapped.set_beta_target(tgt)
        for _ in range(50):
            env.step(np.zeros(na, np.float32))
        print(f"  beta目標{tgt} -> 実位置 {np.round(env.unwrapped.beta_pos, 3)}")

    print("\n--- ③ 2体が爆発せず共存しているか ---")
    hip = env.unwrapped.data.body("hip").xpos.copy()
    print(f"  アルファ hip 位置 = {np.round(hip, 3)}（仰向けで安定なら z≈0.05〜0.2）")
    print(f"  ベータ位置 = {np.round(env.unwrapped.beta_pos, 3)}")
    assert np.all(np.abs(hip) < 3.0) and np.all(np.abs(env.unwrapped.beta_pos) < 3.0), "爆発の疑い"
    print("\nOK: アルファ=フル感覚／ベータ=操縦可能／2体が安定共存")
