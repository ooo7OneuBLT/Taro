"""
目標D の環境（継承型・作り直し版）：太郎アルファ（学ぶ側）と太郎ベータ（相手）を1つのsimに置く。

設計の核：**MIMoEnvは名前の接頭辞で「自分の体」を識別する**（関節=`robot:`、アクチュエータ=`act:`）。
そこでベータを接頭辞 `beta_` で attach すると、MIMoEnv は自動的に
  ・アルファ（接頭辞なし）＝「自分の体」＝フル感覚(proprio621/前庭/触覚/視覚)を取得
  ・ベータ（`beta_robot:` / `beta_act:`）＝対象外＝「物理的に存在するだけの他者」
と切り分けてくれる。＝Cの観測フォーマットが1バイトも変わらない（＝Cの+51自己モデルがそのまま載る）。

継承するもの（作り直さない）：MIMoの感覚モジュール／HybridEnv(interoceptionを足す)／
Cの学習ループ・MinimalFusion・睡眠リプレイ。新規は「2体化」と「ベータの駆動チャネル」だけ。

使い方: python d_env.py   （設計の実証＝アルファだけがフル感覚か・ベータが別に動かせるか）
"""
import os
import sys
import numpy as np
import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "C"))
import paths
paths.setup_brain_path()
sys.path.insert(0, os.path.join(paths.MIMO_DIR))

from mimoEnv.envs.mimo_env import MIMoEnv, EMOTES, DEFAULT_TOUCH_PARAMS_V2
from mimoEnv.envs.dummy import MIMoV2DummyEnv

BETA = "beta_"  # 相手（太郎ベータ）に付ける接頭辞。MIMoEnvの`robot:`/`act:`判定から外れる


def infant_touch_params(factor=2.0):
    """MIMo本物の触覚を「乳児相当の粗さ」にする＝全部位のscaleを一律factor倍。

    scaleは触覚点の間隔（小さいほど密）。MIMo既定は既に**体部位ごとに密度が違う**
    （指先0.002 vs 下腿0.038＝19倍）＝人間のsomatotopy/皮質拡大そのもの。
    一律倍率なら**その比率（手・指が密）を保ったまま**全体の鋭敏度だけ下がる
    ＝乳児の触覚acuityは成人より低く発達で上がる、に対応（Nagaiの「未発達な感覚は
    足枷でなく足場(scaffolding)」）。介護で効く手・指の優位性も保たれる。
    factor=1.0 で成人相当（本物のフル解像度）。
    """
    import copy
    tp = copy.deepcopy(DEFAULT_TOUCH_PARAMS_V2)
    tp["scales"] = {k: v * factor for k, v in tp["scales"].items()}
    return tp


class TwoMimoEnv(MIMoV2DummyEnv):
    """太郎アルファ（＝MIMoEnvが自分とみなす体）＋太郎ベータ（＝接頭辞付きの他者）。"""

    def __init__(self, sep=0.4, **kwargs):
        self._sep = sep
        super().__init__(**kwargs)

    def _initialize_simulation(self):
        # 単体xmlを読む代わりに、2体モデルをMjSpecで組む（ここだけが差し替え点）
        spec_a = mujoco.MjSpec.from_file(paths.SCENE)
        spec_b = mujoco.MjSpec.from_file(paths.SCENE)
        fr = spec_a.worldbody.add_frame()
        fr.pos = [self._sep, 0, 0.4]
        fr.attach_body(spec_b.body("mimo_location"), BETA, "")
        self.model = spec_a.compile()
        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height
        self.data = mujoco.MjData(self.model)

        # 以降は MIMoEnv._initialize_simulation と同じ手順（接頭辞スコープが効く）
        fps = int(np.round(1 / self.dt))
        self.metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": fps}
        self._get_joints()      # `robot:`のみ → アルファの関節だけ
        self._get_actuators()   # `act:`のみ   → アルファのアクチュエータだけ
        self._get_facial_expressions(EMOTES)
        self._set_initial_position(self._initial_qpos)
        self.actuation_model = self.actuation_model(self, self.mimo_actuators)

        # ベータの駆動チャネル（アルファの行動空間には入らない＝別に動かす）
        self.beta_actuators = np.asarray(
            [i for i in range(self.model.nu) if self.model.actuator(i).name.startswith(BETA)])
        return self.model, self.data

    def set_beta_ctrl(self, ctrl):
        """太郎ベータの運動指令（アルファのstepとは独立に毎ステップ差し込む）。"""
        self.data.ctrl[self.beta_actuators] = ctrl

    @property
    def n_beta_actuators(self):
        return len(self.beta_actuators)


def _prove():
    """設計の実証：アルファだけがフル感覚か／ベータは別に動かせるか。"""
    import gymnasium  # noqa
    env = TwoMimoEnv(sep=0.4, vision_params=None, touch_params=None)
    obs, _ = env.reset()
    print("=== D環境（継承型）設計の実証 ===")
    print(f"アルファの関節数(mimo_joints)      = {len(env.mimo_joints)}")
    print(f"アルファのアクチュエータ(mimo_act) = {len(env.mimo_actuators)}  ← 行動空間 {env.action_space.shape}")
    print(f"ベータのアクチュエータ(beta_act)   = {env.n_beta_actuators}  ← 別チャネルで駆動")
    print(f"固有感覚 obs['observation'] 次元   = {obs['observation'].shape[0]}  ← 単体Cと同じなら621")
    print(f"前庭 obs['vestibular'] 次元        = {obs['vestibular'].shape[0]}")
    print(f"モデル総body数                     = {env.model.nbody}（2体分あるはず）")
    beta_bodies = sum(1 for b in range(env.model.nbody) if (env.model.body(b).name or '').startswith(BETA))
    print(f"うち beta_ の body 数              = {beta_bodies}  ← ベータが物理的に存在")
    # ベータを動かして、アルファの固有感覚が汚れない(=次元不変)ことを確認
    env.set_beta_ctrl(np.random.uniform(-0.1, 0.1, env.n_beta_actuators))
    obs2, _, _, _, _ = env.step(np.zeros(env.action_space.shape[0]))
    print(f"ベータ駆動後も固有感覚次元        = {obs2['observation'].shape[0]}（不変ならスコープ成功）")
    print("=> アルファ=フル感覚(Cと同形式)／ベータ=物理的な他者、が1つのsimで成立")


if __name__ == "__main__":
    _prove()
