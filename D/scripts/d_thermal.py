"""
温度感覚（熱受容）を太郎に足す。＝「触れているのは自分の体か、床か」を滑らかに伝える第2の触覚。

【なぜ作るか】2026-07-15 の実測
触覚(接触力)は学べない（persist 97〜104%）。原因を切り分けた結果：
  同じ入力(真の物理状態)・同じデータ・同じ分割・同じ出力次元で対象だけ差し替えると
      対象=固有感覚 → 検証R² **+27.0%**
      対象=触覚     → 検証R² **−65.6%**   （差93ポイント）
＝**触覚という対象に固有の困難**。ただし「情報が無い」のではない（物理が存在を保証する：
関節角度が決まれば体の形は幾何学的に確定し、MuJoCoは決定論的なので接触力は一意に決まる）。
実際は**接触が不連続でカオス的**＝腕が1ミリ動けば接触がON/OFFで反転し力が跳ぶ＝
**決まってはいるが実質的に学習不能**。部位ごとに55次元まで畳んでもダメだった。

→ 打ち手は「情報を足す」ではなく **「予測対象を、暴れる力から滑らかな量に変える」**。
   温度は「何に触れているか」で決まる量なので、腕が1ミリ動いても
   触れている相手が自分の腕なら32℃のまま＝**力が学習不能な理由を構造的に持たない**。

【設計の根拠】
①**温度の値**：人肌=32℃, 室温(床・物体)=22℃。32℃は Ackerley et al. (2014) J Neurosci
  "Human C-Tactile Afferents Are Tuned to the Temperature of a Skin-Stroking Caress"
  （https://www.jneurosci.org/content/34/8/2879）で、人のC触覚線維(CT)が最もよく発火した
  「典型的な皮膚温(neutral)」そのもの。18℃(cool)や42℃(warm)では発火が落ちる。
  ＝**恣意的に決めた定数ではなく、実験で測られた値**を使う。
②**熱慣性（時定数）を入れる**：皮膚には熱容量があり、温度は跳ねずに時定数をもって変化する。
  これが本質＝接触がON/OFFで反転しても温度は滑らかに追従する。「力が学習不能な理由」を
  構造的に持たせない。時定数 tau は既定1.0秒（皮膚の熱的応答のオーダー。**これは根拠の弱い
  仮置き＝逸脱**。感度を見る必要がある）。
③**部位ごと(約55次元)**：CT線維は密度が低く、**指先ではなく前腕・体幹に多い**
  （手のひらは前腕の約1/7：Watkins et al. 2021 J Neurophysiol
  https://journals.physiology.org/doi/full/10.1152/jn.00587.2020）。
  接触力(Aβ)は指先が密(MIMo実測 指先0.002 vs 下腿0.038＝19倍)＝**密度勾配が正反対の2系統**。
  よって per-point の3606次元にするのは人間から遠い。部位ごとが妥当。

【この段階でやらないこと（意図的）】
- CTの逆密度勾配の再現（前腕・体幹を密に）＝まず「滑らかなら学べるか」を確かめてから。
- 自己と他者の区別：**温度では区別できない**（相手の赤ちゃんも32℃）。温度が分けるのは
  「生き物か・物か」。自己/他者は二重接触（触ると両方で感じる）が担う。混同しないこと。

使い方: python d_thermal.py   （単体テスト＝何もしない太郎の温度を測る）
"""
import os
import numpy as np
import gymnasium as gym

SKIN_C = 32.0     # 人肌（Ackerley et al. 2014 のneutral＝典型的な皮膚温）
AMBIENT_C = 22.0  # 室温＝床・物体
_SCALE = 10.0     # (T - AMBIENT)/SCALE → 床=0.0, 人肌=1.0 に正規化


def _is_world(name):
    n = (name or "").lower()
    return n.startswith("world") or "floor" in n or "ground" in n or n == ""


class ThermalWrapper(gym.Wrapper):
    """観測に `thermal`（部位ごとの温度）を足す。物理には一切干渉しない。

    各部位について「今その部位が触れているものの温度」を求め、**熱慣性で滑らかに追従**させる。
      触れているのが太郎の体   → 32℃へ向かう
      触れているのが床・物体   → 22℃へ向かう
      何にも触れていない       → 22℃（外気）へ戻る
    観測は (T - 22)/10 で正規化＝床0.0・人肌1.0。

    Attributes:
        tau: 熱の時定数[秒]。大きいほど滑らか＝予測しやすいが、応答が鈍る。
             **1.0秒は根拠の弱い仮置き（逸脱）。感度を必ず確認すること。**
    """

    def __init__(self, env, tau=1.0):
        super().__init__(env)
        self.tau = tau
        u = env.unwrapped
        self._bodies = sorted(u.touch.sensor_outputs.keys())
        self._names = [u.model.body(b).name for b in self._bodies]
        self._idx = {b: i for i, b in enumerate(self._bodies)}
        self.temp = np.full(len(self._bodies), AMBIENT_C, dtype=np.float64)
        sp = env.observation_space
        if isinstance(sp, gym.spaces.Dict):
            sp.spaces["thermal"] = gym.spaces.Box(-np.inf, np.inf, (len(self._bodies),), np.float64)

    def _target_temp(self):
        """各部位が「今触れているものの温度」。何も触れていなければ外気。"""
        u = self.env.unwrapped
        m, d = u.model, u.data
        tgt = np.full(len(self._bodies), AMBIENT_C, dtype=np.float64)
        for c in range(d.ncon):
            b1 = int(m.geom_bodyid[d.contact[c].geom1])
            b2 = int(m.geom_bodyid[d.contact[c].geom2])
            n1, n2 = m.body(b1).name, m.body(b2).name
            # 相手が太郎の体なら人肌、床・物体なら室温。太郎の体かどうかは
            # 「touchセンサを持つ部位か」で判定する（＝太郎の体表にしかセンサは無い）。
            for me, other, other_name in ((b1, b2, n2), (b2, b1, n1)):
                if me in self._idx:
                    tgt[self._idx[me]] = AMBIENT_C if _is_world(other_name) else (
                        SKIN_C if other in self._idx else AMBIENT_C)
        return tgt

    def _update(self):
        # 熱慣性：dT/dt = (T_target - T)/tau を1 env.step ぶん進める。
        # これが要点＝接触がON/OFFで跳ねても温度は滑らかに追従する。
        dt = self.env.unwrapped.dt
        a = np.exp(-dt / max(self.tau, 1e-6))
        self.temp = a * self.temp + (1 - a) * self._target_temp()

    def _add(self, obs):
        obs = dict(obs)
        obs["thermal"] = (self.temp - AMBIENT_C) / _SCALE   # 床0.0 / 人肌1.0
        return obs

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self.temp = np.full(len(self._bodies), AMBIENT_C, dtype=np.float64)
        return self._add(obs), info

    def step(self, action):
        obs, r, te, tr, info = self.env.step(action)
        self._update()
        return self._add(obs), r, te, tr, info

    @property
    def body_names(self):
        return list(self._names)


def _selftest():
    """何もしない太郎の温度を測る＝『滑らかか』『自分の体を検出できるか』の0コスト確認。"""
    import sys, warnings
    warnings.filterwarnings("ignore")
    _HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _HERE)
    sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "taro_core"))
    import paths
    paths.setup_brain_path()
    sys.path.insert(0, paths.MIMO_DIR)
    import mimoEnv  # noqa
    import torch
    from gymnasium.envs.registration import register
    from hybrid_env import HybridEnv
    from mimoActuation.muscle import MuscleModel
    from test_phase8_motor_learning import rescale_action
    from d_supine_env import SupineMimoEnv, infant_touch_params  # noqa
    register(id="TaroSupine-v0", entry_point="d_supine_env:SupineMimoEnv", max_episode_steps=6000)

    base = HybridEnv(gym.make("TaroSupine-v0", actuation_model=MuscleModel,
                              touch_params=infant_touch_params(2.0), vision_params=None))
    env = ThermalWrapper(base, tau=1.0)
    obs, _ = env.reset(seed=0)
    na = env.action_space.shape[0]
    ctrl = rescale_action(torch.zeros(na), env.action_space)
    hist = []
    for _ in range(30):
        for _ in range(100):
            obs, r, te, tr, info = env.step(ctrl)
            if te or tr:
                break
        hist.append(obs["thermal"].copy())
    H = np.stack(hist)
    print("=== 温度チャンネルの単体テスト（何もしない太郎・仰向け）===")
    print(f"次元={H.shape[1]}（部位ごと）  値域: 床=0.0 / 人肌=1.0\n")
    warm = H[-1] > 0.3
    print(f"最終時点で人肌に温まっている部位: {int(warm.sum())}/{len(warm)}")
    for i in np.argsort(-H[-1])[:8]:
        print(f"   {env.body_names[i]:24s} {H[-1][i]:.3f}")
    # 滑らかさ＝1判断(0.5秒)あたりの変化量。接触力と違って跳ねないはず。
    jump = np.abs(np.diff(H, axis=0)).max()
    print(f"\n1判断あたりの最大変化: {jump:.4f}  ← 小さいほど滑らか（力は不連続に跳ぶ）")
    print(f"温度の変動係数(部位ごとの中央値): "
          f"{float(np.median(H.std(0) / (np.abs(H).mean(0) + 1e-9)))*100:.0f}%")
    env.close()


if __name__ == "__main__":
    _selftest()
