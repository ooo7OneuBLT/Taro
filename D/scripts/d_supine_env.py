"""
仰向け(supine)の太郎の環境。触覚のON/OFFを切り替えられる。

【なぜ作るか】2026-07-15の発見
録画を見たところ、Cの太郎（margin+51＝最高成績）は**立った直後に転倒し、20秒間ずっと床で
もがいていた**。一方D0（座位・自己接触）の太郎は**腕を胸に畳んだまま10秒間ほぼ静止**し、
触覚は persist 96%（＝「何も変わらない」予測と互角＝学ぶものが無い）だった。
接触ペアを直接読むと、**何もしない太郎ですら「自己接触」が100%**（指が前腕に載りっぱなし）。

→ 仮説：**Cが成功したのは、転んで偶然「手足が自由に振れる状態」になったから**。
   自己モデルの成否を決めているのは動機でも触覚でもなく、**行動で信号がどれだけ変わるか**。

仰向けなら3つ同時に解ける：
  ①転倒しない（既に床にいる）②腕が自由（畳まれていない）③新生児の自己接触は仰向けで起きる

【交絡を避けるための設計】
- シーンは **Cと同一の benchmarkv2_scene.xml**（`MIMoV2DummyEnv`の既定）を使う。
  `roll_over_scene.xml` を借りると `<weld body1="head" body2="upper_body"/>` が付いてきて、
  「姿勢」と「首の溶接」の2つが同時に変わり比較が壊れる（落とし穴チェック項1＝交絡）。
  仰向けにする**計算式だけ**を roll_over.py から借りる。
- したがって立位Cとの差は**初期姿勢ただ1つ**。
- `MIMoBenchV2-v0` は max_episode_steps=6000・終了条件は全てFalse＝**課題が人生を打ち切らない**
  （落とし穴チェック項9はここでは起きない。汚染されていたのはD0の MIMoSelfBody-v0 だけ）。

【出典】仰向けの姿勢の作り方は MIMo 同梱の mimoEnv/envs/roll_over.py（STARTING_POSITION="supine"）
"""
import copy
import numpy as np
import mujoco

from mimoEnv.envs.mimo_env import DEFAULT_TOUCH_PARAMS_V2
from mimo_lean import LeanMimoEnv


def infant_touch_params(factor=2.0):
    """全身の触覚の解像度を factor 倍だけ粗くする（＝乳児の触覚acuity）。

    MIMoの既定はsomatotopy（指先0.002 vs 下腿0.038＝19倍の密度差＝皮質拡大に相当）を
    持っている。**その比を保ったまま**全体を粗くしたいので、全部位に同じ係数を掛ける。
    """
    p = copy.deepcopy(DEFAULT_TOUCH_PARAMS_V2)
    p["scales"] = {k: v * factor for k, v in p["scales"].items()}
    return p


class SupineMimoEnv(LeanMimoEnv):
    """仰向けで始まる太郎。シーン・センサ・終了条件は MIMoBenchV2-v0 と同一。

    LeanMimoEnv を継承＝**視覚OFFのときだけ**巨大テクスチャ(976MB)を単色に置き換える
    （物理は不変・実測でqpos合計まで一致）。視覚をONにすると自動で素のテクスチャに戻る
    ので、「絵が消えたまま視覚実験をしてしまう」事故は起きない。詳細は mimo_lean.py。

    Args:
        settle_steps: 開始前に無操作で落ち着かせるstep数（roll_over.pyに倣い100）。
        jitter: リセット時に全関節へ加える一様乱数の幅。毎回わずかに違う仰向けになる
            （同じ初期姿勢を予測するだけで当たる、という汚染を防ぐ＝落とし穴チェック項9の症状）。
    """

    def __init__(self, settle_steps=100, jitter=0.01, **kwargs):
        self._settle_steps = settle_steps
        self._jitter = jitter
        super().__init__(**kwargs)

        # --- 仰向けにする（roll_over.py の supine と同じ式）---
        self.model.body("hip").pos = [0, 0, 0.2]
        self.model.body("hip").quat = np.array([0, -0.7071068, 0, 0.7071068])
        self.model.body("hip").quat *= np.array([1, -1, 1, 1])   # supine（これが無いとprone＝うつ伏せ）

        for _ in range(self._settle_steps):
            mujoco.mj_step(self.model, self.data)
        self.init_position = self.data.qpos.copy()

    def reset_model(self):
        self.set_state(self.init_qpos, self.init_qvel)
        qpos = self.init_position.copy()
        # 関節だけを揺らす。qpos[:7]はfreejoint（体全体の位置と向き）なので触らない。
        qpos[7:] += self.np_random.uniform(low=-self._jitter, high=self._jitter,
                                           size=len(qpos[7:]))
        self.set_state(qpos, np.zeros(self.data.qvel.shape))
        self._set_action(np.zeros(self.action_space.shape))
        mujoco.mj_step(self.model, self.data, nstep=self._settle_steps)
        return self._get_obs()
