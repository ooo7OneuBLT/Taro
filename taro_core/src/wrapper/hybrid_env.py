"""
HybridEnv — MIMo（外側の身体）と太郎の内臓（内側の身体）を束ねるラッパー層。

役割（開発計画.md のPhase 4）：
  - MIMoの gym 環境を無改変のまま包む
  - 太郎からコピーした内臓（bridge/taro_modules/）を同居させ、毎「身体秒」ごとに動かす
  - MIMoの観測に「内受容感覚（空腹など）」を足して返す
  - 恒常性（つらさが下がったら報酬）を計算してMIMoの報酬に足す

設計上の判断：
  - 時間の刻み：MIMoは dt=0.01秒（1ステップ＝1/100秒）。太郎の内臓は「1回＝1秒」前提で
    書かれている（確率・閾値がその想定）。そこで STEPS_PER_BODY_SECOND（既定100）ステップ
    ごとに、内臓を1秒分だけ進める。太郎側の定数は一切いじらない。
  - insula（内受容→脳への変換）は「脳側」の部品なのでここでは動かさない。生の内受容ベクトル
    [hunger, sleepiness, discomfort, arousal] を観測に足すだけ。insula変換は脳と繋ぐPhase 5で。
  - 授乳（親）は本来 parent_sim の役割。Phase 4ではまだ脳も親も繋がないので、「空腹が高い時に
    自動で授乳する」簡易スタンドインを置く（★後で本物の親シミュレータに差し替える）。

内臓の毎秒の配線は Taro の core_b.py tick_body() を忠実に再現：
  胃.tick → 血糖.receive_glucose(吸収量×効率) → 血糖.tick → 内受容.update_from_body → 内受容.tick → 胃.grow
  （肺は発声/聴覚サブシステムの部品なのでPhase 4では除外）
"""

import os
import sys
import numpy as np
import gymnasium
from gymnasium import spaces

# コピーした内臓モジュール（src/body/）を import できるようにする
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "body"))
from internal_state import InternalState
from stomach import Stomach
from blood_vessel import BloodVessel
from adenosine import Adenosine
from homeostasis import Homeostasis


class HybridEnv(gymnasium.Wrapper):
    """MIMo環境を包み、太郎の内受容感覚と恒常性報酬を合成するラッパー。"""

    #: 内臓を1秒進めるのに必要なMIMoステップ数（MIMo dt=0.01 → 100ステップ=1秒）
    STEPS_PER_BODY_SECOND = 100

    #: Taro core_b.py の既定値（血糖効率）
    GLUCOSE_EFFICIENCY = 3.0

    #: 簡易授乳（★スタンドイン）：空腹がこの値を超え、授乳中でなければ授乳する
    FEED_THRESHOLD = 0.6
    FEED_AMOUNT = 0.6

    def __init__(self, env, steps_per_body_second=STEPS_PER_BODY_SECOND):
        super().__init__(env)
        self.steps_per_body_second = steps_per_body_second
        self._step_in_second = 0
        self._init_body()

        # 観測空間に内受容感覚（4次元）を追加する
        base_spaces = dict(self.env.observation_space.spaces)
        base_spaces["interoception"] = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.observation_space = spaces.Dict(base_spaces)

    # ------------------------------------------------------------------
    # 内臓（内側の身体）の管理
    # ------------------------------------------------------------------
    def _init_body(self):
        """内臓を初期状態で作り直す。"""
        self.internal_state = InternalState()
        self.stomach = Stomach()
        self.blood_vessel = BloodVessel()
        self.adenosine = Adenosine()
        self.homeostasis = Homeostasis()
        self._step_in_second = 0

    def advance_body_one_second(self):
        """
        内臓を1秒分だけ進める。Taro core_b.tick_body() の毎秒処理を忠実に再現。
        戻り値：恒常性報酬（つらさが下がった分。下がれば正、上がれば負）。
        """
        # --- 胃 → 血糖 → 空腹 の順に更新（肺はPhase 4では除外） ---
        self.stomach.tick()
        self.blood_vessel.receive_glucose(
            self.stomach.get_last_absorption() * self.GLUCOSE_EFFICIENCY
        )
        self.blood_vessel.tick()
        self.internal_state.update_from_body(self.stomach, self.blood_vessel, None)
        self.internal_state.tick(adenosine=self.adenosine)
        self.stomach.grow()

        # --- 簡易授乳（★後で本物の親シミュレータに差し替える） ---
        if (self.internal_state.hunger > self.FEED_THRESHOLD
                and not self.stomach.is_feeding()):
            self._feed(self.FEED_AMOUNT)

        # --- 恒常性の報酬（つらさが下がったら報酬） ---
        return self.homeostasis.compute_reward(self.internal_state.get_arousal())

    def _feed(self, amount):
        """授乳を開始する（Taro core_b.feed() の内受容に関わる部分のみ）。"""
        self.stomach.start_feeding(amount)
        self.internal_state.on_feed(amount)

    def _interoception_vector(self):
        """観測に足す内受容ベクトル [hunger, sleepiness, discomfort, arousal]。"""
        return np.asarray(self.internal_state.get_state_vector(), dtype=np.float32)

    def _augment_obs(self, obs):
        """MIMoの観測（dict）に内受容感覚を足す。"""
        obs = dict(obs)
        obs["interoception"] = self._interoception_vector()
        return obs

    # ------------------------------------------------------------------
    # gym インターフェース
    # ------------------------------------------------------------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._init_body()
        return self._augment_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # MIMoを100ステップ進めるごとに、内臓を1秒進める
        self._step_in_second += 1
        body_reward = 0.0
        if self._step_in_second >= self.steps_per_body_second:
            self._step_in_second = 0
            body_reward = self.advance_body_one_second()

        reward = float(reward) + float(body_reward)
        info = dict(info)
        info["homeostasis_reward"] = body_reward
        info["hunger"] = self.internal_state.hunger
        return self._augment_obs(obs), reward, terminated, truncated, info
