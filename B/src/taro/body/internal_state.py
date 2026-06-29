"""
内受容感覚 — 身体の信号を統合して脳に渡す

【人間模倣】
新生児に生得的にある状態だけを持つ。
生まれた瞬間はarousal（つらさ）の1次元だけ（Bridges 1932）。

新生児の1日のリズム（86400秒=1日）：
- 16-17時間寝る（1回2-4時間の睡眠を6-8回）
- 8-12回授乳（2-3時間おき、1回20-45分）
- 泣きは数分〜数十分続く

B-2：食べる・泣く・寝入るを「一瞬」から「期間がある行動」に変更。
"""

import random


class InternalState:
    """
    太郎の内受容感覚。身体の信号を統合する。

    行動の状態管理もここで行う：
    - 泣き：始まったら数分間続く。世話されたら止まる
    - 寝入り：うとうとしてから5-10分で寝落ちる
    - 睡眠：2-4時間の期間がある
    """

    def __init__(self):
        self.hunger = 0.3
        self.sleepiness = 0.0
        self.discomfort = 0.0

        # --- 睡眠の状態 ---
        self.sleeping = False        # 寝ているか
        self._sleep_remaining = 0    # 残りの睡眠時間（秒）
        self.drowsy = False          # うとうとしているか（寝入り中）
        self._drowsy_remaining = 0   # 残りのうとうと時間（秒）

        # --- 泣きの状態 ---
        self.crying = False          # 泣いているか
        self._cry_remaining = 0      # 残りの泣き時間（秒）
        self.cry_intensity = 0.0     # 泣きの強さ（0〜1）

    def update_from_body(self, stomach, blood_vessel=None, lungs=None):
        """臓器から状態を受け取る。blood_vessel があれば血糖値から空腹を計算。"""
        if blood_vessel is not None:
            self.hunger = blood_vessel.get_hunger()
        else:
            self.hunger = stomach.get_hunger()

        # 食後の眠気（副交感神経系）はB-4以降で adenosine.py に接続予定

    def get_arousal(self):
        """
        つらさ（0〜1）。寝ている間・うとうと中は0。
        """
        if self.sleeping or self.drowsy:
            return 0.0
        return max(self.hunger, self.sleepiness, self.discomfort)

    def get_arousal_delta(self, prev_arousal):
        """つらさの変化量。下がったら正（ほっとした）。"""
        return prev_arousal - self.get_arousal()

    def apply_care(self, care_type):
        """
        世話を受ける。泣いていたら泣き止む。
        """
        if care_type == "comfort":
            self.discomfort = max(0.0, self.discomfort - 0.5)
        elif care_type == "hold":
            self.discomfort = max(0.0, self.discomfort - 0.3)

        # 世話されたら泣き止む（すぐにではなく、強さが徐々に下がる）
        if self.crying:
            self.cry_intensity *= 0.3
            if self.cry_intensity < 0.1:
                self.crying = False
                self._cry_remaining = 0

    def tick(self, adenosine=None, elapsed_seconds=1):
        """
        1秒分の時間経過。

        睡眠中    → アデノシンを除去。残り時間を減らす。
        うとうと中 → アデノシン基礎産生のみ（覚醒はしているが脳活動は最低限）
        起きている → arousal に比例してアデノシン産生。不快が蓄積。泣きの管理。
        """
        # --- 寝ている ---
        if self.sleeping:
            self._sleep_remaining -= elapsed_seconds
            if self._sleep_remaining <= 0:
                self.sleeping = False
                # sleepiness はアデノシンが管理するためここでリセットしない
                self.discomfort = max(0.0, self.discomfort - 0.3)
            if adenosine:
                adenosine.tick_sleep()
                self.sleepiness = adenosine.get()
            return

        # --- うとうと中（寝入りかけ） ---
        if self.drowsy:
            self._drowsy_remaining -= elapsed_seconds
            if self._drowsy_remaining <= 0:
                # 寝落ちる
                self.drowsy = False
                self.sleeping = True
                self._sleep_remaining = random.randint(7200, 14400)
            if adenosine:
                adenosine.tick_awake(arousal=0.0)  # うとうと中は基礎産生のみ
                self.sleepiness = adenosine.get()
            return

        # --- 起きている ---
        if adenosine:
            adenosine.tick_awake(self.get_arousal())
            self.sleepiness = adenosine.get()
        else:
            # adenosine なしで呼ばれた場合のフォールバック
            self.sleepiness = min(1.0, self.sleepiness + 0.00025 * elapsed_seconds)
        # 不快が上がる（ゆっくり。おむつ濡れなどの蓄積）
        self.discomfort = min(1.0, self.discomfort + 0.00005 * elapsed_seconds)

        # 眠さが高いとうとうとし始める
        if self.sleepiness >= 0.9:
            self.drowsy = True
            self._drowsy_remaining = random.randint(300, 600)
            return

        # --- 泣きの管理 ---
        if self.crying:
            # 泣き続けている間、強さがゆっくり変わる
            self._cry_remaining -= elapsed_seconds
            if self._cry_remaining <= 0 or self.get_arousal() < 0.15:
                self.crying = False
                self._cry_remaining = 0
                self.cry_intensity = 0.0
        else:
            # 新たに泣き始めるかの判定
            arousal = self.get_arousal()
            cry_chance = arousal ** 2 * 0.01
            if random.random() < cry_chance:
                self.crying = True
                self._cry_remaining = random.randint(60, 600)
                self.cry_intensity = min(1.0, arousal * 1.2)

    def can_babble(self):
        """
        喃語が出せる状態かどうか。
        起きていて泣いていなければよい。人間の乳児は空腹でも軽く声を出す。
        """
        if self.sleeping or self.drowsy:
            return False
        return not self.crying

    def is_sleeping(self):
        return self.sleeping

    def is_drowsy(self):
        return self.drowsy

    def is_crying(self):
        return self.crying

    def get_state_vector(self):
        """脳に渡す状態の数値リスト。"""
        return [self.hunger, self.sleepiness, self.discomfort, self.get_arousal()]
