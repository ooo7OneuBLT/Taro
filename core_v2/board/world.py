"""盤面の世界：太郎・大人（ママ・パパ）・ミルク。

1歩＝1分。太郎の行動（何もしない／泣く／手を伸ばす）と、その合図を誰に向けたかを受け取り、
大人がそれに気づけばミルクを持ってくる。大人の決まりは世界の側の決まりで、太郎へのほうびではない
（太郎が受け取るのはお腹が満ちることだけ）。

* 大人はそれぞれ、部屋にいる時といない時、手が空いている時と忙しい時がある。いない・忙しい間は何にも気づかない
* 手が空いていれば、向けられた合図（rule。ふつうは泣き声）に「気づきやすさ×合図の強さ」の割合で気づく
* 泣き声は部屋にいる全員に聞こえる。向けられていない大人は、その3割の気づきやすさで気づく
* 合図でない方の行動（rule が泣き声なら手を伸ばす）には、向けられた大人だけがまれに気づく
* 気づいたら、太郎のお腹が満ちていなければミルクを飲ませる（満ちていれば何もしない）
* 放っておかれてお腹が空のまま長くたつと、部屋にいる大人が自分から様子を見に来る（飢えないため）
* 1段目：大人はママだけ、いつもいる。2段目：ママとパパ。いる時間と気づきやすさが違う
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

ACTIONS = ("nothing", "cry", "reach")

MINUTES_PER_DAY = 24 * 60
EMPTY_AFTER_MIN = 240          # 満腹から空まで4時間
FEED_AMOUNT = 0.7              # 1回の授乳で満ちる量
FEED_IF_BELOW = 0.6            # これより満ちていたら大人はミルクをあげない
BUSY_SHARE = 0.3               # 大人が忙しい時間の割合
BUSY_LEN_MIN = (5, 30)         # 1回の忙しさの長さ（分）
NOTICE_OTHER = 0.03            # 合図でない方の行動に気づく割合
OVERHEAR = 0.3                 # 向けられていない泣き声に気づく割合（向けられた時に比べて）
PRESENCE_BLOCK_MIN = 30        # 部屋にいる・いないが変わる単位
CHECK_AFTER_EMPTY_MIN = 30     # 空のまま放っておかれたら様子を見に来るまで
CRY_COST = 0.002               # 泣くとお腹が少し早く減る（体の負担）


@dataclass
class Adult:
    name: str
    notice: float              # 手が空いている時、強さ1の合図に1分で気づく割合
    present_share: float = 1.0
    busy_left: int = 0
    present: bool = True


def stage1_adults():
    return [Adult("ママ", notice=0.5)]


def stage2_adults(swap: bool = False):
    """ママはよく応え、パパはあまり応えない（swap で逆）。いる時間はママ7割・パパ6割。"""
    good, poor = 0.5, 0.12
    return [Adult("ママ", notice=poor if swap else good, present_share=0.7),
            Adult("パパ", notice=good if swap else poor, present_share=0.6)]


@dataclass
class Board:
    rule: str = "cry"
    seed: int = 0
    adults: list = field(default_factory=stage1_adults)
    fullness: float = 0.8
    t: int = 0
    empty_min: int = 0
    rng: random.Random = field(init=False)

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self._update_presence()

    def _update_presence(self):
        for a in self.adults:
            a.present = self.rng.random() < a.present_share
        if not any(a.present for a in self.adults):
            self.rng.choice(self.adults).present = True      # 誰か1人はいる

    def _busy_tick(self, a: Adult):
        if a.busy_left > 0:
            a.busy_left -= 1
            return
        mean_len = sum(BUSY_LEN_MIN) / 2
        p_start = BUSY_SHARE / (mean_len * (1 - BUSY_SHARE))
        if self.rng.random() < p_start:
            a.busy_left = self.rng.randint(*BUSY_LEN_MIN)

    def present(self) -> list[str]:
        """太郎に見えている大人（部屋にいる人）。"""
        return [a.name for a in self.adults if a.present]

    def step(self, action: str, strength: float, target: str | None = None) -> dict:
        """太郎の1分の行動を受け取り、世界を1分進める。target は合図を向けた大人の名前。"""
        assert action in ACTIONS
        if self.t > 0 and self.t % PRESENCE_BLOCK_MIN == 0:
            self._update_presence()
        for a in self.adults:
            self._busy_tick(a)
        free = [a for a in self.adults if a.present and a.busy_left == 0]
        if target is None and self.adults:
            target = self.adults[0].name

        noticed_by = None
        if action != "nothing":
            for a in free:
                if action == self.rule:
                    p = a.notice * strength * (1.0 if a.name == target else (OVERHEAR if action == "cry" else 0.0))
                else:
                    p = NOTICE_OTHER if a.name == target else 0.0
                if self.rng.random() < p:
                    noticed_by = a.name
                    break
        if self.fullness <= 0.0:
            self.empty_min += 1
        else:
            self.empty_min = 0
        checked_by = None
        if noticed_by is None and free and self.empty_min >= CHECK_AFTER_EMPTY_MIN:
            checked_by = free[0].name

        fed = False
        came = noticed_by or checked_by
        if came and self.fullness < FEED_IF_BELOW:
            self.fullness = min(1.0, self.fullness + FEED_AMOUNT)
            self.empty_min = 0
            fed = True

        self.fullness -= 1.0 / EMPTY_AFTER_MIN
        if action == "cry":
            self.fullness -= CRY_COST * strength
        self.fullness = max(0.0, self.fullness)
        self.t += 1
        target_adult = next((a for a in self.adults if a.name == target), None)
        return {"fullness": self.fullness, "fed": fed, "came": came,
                "noticed": noticed_by is not None, "checked": checked_by is not None,
                "busy": target_adult is None or not target_adult.present or target_adult.busy_left > 0,
                "present": self.present()}
