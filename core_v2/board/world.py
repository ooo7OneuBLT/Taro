"""1段目の盤面：太郎・ママ・ミルク。

1歩＝1分。太郎の行動（泣く・手を伸ばす・何もしない）を受け取り、ママがそれに気づけばミルクを持ってくる。
ママの決まりは世界の側の決まりで、太郎へのほうびではない（太郎が受け取るのはお腹が満ちることだけ）。

* ママは「手が空いている」時と「忙しい」時を行き来する。忙しい間は何にも気づかない
* 手が空いていれば、合図（rule で選ぶ。ふつうは泣き声）に、合図の強さに比べた割合で気づく
* 気づいたら、太郎のお腹が満ちていなければミルクを飲ませる（満ちていれば何もしない）
* 放っておかれてお腹が空のまま長くたつと、ママが自分から様子を見に来る（飢えないため）
* rule="reach" にすると、ママは手を伸ばした方に気づき、泣き声にはほとんど気づかない（対照の世界）
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

ACTIONS = ("nothing", "cry", "reach")

MINUTES_PER_DAY = 24 * 60
EMPTY_AFTER_MIN = 240          # 満腹から空まで4時間
FEED_AMOUNT = 0.7              # 1回の授乳で満ちる量
FEED_IF_BELOW = 0.6            # これより満ちていたらママはミルクをあげない
BUSY_SHARE = 0.3               # ママが忙しい時間の割合
BUSY_LEN_MIN = (5, 30)         # 1回の忙しさの長さ（分）
NOTICE_SIGNAL = 0.5            # 手が空いているとき、強さ1の合図に1分で気づく割合
NOTICE_OTHER = 0.03            # 合図でない方の行動に気づく割合
CHECK_AFTER_EMPTY_MIN = 30     # 空のまま放っておかれたら様子を見に来るまで
CRY_COST = 0.002               # 泣くとお腹が少し早く減る（体の負担）


@dataclass
class Board:
    rule: str = "cry"
    seed: int = 0
    fullness: float = 0.8
    t: int = 0
    mom_busy_left: int = 0
    empty_min: int = 0
    rng: random.Random = field(init=False)

    def __post_init__(self):
        self.rng = random.Random(self.seed)

    def _mom_tick(self):
        if self.mom_busy_left > 0:
            self.mom_busy_left -= 1
            return
        # 手が空いている時に忙しくなる割合：平均の長さから BUSY_SHARE になるように
        mean_len = sum(BUSY_LEN_MIN) / 2
        p_start = BUSY_SHARE / (mean_len * (1 - BUSY_SHARE))
        if self.rng.random() < p_start:
            self.mom_busy_left = self.rng.randint(*BUSY_LEN_MIN)

    def step(self, action: str, strength: float) -> dict:
        """太郎の1分の行動を受け取り、世界を1分進める。太郎に渡すのは観測（お腹・ママが来たか）だけ。"""
        assert action in ACTIONS
        self._mom_tick()
        busy = self.mom_busy_left > 0

        noticed = False
        if not busy and action != "nothing":
            p = NOTICE_SIGNAL * strength if action == self.rule else NOTICE_OTHER
            noticed = self.rng.random() < p
        if self.fullness <= 0.0:
            self.empty_min += 1
        else:
            self.empty_min = 0
        checked = (not busy) and self.empty_min >= CHECK_AFTER_EMPTY_MIN

        fed = False
        if (noticed or checked) and self.fullness < FEED_IF_BELOW:
            self.fullness = min(1.0, self.fullness + FEED_AMOUNT)
            self.empty_min = 0
            fed = True

        self.fullness -= 1.0 / EMPTY_AFTER_MIN
        if action == "cry":
            self.fullness -= CRY_COST * strength
        self.fullness = max(0.0, self.fullness)
        self.t += 1
        return {"fullness": self.fullness, "mom_came": noticed or checked, "fed": fed,
                "busy": busy, "noticed": noticed, "checked": checked}
