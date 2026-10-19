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
* 3段目（言葉）：大人は話す（speech に場面ごとの一言の表を渡したとき。LLM で先に作った物）。
  ミルクをあげる時・呼ばれて来た時・手が離せない時・ときどきの話しかけ。言った一言は obs["heard"]。
  太郎が語を言うと（action="say"）、大人は語の意味で応える（understand_words=False なら言葉に応えない＝対照）：
  - 食べ物の語（まんま・みるく）：向けられた大人が「気づきやすさ×WORD_GAIN×強さ」の割合で気づく（泣き声より伝わりやすい）。
    ほかの大人にも OVERHEAR の割合で聞こえる
  - 人の名前（まま・ぱぱ）：呼ばれた人が同じ割合で気づいて来る（来たら、お腹がすいていればミルク）
  - ほかの語：向けられた大人がまれに気づく（NOTICE_OTHER）
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

ACTIONS = ("nothing", "cry", "reach", "say")

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
WORD_GAIN = 1.8                # 意味の分かる語は泣き声よりこれだけ気づかれやすい
FOOD_WORDS = ("まんま", "みるく")
NAME_WORDS = {"まま": "ママ", "ぱぱ": "パパ"}
BUSY_REPLY = 0.3               # 手が離せない大人が、向けられた合図に一言返す割合
CHAT_PER_MIN = 0.03            # 手の空いている大人が、なんとなく話しかける割合（1分あたり）


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
    speech: dict | None = None             # {"feed:ママ": [一言, ...], ...}（3段目）
    understand_words: bool = True
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

    def _say(self, sit: str, who: str) -> str | None:
        lines = (self.speech or {}).get(f"{sit}:{who}")
        return self.rng.choice(lines) if lines else None

    def _notice_word(self, a: Adult, word: str, strength: float, target: str) -> float:
        if not self.understand_words:
            return NOTICE_OTHER if a.name == target else 0.0
        if word in NAME_WORDS:
            return min(1.0, a.notice * WORD_GAIN * strength) if NAME_WORDS[word] == a.name else 0.0
        if word in FOOD_WORDS:
            return min(1.0, a.notice * WORD_GAIN * strength) * (1.0 if a.name == target else OVERHEAR)
        return NOTICE_OTHER if a.name == target else 0.0

    def step(self, action: str, strength: float, target: str | None = None, word: str | None = None) -> dict:
        """太郎の1分の行動を受け取り、世界を1分進める。target は合図を向けた大人の名前、word は言った語（かな）。"""
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
                if action == "say":
                    p = self._notice_word(a, word or "", strength, target)
                elif action == self.rule:
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

        heard = None
        if self.speech:
            if fed:
                heard = self._say("feed", came)
            elif came:
                heard = self._say("called", came)
            elif action != "nothing" and target_busy(self.adults, target) and self.rng.random() < BUSY_REPLY:
                heard = self._say("busy", target)
            elif free and self.rng.random() < CHAT_PER_MIN:
                heard = self._say("chat", self.rng.choice(free).name)

        self.fullness -= 1.0 / EMPTY_AFTER_MIN
        if action == "cry":
            self.fullness -= CRY_COST * strength
        self.fullness = max(0.0, self.fullness)
        self.t += 1
        target_adult = next((a for a in self.adults if a.name == target), None)
        return {"fullness": self.fullness, "fed": fed, "came": came,
                "noticed": noticed_by is not None, "checked": checked_by is not None,
                "busy": target_adult is None or not target_adult.present or target_adult.busy_left > 0,
                "present": self.present(), "heard": heard}


def target_busy(adults, target) -> bool:
    a = next((a for a in adults if a.name == target), None)
    return a is not None and a.present and a.busy_left > 0
