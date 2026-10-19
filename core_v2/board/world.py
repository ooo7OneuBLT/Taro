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
* 4段目（あれなに？）：objects を渡すと、太郎の目の前に物が置かれる（OBJECT_STAY_MIN 分ほどで替わる。ないときもある）。
  - 問う語（なに・なあに・これ・あれ）：向けられた大人が食べ物の語と同じ割合で気づき、目の前の物の名前を教える一言を言い、
    次の1分にもう一度名前を言う（answer_questions=False なら問う語に応えない＝対照）
  - 目の前に物がある時、手の空いた大人がときどき（NAME_PER_MIN）その物の名前を言う。その半分は「これなあに？」と聞いてから
    次の1分に名前を言う（絵本読みの決まった流れ、Ninio & Bruner 1978）。物が無い時はふつうのおしゃべり
  - 話した大人は obs["spoke"]（太郎の目には、そばに来た人と同じに見える）
* 5段目「断る」（offers=True）：満腹（FULL_FROM 以上）の時、手の空いた大人がときどきミルクをすすめる（obs["offered"]）。
  太郎がその大人に泣く・手で押す（気づきやすさ×強さ）か、断る語（いや・いらない。言葉に応える世界では×WORD_GAIN）を
  向けて気づかれると、やめて「いらないのね」と言う（obs["refused"]。断る語には、すすめの時に大人が言う「もう」「おなかいっぱい」なども入れる：子が言えば大人は断りと受け取る）。OFFER_MIN 分断られなければ飲ませてしまう（obs["overfed"]）
* 5段目「前のことを話す」（news=True）：物が目の前にあった間に部屋にいた大人を覚えておく（seen_by）。太郎が物の名前を
  大人に言って気づかれると、その大人はその物の話をする。自分がいなかった間の物（今は目の前に無い）なら NEWS_JOY の割合で
  喜ぶ（obs["smiled"]）、自分も見ていた物・目の前の物なら PLAIN_JOY（joy_news=False ならいつも PLAIN_JOY＝対照）。
  部屋に戻ってきた大人は「なにかあった？」と聞く（obs["back"]）
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
NAME_PER_MIN = 0.01            # 4段目：目の前に物がある時、聞かれなくても大人がその物の名前を言う割合（1分あたり）［仮置き］
QUIZ_SHARE = 0.5               # そのうち「これなあに？」と聞いてから次の1分に名前を言う割合（Ninio & Bruner 1978）［仮置き］
QUESTION_WORDS = ("なに", "なあに", "これ", "あれ", "なんだ", "なんだい", "なんだろう")
# 5段目「断る」：満腹の時に大人がミルクをすすめる
FULL_FROM = 0.85               # これより満ちていると、大人がときどき「もう少し飲む？」とすすめる
OFFER_PER_MIN = 0.03           # 満腹の時、手の空いた大人がすすめ始める割合（1分あたり）［仮置き］
OFFER_MIN = 3                  # 断られなければ、すすめて3分で飲ませてしまう（満腹なのに飲まされて苦しい）
# 断りとして大人に通じる語（すすめの場面で大人自身が言う語も、子が言えば断りと受け取る）
REFUSE_WORDS = ("いや", "いらない", "ないない", "もう", "もういい", "じゅうぶん", "もうじゅうぶん", "おなかいっぱい", "お腹いっぱい")
# 5段目「前のことを話す」：戻ってきた大人は、自分がいなかった間の物の話を喜ぶ
NEWS_JOY = 0.9                 # 自分がいなかった間の物の名前を言われて、気づいた大人が喜ぶ割合［仮置き］
PLAIN_JOY = 0.2                # 自分も見ていた物・目の前の物の名前の時に喜ぶ割合
OBJECT_STAY_MIN = 15           # 目の前の物が替わるまでの平均（分）
NO_OBJECT = 0.3                # 替わった時に何も置かれない割合


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
    objects: list | None = None            # 4段目：目の前に置かれうる物の名前
    answer_questions: bool = True
    object: str | None = None
    quiz: tuple | None = None              # 4段目：「これなあに？」と聞いた (物, 大人)。次の1分に名前を言う
    offers: bool = False                   # 5段目：満腹の時にミルクをすすめる
    offer: list | None = None              # すすめている [大人, 残りの分]
    news: bool = False                     # 5段目：太郎が物の名前を言うと、大人はその物を知っていたかで応え方を変える
    joy_news: bool = True                  # False なら、いなかった間の物の話も同じにしか喜ばない（対照）
    object_words: dict | None = None       # 物の名前のかな → 物（core_run が渡す）
    seen_by: dict = field(default_factory=dict)   # 物 → その物が目の前にあった間に部屋にいた大人
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
        if word in FOOD_WORDS or (word in QUESTION_WORDS and self.answer_questions and self.object):
            return min(1.0, a.notice * WORD_GAIN * strength) * (1.0 if a.name == target else OVERHEAR)
        return NOTICE_OTHER if a.name == target else 0.0

    def step(self, action: str, strength: float, target: str | None = None, word: str | None = None) -> dict:
        """太郎の1分の行動を受け取り、世界を1分進める。target は合図を向けた大人の名前、word は言った語（かな）。"""
        assert action in ACTIONS
        before = set(self.present())
        if self.t > 0 and self.t % PRESENCE_BLOCK_MIN == 0:
            self._update_presence()
        back = [n for n in self.present() if n not in before]      # 部屋に戻ってきた大人
        if self.object:
            self.seen_by.setdefault(self.object, set()).update(self.present())
        for a in self.adults:
            self._busy_tick(a)
        free = [a for a in self.adults if a.present and a.busy_left == 0]
        if target is None and self.adults:
            target = self.adults[0].name

        noticed_by = None
        refused = overfed = False
        smiled = None
        offerer = self.offer[0] if self.offer else None
        if offerer and not any(a.name == offerer for a in free):
            self.offer, offerer = None, None                       # すすめていた大人が離れた
        tell = None
        if self.news and action == "say" and self.object_words and word in self.object_words:
            tell = self.object_words[word]
        special = (offerer is not None and target == offerer and action != "nothing") or tell is not None
        if special:
            a = next((x for x in free if x.name == target), None)
            if a is not None and offerer == target:
                if action in ("cry", "reach"):                     # いやがる（泣く・押しのける）
                    p = a.notice * strength
                elif action == "say" and word in REFUSE_WORDS:
                    p = min(1.0, a.notice * WORD_GAIN * strength) if self.understand_words else NOTICE_OTHER
                else:
                    p = 0.0
                refused = self.rng.random() < p
            elif a is not None and tell is not None:               # 物の名前を言った
                if self.rng.random() < min(1.0, a.notice * WORD_GAIN * strength):
                    is_news = tell != self.object and a.name not in self.seen_by.get(tell, {a.name})
                    joy = NEWS_JOY if (is_news and self.joy_news) else PLAIN_JOY
                    smiled = a.name if self.rng.random() < joy else None
                    self.seen_by.setdefault(tell, set()).add(a.name)   # 聞いて知った
                    noticed_by = None
                    tell = (tell, a.name, is_news)
                else:
                    tell = None
            if not isinstance(tell, tuple):
                tell = None                                        # 話は届かなかった（気づかれない・すすめの最中）
        elif action != "nothing":
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
        asked = action == "say" and word in QUESTION_WORDS and noticed_by is not None and bool(self.object)
        if asked:
            came = checked_by          # 聞かれて来た大人は名前を教える（ミルクはあげない）
        if came and self.fullness < FEED_IF_BELOW:
            self.fullness = min(1.0, self.fullness + FEED_AMOUNT)
            self.empty_min = 0
            fed = True

        started = False
        if refused:
            self.offer = None
        elif self.offer:
            self.offer[1] -= 1
            if self.offer[1] <= 0:                                 # 断られなかった：飲ませてしまう
                self.fullness, overfed, self.offer = 1.0, True, None
        elif (self.offers and self.fullness >= FULL_FROM and free and not fed
              and self.rng.random() < OFFER_PER_MIN):
            self.offer, started = [self.rng.choice(free).name, OFFER_MIN], True

        heard, spoke = None, None
        attend = False                     # この一言の間、大人が目の前の物を見ている（物の話をしている）
        quiz, self.quiz = self.quiz, None
        if self.speech:
            if refused:
                heard, spoke = self._say("refused", offerer), offerer
            elif overfed:
                heard, spoke = self._say("feed", offerer), offerer
            elif started:
                heard, spoke = self._say("offer", self.offer[0]), self.offer[0]
            elif tell:
                heard, spoke = self._say(f"name:{tell[0]}", tell[1]), tell[1]
            elif asked:
                heard, spoke, attend = self._say(f"name:{self.object}", noticed_by), noticed_by, True
                self.quiz = (self.object, noticed_by)       # 次の1分にもう一度名前を言う（聞かれた名前はくり返して教える）
            elif fed:
                heard = self._say("feed", came)
            elif came:
                heard = self._say("called", came)
            elif action != "nothing" and target_busy(self.adults, target) and self.rng.random() < BUSY_REPLY:
                heard = self._say("busy", target)
            elif quiz and quiz[0] == self.object and any(a.name == quiz[1] for a in free):
                spoke, attend = quiz[1], True           # 「これなあに？」の続き：名前を言う
                heard = self._say(f"name:{self.object}", spoke)
            elif free and self.object and self.rng.random() < NAME_PER_MIN:
                spoke, attend = self.rng.choice(free).name, True   # 目の前の物の話（名前を言う）
                if self.rng.random() < QUIZ_SHARE and (self.speech or {}).get(f"quiz:{spoke}"):
                    heard = self._say("quiz", spoke)
                    self.quiz = (self.object, spoke)
                else:
                    heard = self._say(f"name:{self.object}", spoke)
            elif self.news and back and (self.speech or {}).get(f"back:{back[0]}"):
                spoke = back[0]                              # 戻ってきて「なにかあった？」
                heard = self._say("back", spoke)
            elif free and not self.object and self.rng.random() < CHAT_PER_MIN:
                spoke = self.rng.choice(free).name           # 物が無い時のおしゃべり（ボール遊びの話）
                heard = self._say("chat", spoke)
        if self.objects and self.rng.random() < 1.0 / OBJECT_STAY_MIN:
            self.object = None if self.rng.random() < NO_OBJECT else self.rng.choice(self.objects)
            if self.object:
                self.seen_by[self.object] = set(self.present())      # 新しく置かれた（前に見た人の分は消える）

        self.fullness -= 1.0 / EMPTY_AFTER_MIN
        if action == "cry":
            self.fullness -= CRY_COST * strength
        self.fullness = max(0.0, self.fullness)
        self.t += 1
        target_adult = next((a for a in self.adults if a.name == target), None)
        return {"fullness": self.fullness, "fed": fed, "came": came,
                "noticed": noticed_by is not None, "checked": checked_by is not None,
                "busy": target_adult is None or not target_adult.present or target_adult.busy_left > 0,
                "present": self.present(), "heard": heard, "spoke": spoke, "object": self.object,
                "offered": self.offer is not None, "overfed": overfed, "refused": refused, "smiled": smiled,
                "back": back, "told": tell, "attend": attend}


def target_busy(adults, target) -> bool:
    a = next((a for a in adults if a.name == target), None)
    return a is not None and a.present and a.busy_left > 0
