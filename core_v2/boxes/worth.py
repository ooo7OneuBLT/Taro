"""どれくらい得かを見積もる係（価値と気持ち）。

盤面の世界（2026-10-19、core_v2/board/mind.py の下書きから清書。3段目で語、4段目でわかる、5段目でいや・喜ぶを足した）：
「この人にこの行動を向けたら、すぐ良いことが起きるか」を、欲求ごとに1層のNNで見積もる（頭の中のノート）。
入力は行動の荷札 a（泣く・手を伸ばす・話す・言える語。64個の数）と人の荷札 p（64個の数）、
見込み＝シグモイド(aᵀ W p ＋ u・a ＋ v・p ＋ b)。行動が増えても形は変わらない（選択肢が増えるだけ）。
泣く・手を伸ばす・話すの荷札は生まれつきの符号（ACT_TAGS・TELL_TAGS）、語の荷札は語の荷札（word_tag）。

見込み（HEADS。地図の行 HEAD_ROW）と、そのほうび（本能由来の3つのどれか）：
* food（行0）：ミルクが来る（お腹）。1段目から
* know（行1）：目の前の物の名前がわかるようになる（わかった＝予想の外れが減る）。4段目
* stop（行2）：満腹なのにミルクが口に来る「いや」が、飲まされずに止まる（お腹：苦しくならない）。5段目
* social（行3）：相手が喜んで応える（親が応える）。5段目
どの欲求のために動くかは「選ぶ」係が決め（want→worth の写しの地図の行 r＝1。food は全部0）、
それぞれの見込みは、その欲求のために選んだ行動だけで覚える（ほかの欲求の結果で、その見込みを下げない）。

* social→worth：部屋にいる人の列（荷札, 地図）。地図の [0, 0]＞NEAR の人＝そばに来た・話した人、
  地図の [1, 0]＝1 の人＝太郎に喜んで応えた人（5段目）。列の順番 j が場所
* inner→worth：体の内側の知らせの列（BODY_TAGS の荷札, 地図の [0, 0]＝強さ）：ミルクを飲んだ・いや（満腹なのに口に来る）・
  飲まされた（満腹なのに）。数1つなら「ミルクを飲んだ」（3段目まで）
* wordplan→worth：言える語の列（荷札, 地図の [0, 0]＝まねしたさ）。届いた語は「できる行動」として覚えておき（known_words）、
  選択肢には泣く・手・話す（話せる時）と、できる語を欲求の順（food, know, stop, social, food, …）に1つずつ、
  その欲求の見込みに「まだあまり言っていない分 EXPLORE/√(1＋言った回数)」と ECHO×まねしたさを足した順で選んで出す
  （新しさのおまけが無いと、見込みが同じ語のうち先に覚えた語ばかりが選択肢になり、まんまが一度も試されなかった）（言った回数の少ない行動を試したくなる、新しさのおまけ。Kakade & Dayan 2002。まねしたさは
  今さっき聞いた語ほど大きい）
* eyein→worth：目の前の物（[(物の荷札, 空)]、無ければ空の列）。語の選択肢を欲求の順に選ぶ時、food のほかは
  今その欲求が起こりうるものだけを回す：know＝物がある、stop＝いや（inner）がある、social＝話せる思い出がある
  （起こりえない欲求に語の枠を回すと、3段目ではミルクの語の枠が6つのうち2つになり、みるくが選択肢に入らなかった）
* episodic→worth：話せる思い出（話す行動の荷札, 地図の [0, j]＝1：j 番目の人に話せる）。5段目
* want→worth：選んだ行動の写し（行動の荷札, 地図の [0, j]＝1：j 番目の人に向けた、[r, j]＝1：行 r の欲求のため）
* semantic→worth：わかった（目の前の物の名前を知る度合いが上がった分。意味の辞書が分の終わりに出し、次の分の初めに読む）
* worth→want：行動ごとの見込み（行動の荷札, 地図の [r, j]＝j 番目の人に向けた時の行 r の見込み）。泣く・手・話すは全部の行、
  語はその語を選んだ欲求の行だけ（ほかの行は0＝その欲求の時には思い浮かばない）。0 の所は選べない
* worth→stress：期待したのに来なかった分（この1分の、ミルクのための行動の見込み）
* worth→feel：うれしさ（思ったより良かった分＝1−見込み。ミルク）

覚え方（教える答えは、ほうびが来たかから作る手柄だけ。どの行動が通じるか・誰がよく応えるかは教えない）：
行動の1分あとにほうびが来たら1、1分遅れるごとに RECENCY 倍。ほうびをくれた人に向けた行動だけ手柄で、
別の人に向けた行動は「その人はくれなかった」で0。WINDOW_MIN 分以内に来なければ0。
* food：ミルクを飲んだ時、そばに来た人
ミルクと結び付いた語（paired）：言える語を聞くたび、ミルクを飲んだ前後（PAIR_MIN 分）に聞いたなら1、ほかは0へ寄せる
（聞いた時にミルクが一緒だった割合。古典的条件づけ、Rescorla & Wagner 1972）。お腹がすいた時に思い浮かぶ語（food の選択肢に
出す順）にだけ足す。選ぶのは今までどおり言って通じた見込みで（パブロフ型の手がかりが、どの行動が思い浮かぶかを変える＝
Pavlovian-instrumental transfer、Cartoni et al. 2016。ミルクと一緒に聞く音を赤ちゃんが多く出す、Sundberg et al. 1996）。
見込みそのものに足すと、ママに向けた全部の行動の見込みが上がって（共有の重み）失敗した（2026-10-19）。
* know：わかった時、前の分にそばにいた人（教えてくれた人）
* stop：「いや」が止まって飲まされなかった時、そばにいた人（やめてくれた人）。飲まされたら全部0
* social：喜んで応えてくれた人
1分に1回、rhythm.WORTH_PHASE の回に覚えて見込みを出す（know は分の初め）。
"""

from __future__ import annotations

import math
from collections import deque

import torch

from ..formats import DTYPE, GRID, MAX_OBJECTS, TAG_DIM
from ..rhythm import TICKS_PER_MIN, WANT_PHASE, WORTH_PHASE, phase
from ..tagging import word_tag
from .base import Box

# 生まれつきの行動の荷札（語の荷札と同じ作り方、頭に "act:"）
ACT_TAGS = {"cry": word_tag("act:cry"), "reach": word_tag("act:reach")}
# 5段目：話す（思い出の帳面にある物の名前を言う）。相手がその物を見ていなかった時と、見ていた時
TELL_TAGS = {"tell_new": word_tag("act:tell_new"), "tell_old": word_tag("act:tell_old")}
# 体の内側の知らせ（gutin→inner・inner→worth の列の荷札）
BODY_TAGS = {"fed": word_tag("body:fed"), "aversion": word_tag("body:aversion"), "overfed": word_tag("body:overfed")}
HEADS = ("food", "know", "stop", "social")
HEAD_ROW = {h: i for i, h in enumerate(HEADS)}
_PARAM = {"food": "", "know": "k", "stop": "r", "social": "j"}   # 見込みのNNの重みの名前の後ろ（W, Wk, Wr, Wj）
WINDOW_MIN = 10                # 合図とほうびを結ぶ時間の幅（分）
RECENCY = 0.3                  # ほうびが1分遅れるごとに手柄が何倍になるか（赤ちゃんが結び付けるのは数秒の近さまで、Watson 1979）
PRIOR = 0.1                    # 生まれた時の値（どの合図も・誰にも同じ）
LR = 0.2                       # 学び方の速さ
SAME_PERSON = 0.3              # 荷札の似かたがこれより大きければ同じ人（見るたびに少し揺れるので）
NEAR = 0.9                     # 地図の [0, 0] がこれより大きい人＝そばに来た人
PAIR_MIN = 2                   # ミルクを飲んでから何分の間に届いた語をミルクと結ぶか（耳を通る遅れの分）
PAIR_LR = 0.1                  # 「聞いた時にミルクが一緒だった割合」の覚える速さ
EXPLORE = 0.1                  # 新しさのおまけ（一度も言っていない語）［仮置き］
ECHO = 0.15                    # まねしたさ1（聞いたばかり）のおまけ。まだ言っていない語のおまけより少し大きく、
                               # 覚えた見込みが0.25を超えた語には負ける（まねは選択肢に入れるだけ）［仮置き］


def body(x) -> dict:
    """inner→worth の中身 → {"fed": 強さ, "aversion": 強さ, "overfed": 強さ}。数1つなら「ミルクを飲んだ」。"""
    if x is None:
        return {}
    if isinstance(x, list):
        out = {}
        for t, m in x:
            name = next((k for k, b in BODY_TAGS.items() if float(b @ t) > 0.99), None)
            if name:
                out[name] = float(m[0, 0])
        return out
    return {"fed": float(x)}


class Worth(Box):
    BOX_ID = "worth"

    def __init__(self) -> None:
        super().__init__()
        prior = math.log(PRIOR / (1 - PRIOR))
        for h in HEADS:
            s = _PARAM[h]
            self.register_buffer("W" + s, torch.zeros(TAG_DIM, TAG_DIM, dtype=DTYPE))
            self.register_buffer("u" + s, torch.zeros(TAG_DIM, dtype=DTYPE))
            self.register_buffer("v" + s, torch.zeros(TAG_DIM, dtype=DTYPE))
            self.register_buffer("b" + s, torch.tensor(prior, dtype=DTYPE))
        self.frozen = False    # True なら覚えない（対照）
        self.known_words: list[torch.Tensor] = []   # できる行動として知っている語（reset で消えない）
        self.said: list[int] = []                    # known_words ごとの言った回数
        self.paired: list[float] = []                # known_words ごとの、聞いた時にミルクが一緒だった割合（思い浮かびやすさ）

    def get_extra_state(self):
        return {"known_words": [w.tolist() for w in self.known_words], "said": list(self.said),
                "paired": list(self.paired)}

    def set_extra_state(self, state) -> None:
        self.known_words = [torch.tensor(w, dtype=DTYPE) for w in state.get("known_words", [])]
        self.said = list(state.get("said", [0] * len(self.known_words)))
        self.paired = list(state.get("paired", [0.0] * len(self.known_words)))
        self.__dict__.pop("_found", None)

    def reset(self) -> None:
        self.n = 0
        self.recent = {h: deque() for h in HEADS}   # 欲求ごと：(分, 行動の荷札, 人の荷札, 見込み)
        self.offer = []
        self.people = []
        self.unmet = 0.0
        self.joy = 0.0
        self.aversion = 0.0

    def _params(self, head: str):
        s = _PARAM[head]
        return getattr(self, "W" + s), getattr(self, "u" + s), getattr(self, "v" + s), getattr(self, "b" + s)

    def value(self, act: torch.Tensor, person: torch.Tensor, head: str = "food") -> float:
        W, u, v, b = self._params(head)
        x = float(act @ W @ person) + float(u @ act) + float(v @ person) + float(b)
        return 1.0 / (1.0 + math.exp(-x))

    def learn(self, act: torch.Tensor, person: torch.Tensor, target: float, head: str = "food") -> None:
        if self.frozen:
            return
        W, u, v, b = self._params(head)
        err = LR * (target - self.value(act, person, head))
        W += err * torch.outer(act, person)
        u += err * act
        v += err * person
        b += err

    def _reward(self, head: str, minute: int, giver) -> None:
        """ほうびが minute の分に来た：giver（荷札）に向けた行動に手柄、ほかは0。"""
        for t0, act, who, _ in self.recent[head]:
            by_them = giver is not None and float(who @ giver) > SAME_PERSON
            self.learn(act, who, RECENCY ** (minute - t0 - 1) if by_them else 0.0, head)
        self.recent[head].clear()

    def _expire(self, head: str, minute: int) -> None:
        q = self.recent[head]
        while q and minute - 1 - q[0][0] >= WINDOW_MIN:
            _, act, who, _ = q.popleft()
            self.learn(act, who, 0.0, head)

    def step(self, inbox):
        out = self.passthrough()
        p, minute = phase(self.n), self.n // TICKS_PER_MIN
        if p == 0 and minute > 0:
            # わかった（前の分に聞いた名前で、目の前の物を知る度合いが上がった分。意味の辞書が前の分の終わりに出す）。
            # 前の分の出来事として、教えてくれた人＝前の分にそばにいた人へ向けた行動に手柄
            if float(inbox.get("semantic", 0.0)) > 0.0:
                self._reward("know", minute - 1, self.__dict__.get("_came"))
            self._expire("know", minute - 1)
        found = self.__dict__.setdefault("_found", {})   # 速くするため：同じ荷札の物（id）→ (荷札, 何番目)。毎回全部と比べない
        for word, m in inbox.get("wordplan") or []:
            hit = found.get(id(word))
            if hit is not None and hit[0] is word:
                i = hit[1]
            else:
                i = next((i for i, k in enumerate(self.known_words) if float(word @ k) > 0.99), None)
                if i is not None:
                    found[id(word)] = (word, i)
            if i is None:
                self.known_words.append(word.clone())
                self.said.append(0)
                self.paired.append(0.0)
                i = len(self.known_words) - 1
            self.__dict__.setdefault("_echo", {})[i] = float(m[0, 0])
        if p == WORTH_PHASE:
            people = inbox.get("social") or []
            b = body(inbox.get("inner"))
            fed = b.get("fed", 0.0) > 0.5
            came = next((t for t, m in people if float(m[0, 0]) > NEAR), None)
            smiled = next((t for t, m in people if float(m[1, 0]) > 0.5), None)
            self.__dict__["_came"] = came
            self.unmet, self.joy = 0.0, 0.0
            # ミルクと結び付いた語：聞いた語が、ミルクの1〜2分前後に聞いたものだったか（耳を通る遅れの分 PAIR_MIN）
            if fed and b.get("overfed", 0.0) <= 0.5:
                self.__dict__["_fed_at"] = minute
            near_milk = minute - self.__dict__.get("_fed_at", -99) <= PAIR_MIN
            for i, e in self.__dict__.get("_echo", {}).items():
                if e >= 1.0 and not self.frozen:
                    self.paired[i] += PAIR_LR * ((1.0 if near_milk else 0.0) - self.paired[i])
            if fed:
                if self.recent["food"]:
                    self.joy = 1.0 - self.recent["food"][-1][3]
                self._reward("food", minute, came)
            else:
                self.unmet = sum(v for t0, _, _, v in self.recent["food"] if t0 == minute - 1)
                self._expire("food", minute)
            # いや：止まって飲まされなかった＝やめてくれた人へ手柄、飲まされた＝全部0
            aversion = b.get("aversion", 0.0)
            if b.get("overfed", 0.0) > 0.5:
                self._reward("stop", minute, None)
            elif self.aversion > 0.0 and aversion == 0.0:
                self._reward("stop", minute, came)
            else:
                self._expire("stop", minute)
            self.aversion = aversion
            if smiled is not None:
                self._reward("social", minute, smiled)
            else:
                self._expire("social", minute)
            self.people = [t for t, _ in people]
            ep = inbox.get("episodic")
            tell = ep if isinstance(ep, list) else []
            seen = inbox.get("eyein")
            # 今その欲求が起こりうる見込みだけ、語を思い浮かべる（物がある・いや・話せる思い出がある）
            heads = ["food"] + ["know"] * bool(isinstance(seen, list) and seen) + ["stop"] * (aversion > 0.0) \
                + ["social"] * bool(tell)
            self.offer = self._make_offer(tell, heads)
        elif p == WANT_PHASE + 1:
            for act, m in inbox.get("want") or []:
                j = int(m[0].argmax())
                if j < len(self.people):
                    who = self.people[j]
                    head = next((h for h in HEADS[1:] if float(m[HEAD_ROW[h], j]) > 0.0), "food")
                    self.recent[head].append((minute, act, who, self.value(act, who, head)))
                for i, k in enumerate(self.known_words):
                    if float(k @ act) > 0.99:
                        self.said[i] += 1
        self.n += 1
        self.__dict__["_echo"] = {}
        out["want"] = list(self.offer)
        out["stress"] = torch.tensor(min(1.0, self.unmet), dtype=DTYPE)
        out["feel"] = torch.tensor(self.joy, dtype=DTYPE)
        return out

    def _make_offer(self, tell, heads=HEADS) -> list:
        """選択肢：泣く・手・話す（話せる人だけ）＋言える語を、今起こりうる欲求 heads の順に1つずつ
        （語は選んだ欲求の行だけに見込みを書く）。"""
        echo = self.__dict__.get("_echo", {})
        vals: dict = {}   # この選択肢づくりの間は重みが変わらないので、見込みは1回だけ計算する

        def val(act, j, who, head):
            key = (id(act), j, head)
            if key not in vals:
                vals[key] = self.value(act, who, head)
            return vals[key]

        keys: dict = {}

        def key(head):
            def k(i):
                if (i, head) not in keys:
                    keys[(i, head)] = k0(i)
                return keys[(i, head)]

            def k0(i):
                v = max((val(self.known_words[i], j, who, head) for j, who in enumerate(self.people)), default=0.0)
                k = v + EXPLORE / math.sqrt(1 + self.said[i]) + ECHO * echo.get(i, 0.0)
                return k + self.paired[i] if head == "food" else k
            return k

        acts = [(a, None, HEADS) for a in ACT_TAGS.values()] + [(a, m, HEADS) for a, m in tell]
        n_words = MAX_OBJECTS - len(acts)
        left, pick = list(range(len(self.known_words))), []
        while left and len(pick) < n_words:
            for h in heads:
                if not left or len(pick) >= n_words:
                    break
                i = max(left, key=key(h))
                pick.append((i, h))
                left.remove(i)
        acts += [(self.known_words[i], None, (h,)) for i, h in pick]
        offer = []
        for act, avail, heads in acts:
            m = torch.zeros(GRID, GRID, dtype=DTYPE)
            for j, who in enumerate(self.people[:GRID]):
                if avail is not None and float(avail[0, j]) <= 0.0:
                    continue                    # この人には話せない
                for h in heads:
                    m[HEAD_ROW[h], j] = val(act, j, who, h)
            offer.append((act, m))
        return offer
