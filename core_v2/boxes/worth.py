"""どれくらい得かを見積もる係（価値と気持ち）。

盤面の世界（2026-10-19、core_v2/board/mind.py の下書きから清書。3段目で語を足した）：
「この人にこの行動を向けたら、すぐミルクが来るか」を1層のNNで見積もる（頭の中のノート）。
入力は行動の荷札 a（泣く・手を伸ばす・言える語。64個の数）と人の荷札 p（64個の数）、
見込み＝シグモイド(aᵀ W p ＋ u・a ＋ v・p ＋ b)。行動が増えても形は変わらない（選択肢が増えるだけ）。
泣く・手を伸ばすの荷札は生まれつきの符号（ACT_TAGS）、語の荷札は語の荷札（word_tag）。

* social→worth：部屋にいる人の列（荷札, 地図）。地図が1の人＝そばに来た人（ミルクを持ってきた人）。列の順番 j が場所
* inner→worth：この1分でミルクを飲んだか
* wordplan→worth：言える語の列（荷札, 空の地図）。3段目から
* want→worth：選んだ行動の写し（行動の荷札, 地図の [0, j]＝1：j 番目の人に向けた）
* worth→want：行動ごとの見込み（行動の荷札, 地図の [0, j]＝j 番目の人に向けた時の見込み）
* worth→stress：期待したのに来なかった分（この1分の行動の見込み）
* worth→feel：うれしさ（思ったより良かった分＝1−見込み）

覚え方（教える答えは「ミルクが来たか」から作る手柄だけ。どの行動が通じるか・誰がよく応えるかは教えない）：
行動の1分あとにミルクが来たら1、1分遅れるごとに RECENCY 倍。来てくれた人に向けた行動だけ手柄で、
別の人に向けた行動は「その人は来なかった」で0。WINDOW_MIN 分以内に来なければ0。
1分に1回、rhythm.WORTH_PHASE の回に覚えて見込みを出す。
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
WINDOW_MIN = 10                # 合図とお腹が満ちることを結ぶ時間の幅（分）
RECENCY = 0.3                  # ミルクが1分遅れるごとに手柄が何倍になるか（赤ちゃんが結び付けるのは数秒の近さまで、Watson 1979）
PRIOR = 0.1                    # 生まれた時の値（どの合図も・誰にも同じ）
LR = 0.2                       # 学び方の速さ
SAME_PERSON = 0.3              # 荷札の似かたがこれより大きければ同じ人（見るたびに少し揺れるので）
NEAR = 0.9                     # 地図がこれより大きい人＝そばに来た人


class Worth(Box):
    BOX_ID = "worth"

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("W", torch.zeros(TAG_DIM, TAG_DIM, dtype=DTYPE))
        self.register_buffer("u", torch.zeros(TAG_DIM, dtype=DTYPE))
        self.register_buffer("v", torch.zeros(TAG_DIM, dtype=DTYPE))
        self.register_buffer("b", torch.tensor(math.log(PRIOR / (1 - PRIOR)), dtype=DTYPE))
        self.frozen = False    # True なら覚えない（対照）

    def reset(self) -> None:
        self.n = 0
        self.recent = deque()  # (分, 行動の荷札, 人の荷札, 見込み)
        self.offer = []
        self.people = []
        self.unmet = 0.0
        self.joy = 0.0

    def value(self, act: torch.Tensor, person: torch.Tensor) -> float:
        x = float(act @ self.W @ person) + float(self.u @ act) + float(self.v @ person) + float(self.b)
        return 1.0 / (1.0 + math.exp(-x))

    def learn(self, act: torch.Tensor, person: torch.Tensor, target: float) -> None:
        if self.frozen:
            return
        err = LR * (target - self.value(act, person))
        self.W += err * torch.outer(act, person)
        self.u += err * act
        self.v += err * person
        self.b += err

    def step(self, inbox):
        out = self.passthrough()
        p, minute = phase(self.n), self.n // TICKS_PER_MIN
        if p == WORTH_PHASE:
            people = inbox.get("social") or []
            fed = float(inbox.get("inner", 0.0)) > 0.5
            came = next((t for t, m in people if float(m.max()) > NEAR), None)
            self.unmet, self.joy = 0.0, 0.0
            if fed:
                if self.recent:
                    self.joy = 1.0 - self.recent[-1][3]
                for t0, act, who, _ in self.recent:
                    by_them = came is not None and float(who @ came) > SAME_PERSON
                    self.learn(act, who, RECENCY ** (minute - t0 - 1) if by_them else 0.0)
                self.recent.clear()
            else:
                self.unmet = sum(v for t0, _, _, v in self.recent if t0 == minute - 1)
                while self.recent and minute - 1 - self.recent[0][0] >= WINDOW_MIN:
                    _, act, who, _ = self.recent.popleft()
                    self.learn(act, who, 0.0)
            self.people = [t for t, _ in people]
            acts = list(ACT_TAGS.values()) + [t for t, _ in inbox.get("wordplan") or []]
            self.offer = []
            for act in acts[:MAX_OBJECTS]:
                m = torch.zeros(GRID, GRID, dtype=DTYPE)
                for j, who in enumerate(self.people[:GRID]):
                    m[0, j] = self.value(act, who)
                self.offer.append((act, m))
        elif p == WANT_PHASE + 1:
            for act, m in inbox.get("want") or []:
                j = int(m[0].argmax())
                if j < len(self.people):
                    who = self.people[j]
                    self.recent.append((minute, act, who, self.value(act, who)))
        self.n += 1
        out["want"] = list(self.offer)
        out["stress"] = torch.tensor(min(1.0, self.unmet), dtype=DTYPE)
        out["feel"] = torch.tensor(self.joy, dtype=DTYPE)
        return out

