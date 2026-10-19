"""どれくらい得かを見積もる係（価値と気持ち）。

盤面の世界（2026-10-19、core_v2/board/mind.py の下書きから清書）：
「この人にこの合図を向けたら、すぐミルクが来るか」を、合図ごとの1層のNNで見積もる（頭の中のノート）。
入力は人の荷札（64個の数）、合図ごとに重み64個と切片1つ、出口はシグモイド（0〜1）。

* social→worth：部屋にいる人の列（荷札, 地図）。地図が1の人＝そばに来た人（ミルクを持ってきた人）
* inner→worth：この1分でミルクを飲んだか
* want→worth：選んだ行動の写し（どの人にどの合図を向けたか）
* worth→want：人ごとの見込み。地図の1行目に合図ごとの値（[0, 0]＝泣く、[0, 1]＝手を伸ばす）
* worth→stress：期待したのに来なかった分（この1分の合図の見込み）
* worth→feel：うれしさ（思ったより良かった分＝1−見込み）

覚え方（教える答えは「ミルクが来たか」から作る手柄だけ。どの合図が通じるか・誰がよく応えるかは教えない）：
合図の1分あとにミルクが来たら1、1分遅れるごとに RECENCY 倍。来てくれた人に向けた合図だけ手柄で、
別の人に向けた合図は「その人は来なかった」で0。WINDOW_MIN 分以内に来なければ0。
1分に1回、rhythm.WORTH_PHASE の回に覚えて見込みを出す。
"""

from __future__ import annotations

import math
from collections import deque

import torch

from ..formats import DTYPE, GRID, TAG_DIM
from ..rhythm import TICKS_PER_MIN, WANT_PHASE, WORTH_PHASE, phase
from .base import Box

SIGNALS = ("cry", "reach")
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
        self.register_buffer("w", torch.zeros(len(SIGNALS), TAG_DIM, dtype=DTYPE))
        self.register_buffer("b", torch.full((len(SIGNALS),), math.log(PRIOR / (1 - PRIOR)), dtype=DTYPE))
        self.frozen = False    # True なら覚えない（対照）

    def reset(self) -> None:
        self.n = 0
        self.recent = deque()  # (分, 合図の番号, 荷札, 見込み)
        self.offer = []
        self.unmet = 0.0
        self.joy = 0.0

    def value(self, sig: int, tag: torch.Tensor) -> float:
        return 1.0 / (1.0 + math.exp(-(float(self.w[sig] @ tag) + float(self.b[sig]))))

    def learn(self, sig: int, tag: torch.Tensor, target: float) -> None:
        if self.frozen:
            return
        err = target - self.value(sig, tag)
        self.w[sig] += LR * err * tag
        self.b[sig] += LR * err

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
                for t0, sig, tag, _ in self.recent:
                    by_them = came is not None and float(tag @ came) > SAME_PERSON
                    self.learn(sig, tag, RECENCY ** (minute - t0 - 1) if by_them else 0.0)
                self.recent.clear()
            else:
                self.unmet = sum(v for t0, _, _, v in self.recent if t0 == minute - 1)
                while self.recent and minute - 1 - self.recent[0][0] >= WINDOW_MIN:
                    _, sig, tag, _ = self.recent.popleft()
                    self.learn(sig, tag, 0.0)
            self.offer = []
            for tag, _ in people:
                m = torch.zeros(GRID, GRID, dtype=DTYPE)
                for i in range(len(SIGNALS)):
                    m[0, i] = self.value(i, tag)
                self.offer.append((tag, m))
        elif p == WANT_PHASE + 1:
            for tag, m in inbox.get("want") or []:
                sig = int(m[0].argmax())
                self.recent.append((minute, sig, tag, self.value(sig, tag)))
        self.n += 1
        out["want"] = list(self.offer)
        out["stress"] = torch.tensor(min(1.0, self.unmet), dtype=DTYPE)
        out["feel"] = torch.tensor(self.joy, dtype=DTYPE)
        return out

