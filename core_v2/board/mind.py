"""1段目の太郎の心：体・気持ち・何をするか決める係。

生まれつき（手で書いた、育っても変わらない所）：
* お腹が減ると不快が上がる。不快と「思ったのと違った」（いら立ち）が高いほど、何か行動しやすく、強くなる
* 行動を選ぶ時は、覚えた「その行動のあとにお腹が満ちた割合」に比例して選ぶ（確率マッチング。
  赤ちゃんや動物の選び方として知られる形で、強さのつまみを持たない）
* 期待していたのに満ちなかった時に、いら立ちが上がる（予想の外れ）。満ちたら下がる

覚える所（体験から変わる所）：
* 行動ごとに「その行動のあと WINDOW_MIN 分以内にお腹が満ちたか」を数える。満ちた時は、直前の行動ほど多く
  手柄を分ける（1回の授乳の手柄は合わせて1）。値＝手柄の合計÷その行動をした回数。それだけ
* どの行動がママに通じるかは教えない。ほうびはお腹が満ちることだけ（原則：答えに合わせたほうびは置かない）
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

SIGNALS = ("cry", "reach")     # 「何かする」ときの選択肢（何もしない は不快が低い時の休み）
WINDOW_MIN = 10                # 行動とお腹が満ちることを結ぶ時間の幅
RECENCY = 0.5                  # 1分前の行動は、今の行動の何倍の手柄をもらうか
HUNGRY_FROM = 0.6              # お腹がこれより減ると不快が出る
FRUST_DECAY = 0.9              # いら立ちが1分で残る割合
BASE_STRENGTH = 0.3


@dataclass
class Mind:
    seed: int = 0
    tries: dict = field(default_factory=lambda: {a: 1.0 for a in SIGNALS})
    relief: dict = field(default_factory=lambda: {a: 0.1 for a in SIGNALS})   # 生まれた時はどれも同じ
    frustration: float = 0.0
    recent: deque = field(default_factory=deque)   # (時刻, 行動, その時の期待)
    rng: random.Random = field(init=False)
    t: int = 0
    last: dict = field(default_factory=dict)

    def __post_init__(self):
        self.rng = random.Random(10_000 + self.seed)

    def value(self, a: str) -> float:
        """その行動のあと、お腹が満ちた割合（覚えた値、0〜1）。"""
        return self.relief[a] / self.tries[a]

    @staticmethod
    def discomfort(fullness: float) -> float:
        return max(0.0, (HUNGRY_FROM - fullness) / HUNGRY_FROM)

    def act(self, fullness: float) -> tuple[str, float]:
        disc = self.discomfort(fullness)
        arousal = min(1.0, disc + self.frustration)
        if self.rng.random() >= arousal:
            action, strength = "nothing", 0.0
        else:
            w = [self.value(a) for a in SIGNALS]
            r = self.rng.random() * sum(w)
            action = SIGNALS[0] if r < w[0] else SIGNALS[1]
            strength = min(1.0, BASE_STRENGTH + 0.5 * disc + self.frustration)
            self.recent.append((self.t, action, self.value(action)))
        self.last = {"discomfort": disc, "frustration": self.frustration, "arousal": arousal,
                     "action": action, "strength": strength}
        return action, strength

    def feel(self, obs: dict, fullness_before: float) -> dict:
        """1分の結果を受け取って覚える。満ちた量を、窓の中の行動に分けて結ぶ。"""
        gained = max(0.0, obs["fullness"] - fullness_before + 1e-9) if obs["fed"] else 0.0
        joy = 0.0
        if gained > 0 and self.recent:
            w = [RECENCY ** (self.t - t0) for t0, _, _ in self.recent]
            expected = self.recent[-1][2]
            for wi, (_, a, _) in zip(w, self.recent):
                self.tries[a] += 1
                self.relief[a] += wi / sum(w)
            joy = 1.0 - expected             # 思ったより良かった分＝うれしい
            self.recent.clear()
            self.frustration *= 0.3
        # 窓を出た行動は「満ちなかった」として数え、期待していた分だけいら立つ
        while self.recent and self.t - self.recent[0][0] >= WINDOW_MIN:
            _, a, expected = self.recent.popleft()
            self.tries[a] += 1
            self.frustration += expected
        self.frustration = min(1.0, self.frustration * FRUST_DECAY)
        self.t += 1
        self.last.update({"joy": joy, "gained": gained})
        return self.last
