"""目的を持ち続ける係（3段目 計画）。

段階2・工程3の仮部品（設計書 付録E・仕様書 §3）：棚1つ。
* 意味の辞書から新しい荷札が来て、門番の「入れてよい」が 0.5 を超えていれば棚に置く
* 目を向ける先を選ぶ係（gaze）の「見つけた」が FOUND_THRESHOLD を FOUND_STREAK 回続けて
  超えるか、HOLD_TICKS 回（5秒）たったら棚を空にする
渡す物：gaze へ探している物の荷札（空ならゼロ）、門番へ持っている強さ（持っていれば1、空なら0）。
"""

import torch

from ..formats import DTYPE
from .base import Box

FOUND_THRESHOLD = 0.8
FOUND_STREAK = 2
HOLD_TICKS = 50
SAME_COS = 0.95      # これより似ていれば「同じ探し物がもう一度来た」とみなし、置き換えない


class Goal(Box):
    BOX_ID = "goal"

    def __init__(self) -> None:
        super().__init__()
        self.reset()

    def reset(self) -> None:
        self.held = None
        self.age = 0
        self.streak = 0
        self.cleared_by = ""     # 記録用："found" か "timeout"

    def step(self, inbox):
        out = self.passthrough()
        new = inbox.get("semantic")
        permit = float(inbox.get("gate3", 0.0))
        found = float(inbox.get("gaze", 0.0))
        is_new = new is not None and float(torch.linalg.vector_norm(new)) > 0.0
        if is_new and permit > 0.5 and (self.held is None or float(new @ self.held) < SAME_COS):
            self.held, self.age, self.streak, self.cleared_by = new, 0, 0, ""
        elif self.held is not None:
            self.age += 1
            self.streak = self.streak + 1 if found > FOUND_THRESHOLD else 0
            if self.streak >= FOUND_STREAK:
                self.held, self.cleared_by = None, "found"
            elif self.age >= HOLD_TICKS:
                self.held, self.cleared_by = None, "timeout"
        if self.held is not None:
            out["gaze"] = self.held
            out["gate3"] = torch.tensor(1.0, dtype=DTYPE)
        return out
