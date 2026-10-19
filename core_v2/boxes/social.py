"""他の人を読む係（価値と気持ち）。

盤面の世界（2026-10-19、下書きから清書）：目の入口から届いた「部屋にいる人」の列（人ごとの荷札と地図。
地図が1＝そばに来ている人、0.5＝部屋にいる人、行1の [1, 0]＝笑って応えた人）を social→worth へそのまま渡す。
5段目：部屋にいなかった人が入ってくると「会いたかった人が戻ってきた」強さを1にし、1分ごとに REUNION_DECAY 倍にして
social→want へ渡す（相手に向かいたい欲求のもと。生まれつき）。
人の気持ちを読むのはまだ（下駄なしで後の段で作る）。
"""

import torch

from ..formats import DTYPE
from ..rhythm import phase
from .base import Box

REUNION_DECAY = 0.7
SAME_PERSON = 0.3


class Social(Box):
    BOX_ID = "social"

    def __init__(self) -> None:
        super().__init__()
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.before = None       # 前の分に部屋にいた人の荷札（最初の分は数えない）
        self.reunion = 0.0

    def step(self, inbox):
        out = self.passthrough()
        people = inbox.get("eyein")
        if people:
            out["worth"] = list(people)
            if phase(self.n) == 1:
                now = [t for t, _ in people]
                self.reunion *= REUNION_DECAY
                if self.before is not None and any(all(float(t @ b) <= SAME_PERSON for b in self.before) for t in now):
                    self.reunion = 1.0
                self.before = now
            out["want"] = torch.tensor(self.reunion, dtype=DTYPE)
        self.n += 1
        return out
