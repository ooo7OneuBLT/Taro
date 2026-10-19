"""ストレスの体の反応（全体に効く変数）。

盤面の世界（2026-10-19、下書きから清書）：いら立ち＝予想の外れ。
「この合図ならすぐミルクが来るはず」と期待していたのに来なかった分（worth→stress）が、1分ごとに積もる。
ミルクを飲んだら（feel→stress）大きく下がり、何もなくても1分ごとに少しずつ下がる。
1分に1回、rhythm.STRESS_PHASE の回に働く。stress→feel で今のいら立ちを渡す。生まれつき（学習は無い）。
"""

import torch

from ..formats import DTYPE
from ..rhythm import STRESS_PHASE, phase
from .base import Box

STRESS_DECAY = 0.8     # 1分で残る割合
RELIEF = 0.3           # ミルクを飲んだ時に残る割合


class Stress(Box):
    BOX_ID = "stress"

    def reset(self) -> None:
        self.n = 0
        self.level = 0.0

    def step(self, inbox):
        out = self.passthrough()
        if phase(self.n) == STRESS_PHASE:
            fed = float(inbox.get("feel", 0.0)) > 0.5
            unmet = float(inbox.get("worth", 0.0))
            self.level = self.level * RELIEF if fed else self.level + unmet
            self.level = min(1.0, self.level * STRESS_DECAY)
        self.n += 1
        out["feel"] = torch.tensor(self.level, dtype=DTYPE)
        return out
