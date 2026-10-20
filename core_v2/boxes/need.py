"""体の状態から「欲しい」を作る係（欲求）。

盤面の世界（2026-10-19、下書きから清書）：お腹の満ち具合が HUNGRY_FROM より減ると不快が出る
（減るほど強い、0〜1）。need→vars で全体に効く変数へ。生まれつき（学習は無い）。
"""

import torch

from ..formats import DTYPE
from .base import Box

HUNGRY_FROM = 0.6


class Need(Box):
    BOX_ID = "need"
    PURE = True   # 出すものは届いた値だけで決まる（同じ値が届き続ける回は計算しない、brain.tick）

    def step(self, inbox):
        out = self.passthrough()
        if "gutin" not in inbox:
            return out
        fullness = 1.0 - float(inbox["gutin"])          # 届くのは空き具合
        disc = max(0.0, (HUNGRY_FROM - fullness) / HUNGRY_FROM)
        out["vars"] = torch.tensor(disc, dtype=DTYPE)
        return out
