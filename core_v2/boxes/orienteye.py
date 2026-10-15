"""見えた方へ目を向ける反射（1段目 反射、上丘の役）。

段階2・工程3の仮部品（仕様書 §3）：合わせた地図（prio）の最大のマスが、今の注視点
（地図は網膜の座標なので真ん中 13.5,13.5）から MIN_DIST_CELLS マス以上離れていて、前回の
発射から REFRACTORY_TICKS 回（1.2秒）以上たっていれば、そのマスの方向
((x−13.5)/13.5, (13.5−y)/13.5)（[-1,1]、右・上が正）を目の出口へ渡す。撃たない回はゼロ。

抑える強さ（門番から）が 0.5 を超えるときは、目立ちだけの山には撃たない。
**仕様書からの変更（2026-10-15）**：仕様書は「探し物らしさ<0.2 の山へは撃たない」だが、
この係に届くのは合わせた地図だけで、探し物らしさは別には届かない。合わせた地図の値 m と
w=0.9×抑える強さ から、探し物らしさの下限 (m−(1−w))/w（目立ちが最大の1だった場合）を出し、
それが 0.2 未満なら撃たない。目立ちだけの山は m ≤ 1−w なので必ず止まる。
"""

import math

import torch

from ..formats import GRID, DTYPE
from .base import Box

MIN_DIST_CELLS = 2.0
REFRACTORY_TICKS = 12          # 1.2秒（脳の1回＝0.1秒）
INHIBIT_ON = 0.5
MIN_SEEK = 0.2
W_GOAL = 0.9
CENTER = (GRID - 1) / 2.0      # 13.5


class Orienteye(Box):
    BOX_ID = "orienteye"

    def __init__(self) -> None:
        super().__init__()
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.last_fire = -10 ** 9
        self.last_reason = ""     # 記録用

    def step(self, inbox):
        out = self.passthrough()
        self.t += 1
        m = inbox.get("prio")
        inhibit = float(inbox.get("gate3", 0.0))
        if m is None or float(m.max()) <= 0.0:
            self.last_reason = "empty"
            return out
        r, c = divmod(int(torch.argmax(m)), GRID)
        peak = float(m[r, c])
        if math.hypot(c - CENTER, r - CENTER) < MIN_DIST_CELLS:
            self.last_reason = "centered"
            return out
        if self.t - self.last_fire < REFRACTORY_TICKS:
            self.last_reason = "refractory"
            return out
        if inhibit > INHIBIT_ON:
            w = W_GOAL * inhibit
            if (peak - (1.0 - w)) / w < MIN_SEEK:
                self.last_reason = "inhibited"
                return out
        self.last_fire = self.t
        self.last_reason = "fire"
        out["eyeout"] = torch.tensor([(c - CENTER) / CENTER, (CENTER - r) / CENTER], dtype=DTYPE)
        return out
