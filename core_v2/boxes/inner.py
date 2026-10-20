"""体の内側を感じる係（価値と気持ち）。

盤面の世界（2026-10-19、下書きから清書）：体の内側の入口から体の知らせの列（ミルクを飲んだ・いや・飲まされた）を受け、
* inner→worth：そのまま渡す（ほうびを決める知らせ）
* inner→want：いや（満腹なのにミルクが口に来る強さ。5段目の欲求のもと）
生まれつき。
"""

import torch

from ..formats import DTYPE
from .base import Box


class Inner(Box):
    BOX_ID = "inner"
    PURE = True   # 出すものは届いた値だけで決まる（同じ値が届き続ける回は計算しない、brain.tick）

    def step(self, inbox):
        out = self.passthrough()
        x = inbox.get("gutin")
        if isinstance(x, list):
            from .worth import body
            out["worth"] = x
            out["want"] = torch.tensor(body(x).get("aversion", 0.0), dtype=DTYPE)
        elif x is not None:
            out["worth"] = x
        return out
