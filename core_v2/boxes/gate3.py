"""門番（3段目、目的を入れる・反射を抑える）。

段階2・工程3の仮部品（仕様書 §3）：判断はしない。
* gate3→orienteye（反射を抑える強さ）：目的を持っている強さをそのまま出す
* gate3→goal（入れてよい）：**いつも1**

**仕様書からの変更（2026-10-15）**：仕様書は「持っている強さをそのまま『入れてよい』にも出す」
としていたが、それだと棚が空のとき強さ0→「入れてよい」0→新しい探し物が永久に入らない
（棚が空のまま固まる）。「段階2では判断しない」の意図に合わせ、門は常に開けておく。
"""

import torch

from ..formats import DTYPE
from .base import Box


class Gate3(Box):
    BOX_ID = "gate3"

    def step(self, inbox):
        out = self.passthrough()
        strength = float(inbox.get("goal", 0.0))
        out["goal"] = torch.tensor(1.0, dtype=DTYPE)
        out["orienteye"] = torch.tensor(min(max(strength, 0.0), 1.0), dtype=DTYPE)
        return out
