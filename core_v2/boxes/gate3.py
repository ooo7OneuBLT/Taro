"""門番（3段目、目的を入れる・反射を抑える）。

段階2・工程3の仮部品（仕様書 §3）：判断はしない。
* gate3→orienteye（反射を抑える強さ）：目的を持っている強さを、**1回遅らせて**出す
* gate3→goal（入れてよい）：**いつも1**

**仕様書からの変更（2026-10-15）**：仕様書は「持っている強さをそのまま『入れてよい』にも出す」
としていたが、それだと棚が空のとき強さ0→「入れてよい」0→新しい探し物が永久に入らない
（棚が空のまま固まる）。「段階2では判断しない」の意図に合わせ、門は常に開けておく。

**1回遅らせる理由（2026-10-15、合格テストの本番で見つけた）**：目的が棚に置かれてから orienteye へ届くまで、
抑え（goal→gate3→orienteye）は2回、探し物らしさを重く見た地図（goal→gaze→prio→orienteye）は3回かかる。
そのずれの1回、orienteye は「抑えあり」なのに「目立ちだけの地図」を受け取る。orienteye は地図の値から
探し物らしさを逆算するので（orienteye.py）、目立ちの山（値1.0）を探し物と取り違えて撃っていた。
本番ではボールを見ていた試行で、発話の0.5秒後に妨害物へ跳んだ（例：ボール左・種1、4.6秒）。
抑えを1回遅らせると、入るときも外れるときも地図と同じ回に揃う。
"""

import torch

from ..formats import DTYPE
from .base import Box


class Gate3(Box):
    BOX_ID = "gate3"

    def __init__(self) -> None:
        super().__init__()
        self.reset()

    def reset(self) -> None:
        self.prev_strength = 0.0

    def step(self, inbox):
        out = self.passthrough()
        strength = min(max(float(inbox.get("goal", 0.0)), 0.0), 1.0)
        out["goal"] = torch.tensor(1.0, dtype=DTYPE)
        out["orienteye"] = torch.tensor(self.prev_strength, dtype=DTYPE)
        self.prev_strength = strength
        return out
