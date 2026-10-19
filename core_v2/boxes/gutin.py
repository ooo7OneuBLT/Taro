"""体の内側の入口（入口（感覚））。

盤面の世界（2026-10-19、core_v2/board の下書きから清書）：
* 環境からの入力（inbox[ENV]）：{"fullness": お腹の満ち具合 0〜1, "fed": この1分でミルクを飲んだか}
* gutin→need：お腹の空き具合＝1−満ち具合（AMOUNT。入力が無い世界のゼロが「満腹」と同じになるように空き具合で渡す）
* gutin→feel・gutin→inner：ミルクを飲んだ（AMOUNT、1 か 0）
入力が無い世界（目だけの場面など）では素通し。生まれつき（学習は無い）。
"""

import torch

from ..formats import DTYPE
from .base import ENV, Box


class Gutin(Box):
    BOX_ID = "gutin"

    def step(self, inbox):
        out = self.passthrough()
        env = inbox.get(ENV)
        if not env:
            return out
        out["need"] = torch.tensor(1.0 - float(env["fullness"]), dtype=DTYPE)
        fed = torch.tensor(1.0 if env.get("fed") else 0.0, dtype=DTYPE)
        out["feel"] = fed
        out["inner"] = fed
        return out
