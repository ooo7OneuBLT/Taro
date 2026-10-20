"""体の内側の入口（入口（感覚））。

盤面の世界（2026-10-19、core_v2/board の下書きから清書）：
* 環境からの入力（inbox[ENV]）：{"fullness": お腹の満ち具合 0〜1, "fed": この1分でミルクを飲んだか,
  "offered": ミルクが口に来ているか（5段目）, "overfed": 満腹なのに飲まされたか（5段目）}
* gutin→need：お腹の空き具合＝1−満ち具合（AMOUNT。入力が無い世界のゼロが「満腹」と同じになるように空き具合で渡す）
* gutin→feel：ミルクを飲んだ（AMOUNT、1 か 0）
* gutin→inner：体の内側の知らせの列（worth.BODY_TAGS の荷札, 地図の [0, 0]＝強さ）：ミルクを飲んだ（1/0）・
  いや＝ミルクが口に来ている×満腹の度合い（満ち具合が SATED から1までで0〜1。満腹ならミルクはいやな物になる、
  Cabanac 1971 の体の状態で快・不快が変わる）・飲まされた（1/0）
入力が無い世界（目だけの場面など）では素通し。生まれつき（学習は無い）。
"""

import torch

from ..formats import DTYPE, GRID
from .base import ENV, Box
from .worth import BODY_TAGS

SATED = 0.8        # 満ち具合がこれより上なら、口に来たミルクがいや


def body_list(**amounts) -> list:
    out = []
    for name, v in amounts.items():
        m = torch.zeros(GRID, GRID, dtype=DTYPE)
        m[0, 0] = float(v)
        out.append((BODY_TAGS[name], m))
    return out


class Gutin(Box):
    BOX_ID = "gutin"
    PURE = True   # 出すものは届いた値だけで決まる（同じ値が届き続ける回は計算しない、brain.tick）

    def step(self, inbox):
        out = self.passthrough()
        env = inbox.get(ENV)
        if not env:
            return out
        fullness = float(env["fullness"])
        out["need"] = torch.tensor(1.0 - fullness, dtype=DTYPE)
        fed = 1.0 if env.get("fed") else 0.0
        out["feel"] = torch.tensor(fed, dtype=DTYPE)
        sated = min(1.0, max(0.0, (fullness - SATED) / (1.0 - SATED)))
        out["inner"] = body_list(fed=fed, aversion=sated if env.get("offered") else 0.0,
                                 overfed=1.0 if env.get("overfed") else 0.0)
        return out
