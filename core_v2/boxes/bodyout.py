"""体の出口（出口（筋・内臓））。

盤面の世界（2026-10-19、下書きから清書）：手を伸ばす指令が届いた回だけ、環境へ
{"reach": 強さ, "to": 向けた相手の荷札} を返す（outbox[ENV]）。届かない回は何も返さない。
"""

from .base import ENV, Box


class Bodyout(Box):
    BOX_ID = "bodyout"

    def step(self, inbox):
        out = self.passthrough()
        if inbox.get("reach"):
            tag, m = inbox["reach"][0]
            out[ENV] = {"reach": float(m.max()), "to": tag}
        return out
