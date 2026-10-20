"""体の出口（出口（筋・内臓））。

盤面の世界（2026-10-19、下書きから清書）：手を伸ばす指令が届いた回だけ、環境へ
{"reach": 強さ, "to": j（向けた人の場所＝目に見えている順番）} を返す（outbox[ENV]）。届かない回は何も返さない。
"""

from .base import ENV, Box


class Bodyout(Box):
    BOX_ID = "bodyout"
    PURE = True   # 出すものは届いた値だけで決まる（同じ値が届き続ける回は計算しない、brain.tick）

    def step(self, inbox):
        out = self.passthrough()
        if inbox.get("reach"):
            _, m = inbox["reach"][0]
            out[ENV] = {"reach": float(m.max()), "to": int(m[0].argmax())}
        return out
