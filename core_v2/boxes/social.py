"""他の人を読む係（価値と気持ち）。

盤面の世界（2026-10-19、下書きから清書）：目の入口から届いた「部屋にいる人」の列（人ごとの荷札と地図。
地図が1＝そばに来ている人、0.5＝部屋にいる人）を social→worth へそのまま渡す。
人の気持ちを読むのはまだ（下駄なしで後の段で作る）。
"""

from .base import Box


class Social(Box):
    BOX_ID = "social"

    def step(self, inbox):
        out = self.passthrough()
        if inbox.get("eyein"):
            out["worth"] = list(inbox["eyein"])
        return out
