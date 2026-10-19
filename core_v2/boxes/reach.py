"""手を伸ばす・つかむ地図（2段目 習慣）。

盤面の世界（2026-10-19、下書きから清書）：「選ぶ」係から手を伸ばす指令（行動の荷札, 地図の [0, j]＝強さ。j＝向ける人の場所）が
届いたら、そのまま体の出口へ渡す（reach→bodyout）。どこへどう伸ばすかの地図はまだ（体のある世界で作る）。
"""

from .base import Box


class Reach(Box):
    BOX_ID = "reach"

    def step(self, inbox):
        out = self.passthrough()
        if inbox.get("want"):
            out["bodyout"] = list(inbox["want"])
        return out
