"""生まれつきの声（泣き声）（1段目 反射）。

盤面の世界（2026-10-19、下書きから清書）：「選ぶ」係から泣く指令（行動の荷札, 地図の [0, j]＝強さ。j＝向ける人の場所）が
届いたら、そのまま声の出口へ渡す（cry→voiceout）。生まれつき。
"""

from .base import Box


class Cry(Box):
    BOX_ID = "cry"
    PURE = True   # 出すものは届いた値だけで決まる（同じ値が届き続ける回は計算しない、brain.tick）

    def step(self, inbox):
        out = self.passthrough()
        if inbox.get("want"):
            out["voiceout"] = list(inbox["want"])
        return out
