"""体の内側を感じる係（価値と気持ち）。

盤面の世界（2026-10-19、下書きから清書）：体の内側の入口から「ミルクを飲んだ」を受け、
inner→worth へそのまま渡す（ほうびはお腹が満ちることだけ）。生まれつき。
"""

from .base import Box


class Inner(Box):
    BOX_ID = "inner"

    def step(self, inbox):
        out = self.passthrough()
        if "gutin" in inbox:
            out["worth"] = inbox["gutin"]
        return out
