"""こわい・うれしいの重み付け 〔OXT〕（価値と気持ち）。

盤面の世界（2026-10-19、下書きから清書）：
* feel→stress：ミルクを飲んだ（体の内側の入口から。いら立ちを大きく下げる）
* feel→want：今のいら立ち（いら立ちの係から）。不快と合わせて、何か行動しやすさと合図の強さになる
* worth→feel で届く「うれしさ」（思ったより良かった分）は、今は持っておくだけ（last_joy）
生まれつき（学習は無い）。
"""

from .base import Box


class Feel(Box):
    BOX_ID = "feel"

    def reset(self) -> None:
        self.last_joy = 0.0

    def step(self, inbox):
        out = self.passthrough()
        if "gutin" in inbox:
            out["stress"] = inbox["gutin"]
        if "stress" in inbox:
            out["want"] = inbox["stress"]
        if "worth" in inbox:
            self.last_joy = float(inbox["worth"])
        return out
