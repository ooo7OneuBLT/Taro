"""全体に効く変数（全体に効く変数）。

盤面の世界（2026-10-19、下書きから清書）：欲求の係からの不快（お腹がすいた強さ）を、
vars→want で「選ぶ」係へ渡す（何か行動しやすさと合図の強さのもと）。生まれつき。
"""

from .base import Box


class Vars(Box):
    BOX_ID = "vars"

    def step(self, inbox):
        out = self.passthrough()
        if "need" in inbox:
            out["want"] = inbox["need"]
        return out
