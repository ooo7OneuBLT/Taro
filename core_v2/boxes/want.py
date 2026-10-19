"""「欲しい」を行動へ押す係 〔DA・OXT〕（欲求）。

盤面の世界（2026-10-19、core_v2/board/mind.py の下書きから清書）：何をするか決める係。
* vars→want：不快（お腹がすいた強さ）。feel→want：いら立ち
* worth→want：行動ごとの見込み（行動の荷札, 地図の [0, j]＝j 番目の人に向けた時の見込み）
* 不快＋いら立ちが高いほど何か行動しやすい（その割合で行動し、残りは何もしない）
* 行動するときは、選択肢（行動×向ける相手。部屋にいる人だけ）を見込みに比例して選ぶ（確率マッチング）
* 合図の強さ＝BASE_STRENGTH＋0.5×不快＋いら立ち（1まで）
* want→cry・want→reach・want→wordplan（語を言う。3段目）：(行動の荷札, 地図の [0, j]＝強さ。j＝向ける人の場所)
* want→worth：選んだ行動の写し（行動の荷札, 地図の [0, j]＝1）
1分に1回、rhythm.WANT_PHASE の回に選ぶ。乱数は torch の大域の乱数。生まれつき（学習は worth の側）。
"""

import torch

from ..formats import DTYPE, GRID
from ..rhythm import WANT_PHASE, phase
from ..tagging import word_tag
from .base import Box

BASE_STRENGTH = 0.3
# 生まれつきの行動の荷札（worth.ACT_TAGS と同じ符号）。それ以外の荷札は語として口へ
ACT_OUT = {"cry": word_tag("act:cry"), "reach": word_tag("act:reach")}


class Want(Box):
    BOX_ID = "want"

    def reset(self) -> None:
        self.n = 0
        self.last = {"action": "nothing", "strength": 0.0, "tag": None, "to": None}

    def step(self, inbox):
        out = self.passthrough()
        if phase(self.n) == WANT_PHASE:
            self.last = {"action": "nothing", "strength": 0.0, "tag": None, "to": None}
            disc = float(inbox.get("vars", 0.0))
            stress = float(inbox.get("feel", 0.0))
            options = []
            for act, m in inbox.get("worth") or []:
                for j in range(GRID):
                    if float(m[0, j]) > 0:
                        options.append((act, j, float(m[0, j])))
            arousal = min(1.0, disc + stress)
            if options and float(torch.rand(())) < arousal:
                r = float(torch.rand(())) * sum(o[2] for o in options)
                for act, j, v in options:
                    r -= v
                    if r <= 0:
                        break
                strength = min(1.0, BASE_STRENGTH + 0.5 * disc + stress)
                kind = next((k for k, t in ACT_OUT.items() if float(t @ act) > 0.99), "wordplan")
                self.last = {"action": kind if kind != "wordplan" else "say", "strength": strength, "tag": act, "to": j}
                m = torch.zeros(GRID, GRID, dtype=DTYPE)
                m[0, j] = strength
                out[kind] = [(act, m)]
                copy = torch.zeros(GRID, GRID, dtype=DTYPE)
                copy[0, j] = 1.0
                out["worth"] = [(act, copy)]
        self.n += 1
        return out
