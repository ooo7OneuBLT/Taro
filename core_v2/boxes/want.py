"""「欲しい」を行動へ押す係 〔DA・OXT〕（欲求）。

盤面の世界（2026-10-19、core_v2/board/mind.py の下書きから清書）：何をするか決める係。
* vars→want：不快（お腹がすいた強さ）。feel→want：いら立ち
* worth→want：部屋にいる人ごとの、合図ごとの見込み（地図の1行目）
* 不快＋いら立ちが高いほど何か行動しやすい（その割合で行動し、残りは何もしない）
* 行動するときは、選択肢（合図×向ける相手。部屋にいる人だけ）を見込みに比例して選ぶ（確率マッチング）
* 合図の強さ＝BASE_STRENGTH＋0.5×不快＋いら立ち（1まで）
* want→cry・want→reach：向ける相手の荷札と、強さで塗った地図（選ばなかった方は空）
* want→worth：選んだ行動の写し（荷札と、地図の [0, 合図の番号] が1）
1分に1回、rhythm.WANT_PHASE の回に選ぶ。乱数は torch の大域の乱数。生まれつき（学習は worth の側）。
"""

import torch

from ..formats import DTYPE, GRID
from ..rhythm import WANT_PHASE, phase
from .base import Box

BASE_STRENGTH = 0.3
SIGNALS = ("cry", "reach")     # worth.SIGNALS と同じ並び


class Want(Box):
    BOX_ID = "want"

    def reset(self) -> None:
        self.n = 0
        self.last = {"action": "nothing", "strength": 0.0, "tag": None}

    def step(self, inbox):
        out = self.passthrough()
        if phase(self.n) == WANT_PHASE:
            self.last = {"action": "nothing", "strength": 0.0, "tag": None}
            disc = float(inbox.get("vars", 0.0))
            stress = float(inbox.get("feel", 0.0))
            options = []
            for tag, m in inbox.get("worth") or []:
                for i in range(len(SIGNALS)):
                    options.append((i, tag, float(m[0, i])))
            arousal = min(1.0, disc + stress)
            if options and float(torch.rand(())) < arousal:
                r = float(torch.rand(())) * sum(o[2] for o in options)
                for sig, tag, v in options:
                    r -= v
                    if r <= 0:
                        break
                strength = min(1.0, BASE_STRENGTH + 0.5 * disc + stress)
                self.last = {"action": SIGNALS[sig], "strength": strength, "tag": tag}
                out[SIGNALS[sig]] = [(tag, torch.full((GRID, GRID), strength, dtype=DTYPE))]
                copy = torch.zeros(GRID, GRID, dtype=DTYPE)
                copy[0, sig] = 1.0
                out["worth"] = [(tag, copy)]
        self.n += 1
        return out
