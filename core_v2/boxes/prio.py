"""見る場所の優先度地図（2段目 習慣）。

段階2・工程3の仮部品（仕様書 §3）：目立ち×(1−w) ＋ 探し物らしさ×w。

**仕様書からの変更（2026-10-15）**：仕様書は w = 0.9×（目的を持っている強さ）だが、
配線表では prio に強さが届かない（届くのは eyein の目立ち・gaze の探し物らしさ・social だけ）。
段階2では強さは 0 か 1 で、gaze は目的を持っている間だけ地図を出す（探し物が見えなくても
全マスに小さな床 0.001 を敷く、gaze.py）ので、「探し物らしさの地図に何か載っていれば w=0.9、
空なら 0」で同じ値になる。
"""

from ..formats import Fmt, zero
from .base import Box

W_GOAL = 0.9


class Prio(Box):
    BOX_ID = "prio"

    def step(self, inbox):
        out = self.passthrough()
        sal = inbox.get("eyein")
        seek = inbox.get("gaze")
        sal = zero(Fmt.MAP) if sal is None else sal
        seek = zero(Fmt.MAP) if seek is None else seek
        w = W_GOAL if float(seek.max()) > 0.0 else 0.0
        out["orienteye"] = (sal * (1.0 - w) + seek * w).clamp(0.0, 1.0)
        return out
