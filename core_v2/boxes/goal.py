"""目的を持ち続ける係（3段目 計画）。

段階2・工程3の仮部品（設計書 付録E・仕様書 §3）：棚1つ。
* 意味の辞書から新しい荷札が来て、門番の「入れてよい」が 0.5 を超えていれば棚に置く
* 目を向ける先を選ぶ係（gaze）の「見つけた」が FOUND_THRESHOLD を FOUND_STREAK 回続けて
  超えたら、そこから FOUND_HOLD_TICKS 回（2秒）は持ち続けてから棚を空にする。
  棚に置いてから HOLD_TICKS 回（5秒）たったら、見つけたかどうかに関係なく空にする

FOUND_HOLD_TICKS（2026-10-15、スレッドの責任者の決定）：人間の乳児も、探していた物を見つけたら
しばらくそれを見続ける。最初は見つけた瞬間に棚を空にしていたが、合格テストの本番（818eef8）で、
発話の時点で既にボールを見ていた試行では 0.3〜0.6秒で目的が消え、反射が自由になって妨害物へ
目が跳んだ（ボールの条件B 11/20、失敗9回すべてがこの形）。見つけた後も2秒は目的を持ち続け、
その間は門番を通して反射を抑え続ける。
渡す物：gaze へ探している物の荷札（空ならゼロ）、門番へ持っている強さ（持っていれば1、空なら0）。

FOUND_THRESHOLD（仮部品のルール、2026-10-15 実測で決めた）：仕様書の仮の値は 0.8 だったが、探し物を
見ていても「見つけた」は 0.60〜0.85 で、0.8 を超えるのはまれ（目的が毎回5秒の時間切れで空になっていた）。
目的を持たせたまま目を手で各物へ向けて測った分布（G/scripts/measure_found.py、2つの置き方×2語×2種、
G/logs/見つけた/found_summary.json・found.png）：
  探し物を見ている回 136回：下から5% 0.64・中央 0.73・最大 0.85（切り出しが抜けた4回だけ 0.001）
  探し物以外を見ている回 160回：中央 0.001・最大 0.22
2つの分布の間（0.22 と 0.64 の真ん中あたり）の 0.45 にした。
"""

import torch

from ..formats import DTYPE
from .base import Box

FOUND_THRESHOLD = 0.45
FOUND_STREAK = 2
HOLD_TICKS = 50
FOUND_HOLD_TICKS = 20
SAME_COS = 0.95      # これより似ていれば「同じ探し物がもう一度来た」とみなし、置き換えない


class Goal(Box):
    BOX_ID = "goal"

    def __init__(self) -> None:
        super().__init__()
        self.reset()

    def reset(self) -> None:
        self.held = None
        self.age = 0
        self.streak = 0
        self.found_age = None    # 見つけた回からの回数（見つけていなければ None）
        self.cleared_by = ""     # 記録用："found" か "timeout"

    def step(self, inbox):
        out = self.passthrough()
        new = inbox.get("semantic")
        permit = float(inbox.get("gate3", 0.0))
        found = float(inbox.get("gaze", 0.0))
        is_new = new is not None and float(torch.linalg.vector_norm(new)) > 0.0
        if is_new and permit > 0.5 and (self.held is None or float(new @ self.held) < SAME_COS):
            self.held, self.age, self.streak, self.cleared_by = new, 0, 0, ""
            self.found_age = None
        elif self.held is not None:
            self.age += 1
            if self.found_age is None:
                self.streak = self.streak + 1 if found > FOUND_THRESHOLD else 0
                if self.streak >= FOUND_STREAK:
                    self.found_age = 0
            else:
                self.found_age += 1
            if self.found_age is not None and self.found_age >= FOUND_HOLD_TICKS:
                self.held, self.cleared_by = None, "found"
            elif self.age >= HOLD_TICKS:
                self.held, self.cleared_by = None, "timeout"
        if self.held is not None:
            out["gaze"] = self.held
            out["gate3"] = torch.tensor(1.0, dtype=DTYPE)
        return out
