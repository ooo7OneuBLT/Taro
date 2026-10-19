"""言う語を組み立てる係（3段目 目的を持ち続ける）。

**段階7（2026-10-17、ユーザー承認の仕様書 `2026-10-17_段階7_仕様書_言える.md` §2-2）**：
意味の辞書から、見ている物から逆に引いた語の荷札 Wᵀa が届く。これを、聞き分ける係から届いたことのある
「知っている語」の荷札のうち、いちばん似ているものに合わせ（1回ごとの雑音や、ほかの語の混ざりを落とす）、
届いている間ずっと声の係へ送り続ける（言いたい語を持ち続ける）。届かなくなったらゼロ。
知っている語をまだ1つも聞いていなければ何も送らない。

言うか黙るか（つながり M が十分強いか）は、M を持っている声の係が決める。この係は M を読まない
（係どうしは矢印以外で中身を見ない、base.py）。

受け取るもの：semantic（TAG：見ている物から引いた語）、listen（TAG：聞いた知っている語）。
出すもの：voice（TAG：言いたい語、言わないときはゼロ）、listen（使わない、ゼロ）。
生まれつき（しくみ）。経験で変わるのは「聞いたことのある語の荷札」の一覧だけ（reset で消えない）。

盤面の世界（2026-10-19、3段目「言葉で頼む」。use_board() で入る）：
* 聞いた語ごとに聞いた回数を数え、SAYABLE_AFTER 回以上聞いた語を「言える語」として、多く聞いた順に
  MAX_SAYABLE 個まで wordplan→worth へ送る（何をするか決める係の選択肢になる）。人も同じ回数で語を言い始める
  （3回で覚える。Horst & Samuelson 2008 など、名前の段で使ったのと同じ線）
* 「選ぶ」係から言うと決めた語（want→wordplan）が届いたら、そのまま声の出口へ渡す（wordplan→voiceout）。
  声の係を通して口の形にするのは、口の下駄を外す時
""" 

from __future__ import annotations

import torch

from ..formats import DTYPE, GRID, make_tag
from .base import Box

SAME_COS = 0.99   # これより似ていれば同じ語として一覧に足さない（荷札は語ごとに決まっているので同じ語なら1）
SAYABLE_AFTER = 3     # 盤面：何回聞いたら言える語になるか
MAX_SAYABLE = 6       # 盤面：選択肢にする言える語の数（泣く・手と合わせて MAX_OBJECTS 以内）


class Wordplan(Box):
    BOX_ID = "wordplan"

    def __init__(self) -> None:
        super().__init__()
        self.words: list[torch.Tensor] = []   # 聞いたことのある知っている語の荷札
        self.counts: list[int] = []           # 盤面：語ごとの聞いた回数（words と同じ並び）
        self.board = False
        self.reset()

    def use_board(self) -> None:
        self.board = True

    def reset(self) -> None:
        self.held = None

    def get_extra_state(self):
        return {"words": [w.tolist() for w in self.words], "counts": list(self.counts)}

    def set_extra_state(self, state) -> None:
        self.words = [make_tag(w) for w in state.get("words", [])]
        self.counts = list(state.get("counts", [0] * len(self.words)))

    def step(self, inbox):
        out = self.passthrough()
        q = inbox.get("listen")
        if q is not None and bool(q.any()):
            if not self.words or float((self._stack() @ q).max()) <= SAME_COS:
                self.words.append(q.clone())
                self.counts.append(0)
                self._stacked = None
            self.counts[int((self._stack() @ q).argmax())] += 1
        if self.board:
            order = sorted(range(len(self.words)), key=lambda i: -self.counts[i])
            out["worth"] = [(self.words[i], torch.zeros(GRID, GRID, dtype=DTYPE))
                            for i in order[:MAX_SAYABLE] if self.counts[i] >= SAYABLE_AFTER]
            if inbox.get("want"):
                out["voiceout"] = list(inbox["want"])
        r = inbox.get("semantic")
        self.held = None
        if r is not None and bool(r.any()) and self.words:
            self.held = self.words[int((self._stack() @ r).argmax())]
            out["voice"] = self.held
        return out

    def _stack(self) -> torch.Tensor:
        if self.__dict__.get("_stacked") is None or len(self._stacked) != len(self.words):
            self.__dict__["_stacked"] = torch.stack(self.words)
        return self._stacked
