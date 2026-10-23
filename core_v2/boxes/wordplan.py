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
"""

from __future__ import annotations

import torch

from ..formats import make_tag
from .base import Box

SAME_COS = 0.99   # これより似ていれば同じ語として一覧に足さない（荷札は語ごとに決まっているので同じ語なら1）


class Wordplan(Box):
    BOX_ID = "wordplan"

    def __init__(self) -> None:
        super().__init__()
        self.words: list[torch.Tensor] = []   # 聞いたことのある知っている語の荷札
        self.reset()

    def reset(self) -> None:
        self.held = None

    def get_extra_state(self):
        return {"words": [w.tolist() for w in self.words]}

    def set_extra_state(self, state) -> None:
        self.words = [make_tag(w) for w in state.get("words", [])]

    def step(self, inbox):
        out = self.passthrough()
        q = inbox.get("listen")
        if q is not None and bool(q.any()):
            if not self.words or float((self._stack() @ q).max()) <= SAME_COS:
                self.words.append(q.clone())
                self._stacked = None
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
