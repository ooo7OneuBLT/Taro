"""意味の辞書（3段目 計画）。

段階2・工程3の仮部品（設計書 付録E）：{語の荷札 → 物の見た目の荷札} の表を引く。
表は手で1件ずつ登録する（`core_v2/tools/register_appearance.py` が、場面で目の入口が
その物に出した見た目を記録して `G/models/semantic_table.pt` に書く）。
来た語の荷札が表のどの語とも十分に似ていなければ（cos < MATCH_COS）ゼロ。
出す荷札には雑音を足す（tagging.noisy、リスク1）。
段階3で学習に替える対象（逸脱：辞書の中身を手で登録した）。
"""

import torch

from ..formats import make_tag
from ..tagging import noisy, word_tag
from .base import Box

MATCH_COS = 0.9


class Semantic(Box):
    BOX_ID = "semantic"

    def __init__(self) -> None:
        super().__init__()
        self.table: dict[str, torch.Tensor] = {}   # {語: 見た目の荷札}

    def register(self, word: str, appearance_tag) -> None:
        self.table[word] = make_tag(appearance_tag)

    def load_table(self, path) -> dict:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        for w, tag in blob["table"].items():
            self.register(w, tag)
        return blob.get("meta", {})

    def step(self, inbox):
        out = self.passthrough()
        q = inbox.get("listen")
        if q is None or float(torch.linalg.vector_norm(q)) == 0.0 or not self.table:
            return out
        best, best_cos = None, MATCH_COS
        for w, tag in self.table.items():
            c = float(word_tag(w) @ q)
            if c >= best_cos:
                best, best_cos = w, c
        if best is not None:
            out["goal"] = noisy(self.table[best])
        return out
