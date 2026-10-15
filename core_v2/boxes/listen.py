"""音を語に切る・聞き分ける係（2段目 習慣）。

段階2・工程3の仮部品（設計書 付録E）：耳の入口から来た文字列に、知っている物の名前
（KNOWN_WORDS）が入っていたら、その語の荷札（語ごとに決まった乱数、tagging.word_tag）を
意味の辞書（semantic）へ渡す。入っていなければゼロ。
「ボールはどこ？」は耳の入口で「ぼおるはどこ？」になって届く（長音の展開）。
listen→goal は段階2では使わない（ゼロ）。段階3で学習に替える対象。
"""

from ..tagging import word_tag
from .base import Box

KNOWN_WORDS = ("ぼおる", "こっぷ")


class Listen(Box):
    BOX_ID = "listen"

    def step(self, inbox):
        out = self.passthrough()
        text = inbox.get("earin") or ""
        hits = [(text.find(w), w) for w in KNOWN_WORDS if w in text]
        if hits:
            out["semantic"] = word_tag(min(hits)[1])   # 2つ入っていたら先に言われた方
        return out
