"""耳の入口（入口（感覚））。

段階2・工程2：親の発話の文字列を、旧太郎の耳と同じ正規化（カタカナ→ひらがな→長音展開、
`senses/hearing.py` の normalize_kana）をかけて聞き分ける係（listen）へ渡す。
書式の例外（TEXT）。トークン列にはしない（語の荷札を作るのは listen の仕事）。

環境からの入力（inbox[ENV]）：その回の発話の文字列（無ければ "" か None）。
発話は環境の1歩にだけ出るので、run_v2 が脳の1回ぶん（10歩）の info を全部見て集めて渡す。
それ以外の出ていく矢印は素通し（ゼロ）。
"""

import os
import sys

from .base import ENV, Box

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


def normalize_kana(text: str) -> str:
    core_src = os.path.join(_REPO, "taro_core", "src")
    if core_src not in sys.path:
        sys.path.insert(0, core_src)
    from senses.hearing import normalize_kana as _nk
    return _nk(text)


class Earin(Box):
    BOX_ID = "earin"

    def step(self, inbox):
        out = self.passthrough()
        text = inbox.get(ENV)
        if text:
            out["listen"] = normalize_kana(str(text))
        return out
