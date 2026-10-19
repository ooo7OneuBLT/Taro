"""声の出口（出口（筋・内臓））。

**段階7（2026-10-17、ユーザー承認の仕様書 `2026-10-17_段階7_仕様書_言える.md` §2-1）**：前の太郎の口の表を移した
（`core_v2/mouth.py`）。声の係（voice）から届いた口の指令（MOTOR 22＝調音点7・調音法7・声帯2・母音6）の
4つの区切りごとに一番大きいものを選び、その月齢の体でできる形に直して（段階・連動）、表を引いて1文字を出す。
指令が全部ゼロなら黙る（""）。

* 環境からの入力（inbox[ENV]）：月齢（か月、float）。口の成長は体の側の話なので、環境（試験場・体）が渡す。
  無ければ 0（母音だけ）
* 環境への指令（outbox[ENV]）：出たかな1文字か ""。環境はそれを次の回に耳の入口へ戻す（自分の声を聞く）
* 泣き（cry→voiceout、盤面の世界 2026-10-19）：泣く指令（向ける相手の荷札と、強さで塗った地図）が届いた回だけ、
  環境への指令を {"kana": 出たかな, "cry": 強さ, "to": 向けた相手の荷札} にする（届かない回は今までどおりかなの文字列）

生まれつき（手で書いた表と月齢の段階）。学習は無い。
"""

from .. import mouth
from .base import ENV, Box


class Voiceout(Box):
    BOX_ID = "voiceout"

    def reset(self) -> None:
        self.last_char = ""

    def step(self, inbox):
        out = self.passthrough()
        params = mouth.from_motor(inbox["voice"]) if "voice" in inbox else None
        month = float(inbox.get(ENV) or 0.0)
        self.last_char = mouth.speak(params, month) if params is not None else ""
        out[ENV] = self.last_char
        if inbox.get("cry"):
            tag, m = inbox["cry"][0]
            out[ENV] = {"kana": self.last_char, "cry": float(m.max()), "to": tag}
        return out
