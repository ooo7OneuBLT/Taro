"""親の話しかけ（段階5、仕様書 §3）。名づけの前に、太郎に耳だけで聞かせる文を作る。

赤ちゃんは名前を教わる前から、たくさんの話しかけを聞いている。語の切れ目の手がかり（続きやすさ）は
その中にある（Saffran・Aslin・Newport 1996）。また、親の話しかけの約1割は語1つだけの発話
（Brent・Siskind 2001。記憶ベース・要確認）で、これも切れ目の足がかりになる（前の太郎の分節第2案と同じ）。

文は耳の入口の正規化のあと（ひらがな・長音は母音に展開）の形で書く。
「でんしゃ」はここに入れない（知らない語のテストに使う）。名づけ・合格テストの場面の文（「〇〇だね」
「〇〇はどこ？」）と同じ型の文も含むが、それは実際の親も言うため。
"""

from __future__ import annotations

import random

WORDS = ["ぼおる", "こっぷ", "わんわん", "にゃんにゃん", "ぶうぶう", "くつ", "ばなな", "りんご", "おさら", "ぼうし"]
UNHEARD = ["でんしゃ", "とけい"]           # 聞かせない語（知らない語のテスト用）

FRAMES = [
    "{w}だね", "{w}をみて", "{w}はどこ？", "{w}があるよ", "{w}とってね", "あ、{w}", "{w}だよ",
    "おおきい{w}", "ちいさい{w}", "{w}かわいいね", "ほら{w}", "{w}どうぞ", "{w}すきだね",
]
FILLERS = ["いいこだね", "おいしいね", "みてごらん", "すごいね", "ねんねしようね", "おはよう", "はいどうぞ"]
P_ALONE = 0.1      # 語1つだけの発話の割合［Tier2・記憶ベース］
P_FILLER = 0.15    # 物の名前の入らない文の割合［仮置き］


def make_corpus(n: int, seed: int = 0, words=None) -> list[str]:
    """n 文の話しかけを作る（決まった種で毎回同じ）。"""
    rng = random.Random(seed)
    words = list(words or WORDS)
    out = []
    for _ in range(n):
        r = rng.random()
        if r < P_ALONE:
            out.append(rng.choice(words))
        elif r < P_ALONE + P_FILLER:
            out.append(rng.choice(FILLERS))
        else:
            out.append(rng.choice(FRAMES).format(w=rng.choice(words)))
    return out


def gold_words(sentence: str) -> list[str]:
    """答え合わせ用：文を正しい語の並びに切ったもの（太郎には渡さない）。"""
    for w in sorted(WORDS + UNHEARD, key=len, reverse=True):
        if sentence == w:
            return [w]
    for f in FRAMES:
        pre, post = f.split("{w}")
        if sentence.startswith(pre) and sentence.endswith(post):
            mid = sentence[len(pre):len(sentence) - len(post)] if post else sentence[len(pre):]
            if mid in WORDS + UNHEARD:
                return [x for x in (pre.rstrip("、"), mid, post) if x]
    return [sentence]
