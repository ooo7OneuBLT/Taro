"""耳（Hearing） — 親の発話文字列を、脳が扱えるトークン列に変える薄い入口。

【出典】Vocabulary クラスは B/src/taro/brain/cortex.py:18-42 からの移植
（2026-08-18）。原本はB側に残すが、今後使うのはこちら（F2「耳の移植」）。
学習則・分節ロジックは一切変えていない（動的にかな1文字単位でIDを割り振るだけ）。

【設計】F2仕様書（F/docs/仕様_F2_耳の移植.md）より：
  - トークンはかな1文字単位の動的語彙（固定音素集合ではない）
  - hear(text) は Vocabulary.encode の薄いラッパーで、環境から受け取った
    親の発話文字列をトークンID列に変える「耳」の役割だけを持つ
  - BOS/EOSの付与や連合（Lexicon）への橋渡しはこのファイルの責務ではない
    （呼び出し側が taro_core/src/brain/lexicon.py と組み合わせて使う）
"""


class Vocabulary:
    """見た文字から動的に語彙を構築する。

    B/src/taro/brain/cortex.py:18-42 からの移植（2026-08-18）。中身は無変更。
    """

    def __init__(self):
        self.char2idx = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2}
        self.idx2char = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>"}
        self.size = 3

    def encode(self, text):
        indices = []
        for ch in text:
            if ch not in self.char2idx:
                self.char2idx[ch] = self.size
                self.idx2char[self.size] = ch
                self.size += 1
            indices.append(self.char2idx[ch])
        return indices

    def decode(self, indices):
        chars = []
        for idx in indices:
            ch = self.idx2char.get(idx, "?")
            if ch not in ("<PAD>", "<BOS>", "<EOS>"):
                chars.append(ch)
        return "".join(chars)


class Hearing:
    """耳。環境から親の発話文字列を受け、トークン列にして脳へ渡す入口。

    Vocabulary をラップするだけで、分節（Lexicon.segment）や連合（Lexicon.observe）
    には踏み込まない（F2仕様書の役割分担どおり：耳＝トークン化、連合器＝学習）。
    """

    def __init__(self):
        self.vocab = Vocabulary()

    def hear(self, text):
        """発話文字列text を受け取り、トークンID列を返す。"""
        return self.vocab.encode(text)
