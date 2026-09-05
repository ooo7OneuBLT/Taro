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


# --- 長音「ー」の展開（2026-08-27・F2-10） -----------------------------------
# 【なぜ要るか】太郎の耳と口が扱う単位はモーラ（かな1文字）で、口は
#   「調音点・調音法・声帯・母音」の4つの数字で音を作る。長音「ー」は
#   *前の母音を伸ばす* という時間の現象で、この4つでは表せない
#   （実際 vocal_tract.py の _CHAR_TO_PARAMS["ー"] は「あ」と同じ座標に
#   なっており、太郎の口からは原理的に「ー」が出せない。実測：F2-9Cの
#   産出テストで「ぶーぶー」の一致は 0/104、出力は「ぶあぶあ」等）。
#
# 【なぜ耳で直すのが正しいか】「ー」は *書き言葉の記号* であって音ではない。
#   人間の耳に届くのは伸びた母音の音で、日本語の音韻論でも長音は同じ母音の
#   連続（/buubuu/＝4モーラ）として数える。よって太郎の中身を変えるのでは
#   なく、耳に届く時点で音のとおりに直す＝人間模倣からの逸脱の *解消*。
#   親の発話テキストの表記（ログ・Viewer表示）は「ぶーぶー」のまま変えない。
#
# 【規則】「ー」＝直前のかなの母音を1つ足す。母音を持たないかな（ん・っ）の
#   後ろに来た場合は伸ばしようがないので「ー」を落とす（日本語では起きない）。
_VOWEL_OF = {}
for _row, _v in (
    (u"あかさたなはまやらわがざだばぱぁゃゎ", u"あ"),
    (u"いきしちにひみりぎじぢびぴぃ",         u"い"),
    (u"うくすつぬふむゆるぐずづぶぷぅゅ",     u"う"),
    (u"えけせてねへめれげぜでべぺぇ",         u"え"),
    (u"おこそとのほもよろごぞどぼぽぉょを",   u"お"),
):
    for _c in _row:
        _VOWEL_OF[_c] = _v


def expand_long_vowel(text):
    """「ー」を直前の母音に展開する（"ぶーぶー" → "ぶうぶう"）。"""
    out = []
    for ch in text:
        if ch in (u"ー", u"ー", u"ｰ"):
            v = _VOWEL_OF.get(out[-1]) if out else None
            if v is not None:
                out.append(v)
        else:
            out.append(ch)
    return u"".join(out)


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

    def add_special(self, token):
        """複数文字の特殊トークン（"<PARENT>" 等）を1つのIDとして登録する
        （2026-08-30・文脈設計の決定3「話者の印」）。

        encode() は文字単位に分解するので特殊トークンには使えない。
        既に登録済みなら何もしないでそのIDを返す。
        """
        if token not in self.char2idx:
            self.char2idx[token] = self.size
            self.idx2char[self.size] = token
            self.size += 1
        return self.char2idx[token]

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
        """発話文字列text を受け取り、トークンID列を返す。

        【2026-08-27・F2-10】長音「ー」はここで直前の母音に展開する
        （"ぶーぶー" → "ぶうぶう"）。理由はこのファイル上部の
        expand_long_vowel のコメントを参照。
        """
        return self.vocab.encode(expand_long_vowel(text))
