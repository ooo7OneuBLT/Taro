"""
声道シミュレータ — 太郎の「口」

【人間模倣・身体的制約】
人間の赤ちゃんは口・舌・喉・肺という身体を持ち、
4つのパラメータ（調音点・調音法・声帯振動・母音）の組み合わせで音を作る。

太郎には物理的な身体がないが、この構造をシミュレーションで再現する。
これにより「ま」と「ぱ」がパラメータ1つ分しか違わないという
身体的な近さが脳の中に自然に生まれる。

調音音声学に基づく日本語ひらがなの分類表を内蔵する。
"""

# --- 4つのパラメータの定義 ---

# 調音点（どこで音を作るか）
PLACES = ["なし", "両唇", "歯茎", "歯茎硬口蓋", "硬口蓋", "軟口蓋", "声門"]

# 調音法（どうやって音を作るか）
MANNERS = ["なし", "鼻音", "破裂音", "摩擦音", "破擦音", "弾き音", "半母音"]

# 声帯振動
VOICINGS = ["無声", "有声"]

# 母音（口の形）
VOWELS = ["あ", "い", "う", "え", "お"]

# パラメータ数
NUM_PLACE = len(PLACES)      # 7
NUM_MANNER = len(MANNERS)    # 7
NUM_VOICING = len(VOICINGS)  # 2
NUM_VOWEL = len(VOWELS)      # 5
TOTAL_PARAMS = NUM_PLACE + NUM_MANNER + NUM_VOICING + NUM_VOWEL  # 21

# --- ひらがな ↔ 調音パラメータの変換表 ---
# (調音点, 調音法, 声帯, 母音) → ひらがな
# 調音音声学の分類に基づく

_ARTICULATION_TABLE = {
    # 母音のみ（子音なし）
    (0, 0, 1, 0): "あ", (0, 0, 1, 1): "い", (0, 0, 1, 2): "う",
    (0, 0, 1, 3): "え", (0, 0, 1, 4): "お",

    # か行（軟口蓋・破裂音・無声）
    (5, 2, 0, 0): "か", (5, 2, 0, 1): "き", (5, 2, 0, 2): "く",
    (5, 2, 0, 3): "け", (5, 2, 0, 4): "こ",

    # が行（軟口蓋・破裂音・有声）
    (5, 2, 1, 0): "が", (5, 2, 1, 1): "ぎ", (5, 2, 1, 2): "ぐ",
    (5, 2, 1, 3): "げ", (5, 2, 1, 4): "ご",

    # さ行（歯茎・摩擦音・無声）※し=歯茎硬口蓋
    (2, 3, 0, 0): "さ", (3, 3, 0, 1): "し", (2, 3, 0, 2): "す",
    (2, 3, 0, 3): "せ", (2, 3, 0, 4): "そ",

    # ざ行（歯茎・破擦音/摩擦音・有声）※じ=歯茎硬口蓋
    (2, 4, 1, 0): "ざ", (3, 4, 1, 1): "じ", (2, 4, 1, 2): "ず",
    (2, 4, 1, 3): "ぜ", (2, 4, 1, 4): "ぞ",

    # た行（歯茎・破裂音・無声）※ち=歯茎硬口蓋破擦、つ=歯茎破擦
    (2, 2, 0, 0): "た", (3, 4, 0, 1): "ち", (2, 4, 0, 2): "つ",
    (2, 2, 0, 3): "て", (2, 2, 0, 4): "と",

    # だ行（歯茎・破裂音・有声）
    (2, 2, 1, 0): "だ", (2, 2, 1, 1): "ぢ", (2, 2, 1, 2): "づ",
    (2, 2, 1, 3): "で", (2, 2, 1, 4): "ど",

    # な行（歯茎・鼻音・有声）※に=硬口蓋
    # 「な」と「ん」を区別するため、「な」は歯茎(2)、「ん」は軟口蓋(5)で鼻音
    (2, 1, 1, 0): "な", (4, 1, 1, 1): "に", (2, 1, 1, 2): "ぬ",
    (2, 1, 1, 3): "ね", (2, 1, 1, 4): "の",

    # は行（声門・摩擦音・無声）※ひ=硬口蓋、ふ=両唇
    (6, 3, 0, 0): "は", (4, 3, 0, 1): "ひ", (1, 3, 0, 2): "ふ",
    (6, 3, 0, 3): "へ", (6, 3, 0, 4): "ほ",

    # ば行（両唇・破裂音・有声）
    (1, 2, 1, 0): "ば", (1, 2, 1, 1): "び", (1, 2, 1, 2): "ぶ",
    (1, 2, 1, 3): "べ", (1, 2, 1, 4): "ぼ",

    # ぱ行（両唇・破裂音・無声）
    (1, 2, 0, 0): "ぱ", (1, 2, 0, 1): "ぴ", (1, 2, 0, 2): "ぷ",
    (1, 2, 0, 3): "ぺ", (1, 2, 0, 4): "ぽ",

    # ま行（両唇・鼻音・有声）
    (1, 1, 1, 0): "ま", (1, 1, 1, 1): "み", (1, 1, 1, 2): "む",
    (1, 1, 1, 3): "め", (1, 1, 1, 4): "も",

    # や行（硬口蓋・半母音・有声）
    (4, 6, 1, 0): "や", (4, 6, 1, 2): "ゆ", (4, 6, 1, 4): "よ",

    # ら行（歯茎・弾き音・有声）
    (2, 5, 1, 0): "ら", (2, 5, 1, 1): "り", (2, 5, 1, 2): "る",
    (2, 5, 1, 3): "れ", (2, 5, 1, 4): "ろ",

    # わ行（両唇・半母音・有声）
    (1, 6, 1, 0): "わ", (1, 6, 1, 4): "を",

    # ん（軟口蓋・鼻音・有声・特殊：母音なしだが便宜上「あ」母音の位置）
    (5, 1, 1, 0): "ん",
}

# 逆引き：ひらがな → パラメータ
_CHAR_TO_PARAMS = {}
for params, char in _ARTICULATION_TABLE.items():
    _CHAR_TO_PARAMS[char] = params

# 特殊文字
_CHAR_TO_PARAMS["っ"] = (2, 2, 0, 0)  # 促音＝歯茎破裂（「た」に近い）
_CHAR_TO_PARAMS["ー"] = (0, 0, 1, 0)  # 長音＝母音の延長


# --- 成熟ステージごとに使える値の定義 ---
# Stage 0: 母音のみ（調音点=なし, 調音法=なし, 声帯=有声に固定）
# Stage 1: +両唇、+鼻音・破裂音（「ばばば」「まままま」）
# Stage 2: +歯茎・歯茎硬口蓋・硬口蓋（「だだだ」「ななな」）
# Stage 3: +声帯の無声（「ぱ」と「ば」の使い分け）。全パラメータ解放

STAGE_ALLOWED_PLACE = {
    0: [0],              # なし（母音のみ）
    1: [0, 1],           # +両唇
    2: [0, 1, 2, 3, 4],  # +歯茎・歯茎硬口蓋・硬口蓋
    3: list(range(NUM_PLACE)),  # 全解放
}
STAGE_ALLOWED_MANNER = {
    0: [0],              # なし（母音のみ）
    1: [0, 1, 2],        # +鼻音・破裂音
    2: [0, 1, 2, 3, 4, 5, 6],  # +摩擦音・破擦音・弾き音・半母音
    3: list(range(NUM_MANNER)),
}
STAGE_ALLOWED_VOICING = {
    0: [1],              # 有声のみ（固定）
    1: [1],              # まだ有声のみ
    2: [1],              # まだ有声のみ
    3: [0, 1],           # 無声も解放
}
# 母音は最初から全部使える（顎の開閉）
STAGE_ALLOWED_VOWEL = {s: list(range(NUM_VOWEL)) for s in range(4)}

# --- A2-6：調音パラメータの連動パターン ---
# 赤ちゃんの口は最初、調音点と調音法が「くっついて」一緒に動く。
# 調音点を選ぶと調音法が自動的に決まる（独立に選べない）。
# 発達に応じて独立制御が可能になる。
#
# coupled=True のとき、脳は調音点だけを選び、調音法は以下のマッピングで決まる：
#   なし(0) → なし(0)        ＝母音のみ
#   両唇(1) → 鼻音(1)       ＝「ま」系が出る（赤ちゃんの最初の子音）
#   歯茎(2) → 破裂音(2)     ＝「だ」系が出る
#   歯茎硬口蓋(3) → 破擦音(4)
#   硬口蓋(4) → 半母音(6)   ＝「や」系
#   軟口蓋(5) → 破裂音(2)   ＝「が」系
#   声門(6) → 摩擦音(3)     ＝「は」系

COUPLED_PLACE_TO_MANNER = {
    0: 0,  # なし→なし（母音のみ）
    1: 1,  # 両唇→鼻音（ま行）
    2: 2,  # 歯茎→破裂音（だ行）
    3: 4,  # 歯茎硬口蓋→破擦音
    4: 6,  # 硬口蓋→半母音（や行）
    5: 2,  # 軟口蓋→破裂音（が行）
    6: 3,  # 声門→摩擦音（は行）
}


class VocalTract:
    """
    太郎の口。4つのパラメータから文字を作る。

    赤ちゃんの口・舌・喉の物理的構造をテキスト世界で再現する。
    A2-3：身体の成熟に応じてパラメータが段階的に解放される。
    A2-6：調音点と調音法の連動→独立の発達を追加。
    """

    def __init__(self):
        self.params_to_char = dict(_ARTICULATION_TABLE)
        self.char_to_params = dict(_CHAR_TO_PARAMS)
        self.stage = 0
        self.coupled = True  # True=調音点と調音法が連動（初期状態）

    def speak(self, place, manner, voicing, vowel):
        """
        口のパラメータ → 文字。

        見つからない場合は最も近い組み合わせの文字を返す。
        （赤ちゃんが中途半端な口の形をしても何かしらの音は出る）
        """
        key = (place, manner, voicing, vowel)
        if key in self.params_to_char:
            return self.params_to_char[key]
        return self._nearest(place, manner, voicing, vowel)

    def hear(self, char):
        """
        文字 → 口のパラメータ。

        聞いた音を「この口の動かし方で出る音だ」と認識する。
        未知の文字はNoneを返す。
        """
        return self.char_to_params.get(char, None)

    def get_all_chars(self):
        """声道で発声可能な全文字を返す。"""
        return list(self.char_to_params.keys())

    def param_distance(self, char_a, char_b):
        """
        2つの文字の調音的な距離（0〜4）。
        4つのパラメータのうち何個が異なるかを数える。

        「ま」と「ぱ」→ 1（調音法だけ違う）
        「ま」と「き」→ 4（全部違う）
        """
        pa = self.char_to_params.get(char_a)
        pb = self.char_to_params.get(char_b)
        if pa is None or pb is None:
            return 4
        return sum(1 for a, b in zip(pa, pb) if a != b)

    def _nearest(self, place, manner, voicing, vowel):
        """最も近い組み合わせの文字を探す。"""
        best_char = "あ"
        best_dist = 999
        target = (place, manner, voicing, vowel)
        for key, char in self.params_to_char.items():
            dist = sum(1 for a, b in zip(target, key) if a != b)
            if dist < best_dist:
                best_dist = dist
                best_char = char
        return best_char

    def update_stage(self, sim_seconds, stage1_time, stage2_time, stage3_time,
                     decouple_time=None):
        """sim時間に応じて成熟ステージと連動/独立を更新する。"""
        if sim_seconds >= stage3_time:
            self.stage = 3
        elif sim_seconds >= stage2_time:
            self.stage = 2
        elif sim_seconds >= stage1_time:
            self.stage = 1
        else:
            self.stage = 0

        # A2-6：調音パラメータの独立制御
        if decouple_time is not None and sim_seconds >= decouple_time:
            self.coupled = False

    def get_allowed(self):
        """現在のステージで使えるパラメータ値を返す。"""
        return (
            STAGE_ALLOWED_PLACE[self.stage],
            STAGE_ALLOWED_MANNER[self.stage],
            STAGE_ALLOWED_VOICING[self.stage],
            STAGE_ALLOWED_VOWEL[self.stage],
        )

    def get_manner_for_place(self, place):
        """
        連動モード時：調音点から調音法を自動決定する。
        【人間模倣・身体的制約】赤ちゃんの口は最初、全部くっついて動く。
        """
        return COUPLED_PLACE_TO_MANNER.get(place, 0)

    def force_decouple(self):
        """アブレーション用：最初から独立制御にする。"""
        self.coupled = False

    def is_coupled(self):
        return self.coupled

    def clamp_to_stage(self, place, manner, voicing, vowel):
        """ロックされたパラメータを許可値に制限する（身体的制約）。"""
        allowed = self.get_allowed()
        if place not in allowed[0]:
            place = allowed[0][0]
        if manner not in allowed[1]:
            manner = allowed[1][0]
        if voicing not in allowed[2]:
            voicing = allowed[2][0]
        if vowel not in allowed[3]:
            vowel = allowed[3][0]
        return place, manner, voicing, vowel

    @staticmethod
    def num_params():
        """各パラメータの選択肢数をタプルで返す。"""
        return (NUM_PLACE, NUM_MANNER, NUM_VOICING, NUM_VOWEL)
