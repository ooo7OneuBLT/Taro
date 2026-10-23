"""口の表（声道）と、2つの声の似かたの物差し（段階7、2026-10-17）。

前の太郎の `taro_core/src/body/vocal_tract.py`（口の表・月齢の段階・連動）と
`taro_core/src/brain/drives/imitation_reward.py` の `_sub_cost`・`_weighted_edit_distance`（似かた）を移した。
中身（表の字・段階・連動の対応）は1つも変えていない。

* 声は本物の音ではなく**ひらがな1文字**（ユーザー決定 2026-10-17。逸脱リスト その55）。
  口の4つの数（どこで＝調音点7・どうやって＝調音法7・声帯2・母音6）を決めると、表を引いて1文字が出る
* 月齢で出せる音が増える（6か月＝唇と鼻音・破裂音、9か月＝舌、12か月＝無声と軟口蓋・声門）。
  11か月までは調音点を選ぶと調音法が決まる（連動）。時刻は前の太郎 B の設定と同じ
  （`B/config/config.yaml` vocal_maturation）［Tier2・カレンダー決め打ち（逸脱リスト C3）］
* 似かた：2つのかな列の、口の形で重みを付けた編集距離を長さで割って 1 から引く（0〜1）。
  前の太郎では「まねのほうび」の名前でほうびに使っていた。ここでは**ほうびに使わない**。
  声の係が「親の語」と「自分の声」の似かたを測るだけ（生まれつきの物差し）

生まれつき（手で書いた表）。学習は無い。声の出口（voiceout）が口の表を、声の係（voice）が似かたを使う。
**注意**：「っ」の口の形 (2,2,0,0) は表では「た」なので、口からは「っ」は出ない（「ー」と同じく、
口の形ではなく時間の現象）。「ゃ」「ゅ」「ょ」も表に無い（にゃんにゃん・でんしゃは言えない）。
"""

from __future__ import annotations

import torch

from .formats import DTYPE

PLACES = ["なし", "両唇", "歯茎", "歯茎硬口蓋", "硬口蓋", "軟口蓋", "声門"]
MANNERS = ["なし", "鼻音", "破裂音", "摩擦音", "破擦音", "弾き音", "半母音"]
VOICINGS = ["無声", "有声"]
VOWELS = ["あ", "い", "う", "え", "お", "（母音なし）"]   # 5＝「ん」専用
SIZES = (len(PLACES), len(MANNERS), len(VOICINGS), len(VOWELS))   # (7, 7, 2, 6)
N_MOTOR = sum(SIZES)                                               # 22：声の係→声の出口の MOTOR の長さ
_OFFSETS = (0, SIZES[0], SIZES[0] + SIZES[1], SIZES[0] + SIZES[1] + SIZES[2])

# (調音点, 調音法, 声帯, 母音) → ひらがな（前の太郎の _ARTICULATION_TABLE そのまま）
TABLE = {
    (0, 0, 1, 0): "あ", (0, 0, 1, 1): "い", (0, 0, 1, 2): "う", (0, 0, 1, 3): "え", (0, 0, 1, 4): "お",
    (5, 2, 0, 0): "か", (5, 2, 0, 1): "き", (5, 2, 0, 2): "く", (5, 2, 0, 3): "け", (5, 2, 0, 4): "こ",
    (5, 2, 1, 0): "が", (5, 2, 1, 1): "ぎ", (5, 2, 1, 2): "ぐ", (5, 2, 1, 3): "げ", (5, 2, 1, 4): "ご",
    (2, 3, 0, 0): "さ", (3, 3, 0, 1): "し", (2, 3, 0, 2): "す", (2, 3, 0, 3): "せ", (2, 3, 0, 4): "そ",
    (2, 4, 1, 0): "ざ", (3, 4, 1, 1): "じ", (2, 4, 1, 2): "ず", (2, 4, 1, 3): "ぜ", (2, 4, 1, 4): "ぞ",
    (2, 2, 0, 0): "た", (3, 4, 0, 1): "ち", (2, 4, 0, 2): "つ", (2, 2, 0, 3): "て", (2, 2, 0, 4): "と",
    (2, 2, 1, 0): "だ", (2, 2, 1, 1): "ぢ", (2, 2, 1, 2): "づ", (2, 2, 1, 3): "で", (2, 2, 1, 4): "ど",
    (2, 1, 1, 0): "な", (4, 1, 1, 1): "に", (2, 1, 1, 2): "ぬ", (2, 1, 1, 3): "ね", (2, 1, 1, 4): "の",
    (6, 3, 0, 0): "は", (4, 3, 0, 1): "ひ", (1, 3, 0, 2): "ふ", (6, 3, 0, 3): "へ", (6, 3, 0, 4): "ほ",
    (1, 2, 1, 0): "ば", (1, 2, 1, 1): "び", (1, 2, 1, 2): "ぶ", (1, 2, 1, 3): "べ", (1, 2, 1, 4): "ぼ",
    (1, 2, 0, 0): "ぱ", (1, 2, 0, 1): "ぴ", (1, 2, 0, 2): "ぷ", (1, 2, 0, 3): "ぺ", (1, 2, 0, 4): "ぽ",
    (1, 1, 1, 0): "ま", (1, 1, 1, 1): "み", (1, 1, 1, 2): "む", (1, 1, 1, 3): "め", (1, 1, 1, 4): "も",
    (4, 6, 1, 0): "や", (4, 6, 1, 2): "ゆ", (4, 6, 1, 4): "よ",
    (2, 5, 1, 0): "ら", (2, 5, 1, 1): "り", (2, 5, 1, 2): "る", (2, 5, 1, 3): "れ", (2, 5, 1, 4): "ろ",
    (1, 6, 1, 0): "わ", (1, 6, 1, 4): "を",
    (1, 1, 1, 5): "ん",
}
# ひらがな → 口の形。似かたを測る物差しにだけ使う（口の計画には使わない＝大人の正解表で埋めない）
CHAR_TO_PARAMS = {c: p for p, c in TABLE.items()}
CHAR_TO_PARAMS["っ"] = (2, 2, 0, 0)   # 促音＝歯茎破裂（「た」と同じ形。口からは「た」が出る）
CHAR_TO_PARAMS["ー"] = (0, 0, 1, 0)   # 長音（耳の入口が母音に展開するので通常は来ない）

# 月齢の段階（前の太郎と同じ）。段階ごとに使える値
STAGE_MONTHS = (6.0, 9.0, 12.0)      # 段階1・2・3 になる月齢（B：180日・270日・365日）［Tier2・逸脱 C3］
DECOUPLE_MONTH = 11.0                # 調音点と調音法を別々に選べるようになる月齢（B：330日）［Tier2・逸脱 C3］
ALLOWED = {   # 段階: (調音点, 調音法, 声帯, 母音)
    0: ([0], [0], [1], [0, 1, 2, 3, 4]),
    1: ([0, 1], [0, 1, 2], [1], [0, 1, 2, 3, 4, 5]),
    2: ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5, 6], [1], [0, 1, 2, 3, 4, 5]),
    3: (list(range(7)), list(range(7)), [0, 1], [0, 1, 2, 3, 4, 5]),
}
COUPLED_PLACE_TO_MANNER = {0: 0, 1: 1, 2: 2, 3: 4, 4: 6, 5: 2, 6: 3}   # 連動：両唇→鼻音、歯茎→破裂音…
MONTH_S = 30.4 * 24 * 3600.0         # 1か月の秒数（365日÷12）


def stage_of(month: float) -> int:
    return sum(1 for m in STAGE_MONTHS if month >= m)


def body_shape(params, month: float) -> tuple[int, int, int, int]:
    """口の指令を、その月齢の体で実際にできる形にする（前の太郎の clamp_to_stage と連動）。"""
    place, manner, voicing, vowel = params
    allowed = ALLOWED[stage_of(month)]
    if month < DECOUPLE_MONTH:
        manner = COUPLED_PLACE_TO_MANNER[place]
    out = []
    for v, ok in zip((place, manner, voicing, vowel), allowed):
        out.append(v if v in ok else ok[0])
    return tuple(out)


def speak(params, month: float) -> str:
    """口の指令 → 出た1文字。表に無い組は、4つの数のうち違う数がいちばん少ない字（前の太郎の _nearest）。"""
    key = body_shape(params, month)
    if key in TABLE:
        return TABLE[key]
    best, best_d = "あ", 99
    for k, c in TABLE.items():
        d = sum(1 for a, b in zip(key, k) if a != b)
        if d < best_d:
            best, best_d = c, d
    return best


# ---- MOTOR(22) との変換 ----
def to_motor(params) -> torch.Tensor:
    m = torch.zeros(N_MOTOR, dtype=DTYPE)
    for off, v in zip(_OFFSETS, params):
        m[off + v] = 1.0
    return m


def from_motor(m: torch.Tensor):
    """4つの区切りごとに一番大きいものを選ぶ。全部0（黙る）なら None。"""
    if float(m.abs().sum()) == 0.0:
        return None
    return tuple(int(m[off:off + n].argmax()) for off, n in zip(_OFFSETS, SIZES))


# ---- 似かた（前の太郎の imitation_reward.py の物差し。ほうびには使わない） ----
def param_distance(a: str, b: str) -> int:
    pa, pb = CHAR_TO_PARAMS.get(a), CHAR_TO_PARAMS.get(b)
    if pa is None or pb is None:
        return 4
    return sum(1 for x, y in zip(pa, pb) if x != y)


def _sub_cost(a: str, b: str) -> float:
    return 0.0 if a == b else param_distance(a, b) / 4.0


def weighted_edit_distance(a: str, b: str) -> float:
    n, m = len(a), len(b)
    dp = [float(j) for j in range(m + 1)]
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], float(i)
        for j in range(1, m + 1):
            cost = _sub_cost(a[i - 1], b[j - 1])
            prev, dp[j] = dp[j], min(dp[j] + 1.0, dp[j - 1] + 1.0, prev + cost)
    return dp[m]


def similarity(a: str, b: str) -> float:
    """2つのかな列の似かた（0〜1）。1−距離/長いほうの長さ（前の太郎の compute_imitation_reward と同じ式）。"""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return max(0.0, 1.0 - weighted_edit_distance(a, b) / max(len(a), len(b)))
