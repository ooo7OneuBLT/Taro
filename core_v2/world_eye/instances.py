"""段階9：20語×4個体の物の一覧（仕様書 `2026-10-18_段階9_仕様書_名詞を厚く.md` §2、ユーザー決定：全部自作の形）。

* 語（太郎が聞く形）・種類（objects.noun_geoms の形の作り方）・部屋・物の音の有無を NOUNS に1行ずつ書く
* 個体は「種類＋個体の番号」から決まった種で作る（いつ作っても同じ）：
    個体0：色A・大きさ1.0倍・無地        （覚える）
    個体1：色B・大きさ0.7倍・無地        （覚える）
    個体2：色A・大きさ1.3倍・模様つき（色Aと白の格子か点。種類ごとに交互）（覚える）
    個体3：色C・大きさ1.0倍・無地        （試験だけ。覚える3つに無い色）
  部品の長さは全部の個体で ±20%（一様乱数）。色は PALETTE の8色から種類ごとに A・B・C を重ならないように引く。
  色の数（8）は、20種類×覚える2色で各色がおよそ5種類に出て、試験 B（形と色がぶつかる2択）の「同じ色の別の種類」が
  いつも見つかるように選んだ切りのよい数［Tier3・モデル上の選択］
* 大きさの基準 BASE_R＝3cm（段階2〜8の物と同じ。30cm 先で 6〜10cm の見かけ）
"""

from __future__ import annotations

import hashlib

import numpy as np

from .objects import _f, noun_geoms

BASE_R = 0.03
N_INST = 4
TRAIN_INST = (0, 1, 2)
TEST_INST = 3
SCALES = (1.0, 0.7, 1.3, 1.0)
PART_JITTER = 0.2

# (語, 種類, 部屋, 物の音があるか)。部屋と語は仕様書 §2 の表のとおり
NOUNS = [
    ("こっぷ", "cup", "食事", False), ("すぷうん", "spoon", "食事", True), ("おさら", "plate", "食事", False),
    ("ぱん", "bread", "食事", False), ("ばなな", "banana", "食事", False), ("りんご", "apple", "食事", False),
    ("があがあ", "duck", "お風呂", True), ("さかな", "fish", "お風呂", False), ("たおる", "towel", "お風呂", False),
    ("ぼおる", "ball", "お風呂", False),
    ("わんわん", "dog", "遊び", True), ("にゃんにゃん", "cat", "遊び", True), ("ぶうぶう", "car", "遊び", True),
    ("でんしゃ", "train", "遊び", True), ("ばす", "bus", "遊び", False), ("がおお", "lion", "遊び", True),
    ("くつ", "shoe", "遊び", False), ("ぼうし", "hat", "遊び", False), ("つみき", "blocks", "遊び", False),
    ("くま", "bear", "遊び", False),
]
WORDS = [n[0] for n in NOUNS]
KIND_OF = {n[0]: n[1] for n in NOUNS}
ROOM_OF = {n[0]: n[2] for n in NOUNS}
SOUND_WORDS = [n[0] for n in NOUNS if n[3]]
# 試験 C の「音を持つ6語」（仕様書 §4・設計図）。すぷうんの「チン」（皿をたたく音）は §2 の表にあるので音は鳴らすが、
# 合格の線の6語には入っていない（別に数える）
C_WORDS = ["わんわん", "にゃんにゃん", "ぶうぶう", "がおお", "があがあ", "でんしゃ"]

PALETTE = {
    "赤": (0.85, 0.15, 0.15), "青": (0.15, 0.3, 0.85), "緑": (0.15, 0.62, 0.25), "黄": (0.95, 0.8, 0.1),
    "橙": (0.95, 0.5, 0.1), "紫": (0.55, 0.25, 0.75), "桃": (0.95, 0.55, 0.7), "茶": (0.5, 0.3, 0.15),
}
PART_NAMES = ("body", "handle", "rim", "width", "height", "length", "stem", "neck", "tail", "size", "band",
              "leg", "ear", "mane", "cabin", "window", "chimney", "brim", "crown", "block")


def _seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256("/".join(map(str, parts)).encode("utf-8")).digest()[:8], "little")


def kind_colors(word: str) -> tuple[str, str, str]:
    """種類ごとの色 A・B・C（重ならない）。"""
    g = np.random.default_rng(_seed("colors", word))
    names = list(PALETTE)
    a, b, c = g.choice(len(names), size=3, replace=False)
    return names[a], names[b], names[c]


def instance(word: str, k: int) -> dict:
    """個体 k の作り方：{word, kind, room, k, name, color, rgb, scale, r, pattern, L, train}。"""
    ca, cb, cc = kind_colors(word)
    color = (ca, cb, ca, cc)[k]
    g = np.random.default_rng(_seed("inst", word, k))
    L = {p: float(1.0 + PART_JITTER * (2.0 * g.random() - 1.0)) for p in PART_NAMES}
    pattern = None
    if k == 2:
        pattern = "格子" if WORDS.index(word) % 2 == 0 else "点"
    return {"word": word, "kind": KIND_OF[word], "room": ROOM_OF[word], "k": k,
            "name": f"n9_{KIND_OF[word]}_{k}", "color": color, "rgb": PALETTE[color], "scale": SCALES[k],
            "r": BASE_R * SCALES[k], "pattern": pattern, "L": L, "train": k in TRAIN_INST}


def all_instances() -> list[dict]:
    return [instance(w, k) for w in WORDS for k in range(N_INST)]


def assets_xml(inst: dict) -> list[str]:
    """模様つきの個体の材質（MuJoCo の組み込みの模様：格子＝checker、点＝flat に random の点）。無地なら空。"""
    if inst["pattern"] is None:
        return []
    nm = inst["name"]
    rgb = _f(inst["rgb"])
    if inst["pattern"] == "格子":
        tex = (f'<texture name="{nm}_tex" type="cube" builtin="checker" width="64" height="64" '
               f'rgb1="{rgb}" rgb2="0.95 0.95 0.92"/>')
        rep = "3 3"
    else:
        tex = (f'<texture name="{nm}_tex" type="cube" builtin="flat" mark="random" random="0.15" width="10" '
               f'height="10" rgb1="{rgb}" rgb2="{rgb}" markrgb="0.95 0.95 0.92"/>')
        rep = "1 1"
    return [tex, f'<material name="{nm}_mat" texture="{nm}_tex" texuniform="true" texrepeat="{rep}"/>']


def look_attr(inst: dict) -> str:
    if inst["pattern"] is not None:
        return f'material="{inst["name"]}_mat"'
    return f'rgba="{_f(list(inst["rgb"]) + [1.0])}"'


def body_xml(inst: dict, name: str | None = None, pos=(0.0, 0.0, -5.0)) -> str:
    """個体の体の XML（自由な関節つき。objects.body_xml と同じ形）。"""
    name = name or inst["name"]
    inner = "\n      ".join(noun_geoms(inst["kind"], inst["r"], look_attr(inst), inst["L"]))
    return (f'    <body name="{name}" pos="{_f(pos)}">\n      <freejoint name="{name}_free"/>\n'
            f'      {inner}\n    </body>')
