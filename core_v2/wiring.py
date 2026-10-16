"""配線：どの係からどの係へ、どの書式で渡すか。

* 矢印の本体は構造図から自動生成した wiring_table.py（117本）
* ここで手で足すのは ADDED_EDGES（図の書き漏れの修正。`added=True` の印が付く）
* 書式は、「ボールはどこ？」の通り道の矢印だけ PATH_FORMATS で決める。
  それ以外は仮に荷札（素通しなので中身はゼロ）。係に中身を入れるときにここを直す

配線を変えると wiring_hash() が変わり、古い保存ファイルは読めなくなる（save.py）。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .formats import GRID, MAX_OBJECTS, TAG_DIM, Fmt
from .wiring_table import BOXES as _TABLE_BOXES
from .wiring_table import EDGES as _TABLE_EDGES

# 図に無いが足した係: (id, 名前, 段)。段階4（2026-10-16、ユーザー承認）：物の名簿（頭頂葉）
ADDED_BOXES = [
    ("objfile", "物の名簿（頭頂葉）", "入口（感覚）"),
]

# 図に無いが足した矢印: (出す係, 受ける係, 理由)
ADDED_EDGES = [
    ("listen", "semantic",
     "聞いた語から物の見た目を引くため。人間でも聞いた語は側頭葉で意味と結び付く（図の書き漏れ、2026-10-15決定）"),
    ("eyein", "objfile", "段階4：見つけた物を名簿に書く（物ごとの荷札と地図、目の動きの地図）"),
    ("prio", "objfile", "段階4：脱馴化で、行の位置の目立ちの「ふだん」と比べる"),
    ("orienteye", "objfile", "段階4：撃った指令の写し。撃った先の物に注意を移す"),
    ("objfile", "orienteye", "段階4：行ごとの「もう見た」の山と、今見ている行（慣れ）"),
    ("objfile", "semantic", "段階4：名前を結び付ける行の荷札（見た目の平均）"),
]

# 通り道の矢印の書式（設計書 技術付録D）。(書式, MOTOR のときの筋の数)
PATH_FORMATS: dict[tuple[str, str], tuple[Fmt, int | None]] = {
    ("earin", "listen"): (Fmt.TEXT, None),          # 入口の例外：normalize_kana 済みの文字列
    ("listen", "semantic"): (Fmt.TAG, None),        # 語の荷札
    ("listen", "goal"): (Fmt.TAG, None),            # 段階2では使わない（ゼロ）
    ("semantic", "goal"): (Fmt.TAG, None),          # 見た目の荷札
    ("goal", "gaze"): (Fmt.TAG, None),              # 探している物の荷札
    ("goal", "gate3"): (Fmt.AMOUNT, None),          # 持っている強さ
    ("gate3", "goal"): (Fmt.AMOUNT, None),          # 入れてよい
    ("gate3", "orienteye"): (Fmt.AMOUNT, None),     # 反射を抑える強さ
    ("eyein", "gaze"): (Fmt.TAG_AND_MAP, None),     # 物ごとの荷札と場所
    ("eyein", "prio"): (Fmt.MAP, None),             # 目立ちの地図
    ("eyein", "semantic"): (Fmt.TAG_AND_MAP, None), # 段階3：名前を聞いた瞬間に見ている物（覚えるとき）
    ("eyein", "orienteye"): (Fmt.TAG_AND_MAP, None), # 段階3：「もう見た」の印を物に付けて動かす
    ("gaze", "prio"): (Fmt.MAP, None),              # 探し物らしさの地図
    ("gaze", "goal"): (Fmt.AMOUNT, None),           # 見つけた度合い
    ("prio", "orienteye"): (Fmt.MAP, None),         # 合わせた地図
    ("orienteye", "eyeout"): (Fmt.MOTOR, 2),        # 目を向ける方向 (h, v)、[-1,1]、右・上が正
    ("eyein", "objfile"): (Fmt.TAG_AND_MAP, None),  # 段階4
    ("prio", "objfile"): (Fmt.MAP, None),
    ("orienteye", "objfile"): (Fmt.MOTOR, 2),       # 撃った指令の写し
    ("objfile", "orienteye"): (Fmt.TAG_AND_MAP, None),
    ("objfile", "semantic"): (Fmt.TAG, None),
}


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    fmt: Fmt
    size: int | None = None  # MOTOR のときだけ
    added: bool = False      # 図に無く手で足した矢印

    @property
    def key(self) -> tuple[str, str]:
        return (self.src, self.dst)


@dataclass(frozen=True)
class BoxInfo:
    id: str
    label: str
    group: str


BOXES: list[BoxInfo] = [BoxInfo(*b) for b in list(_TABLE_BOXES) + ADDED_BOXES]
BOX_IDS: list[str] = [b.id for b in BOXES]


def _build_edges() -> list[Edge]:
    pairs = [(s, d, False) for s, d in _TABLE_EDGES] + [(s, d, True) for s, d, _ in ADDED_EDGES]
    edges = []
    for s, d, added in pairs:
        fmt, size = PATH_FORMATS.get((s, d), (Fmt.TAG, None))
        edges.append(Edge(s, d, fmt, size, added))
    return edges


EDGES: list[Edge] = _build_edges()


def validate() -> None:
    """配線表そのものの矛盾を調べる。読み込み時に毎回走る。"""
    ids = set(BOX_IDS)
    if len(ids) != len(BOX_IDS):
        raise ValueError("係の id が重複している")
    keys = [e.key for e in EDGES]
    if len(set(keys)) != len(keys):
        raise ValueError("同じ向きの矢印が2本ある")
    for e in EDGES:
        if e.src not in ids or e.dst not in ids:
            raise ValueError(f"知らない係への矢印: {e.src} -> {e.dst}")
        if e.src == e.dst:
            raise ValueError(f"自分への矢印: {e.src}")
    for key in PATH_FORMATS:
        if key not in set(keys):
            raise ValueError(f"PATH_FORMATS に図に無い矢印がある: {key}")


def incoming(box_id: str) -> list[Edge]:
    return [e for e in EDGES if e.dst == box_id]


def outgoing(box_id: str) -> list[Edge]:
    return [e for e in EDGES if e.src == box_id]


def wiring_hash() -> str:
    """配線と書式の指紋。保存ファイルに書き、読み込み時に照合する。"""
    payload = {
        "tag_dim": TAG_DIM,
        "grid": GRID,
        "max_objects": MAX_OBJECTS,
        "boxes": sorted(BOX_IDS),
        "edges": sorted([e.src, e.dst, e.fmt.value, e.size, e.added] for e in EDGES),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


validate()
