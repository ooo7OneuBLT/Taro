"""1回で覚える記憶帳 〔DA〕（記憶）。

段階2・工程1：素通し（出ていく矢印すべてに書式のゼロを渡す）。

盤面の世界（5段目「前のことを話す」、2026-10-19）：手書きの下駄の「思い出の帳面」。あとで本物の記憶帳（海馬の1回で
覚える記憶）に替える。1分に1回（rhythm.WORTH_PHASE の回。目の入口と意味の辞書の知らせがそろう回）：
* eyein→episodic：[(目の前の物の荷札, 空), (部屋にいる人の荷札, 地図), …]（物が無ければ荷札ゼロ）
* semantic→episodic：目の前の物の名前の語（知らなければゼロ）
目の前に物が置かれている間を1つの思い出にし、その物・名前の語・その間に部屋にいた人を書く（新しい順に DIARY_N 個、
DIARY_MIN 分まで）。名前を知らない物の思い出は話せない。
* episodic→worth：話す行動（worth.TELL_TAGS, 地図の [0, j]＝1：j 番目の人に話せる）。tell_new＝その人がいなかった間の物
  （今は目の前に無い）、tell_old＝その人もいた時の物（今は目の前に無い）
* episodic→wordplan：話せる語（荷札, [0, j]＝j がいなかった時の物、[1, j]＝j もいた時の物）。新しい思い出から順
* wordplan→episodic：話した語（荷札, [0, j]＝1）。話した相手は、その思い出を知っている人として書き足す
生まれつき（帳面のつけ方は手書き）。何を誰に話すと良いかは worth が覚える。
"""

import torch

from ..formats import DTYPE, GRID
from ..rhythm import TICKS_PER_MIN, WORTH_PHASE, phase
from .base import Box

DIARY_N = 6          # 覚えておく思い出の数
DIARY_MIN = 180      # 思い出を話せる長さ（分）
SAME_THING = 0.9     # 物の荷札がこれより似ていれば同じ物
SAME_PERSON = 0.3    # 人の荷札がこれより似ていれば同じ人（worth と同じ）


def _has(people, tag) -> bool:
    return any(float(p @ tag) > SAME_PERSON for p in people)


class Episodic(Box):
    BOX_ID = "episodic"

    def __init__(self) -> None:
        super().__init__()
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.diary = []          # 新しい順：{"obj", "word", "seen": [人の荷札], "t", "open"}
        self.out_worth, self.out_words = [], []

    def step(self, inbox):
        out = self.passthrough()
        seen = inbox.get("eyein")
        if not isinstance(seen, list) or not seen:
            self.n += 1
            return out
        if not hasattr(self, "diary"):
            self.reset()
        minute = self.n // TICKS_PER_MIN
        for word, m in (inbox.get("wordplan") or []) if isinstance(inbox.get("wordplan"), list) else []:
            j = int(m[0].argmax())
            people = [t for t, _ in seen[1:]]
            for e in self.diary:
                if e["word"] is not None and float(e["word"] @ word) > 0.9 and j < len(people):
                    e["seen"].append(people[j])      # 話した相手も知っている
        if phase(self.n) == WORTH_PHASE:
            obj = seen[0][0]
            people = [t for t, _ in seen[1:]]
            name = inbox.get("semantic")
            here = float(torch.linalg.vector_norm(obj)) > 0.0
            cur = self.diary[0] if self.diary and self.diary[0]["open"] else None
            if cur is not None and (not here or float(cur["obj"] @ obj) < SAME_THING):
                cur["open"], cur = False, None
            if here:
                if cur is None:
                    cur = {"obj": obj.clone(), "word": None, "seen": [], "t": minute, "open": True}
                    self.diary.insert(0, cur)
                    del self.diary[DIARY_N:]
                cur["seen"] += [p for p in people if not _has(cur["seen"], p)]
                cur["t"] = minute
                if isinstance(name, torch.Tensor) and float(torch.linalg.vector_norm(name)) > 0.0:
                    cur["word"] = name.clone()
            from .worth import TELL_TAGS
            new = torch.zeros(GRID, GRID, dtype=DTYPE)
            old = torch.zeros(GRID, GRID, dtype=DTYPE)
            self.out_words = []
            for e in self.diary:
                if e["open"] or e["word"] is None or minute - e["t"] > DIARY_MIN:
                    continue
                m = torch.zeros(GRID, GRID, dtype=DTYPE)
                for j, p in enumerate(people[:GRID]):
                    row = 1 if _has(e["seen"], p) else 0
                    m[row, j] = 1.0
                    (old if row else new)[0, j] = 1.0
                self.out_words.append((e["word"], m))
            self.out_worth = [(TELL_TAGS[k], m) for k, m in (("tell_new", new), ("tell_old", old)) if float(m.sum()) > 0]
        self.n += 1
        out["worth"] = list(self.out_worth)
        out["wordplan"] = list(self.out_words)
        return out
