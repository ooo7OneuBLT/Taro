"""learn_names の数え方が、意味の辞書の覚える回（覚え方Bは名残りが消えた TRACE_TICKS 回あと）に合っているか。
1cfe361 で覚える回が聞いて約5秒後に移ったのに、learn_names は「覚えた回の3回前まで」に聞いた語を探していたので、
体ありで 11回覚えて 11回とも取り違え・回数0になった（c42feda、PC）。体なしで同じ回し方をして確かめる。"""

import torch

from core_v2.boxes.semantic import Semantic
from core_v2.formats import DTYPE, TAG_DIM, make_tag
from core_v2.tagging import word_tag
from core_v2.tools.learn_names import OBJ_OF_WORD, learn_lag, word_of_learn

GRID = 32


def _lst(tag):
    z = torch.zeros(GRID, GRID, dtype=DTYPE)
    m = z.clone()
    m[0, 0] = 1.0
    return [(tag, m), (tag, z), (tag, z)]


def _run(sem):
    g = torch.Generator().manual_seed(3)
    tags = {o: make_tag(torch.randn(TAG_DIM, generator=g)) for o in ("ball", "cup")}
    plan = []
    for _ in range(3):
        for obj, w in (("ball", "ぼおる"), ("cup", "こっぷ")):
            plan += [(obj, "")] * 30 + [(obj, w)] + [(obj, "")] * 70
    pipe = [None, None]            # earin→listen→semantic で2回遅れる
    heard_hist, got = [], []
    for tick, (obj, w) in enumerate(plan):
        pipe.append(word_tag(w) if w else torch.zeros(TAG_DIM, dtype=DTYPE))
        sem.step({"listen": pipe.pop(0), "objfile": _lst(tags[obj])})
        if w:
            heard_hist.append((tick, w, obj))
        if sem.last_event == "learn":
            got.append(word_of_learn(heard_hist, tick, learn_lag(sem)))
    return got


def test_every_learn_is_attributed_to_the_word_heard_mode_b():
    got = _run(Semantic(learning=True, learn_mode="B"))
    assert len(got) == 6
    assert all(w in OBJ_OF_WORD and OBJ_OF_WORD[w] == obj for w, obj in got), got


def test_every_learn_is_attributed_to_the_word_heard_mode_a():
    got = _run(Semantic(learning=True, learn_mode="A"))
    assert len(got) == 6
    assert all(w in OBJ_OF_WORD and OBJ_OF_WORD[w] == obj for w, obj in got), got
