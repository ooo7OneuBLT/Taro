"""段階3〜4：意味の辞書の学習（semantic）の確かめ。環境なしで回る。
段階4から、覚える物は物の名簿（objfile）が決めて荷札で送ってくる（名簿の側は test_objfile.py）。"""

import torch

from core_v2.boxes.semantic import RECALL_MIN, Semantic
from core_v2.formats import TAG_DIM, make_tag
from core_v2.tagging import noisy, word_tag

BALL, CUP = word_tag("ぼおる"), word_tag("こっぷ")
NONE = torch.zeros(TAG_DIM)


def name(box, word_q, tag):
    """親が名前を言った1回（語は1回だけ届く）と、その次の黙った1回。tag は名簿から届く荷札。"""
    out = box.step({"listen": word_q, "objfile": tag})
    box.step({"listen": torch.zeros(TAG_DIM), "objfile": tag})
    return out["goal"]


def appearance(seed):
    g = torch.Generator().manual_seed(seed)
    return make_tag(torch.randn(TAG_DIM, generator=g))


def test_empty_says_nothing():
    box = Semantic(learning=True)
    out = box.step({"listen": BALL, "objfile": NONE})
    assert float(torch.linalg.vector_norm(out["goal"])) == 0.0


def test_three_namings_are_enough_two_are_not():
    torch.manual_seed(0)
    ball = appearance(1)
    box = Semantic(learning=True)
    for _ in range(2):
        name(box, BALL, noisy(ball))
    assert box.recall(BALL) is None           # 2回ではまだ出さない
    name(box, BALL, noisy(ball))
    r = box.recall(BALL)
    assert r is not None and float(r @ ball) > 0.95
    assert int(box.n_learn) == 3


def test_two_words_are_not_mixed_up():
    torch.manual_seed(0)
    ball, cup = appearance(1), appearance(2)
    box = Semantic(learning=True)
    for _ in range(5):
        for g, w in ((ball, BALL), (cup, CUP)):
            name(box, w, noisy(g))
    assert float(box.recall(BALL) @ ball) > 0.95
    assert float(box.recall(CUP) @ cup) > 0.95
    assert float(box.recall(CUP) @ ball) < 0.5


def test_nothing_from_the_roster_is_not_learned():
    box = Semantic(learning=True)
    for _ in range(5):
        name(box, BALL, NONE)
    assert int(box.n_learn) == 0 and box.last_event == ""


def test_one_utterance_is_learned_once():
    box = Semantic(learning=True)
    tag = noisy(appearance(1))
    for _ in range(4):                         # 同じ語が続けて届いても1回
        box.step({"listen": BALL, "objfile": tag})
    assert int(box.n_learn) == 1


def test_recalled_tag_goes_to_goal():
    torch.manual_seed(0)
    ball = appearance(1)
    box = Semantic(learning=True)
    for _ in range(3):
        name(box, BALL, noisy(ball))
    box.learning = False
    out = box.step({"listen": BALL, "objfile": NONE})
    assert float(out["goal"] @ ball) > 0.85 and box.last_recall_norm >= RECALL_MIN


def test_not_learning_when_off_and_W_survives_reset(tmp_path):
    box = Semantic(learning=False)
    name(box, BALL, noisy(appearance(1)))
    assert int(box.n_learn) == 0
    box.learning = True
    for _ in range(3):
        name(box, BALL, noisy(appearance(1)))
    box.reset()
    assert box.recall(BALL) is not None        # 試行の切り替わりで忘れない
    p = tmp_path / "w.pt"
    box.save_learned(p, {"n": 3})
    other = Semantic()
    assert other.load_learned(p) == {"n": 3}
    assert torch.equal(other.W, box.W)
