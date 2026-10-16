"""段階3：意味の辞書の学習（semantic、仕様書 付録B）の確かめ。環境なしで回る。"""

import torch

from core_v2.boxes.eyein import bump
from core_v2.boxes.semantic import RECALL_MIN, Semantic
from core_v2.formats import TAG_DIM, make_tag
from core_v2.tagging import noisy, word_tag

BALL, CUP = word_tag("ぼおる"), word_tag("こっぷ")


def look(g, center: bool = True):
    """見た目の荷札 g の物が、真ん中（112,112画素）か端（30,30画素）にある、目の入口の出力。"""
    pos = (112.0, 112.0) if center else (30.0, 30.0)
    return [(noisy(g), bump(*pos))]


def name(box, word_q, objs):
    """親が名前を言った1回（語は1回だけ届く）と、その次の黙った1回。"""
    out = box.step({"listen": word_q, "eyein": objs})
    box.step({"listen": torch.zeros(TAG_DIM), "eyein": objs})
    return out["goal"]


def appearance(seed):
    g = torch.Generator().manual_seed(seed)
    return make_tag(torch.randn(TAG_DIM, generator=g))


def test_empty_says_nothing():
    box = Semantic(learning=True)
    out = box.step({"listen": BALL, "eyein": []})
    assert float(torch.linalg.vector_norm(out["goal"])) == 0.0


def test_three_namings_are_enough_two_are_not():
    torch.manual_seed(0)
    ball = appearance(1)
    box = Semantic(learning=True)
    for _ in range(2):
        name(box, BALL, look(ball))
    assert box.recall(BALL) is None           # 2回ではまだ出さない
    name(box, BALL, look(ball))
    r = box.recall(BALL)
    assert r is not None and float(r @ ball) > 0.95   # 覚えたときの荷札の雑音（σ=0.05）が平均で残る
    assert int(box.n_learn) == 3


def test_two_words_are_not_mixed_up():
    torch.manual_seed(0)
    ball, cup = appearance(1), appearance(2)
    box = Semantic(learning=True)
    for _ in range(5):
        name(box, BALL, look(ball))
        name(box, CUP, look(cup))
    assert float(box.recall(BALL) @ ball) > 0.95
    assert float(box.recall(CUP) @ cup) > 0.95
    assert float(box.recall(CUP) @ ball) < 0.5


def test_nothing_in_the_center_is_not_learned():
    box = Semantic(learning=True)
    for _ in range(5):
        name(box, BALL, look(appearance(1), center=False))
    assert int(box.n_learn) == 0 and box.last_event == ""


def test_one_utterance_is_learned_once():
    box = Semantic(learning=True)
    objs = look(appearance(1))
    for _ in range(4):                         # 同じ語が続けて届いても1回
        box.step({"listen": BALL, "eyein": objs})
    assert int(box.n_learn) == 1


def test_not_learning_when_off_and_W_survives_reset(tmp_path):
    box = Semantic(learning=False)
    name(box, BALL, look(appearance(1)))
    assert int(box.n_learn) == 0
    box.learning = True
    for _ in range(3):
        name(box, BALL, look(appearance(1)))
    box.reset()
    assert box.recall(BALL) is not None        # 試行の切り替わりで忘れない
    p = tmp_path / "w.pt"
    box.save_learned(p, {"n": 3})
    other = Semantic()
    assert other.load_learned(p) == {"n": 3}
    assert torch.equal(other.W, box.W)


def test_recalled_tag_goes_to_goal():
    torch.manual_seed(0)
    ball = appearance(1)
    box = Semantic(learning=True)
    for _ in range(3):
        name(box, BALL, look(ball))
    box.learning = False
    out = box.step({"listen": BALL, "eyein": []})
    assert float(out["goal"] @ ball) > 0.85   # 出すときにも雑音を足す and box.last_recall_norm >= RECALL_MIN


def test_learns_an_object_a_little_off_center():
    """跳びが足りず物が真ん中から約6度（2マス＝16画素）ずれていても、親の条件（10度以内）に合わせて覚える。
    5マス（40画素）ずれた物は覚えない（2026-10-16、真ん中の窓を半径3マスに広げた）。"""
    a = appearance(41)
    box = Semantic(learning=True)
    name(box, BALL, [(noisy(a), bump(112.0 + 16.0, 112.0 - 8.0))])
    assert int(box.n_learn) == 1
    far = Semantic(learning=True)
    name(far, BALL, [(noisy(a), bump(112.0 + 40.0, 112.0))])
    assert int(far.n_learn) == 0
