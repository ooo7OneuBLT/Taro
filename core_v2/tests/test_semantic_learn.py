"""段階3〜4：意味の辞書の学習（semantic）の確かめ。環境なしで回る。
段階4から、覚える物は物の名簿（objfile）が決めて荷札で送ってくる（名簿の側は test_objfile.py）。
段階6から、名簿は行ごとの (荷札, 見ていた割合の山) の列＋最後に名前を結び付ける行を送る。"""

import random

import torch

from core_v2.boxes.semantic import RECALL_MIN, Semantic
from core_v2.boxes.objfile import bump_at
from core_v2.formats import GRID, TAG_DIM, make_tag
from core_v2.tagging import noisy, word_tag

BALL, CUP = word_tag("ぼおる"), word_tag("こっぷ")
ZMAP = torch.zeros(GRID, GRID)
NONE = [(torch.zeros(TAG_DIM), ZMAP)]


def roster(*rows, named=None):
    """名簿の列。rows は (荷札, 見ていた割合)。named を省くと最初の行を名前を結び付ける行にする。"""
    lst = [(t, s * bump_at(10.0, 10.0 + 4 * i)) for i, (t, s) in enumerate(rows)]
    nt = named if named is not None else (rows[0][0] if rows else torch.zeros(TAG_DIM))
    return lst + [(nt, ZMAP)]


def name(box, word_q, tag):
    """親が名前を言った1回（語は1回だけ届く）と、その次の黙った1回。tag は名簿から届く荷札（ずっと見ていた物）。"""
    if not isinstance(tag, list):
        tag = roster((tag, 1.0))
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
    tag = roster((noisy(appearance(1)), 1.0))
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


def test_cross_situational_B_learns_ambiguous_pairs():
    """段階6：2つの物が見えていて2つの語を言う（どちらがどちらかは示さない）。Bは各組5回で組を当てる。
    A（見ていた方だけ）も、見る物が場面ごとに偶然で変わるなら、正しい組の方が多く積み上がる。"""
    torch.manual_seed(0)
    random.seed(0)
    objs = [appearance(i) for i in range(6)]
    words = [word_tag(w) for w in ("ぼさ", "がっさ", "まぬ", "これっと", "かき", "れぐり")]
    boxes = {"A": Semantic(learning=True, learn_mode="A"), "B": Semantic(learning=True, learn_mode="B")}
    for _ in range(5):
        order = list(range(6))
        random.shuffle(order)
        for k in range(0, 6, 2):            # 1場面に2組。5周×3場面×2組＝各組5回
            i, j = order[k], order[k + 1]
            look = random.choice([i, j])     # 太郎はどちらか一方を長く見ている
            rows = [(noisy(objs[i]), 0.7 if look == i else 0.3), (noisy(objs[j]), 0.7 if look == j else 0.3)]
            for w in random.sample([i, j], 2):
                for b in boxes.values():
                    b.step({"listen": words[w], "objfile": roster(*rows, named=objs[look])})
                    b.step({"listen": torch.zeros(TAG_DIM), "objfile": roster(*rows, named=objs[look])})
    hits = {}
    for k, b in boxes.items():
        hits[k] = sum(int(max(range(6), key=lambda o: float(b.W @ words[w] @ objs[o])) == w) for w in range(6))
    assert hits["B"] == 6
