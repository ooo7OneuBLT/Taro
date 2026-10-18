"""段階8a：驚きの係（仮部品）と、名簿・意味の辞書・聞き分ける係の変更の確かめ。環境なしで回る。"""

import torch

from core_v2.boxes.eyein import ego_map
from core_v2.boxes.listen import Listen
from core_v2.boxes.objfile import ABSENT_TICKS, KEEP_S, SEATS, Objfile, bump_at
from core_v2.boxes.semantic import RECALL_MIN, SURPRISE_TAG, Semantic
from core_v2.boxes.surprise import Surprise, centre_weight
from core_v2.formats import TAG_DIM, Fmt, check, make_tag
from core_v2.tagging import state_tag, word_tag
from core_v2.tools import nai_bed as nb
from core_v2.tools.parent_speech import make_corpus

Z2 = torch.zeros(2)
C = 13.5


def tag(seed):
    g = torch.Generator().manual_seed(seed)
    return make_tag(torch.randn(TAG_DIM, generator=g))


def run(steps, sp=None, of=None):
    """steps: 1回ごとの [(荷札, 行, 列)]（見えている物）と目の動き（画素）と目立ちの地図を出すか。驚きの強さの列を返す。"""
    of = of or Objfile()
    sp = sp or Surprise()
    us = []
    for things, ego, with_prio in steps:
        eyein = [(t, bump_at(r, c, sigma=1.0)) for t, r, c in things] + [(torch.zeros(TAG_DIM), ego_map(*ego))]
        sal = torch.zeros(28, 28)
        for _, r, c in things:
            sal = torch.maximum(sal, bump_at(r, c))
        inbox = {"eyein": eyein, "orienteye": Z2}
        if with_prio:
            inbox["prio"] = sal
        out = of.step(inbox)
        check(Fmt.TAG_AND_MAP, out["surprise"], name="objfile→surprise")
        s = sp.step({"objfile": out["surprise"]})
        check(Fmt.AMOUNT, s["semantic"], name="surprise→semantic")
        us.append(float(s["semantic"]))
    return us


def show(t, n, r=C, c=C, prio=True):
    return [([(t, r, c)], (0.0, 0.0), prio)] * n


def blank(n, prio=True):
    return [([], (0.0, 0.0), prio)] * n


def test_fixated_vanish_surprises_until_the_row_is_gone():
    us = run(show(tag(1), 50) + blank(40))
    assert max(us[:50]) == 0.0
    after = us[50:]
    first = next(i for i, u in enumerate(after) if u > 0)
    assert first <= ABSENT_TICKS + 1                              # 名簿が「見えない」と決めたらすぐ
    assert all(u > 0 for u in after[first:int(KEEP_S / 0.1)])      # 行が消えるまで続く
    assert after[-1] == 0.0


def test_controls_do_not_surprise():
    t = tag(2)
    stay = show(t, 90)
    move = show(t, 50) + [([(t, C, C + k)], (0.0, 0.0), True) for k in range(1, 15)] + blank(30)
    look = show(t, 50) + [([], (-160.0, 0.0), True)] + blank(20) + [([], (160.0, 0.0), True)] + blank(30)
    for steps in (stay, move, look):
        assert max(run(steps)) == 0.0


def test_longer_look_stronger_surprise():
    short = max(run(show(tag(3), 10) + blank(10)))
    long = max(run(show(tag(3), 100) + blank(10)))
    assert 0.0 < short < long < 1.0


def test_peripheral_vanish_is_weak_and_needs_a_salience_drop():
    t = tag(4)
    # 中心で見たあと、目が動いて物が周辺（8マス右）に。そこで消える
    seq = show(t, 50) + [([(t, C, C + 8)], (64.0, 0.0), True)] + show(t, 10, c=C + 8)
    centre = max(run(show(t, 50) + blank(10)))
    with_drop = max(run(seq + blank(10)))
    no_prio = max(run([(a, b, False) for a, b, _ in seq] + blank(10, prio=False)))
    assert 0.0 < with_drop < centre
    assert no_prio == 0.0
    assert centre_weight(0.0) == 1.0 and 0.0 < centre_weight(20.0) < 0.3


def test_switches():
    t = tag(5)
    seq = show(t, 30) + blank(10)
    assert max(run(seq, Surprise(use_enc=False))) == 1.0
    assert max(run(seq, Surprise(enabled=False))) == 0.0
    per = show(t, 50) + [([(t, C, C + 8)], (64.0, 0.0), True)] + show(t, 10, c=C + 8) + blank(10)
    assert max(run(per, Surprise(use_centre=False))) > max(run(per))


def test_a_freed_seat_is_not_reused_at_once():
    of = Objfile()
    seq = show(tag(6), 5) + [([(tag(7), 5.0, 5.0)], (0.0, 0.0), True)] * 3
    run(seq, of=of)
    seats = {rw.seat for rw in of.rows}
    assert len(seats) == len(of.rows) and max(seats) < SEATS


def test_surprise_tag_is_innate_and_not_a_word():
    assert torch.allclose(SURPRISE_TAG, state_tag("予想外"))
    assert abs(float(SURPRISE_TAG @ word_tag("ない"))) < 0.5
    assert abs(float(SURPRISE_TAG @ word_tag("予想外"))) < 0.5


def test_semantic_learns_surprise_only_from_experience():
    sem = Semantic(learning=True)
    assert float(torch.linalg.vector_norm(sem.W.T @ SURPRISE_TAG)) == 0.0   # 生まれたときは何も結び付いていない
    a = tag(8)
    lst = [(a, bump_at(C, C)), (a, torch.zeros(28, 28)), (a, torch.zeros(28, 28))]
    q = word_tag("ない")
    for k in range(8):                      # 物が消えて驚いている間に「ない」を聞く、を8回（物は毎回ちがう）
        lst = [(tag(20 + k), bump_at(C, C)), (tag(20 + k), torch.zeros(28, 28)), (tag(20 + k), torch.zeros(28, 28))]
        for i in range(80):
            sem.step({"objfile": lst, "surprise": torch.tensor(1.0 if i > 20 else 0.0),
                      "listen": q if i == 30 else torch.zeros(TAG_DIM)})
    r = sem.recall_word([], 1.0)
    assert r is not None and float(r @ q) > 0.9
    assert float(torch.linalg.vector_norm(sem.W.T @ SURPRISE_TAG)) >= RECALL_MIN
    # 驚きが無ければ、今まで（段階7）と同じ引き方
    assert torch.allclose(sem._seen_vec(lst, 0.0), make_tag(lst[-2][0]))


def test_listen_sends_every_known_word_in_order():
    ear = Listen(learning=True)
    for s in make_corpus(1000, seed=1):
        ear.step({"earin": s})
    for s in ["ない", "ない", "ぼおるないね", "こっぷないね"]:
        ear.step({"earin": s})
    ear.learning = False
    ear.reset()                             # 前の発話の送り残しを空にする
    assert "ない" in ear.known()
    got = [ear.step({"earin": "ぼおるないね"})["semantic"], ear.step({"earin": ""})["semantic"],
           ear.step({"earin": ""})["semantic"]]
    assert torch.allclose(got[0], word_tag("ぼおる"))
    assert torch.allclose(got[1], word_tag("ない"))
    assert float(torch.linalg.vector_norm(got[2])) == 0.0


def test_bed_trials_pass_line_a_by_construction():
    b = nb._probe_copy(nb.NaiBed(0))
    t = nb.untaught_tags()["とけい"]
    assert nb.run_trial(b, t, "vanish", 3.0)["u_max"] > 0
    for k in ("stay", "move", "look"):
        assert nb.run_trial(b, t, k, 3.0)["u_max"] == 0.0
