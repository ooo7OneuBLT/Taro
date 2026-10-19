"""盤面の4段目（あれなに？）・5段目（断る・前のことを話す）の部品のテスト。"""

import torch

from core_v2.board.core_run import new_brain
from core_v2.board.world import Board, OFFER_MIN, stage2_adults
from core_v2.boxes.worth import BODY_TAGS, HEAD_ROW, TELL_TAGS, body
from core_v2.formats import DTYPE, GRID
from core_v2.rhythm import TICKS_PER_MIN
from core_v2.tagging import word_tag


def _full_board(**kw):
    b = Board(seed=0, adults=stage2_adults(), fullness=1.0, objects=["ボール"], offers=True, news=True,
              object_words={"ぼおる": "ボール"}, **kw)
    for a in b.adults:
        a.present, a.busy_left = True, 0
    return b


def test_offer_forces_milk_unless_refused():
    b = _full_board()
    while not b.offer:
        b.fullness = 1.0
        b.step("nothing", 0.0)
    for _ in range(OFFER_MIN):
        obs = b.step("nothing", 0.0)
    assert obs["overfed"] and b.fullness > 0.99


def test_refuse_word_stops_offer():
    b = _full_board()
    while not b.offer:
        b.fullness = 1.0
        b.step("nothing", 0.0)
    who = b.offer[0]
    b.adults[[a.name for a in b.adults].index(who)].notice = 1.0
    obs = b.step("say", 1.0, who, "いらない")
    assert obs["refused"] and not obs["overfed"] and b.offer is None


def test_news_makes_adult_smile_more_than_old():
    smiles = {True: 0, False: 0}
    for seed in range(200):
        b = _full_board()
        b.rng.seed(seed)
        b.adults[0].notice = 1.0
        b.seen_by = {"ボール": {"パパ"} if seed % 2 else {"ママ", "パパ"}}
        b.object = None
        obs = b.step("say", 1.0, "ママ", "ぼおる")
        smiles[seed % 2 == 1] += obs["smiled"] == "ママ"
    assert smiles[True] > 3 * smiles[False]


def test_body_list_roundtrip():
    from core_v2.boxes.gutin import body_list
    b = body(body_list(fed=1.0, aversion=0.5, overfed=0.0))
    assert b == {"fed": 1.0, "aversion": 0.5, "overfed": 0.0}
    assert body(torch.tensor(1.0)) == {"fed": 1.0}


def test_offered_milk_when_full_makes_aversion_drive():
    brain = new_brain(0)
    ext = {"gutin": {"fullness": 1.0, "fed": False, "offered": True}, "eyein": {"people": [("ママ", True)]}}
    for _ in range(TICKS_PER_MIN * 2):
        brain.tick(ext)
    out = brain.boxes["inner"].step({"gutin": brain.boxes["gutin"].step({"env": ext["gutin"]})["inner"]})
    assert float(out["want"]) == 1.0


def test_diary_offers_tell_new_to_person_who_was_away():
    brain = new_brain(0)
    ep = brain.boxes["episodic"]
    ep.reset()
    obj, papa, mama = word_tag("obj:ボール"), word_tag("person:パパ"), word_tag("person:ママ")
    name = word_tag("ぼおる")
    z = torch.zeros(GRID, GRID, dtype=DTYPE)
    for t in range(TICKS_PER_MIN * 3):          # ボールが目の前にある間、ママだけいた
        ep.step({"eyein": [(obj, z), (mama, z)], "semantic": name})
    for t in range(TICKS_PER_MIN * 2):          # ボールが片付いて、パパが戻ってきた
        ep.step({"eyein": [(torch.zeros_like(obj), z), (mama, z), (papa, z)], "semantic": torch.zeros_like(name)})
    acts = {k: m for t, m in ep.out_worth for k, x in TELL_TAGS.items() if float(x @ t) > 0.99}
    assert float(acts["tell_new"][0, 1]) == 1.0 and float(acts["tell_new"][0, 0]) == 0.0   # パパ（1番目）に新しい話
    assert float(acts["tell_old"][0, 0]) == 1.0                                             # ママは見ていた


def test_worth_heads_learn_separately():
    brain = new_brain(0)
    w = brain.boxes["worth"]
    a, p = word_tag("いらない"), word_tag("person:ママ")
    for _ in range(20):
        w.learn(a, p, 1.0, "stop")
    assert w.value(a, p, "stop") > 0.8 and abs(w.value(a, p, "food") - 0.1) < 1e-6
    assert HEAD_ROW["social"] == 3 and set(BODY_TAGS) == {"fed", "aversion", "overfed"}
