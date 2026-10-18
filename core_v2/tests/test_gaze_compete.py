"""段階9：目を向ける係の「まちがえそうな物と比べる」決まり（gaze.RULE="compete"）と、意味の辞書が一緒に思い出す
まちがえそうな物（semantic.recall_list）の確かめ。環境なしで回る。"""

import torch

from core_v2.boxes import gaze as gz
from core_v2.boxes.goal import Goal
from core_v2.boxes.objfile import bump_at
from core_v2.boxes.semantic import N_RIVALS, Semantic
from core_v2.formats import GRID, MAX_OBJECTS, TAG_DIM, make_tag
from core_v2.tagging import word_tag

from .test_semantic_learn import appearance, name

ZMAP = torch.zeros(GRID, GRID, dtype=torch.float32)


def _mix(a, b, w):
    return make_tag(w * a + (1 - w) * b)


def test_compete_takes_target_below_old_line_and_rejects_rival():
    target, rival = appearance(1), appearance(2)
    obj = _mix(target, appearance(3), 0.45)          # 探し物だが、線 0.7 には届かない見え
    assert float(obj @ target) < gz.MATCH_COS
    other = _mix(rival, target, 0.6)               # まちがえそうな物に近い別の物
    got = gz.match(target, [rival], [obj, other], "compete")
    assert got[0] is not None and got[1] is None
    old = gz.match(target, [rival], [obj, other], "gate")
    assert old[0] is None


def test_without_rivals_uses_old_line():
    target = appearance(1)
    obj = _mix(target, appearance(3), 0.45)
    assert gz.match(target, [], [obj], "compete") == [None]


def test_gaze_box_reads_list_and_plain_tag():
    target, rival = appearance(1), appearance(2)
    obj = _mix(target, appearance(3), 0.45)
    where = bump_at(13.5, 13.5)
    box = gz.Gaze()
    out = box.step({"goal": [(target, ZMAP), (rival, ZMAP)], "kind": [(obj, where)]})
    assert float(out["goal"]) > 0.4                 # 真ん中の物を探し物として見つけた
    out = box.step({"goal": target, "kind": [(obj, where)]})     # 荷札1つ（前の形）は線 0.7
    assert float(out["goal"]) < 0.01


def test_goal_holds_the_list_and_accepts_plain_tag():
    target, rival = appearance(1), appearance(2)
    g = Goal()
    out = g.step({"semantic": [(target, ZMAP), (rival, ZMAP)], "gate3": torch.tensor(1.0)})
    assert len(out["gaze"]) == 2 and torch.equal(out["gaze"][0][0], target)
    g = Goal()
    out = g.step({"semantic": target, "gate3": torch.tensor(1.0)})
    assert len(out["gaze"]) == 1


def test_recall_list_brings_rivals_but_not_the_word_itself():
    torch.manual_seed(0)
    words = ["ぼおる", "こっぷ", "くま", "ばなな", "くつ", "ばす", "さかな", "つみき", "ぱん"]
    box = Semantic(learning=True)
    looks = {w: appearance(10 + i) for i, w in enumerate(words)}
    for _ in range(3):
        for w in words:
            name(box, word_tag(w), looks[w])
    box.learning = False
    lst = box.recall_list(word_tag("ぼおる"))
    assert lst is not None and len(lst) <= MAX_OBJECTS
    assert len(lst) == 2 + N_RIVALS                 # 聞いた語・物ふつうの見た目・まちがえそうな語 N_RIVALS
    target = lst[0][0]
    assert float(target @ looks["ぼおる"]) > 0.85
    for t, m in lst[1:]:
        assert float(t @ looks["ぼおる"]) < 0.8     # 聞いた語そのものは入らない
        assert float(m.abs().sum()) == 0.0
    # 物ふつうの見た目は、見た物の平均に近い
    mean = make_tag(torch.stack(list(looks.values())).mean(0))
    assert float(lst[1][0] @ mean) > 0.8
    # 知らない語は何も出さない
    assert box.recall_list(word_tag("でんしゃ")) is None
    assert TAG_DIM == 64
