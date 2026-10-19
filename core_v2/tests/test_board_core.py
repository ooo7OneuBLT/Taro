"""盤面の世界を本体の脳（core_v2）で回すテスト（下書き core_v2/board/mind.py の清書）。"""

import torch

from core_v2.board.core_run import grow_core, new_brain
from core_v2.rhythm import TICKS_PER_MIN


def _minute(brain, fullness, people, fed=False):
    ext = {"gutin": {"fullness": fullness, "fed": fed}, "eyein": {"people": people}}
    for _ in range(TICKS_PER_MIN):
        to_env = brain.tick(ext)
    return to_env


def test_full_baby_does_nothing():
    brain = new_brain(0)
    for _ in range(30):
        to_env = _minute(brain, 1.0, [("ママ", False)])
        assert not isinstance(to_env.get("voiceout"), dict)
        assert "bodyout" not in to_env


def test_hungry_baby_signals_to_someone_in_room():
    brain = new_brain(0)
    acts = 0
    for _ in range(30):
        to_env = _minute(brain, 0.0, [("パパ", False)])
        out = to_env.get("voiceout") if isinstance(to_env.get("voiceout"), dict) else to_env.get("bodyout")
        if out:
            acts += 1
            assert out["to"] == 0      # 部屋にいるのはパパだけ（目に見えている0番目）
    assert acts >= 20                  # 空腹の不快1なら毎分なにかする


def test_worth_learns_cry_to_mom():
    r = grow_core("cry", 0, 2)
    worth = r["brain"].boxes["worth"]
    from core_v2.boxes.worth import ACT_TAGS
    from core_v2.tagging import word_tag
    mama = word_tag("person:ママ")
    assert worth.value(ACT_TAGS["cry"], mama) > worth.value(ACT_TAGS["reach"], mama)   # 泣く > 手を伸ばす
    assert worth.value(ACT_TAGS["cry"], mama) > 0.1                                   # 生まれた時より上


def test_frozen_worth_does_not_change():
    r = grow_core("cry", 0, 1, frozen=True)
    worth = r["brain"].boxes["worth"]
    assert torch.all(worth.W == 0) and torch.all(worth.u == 0)
