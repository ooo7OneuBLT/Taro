"""6段目「親役で話す」の道具（core_v2/board/play.py）のテスト。"""

from core_v2.board.core_run import new_brain
from core_v2.board.play import Play


def _play(fullness):
    p = Play.new(new_brain(0))
    p.fullness = fullness
    return p


def test_feed_when_hungry_and_offer_when_full():
    p = _play(0.3)
    r = p.turn("ミルクをあげる", wait=1)
    assert "ミルクを飲ませた" in r["notes"] and p.fullness > 0.9
    p.turn("ミルクをあげる", wait=1)
    assert p.offered
    p.turn("やめる", wait=1)
    assert not p.offered


def test_words_and_objects():
    p = _play(0.9)
    r = p.turn("ボールを見せて「ボールだよ」と言う", wait=1)
    assert p.object == "ボール"
    assert any("ぼおる" in n for n in r["notes"])
    p.turn("片付ける", wait=1)
    assert p.object is None


def test_hungry_baby_acts_and_shows_why():
    p = _play(0.0)
    r = p.turn("待つ 30分")
    acted = [a for a in r["steps"] if a["action"] != "nothing"]
    assert acted and acted[0]["to"] == "ママ"
    why = acted[0]["why"]
    assert why["head"] == "food" and why["drives"]["food"] > 0 and why["options"]


def test_return_after_leaving_alone_is_a_reunion():
    p = _play(0.9)
    p.turn("出ていく", wait=3)
    r = p.turn("戻る", wait=1)
    assert r["steps"][0]["why"]["drives"]["social"] > 0.5
