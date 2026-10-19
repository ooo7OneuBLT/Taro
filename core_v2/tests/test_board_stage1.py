"""新1段目「泣いて頼む」（盤面の世界）のテスト。数秒で終わる人数・日数で回す。"""

from core_v2.board.mind import Mind
from core_v2.board.world import Board
from core_v2.tools.board_stage1 import grow, summarize


def test_world_feeds_only_when_hungry():
    b = Board(rule="cry", seed=0, fullness=0.9)
    for _ in range(20):
        obs = b.step("cry", 1.0)
        assert not obs["fed"]          # 満ちている間はママはミルクをあげない


def test_mind_starts_even():
    m = Mind(seed=0)
    assert m.value("cry") == m.value("reach")


def test_learns_signal_mom_answers():
    for rule in ("cry", "reach"):
        runs = [grow(rule, s, days=3) for s in range(6)]
        s = summarize(runs, rule)
        assert s["last"] > 0.7, (rule, s["last"])


def test_frozen_stays_near_half():
    runs = [grow("cry", s, days=3, frozen=True) for s in range(6)]
    assert summarize(runs, "cry")["last"] < 0.65
