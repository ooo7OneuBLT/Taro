"""段階9：場面・試しの頭ごとに、親が太郎を机の方へ向けて座らせる（目を最初の向き (0, EL0) に置く）。

PC の段3（本物の物・本物の目、300場面）で、80場面から先、名づけが1回も増えなかった。部屋が替わっても目は前の
向きのまま次の場面が始まり、目は可動域の端（左右45度・上下33度）に寄ったままだった。物が視野（片側30度）の
外にあると目立つ山が無く、目を動かすきっかけが無いので、目は端から戻らない（親は太郎が物を見ないと名前を言わない）。

回す前の予想（2026-10-18）：
* 目を端（−45, 33）へ置いてから場面を始めても、場面の最初の回の目は (0, EL0)
* 目を端へ置いてから場面を4つ回すと、どの場面でも太郎は物を見る（親の「見た」が場面ごとに増える）
* 体ありの試し（noun_body）も、前の試しの目の向きに関係なく (0, EL0) から始まる
"""

import os

import pytest

os.environ.setdefault("TARO_EYE_LOG", "off")

from core_v2.world_eye import noun_scene as ns  # noqa: E402


def _record_first_tick(t):
    seen = []
    orig = t.tick

    def tick():
        o = t.env._orienting
        seen.append((round(o.pan, 3), round(o.tilt, 3)))
        return orig()
    t.tick = tick
    return seen


@pytest.fixture
def grower(tmp_path):
    from core_v2.tools import grow_names
    g = grow_names.Grower(0, "made", "table", None, log_root=str(tmp_path / "g"), start="birth")
    yield g
    g.close()


def test_each_scene_starts_facing_the_table_and_taro_looks(grower):
    g = grower
    seen = _record_first_tick(g)
    per_scene = int(round((ns.SCENE_S + ns.GAP_S) / ns.TICK_S))
    looks = []
    for s in range(4):
        if g.env is not None:
            g.env.set_eye(-45.0, 33.0, cause="test_edge")      # 前の場面が端で終わったことにする
        before = g.parent.stats["looks"]
        g.scene()
        looks.append(g.parent.stats["looks"] - before)
        assert seen[s * per_scene] == (0.0, ns.EL0), seen[s * per_scene]
    assert all(n >= 1 for n in looks), looks


def test_noun_body_trials_start_facing_the_table(tmp_path):
    import random
    from core_v2.tools import noun_body
    from core_v2.world_eye import instances as ins
    t = noun_body.Tester(0, "made", "table", None, log_root=str(tmp_path / "t"), start="birth")
    try:
        seen = _record_first_tick(t)
        i = ins.WORDS.index("わんわん")
        t.use_room(ins.ROOM_OF["わんわん"]).set_eye(-13.0, -7.0, cause="test_prev")
        t.look_trial("わんわん", [(i, 0), (ins.WORDS.index("ぶうぶう"), 0)], "B2", random.Random(0))
        assert seen[0] == (0.0, ns.EL0)
    finally:
        t.close()
