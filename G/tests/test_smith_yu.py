"""段階6（スミスとユーの再現）の環境側のテスト：予定表を作る道具と、物を2つずつ出す仕組み。

  python -m pytest -q G/tests/test_smith_yu.py
"""
import copy
import os
import sys
from collections import Counter

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
for p in (REPO, os.path.join(REPO, "G", "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from smith_yu_schedule import OBJECTS, WORDS, make, pairs_world  # noqa: E402

SCENE = "段階6_スミスとユー_6物体_2026-10-17"


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_schedule_counts(seed):
    sy = make(seed)
    train = [s for s in sy["slides"] if s["phase"] == "train"]
    test = [s for s in sy["slides"] if s["phase"] == "test"]
    assert len(train) == 30 and len(test) == 12
    # 学ぶ画面：1画面4秒・物2つ・語2つ（出て0.5秒と1.0秒）、各物10回、各組（物どうし）2回
    seen = Counter()
    pairs = Counter()
    for s in train:
        assert s["t1"] - s["t0"] == pytest.approx(4.0)
        assert s["left"] != s["right"]
        assert [round(w["t"] - s["t0"], 3) for w in s["words"]] == [0.5, 1.0]
        assert {w["object"] for w in s["words"]} == {s["left"], s["right"]}
        seen.update([s["left"], s["right"]])
        pairs[frozenset((s["left"], s["right"]))] += 1
    assert set(seen.values()) == {10}
    assert len(pairs) == 15 and set(pairs.values()) == {2}
    assert sy["no_repeat"]
    for a, b in zip(train, train[1:]):
        assert not ({a["left"], a["right"]} & {b["left"], b["right"]})
    # 語と物の対応は1対1、語は正しい物の名前
    assert sorted(sy["word_of"].values()) == sorted(WORDS)
    for s in sy["slides"]:
        for w in s["words"]:
            assert sy["word_of"][w["object"]] == w["word"]
    # テスト：8秒、語4回、各物が正解2回・別の物2回、正解の左右は1語につき左右1回ずつ
    tg, ds, side = Counter(), Counter(), {}
    for s in test:
        assert s["t1"] - s["t0"] == pytest.approx(8.0)
        assert len(s["words"]) == 4 and {w["object"] for w in s["words"]} == {s["target"]}
        assert s["target"] != s["distractor"] and {s["left"], s["right"]} == {s["target"], s["distractor"]}
        tg[s["target"]] += 1
        ds[s["distractor"]] += 1
        side.setdefault(s["target"], []).append(s["left"] == s["target"])
    assert set(tg.values()) == {2} and set(ds.values()) == {2}
    assert all(sorted(v) == [False, True] for v in side.values())
    # 予定表の形：区間が重ならない・語は時刻順・語どうしは0.1秒（脳1回）以上あく
    ps = sy["pairs_schedule"]
    assert all(b["t0"] >= a["t1"] for a, b in zip(ps, ps[1:]))
    ts = [w["t"] for w in sy["word_schedule"]]
    assert ts == sorted(ts) and min(np.diff(ts)) >= 0.1
    assert sy["t_end"] == ps[-1]["t1"]


def test_schedule_seed_changes_and_repeats():
    a, b, a2 = make(0), make(1), make(0)
    assert a["pairs_schedule"] == a2["pairs_schedule"] and a["word_of"] == a2["word_of"]
    assert a["pairs_schedule"] != b["pairs_schedule"]


def test_pairs_world_matches_objects():
    pw = pairs_world()
    assert len(pw["objects"]) == len(OBJECTS) == 6 and pw["schedule"] == []
    kinds = {(o["shape"], tuple(o["rgba"])) for o in OBJECTS}
    assert len(kinds) == 6 and len({tuple(o["rgba"]) for o in OBJECTS}) == 6


def _build(schedule):
    from run.scene_tools import scene_io
    sc = scene_io.load(SCENE)
    sc = copy.deepcopy(sc)
    sc["world"]["pairs"] = dict(sc["world"]["pairs"], schedule=schedule)
    env, _hands = scene_io.build(sc, vision=False)
    return env


def test_env_places_pairs():
    """予定表どおりに左右へ置き、ほかは地下の遠方。区間の外は何も出さない。toy1・toy2 は退避。"""
    pytest.importorskip("mujoco")
    from run.trainer import close_env
    # 時計は env.data.time（リセットの落ち着かせのあと 0.5秒から始まる。語の予定表と同じ時計）
    sched = [{"t0": 0.5, "t1": 0.7, "left": 2, "right": 4},
             {"t0": 0.8, "t1": 1.0, "left": 5, "right": 0}]
    env = _build(sched)
    try:
        u = env.unwrapped
        act = np.zeros(env.action_space.shape[0], dtype=np.float32)
        env.reset(seed=0)
        rest_l, rest_r = u._rest_pos.copy(), u._rest_pos2.copy()

        def pos(k):
            return np.array(u.data.body(f"sy_obj{k}").xpos)

        def far(k):      # 出していない物＝左右どちらの置き場所からも1m以上離れている（遠方）
            return min(np.linalg.norm(pos(k) - rest_l), np.linalg.norm(pos(k) - rest_r)) > 1.0

        def step_until(t):
            while float(u.data.time) < t:
                env.step(act)
        step_until(0.6)
        assert u.pairs_showing() == (2, 4)
        assert np.allclose(pos(2), rest_l, atol=0.01) and np.allclose(pos(4), rest_r, atol=0.01)
        for k in (0, 1, 3, 5):
            assert far(k)
        assert np.linalg.norm(np.array(u.data.body("test_object1").xpos) - rest_l) > 1.0
        assert np.linalg.norm(np.array(u.data.body("test_object2").xpos) - rest_r) > 1.0
        step_until(0.75)
        assert u.pairs_showing() == (None, None)
        assert all(far(k) for k in range(6))
        step_until(0.9)
        assert u.pairs_showing() == (5, 0)
        assert np.allclose(pos(5), rest_l, atol=0.01) and np.allclose(pos(0), rest_r, atol=0.01)
        # 形と色は objects どおり
        g = int(u.model.body("sy_obj1").geomadr[0])
        assert np.allclose(u.model.geom_rgba[g], OBJECTS[1]["rgba"])
    finally:
        close_env(env)


def test_pairs_bad_config_raises():
    pytest.importorskip("mujoco")
    with pytest.raises(ValueError):
        _build([{"t0": 0.0, "t1": 1.0, "left": 1, "right": 1}])
    with pytest.raises(ValueError):
        _build([{"t0": 0.0, "t1": 1.0, "left": 0, "right": 9}])
    with pytest.raises(ValueError):
        _build([{"t0": 0.0, "t1": 1.0, "left": 0, "right": 1}, {"t0": 0.5, "t1": 2.0, "left": 2, "right": 3}])
