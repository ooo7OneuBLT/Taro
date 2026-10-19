"""家の場面（AI2-THOR）の確かめ。本物の AI2-THOR はクラウドに無いので、偽物の家（world_thor/fake.py）で回す。

回す前の予想（2026-10-19）：
* reset で、目（三人目のカメラ）がエージェントの 0.3m 前・床から 0.6m・下へ10度・画角60度に置かれる
* 目を右へ 0.5（15度）向けると、カメラの向き y が 15度になり、画面の中身が左へ 112×tan15°/tan30° ≈ 52画素ずれる
  （目の入口の指令の写し command_shift の予想と同じ。ずれは画像の位相相関で測る）
* 環境を何歩進めても、画像を読まなければ AI2-THOR は呼ばない。読むと、目が動いた分だけカメラを1回動かす
* 座る場所（place="auto"）：冷蔵庫の扉の前で壁を向いた始めの場所（名前の物が目に入らない）から、歩ける所と向きを
  幾何で数えて、名前の物が3つ以上目に入る所へ Teleport する。place="start" では動かない
* ひとりで見せる（bring-in）：名前の物を見ないまま5秒（50回目）で、親が物を目の前へ持ってくる。太郎が見るまで名前は言わず、
  見はじめて 0.5秒で言う。見たら5秒で戻す。一度も見なければ8秒で、言わずに戻す
* 最初は真正面やや下のりんごが目の真ん中にある。親は1秒見続けたら目の前 0.3m へ持ってきて、0.5秒で「りんご」を含む文を言い、
  見続けていれば 2秒ごとに2回言い直し（計3回）、5秒で元の場所へ戻す。戻した物は3秒は持ってこない
"""

import json
import os
import random
import subprocess
import sys

import numpy as np
import pytest

from core_v2.boxes.eyein import command_shift, image_shift
from core_v2.world_thor import parent as tp
from core_v2.world_thor.env import cam_basis, vec
from core_v2.world_thor.fake import fake_env

APPLE = "Apple|+00.00|+00.50|+01.00"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


@pytest.fixture
def env():
    e = fake_env(toys=0)          # おもちゃを並べない（物は DEFAULT_OBJECTS の場所のまま）
    yield e
    e.close()


def test_import_does_not_need_ai2thor():
    code = ("import sys, core_v2.world_thor, core_v2.world_thor.taro, core_v2.tools.thor_watch; "
            "assert 'ai2thor' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO)


def test_reset_places_infant_eye(env):
    names = [c[0] for c in env.controller.calls]
    assert names == ["GetReachablePositions", "AddThirdPartyCamera"]     # 到達点が1つ → 動かない
    action, kw = env.controller.calls[1]
    assert kw["position"] == pytest.approx({"x": 0.0, "y": 0.6, "z": 0.3})
    assert kw["rotation"]["x"] == pytest.approx(10.0) and kw["rotation"]["y"] == pytest.approx(0.0)
    assert kw["fieldOfView"] == pytest.approx(60.0)
    assert env.controller.kw == {} and env.px == 224
    assert sorted(env.word_of(i) for i in env.targets) == ["おさら", "こっぷ", "ほん", "りんご"]


def test_eye_command_turns_camera_like_efference_copy(env):
    a = env.render_eye().astype(np.float32).mean(axis=2)
    env._orienting.set_map_target(0.5, 0.0)
    for _ in range(30):
        env.step()
    assert env._orienting.pan == pytest.approx(15.0)
    b = env.render_eye().astype(np.float32).mean(axis=2)
    action, kw = env.controller.calls[-1]
    assert action == "UpdateThirdPartyCamera"
    assert kw["rotation"]["y"] == pytest.approx(15.0) and kw["rotation"]["x"] == pytest.approx(10.0)
    ix, iy, _ = image_shift(a, b)
    px, py = command_shift(0.5, 0.0)
    assert px < -40 and abs(ix - px) < 6 and abs(iy) < 4
    # 上へ 0.2（6度）：カメラは下向き 10→4度、中身は下へ
    env._orienting.set_map_target(0.0, 0.2)
    for _ in range(30):
        env.step()
    env.render_eye()
    assert env.controller.calls[-1][1]["rotation"]["x"] == pytest.approx(4.0)


def test_controller_called_only_when_image_is_read(env):
    n0 = len(env.controller.calls)
    for _ in range(20):
        env.step()
    assert len(env.controller.calls) == n0
    obs, *_ = env.step()
    obs["eye_left"]
    assert len(env.controller.calls) == n0          # 目が動いていない → 前の絵のまま
    env._orienting.set_map_target(-0.3, 0.0)
    for _ in range(10):
        obs, *_ = env.step()
    assert obs["eye_left"].shape == (224, 224, 3)
    assert [c[0] for c in env.controller.calls[n0:]] == ["UpdateThirdPartyCamera"]


def test_centre_object_from_segmentation(env):
    assert env.centre_object() == APPLE and env.word_of(APPLE) == "りんご"
    c, r, n = env.object_centres()[APPLE]
    assert abs(c - 112) < 3 and abs(r - 112) < 6 and n > 200
    env.set_eye(-30.0, -10.0)
    assert env.centre_object() != APPLE


def test_parent_brings_shakes_names_and_puts_back(env):
    par = tp.FollowInParent(random.Random(0))
    said, did_log, where = [], [], []
    for k in range(100):                       # 10秒。太郎の目は動かない（脳はつながない）
        did = par.tick(env)
        did_log.append(did)
        for _ in range(10):
            _o, _r, _t, _tr, info = env.step()
            if "parent_utterance" in info:
                said.append((k, info["parent_utterance"]["text"]))
        env.render_eye()
        where.append(vec(env.objects[APPLE]["position"]))
        if k == 15:                            # 持ってきて0.6秒：目の前 0.3m（振っている分 ±3cm）
            g = cam_basis(env.cam_yaw, env.cam_pitch)[0]
            want = env.eye_pos + 0.3 * g
            got = vec(env.objects[APPLE]["axisAlignedBoundingBox"]["center"])
            assert np.linalg.norm(got - want) < 0.035
    brought = [k for k, d in enumerate(did_log) if "brought" in d]
    put = [k for k, d in enumerate(did_log) if "put" in d]
    assert brought[0] == 9                     # 10回目（1秒）で持ってくる
    assert put[0] == brought[0] + 55           # 5秒で戻す。ただし最後の言い直し（4.5秒）から1秒は持っている
    assert [k for k, _ in said[:3]] == [brought[0] + 5, brought[0] + 25, brought[0] + 45]
    assert all("りんご" in t for _, t in said[:3])
    assert par.stats["repeats"] == 2
    assert np.allclose(where[put[0]], env.home[APPLE][0])
    # 戻したあと3秒は持ってこない（りんごは真ん中に戻って、まだ見えている。3秒たったらまた持ってくる）
    assert brought[1] == put[0] + 30
    tele = [c for c in env.controller.calls if c[0] == "TeleportObject"]
    assert tele and all(c[1]["forceKinematic"] and c[1]["forceAction"] for c in tele)


def test_say_sound_and_voice(env):
    env.say("ほんだよ", target="Book", cause="test")
    env.object_sound("Book", np.zeros(1600))
    _o, _r, _t, _tr, info = env.step()
    assert info["parent_utterance"]["text"] == "ほんだよ" and info["object_sound"].shape == (1600,)
    env.taro_said("ほ")
    assert env._taro_voice_signal and env._taro_voice_text == "ほ"


def test_look_log():
    lg = tp.LookLog()
    for oid, w in [("a", "りんご")] * 12 + [(None, "")] * 3 + [("b", "ほん")] * 4:
        lg.add(oid, w)
    lg.flush()
    s = lg.summary()
    assert s["looks"] == 2 and s["max_s"] == pytest.approx(1.2) and s["n_ge_1s"] == 1
    assert s["by_word"]["ほん"]["total_s"] == pytest.approx(0.4)


def test_grower_with_brain_moves_eye_and_parent_names():
    from core_v2.world_thor.taro import ThorGrower
    g = ThorGrower(0, "seg", env=fake_env(), start="birth")
    try:
        g.run(15.0)
    finally:
        g.close()
    s = g.summary()
    assert s["ticks"] == 150 and s["saccades"] >= 1
    assert s["parent"]["brings"] >= 1 and s["parent"]["names"] >= 1
    assert s["looks"]["looks"] >= 1 and s["thor_fail"] == 0


def test_thor_watch_writes_video(tmp_path):
    from core_v2.tools import watch_scene
    out = str(tmp_path / "v.mp4")
    res = watch_scene.run(["--world", "thor", "--fake", "--eye", "seg", "--from", "birth", "--max-ticks", "5",
                           "--stills", "1", "--render-px", "224", "--out", out])
    assert res["frames"] == 5 and os.path.getsize(res["out"]) > 0
    assert len(res["stills"]) == 1 and os.path.isfile(res["stills"][0])
    with open(os.path.splitext(res["out"])[0] + "_まとめ.json", encoding="utf-8") as f:
        assert json.load(f)["fake"] is True


# ---------------------------------------------------------------- 座る場所（place）
FRIDGE = ("Fridge|+00.00|+00.60|-00.55", "Fridge", (0.0, 0.6, -0.55), 0.9, (200, 200, 205))


def _room(place):
    """始めの場所 (0, 0.9, 0) は −z（冷蔵庫）を向く。名前の物は +z 側の 1m ほど先。歩ける所は 0.25m の格子。"""
    from core_v2.world_thor.fake import DEFAULT_OBJECTS
    grid = [(x * 0.25, 0.9, z * 0.25) for x in range(-4, 5) for z in range(-8, 4)]
    return fake_env(place=place, toys=0, controller_kw={"objects": DEFAULT_OBJECTS + [FRIDGE], "agent_yaw": 180.0,
                                                         "reachable": grid})


def test_placement_picks_spot_with_nameable_objects_in_view():
    bad = _room("start")
    good = _room("auto")
    try:
        assert bad.placement["n_in_view"] == 0 and bad.placement["penalty"] >= 3.0
        assert bad.agent_yaw == pytest.approx(180.0)
        pl = good.placement
        assert pl["start_n_in_view"] == 0 and pl["n_in_view"] >= 3 and pl["penalty"] == 0.0
        assert pl["n_candidates"] == 1 + 60 * 4 and pl["ahead_m"] >= 1.5
        tele = [c for c in good.controller.calls if c[0] == "Teleport"]
        assert len(tele) == 1 and tele[0][1]["standing"] is True
        assert tele[0][1]["rotation"]["y"] == pytest.approx(pl["yaw"])
        # 目のカメラは Teleport のあと、新しい場所に置かれる
        names = [c[0] for c in good.controller.calls]
        assert names.index("AddThirdPartyCamera") > names.index("Teleport")
        cam = good.controller.cams[0]
        assert cam["position"] == pytest.approx(xyz_list(good.eye_pos))
        # 数えた物は本当に絵に写っている
        seen = set(good.object_centres())
        assert len(seen) >= 3 and {good.word_of(i) for i in seen} >= set(pl["words"]) - {""}
        # 同じ種なら同じ場所
        again = _room("auto")
        assert again.placement["position"] == pl["position"] and again.placement["yaw"] == pl["yaw"]
        again.close()
    finally:
        bad.close()
        good.close()


def xyz_list(v):
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def test_placement_skips_objects_inside_closed_fridge():
    from core_v2.world_thor.fake import DEFAULT_OBJECTS
    closed = ("Fridge|+00.00|+00.60|+03.00", "Fridge", (0.0, 0.6, 3.0), 0.9, (200, 200, 205),
              {"openable": True, "isOpen": False})
    inside = ("Apple|+00.00|+00.50|+01.40", "Apple", (0.0, 0.5, 1.4), 0.09, (200, 30, 30),
              {"parentReceptacles": [closed[0]]})
    e = fake_env(toys=0, controller_kw={"objects": DEFAULT_OBJECTS[1:] + [closed, inside]})
    try:
        r = e.score_spot(np.array([0.0, 0.9, 0.0]), 0.0)
        assert inside[0] not in r["seen"] and len(r["seen"]) >= 2
    finally:
        e.close()


# ---------------------------------------------------------------- ひとりで見せる（bring-in）
def _run_parent(env, par, ticks, eye=None, said=None, did_log=None):
    for _k in range(ticks):
        if eye is not None:
            env.set_eye(*eye)
        did = par.tick(env)
        if did_log is not None:
            did_log.append(did)
        for _ in range(10):
            _o, _r, _t, _tr, info = env.step()
            if said is not None and "parent_utterance" in info:
                said.append((len(did_log or []) - 1, info["parent_utterance"]))
        env.render_eye()


def test_bring_in_after_5s_names_only_after_taro_looks(env):
    par = tp.FollowInParent(random.Random(0))
    away = (-45.0, 33.0)                         # 左上の壁：名前の物は目の真ん中に無い
    other = (45.0, 33.0)
    said, did_log = [], []
    _run_parent(env, par, 50, eye=away, said=said, did_log=did_log)
    assert env.centre_object() is None or "brought" in did_log[-1]
    b = [k for k, d in enumerate(did_log) if "brought" in d]
    assert b == [49] and did_log[49]["mode"] == "bringin"
    oid = par.held
    assert oid in env.targets and par.mode == "bringin"
    # 太郎はよそを見ている（2秒）：持って振っているが、名前は言わない
    _run_parent(env, par, 20, eye=other, said=said, did_log=did_log)
    assert said == [] and par.held == oid and par.first_look is None
    # 持ってきた所（視線の先 0.3m）を見る → はじめて見た回から 0.5秒で言う
    _run_parent(env, par, 1, eye=away, said=said, did_log=did_log)
    assert env.centre_object() == oid
    _run_parent(env, par, 29, eye=away, said=said, did_log=did_log)
    first_look = 70                              # 71回目（k=70）の tick で見つける
    assert par.first_look == first_look - 49
    assert said and said[0][0] == first_look + 5
    assert env.word_of(oid) in said[0][1]["text"] and said[0][1]["cause"] == "parent_name_bringin"
    assert said[0][1]["target"] == oid
    assert said[1][0] == first_look + 25          # 見続けていれば2秒ごとに言い直す
    # 持って5秒（50回）で戻す
    _run_parent(env, par, 10, eye=away, said=said, did_log=did_log)
    put = [k for k, d in enumerate(did_log) if "put" in d]
    assert put == [49 + 56]                       # 最後の名づけ（持って4.6秒）から1秒
    assert np.allclose(vec(env.objects[oid]["position"]), env.home[oid][0])
    s = par.stats
    assert s["brings_bringin"] == 1 and s["brings_follow"] == 0 and s["names_bringin"] == 2 and s["bringin_looked"] == 1


def test_bring_in_never_looked_puts_back_after_8s_without_naming(env):
    par = tp.FollowInParent(random.Random(0))
    said, did_log = [], []
    _run_parent(env, par, 50, eye=(-45.0, 33.0), said=said, did_log=did_log)
    first = par.held
    assert first is not None
    _run_parent(env, par, 120, eye=(45.0, 33.0), said=said, did_log=did_log)
    put = [k for k, d in enumerate(did_log) if "put" in d]
    assert put[0] == 49 + 80 and [k for k, _ in said if k <= put[0]] == []
    assert par.stats["bringin_unlooked"] == 1
    # 戻して3秒（手が空くまで）は持ってこない。次は別の物（ここで見せた物は選ばない）
    b = [k for k, d in enumerate(did_log) if "brought" in d]
    assert b[1] == put[0] + 30
    assert did_log[b[1]]["brought"] and par.recent[0] == first and par.recent[1] != first
    # 2つめは今の視線の先（よそを見ている所）へ持ってくるので、太郎はすぐ見て、0.5秒あとに名前を聞く
    assert said and said[0][0] == b[1] + 1 + 5 and said[0][1]["target"] == par.recent[1]


def test_bring_in_off_keeps_follow_only(env):
    par = tp.FollowInParent(random.Random(0), bring_in_after=None)
    did_log = []
    _run_parent(env, par, 80, eye=(-45.0, 33.0), did_log=did_log)
    assert par.stats["brings"] == 0


# ---------------------------------------------------------------- 依頼17 の結果を受けて：おもちゃを並べる・名づけは持って見ている間だけ・座る場所
def _demo(**kw):
    from core_v2.world_thor.fake import DEMO_ROOM
    ckw = dict(DEMO_ROOM)
    ckw.update(kw.pop("controller_kw", {}))
    return fake_env(controller_kw=ckw, **kw)


def _aim(env, oid):
    """目を物の見かけの真ん中へ向ける（道具の set_eye。太郎の跳びではない）。"""
    v = env._centre(env.objects[oid]) - env.eye_pos
    pan = np.degrees(np.arctan2(v[0], v[2])) - env.agent_yaw
    tilt = np.degrees(np.arctan2(v[1], np.hypot(v[0], v[2])))
    env.set_eye(float(pan), float(tilt))


def test_parent_sets_out_toys_on_floor_in_view():
    e = _demo(place=(0.0, 0.0, 0.0))
    try:
        assert len(e.toys) == 4
        words = [e.word_of(i) for i in e.toys]
        assert len(set(words)) == 4 and "Apple|+00.00|+00.60|-01.60" not in e.toys   # 冷蔵庫の中のりんごは後回し
        assert [t["angle"] for t in e.toy_info] == [-20.0, -7.0, 7.0, 20.0]
        for t in e.toy_info:
            o = e.objects[t["id"]]
            assert o["position"]["y"] == pytest.approx(0.0, abs=1e-6)          # 床の上（物の底）
            d = np.hypot(o["position"]["x"] - e.eye_pos[0], o["position"]["z"] - e.eye_pos[2])
            assert 0.6 - 1e-6 <= d <= 1.2 + 1e-6 and t["method"] == "PlaceObjectAtPoint"
            assert np.allclose(e.home[t["id"]][0], vec(o["position"]))          # 戻す所は床の置いた所
        pl = e.placement
        assert pl["toys_in_view"] == 4 and pl["toys_seen"] == 4
        assert set(e.object_centres()) >= set(e.toys)
        cols = sorted(e.object_centres()[i][0] for i in e.toys)
        assert cols[0] < 70 and cols[-1] > 154                                 # 視野に散らばる
        assert sum(c[0] == "PlaceObjectAtPoint" for c in e.controller.calls) == 4
    finally:
        e.close()


def test_toys_fall_back_to_teleport_without_place_at_point():
    e = _demo(place=(0.0, 0.0, 0.0), controller_kw={"place_at_point": False})
    try:
        assert len(e.toys) == 4 and all(t["method"] == "TeleportObject(控え)" for t in e.toy_info)
        tele = [c for c in e.controller.calls if c[0] == "TeleportObject"]
        assert len(tele) == 4 and all(c[1]["forceKinematic"] is False for c in tele)
        for i in e.toys:
            assert e.objects[i]["position"]["y"] == pytest.approx(0.0, abs=1e-6)
        assert e.placement["toys_in_view"] == 4
    finally:
        e.close()


def test_put_back_returns_toy_to_its_floor_spot():
    e = _demo(place=(0.0, 0.0, 0.0))
    try:
        toy = e.toys[1]
        floor = vec(e.objects[toy]["position"]).copy()
        _aim(e, toy)
        e.render_eye()
        assert e.centre_object() == toy
        par = tp.FollowInParent(random.Random(0), bring_in_after=None)
        did_log = []
        _run_parent(e, par, 70, did_log=did_log)       # 目は動かさない（持ってきた物は視線の先）
        b = [k for k, d in enumerate(did_log) if "brought" in d]
        put = [k for k, d in enumerate(did_log) if "put" in d]
        assert b[0] == 9 and put[0] == 9 + 55
        assert np.allclose(vec(e.objects[toy]["position"]), floor)
        assert e.objects[toy]["position"]["y"] == pytest.approx(0.0, abs=1e-6)
    finally:
        e.close()


def test_no_naming_while_taro_looks_away_reproduces_pc_bug(env):
    """依頼17 の PC：持ってきた物から太郎が目をそらしても、0.5秒目の名づけは言っていた（k==0 は見ていなくてよかった）。"""
    par = tp.FollowInParent(random.Random(0))
    said, did_log = [], []
    _run_parent(env, par, 10, said=said, did_log=did_log)
    assert "brought" in did_log[9] and par.held == APPLE
    _run_parent(env, par, 20, eye=(-45.0, 33.0), said=said, did_log=did_log)   # よそ（壁）を見ている 2秒
    assert said == [] and par.held == APPLE                                     # 前はここで「りんご」と言っていた
    _run_parent(env, par, 1, eye=(0.0, env.base_tilt), said=said, did_log=did_log)
    assert env.centre_object() == APPLE
    _run_parent(env, par, 3, said=said, did_log=did_log)
    assert len(said) == 1 and said[0][0] == 30 and said[0][1]["target"] == APPLE and "りんご" in said[0][1]["text"]
    assert all(d.get("said") is None or d["looking"] == "りんご" for d in did_log)


def test_no_naming_after_put_back_or_when_hand_empty(env):
    # 環境の側：持っていない物の名づけは届けない（env.say に残っていても）
    env.say("りんごだよ", target=APPLE, cause="parent_name_follow")
    _o, _r, _t, _tr, info = env.step()
    assert "parent_utterance" not in info
    env.hold(APPLE, env.eye_pos + 0.3 * env.gaze_dir())
    env.say("りんごだよ", target=APPLE, cause="parent_name_follow")
    env.put_back(APPLE)
    _o, _r, _t, _tr, info = env.step()
    assert "parent_utterance" not in info
    env.hold(APPLE, env.eye_pos + 0.3 * env.gaze_dir())
    env.say("りんごだよ", target=APPLE, cause="parent_name_follow")
    _o, _r, _t, _tr, info = env.step()
    assert info["parent_utterance"]["text"] == "りんごだよ"
    env.put_back(APPLE)
    env.render_eye()
    # 親の側：言った回は、いつも持っていて、太郎がその物を見ている回。最後の名づけから1秒は戻さない
    par = tp.FollowInParent(random.Random(1))
    did_log, said = [], []
    rng = np.random.default_rng(0)
    for k in range(300):
        if k % 7 == 0:
            env.set_eye(float(rng.uniform(-20, 20)), float(rng.uniform(-25, 5)))
        _run_parent(env, par, 1, said=said, did_log=did_log)
    names = [k for k, d in enumerate(did_log) if "said" in d]
    assert names, "一度も言わなかった（確かめにならない）"
    for k in names:
        d = did_log[k]
        assert d["holding"] and d["looking"] == d["holding"]
    puts = [k for k, d in enumerate(did_log) if "put" in d]
    for k in names:
        nxt = [p for p in puts if p > k]
        assert not nxt or nxt[0] - k >= 10


def test_bring_in_prefers_toys_on_floor():
    e = _demo(place=(0.0, 0.0, 0.0))
    try:
        par = tp.FollowInParent(random.Random(0))
        did_log = []
        _run_parent(e, par, 50, eye=(-45.0, 33.0), did_log=did_log)
        assert did_log[49].get("mode") == "bringin" and par.held in e.toys
    finally:
        e.close()


def test_placement_requires_open_floor_ahead():
    from core_v2.world_thor.fake import DEFAULT_OBJECTS
    # 奥行き 1m しかない細長い部屋：物は +z 側にあるが、+z を向くと 1.5m 先が歩けない → 左右（90・270度）を向く所だけが候補
    grid = [(x * 0.25, 0.9, z * 0.25) for x in range(-8, 9) for z in range(-1, 4)]
    for toys in (0, 4):
        e = fake_env(toys=toys, controller_kw={"objects": DEFAULT_OBJECTS, "reachable": grid})
        try:
            pl = e.placement
            assert pl["ahead_m"] >= 1.5 and pl["yaw"] in (90.0, 270.0) and pl["mode"] == "auto"
            assert pl["n_open"] < pl["n_candidates"]
        finally:
            e.close()
    # どこも前が開いていない → 始めの場所のまま
    tiny = [(x * 0.25, 0.9, z * 0.25) for x in range(-2, 3) for z in range(-1, 2)]
    e = fake_env(controller_kw={"reachable": tiny})
    try:
        assert e.placement["n_open"] == 0 and "前が開いた所なし" in e.placement["mode"]
        assert not any(c[0] == "Teleport" for c in e.controller.calls)
    finally:
        e.close()


def test_spot_by_hand_teleports_exactly_and_snaps_when_unreachable():
    e = _demo(place="1.0,0.5,270")
    try:
        assert e.placement["mode"] == "spot" and e.placement["position"] == [1.0, 0.9, 0.5] and e.placement["yaw"] == 270.0
        tele = [c for c in e.controller.calls if c[0] == "Teleport"]
        assert tele[0][1]["position"] == {"x": 1.0, "y": 0.9, "z": 0.5}
        assert len(e.toys) >= 3                                         # 手で選んだ所でも並べる
    finally:
        e.close()
    e = _demo(place=(1.1, 0.4, 0.0))
    try:
        assert e.placement["mode"] == "spot(近い到達点へ)" and e.placement["position"] == [1.0, 0.9, 0.5]
    finally:
        e.close()


def test_first_frame_tool_writes_eye_and_topdown(tmp_path):
    from core_v2.tools import thor_watch
    res = thor_watch.run(["--first-frame", "--fake", "--scene", "FloorPlan1", "--render-px", "224",
                          "--out", str(tmp_path)])
    assert os.path.isfile(tmp_path / "家の場面3_最初_FloorPlan1.png")
    assert os.path.isfile(tmp_path / "家の場面3_上から_FloorPlan1.png")
    assert os.path.isfile(tmp_path / "家の場面3_最初_FloorPlan1.json")
    assert res["placement"]["toys_in_view"] == 4 and res["n_reach"] > 100 and res["thor_fail"] == 0
    res2 = thor_watch.run(["--first-frame", "--fake", "--render-px", "224", "--spot=-1,1,90", "--out", str(tmp_path)])
    assert res2["placement"]["position"][0] == -1.0 and res2["placement"]["yaw"] == 90.0
