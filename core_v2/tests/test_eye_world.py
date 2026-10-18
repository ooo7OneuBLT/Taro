"""目だけの場面の確かめ（描いた画像で確かめる。DINOv2 は使わない）。

回す前の予想（2026-10-18）：
* ボールを隠すと、ボールの所の画素だけが変わる（ボールは30cm先の半径3cm＝画像で直径約35画素、約900画素。
  影は床の遠くに落ちるので、変わる画素は 400〜2000）。出し直すと、隠す前と1画素も違わない
* 目を右へ 0.2（6度）向けると、画面の中身が左へ tan(6°)/tan(30°)×112 ≈ 20画素ずれる。上へ 0.2 なら上下に約20画素
* 0.25 秒より短い間に2回指令しても、2回目は前に撃ってから 0.25 秒たつまで撃たない
"""

import csv
import os

import numpy as np
import pytest

os.environ.setdefault("TARO_EYE_LOG", "off")

from core_v2 import world_eye  # noqa: E402
from core_v2.boxes.eyein import image_shift  # noqa: E402
from core_v2.world_eye import eye as eye_mod  # noqa: E402

SCENE2 = "段階2b_3物体_ボール左_2026-10-15"
SCENE6 = "段階6_スミスとユー_6物体_2026-10-17"


@pytest.fixture
def env(tmp_path):
    e = world_eye.build_env(SCENE2, None, seed=0, silent=True, log_dir=str(tmp_path))
    e._toy3_flash_move = None        # 妨害物の揺れと点滅を止める（比べる画像が時間で変わらないように）
    yield e
    e.close()


def test_hide_changes_only_the_object_pixels(env, tmp_path):
    env.reset(seed=0)
    a = env.render_eye().astype(int)
    assert env._gaze_angle_to("test_object1") is not None
    env.hide("test_object1", cause="test")
    b = env.render_eye().astype(int)
    changed = np.abs(a - b).sum(axis=2) > 0
    assert 400 <= int(changed.sum()) <= 2000, int(changed.sum())
    assert env._gaze_angle_to("test_object1") is None          # 見えない物は見られない
    env.show("test_object1", cause="test")
    c = env.render_eye().astype(int)
    assert np.array_equal(a, c)
    # 前後の画像が残り、events.csv に書いてある
    ep = sorted(d for d in os.listdir(tmp_path) if d.startswith("ep"))[-1]
    rows = list(csv.DictReader(open(os.path.join(tmp_path, ep, "events.csv"), encoding="utf-8")))
    hide = [r for r in rows if r["event"] == "hide"]
    assert hide and hide[0]["object"] == "test_object1" and hide[0]["visible"] == "0"
    for col in ("before", "after"):
        assert os.path.isfile(os.path.join(tmp_path, ep, hide[0][col]))


def test_eye_pan_and_tilt_move_the_image(env):
    env.reset(seed=0)
    a = env.render_eye().astype(np.float32).mean(axis=2)
    u = env.unwrapped
    u._orienting.set_map_target(0.2, 0.0)
    for _ in range(10):
        env.step(None)
    b = env.render_eye().astype(np.float32).mean(axis=2)
    dx, dy, _ = image_shift(a, b)
    want = -np.tan(np.radians(6.0)) / np.tan(np.radians(30.0)) * 112.0
    assert abs(dx - want) <= 3.0 and abs(dy) <= 2.0, (dx, dy, want)
    assert abs(u._orienting.pan - 6.0) < 1e-9
    u._orienting.set_map_target(0.0, 0.2)
    for _ in range(30):
        env.step(None)
    c = env.render_eye().astype(np.float32).mean(axis=2)
    dx, dy, _ = image_shift(b, c)
    # 上を向くと中身は下へ（画像の y は下が正）。pan 6度のまま上へ6度なので、ずれはおよそ +20 画素
    assert abs(dx) <= 3.0 and 15.0 <= dy <= 25.0, (dx, dy)
    # 視線の向きと、物への角度の計算が合っている：ボールの方へ向けると、ボールへの角度がほぼ0になる
    pos = env.data.body("test_object1").xpos - env.data.cam_xpos[env._cam_id]
    pan = np.degrees(np.arctan2(pos[1], -pos[0]))
    tilt = np.degrees(np.arctan2(pos[2], np.hypot(pos[0], pos[1])))
    env.set_eye(pan, tilt)
    assert env._gaze_angle_to("test_object1") < 0.1


def test_saccade_latency_and_main_sequence():
    k = eye_mod.EyeKinematics()
    k.set_map_target(0.5, 0.0)            # 15度
    k.step(0.01)
    assert k.is_saccading()
    dur = eye_mod.saccade_duration(15.0)  # 21 + 2.2×15 = 54 ミリ秒
    assert abs(dur - 0.054) < 1e-9
    n = 1
    while k.is_saccading():
        k.step(0.01)
        n += 1
    assert abs(k.pan - 15.0) < 1e-9 and n == 6     # 54ミリ秒は6歩目で着く
    k.set_map_target(-0.5, 0.0)           # 撃ったのは 0.01 秒。0.26 秒より前は撃たない
    for _ in range(19):
        k.step(0.01)
    assert not k.is_saccading() and abs(k.pan - 15.0) < 1e-9
    k.step(0.01)                          # 0.26 秒（撃ってから 0.25 秒）の歩で撃つ
    assert k.is_saccading()


def test_eye_stops_at_its_range():
    k = eye_mod.EyeKinematics()
    for _ in range(3):
        k.set_map_target(1.0, 0.0)
        for _ in range(30):
            k.step(0.01)
    assert k.pan == eye_mod.PAN_LIMIT_DEG and k.blocked_target_n >= 1


def test_pairs_show_and_hide_on_schedule(tmp_path):
    sched = [{"t0": 0.5, "t1": 1.0, "left": 0, "right": 1}, {"t0": 1.5, "t1": 2.0, "left": 2, "right": 3}]
    e = world_eye.build_env(SCENE6, None, seed=0, world_extra={"pairs": {"schedule": sched}},
                            log_dir=str(tmp_path))
    try:
        e.reset(seed=0)
        assert not e.visible("test_object1") and not e.visible("sy_obj0")
        seen = {}
        for i in range(220):
            e.step(None)
            seen[round(float(e.data.time), 2)] = (e.visible("sy_obj0"), e.visible("sy_obj2"), e.pairs_showing())
        assert seen[0.6] == (True, False, (0, 1))
        assert seen[1.2] == (False, False, (None, None))
        assert seen[1.6] == (False, True, (2, 3))
        assert np.allclose(e.pose("sy_obj2")[0], e._rest_pos)
    finally:
        e.close()


def _labeling_world(**over):
    pl = {"enabled": True, "utterances": {"toy1": "ぼおるだね", "toy2": "こっぷだね"}, "gaze_hold_sec": 3.0,
          "attract_timeout_sec": 12.0, "respond_prob": 1.0, "repeat_labels": 0, "solo_presentation": False,
          "follow_gaze": False, "vanish_utterance": False, "vanish_wait_notice": False, "cycle_pick": True,
          "present_angles": {}}
    pl.update(over)
    return {"word_test": {"schedule": []}, "parent_labeling": pl}


def _look_at(e, body):
    pos = e._rest_pos if body == "test_object1" else e._rest_pos2
    v = pos - e.data.cam_xpos[e._cam_id]
    e.set_eye(np.degrees(np.arctan2(v[1], -v[0])), np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1]))))


def test_parent_names_the_object_taro_looks_at():
    e = world_eye.build_env(SCENE2, None, seed=1, world_extra=_labeling_world(), log_dir=None)
    try:
        e.reset(seed=1)
        e.step(None)
        tgt = e._parent_labeling._target
        _look_at(e, "test_object1" if tgt == "toy1" else "test_object2")
        said = []
        for _ in range(400):
            _o, _r, _t, _tr, info = e.step(None)
            if "parent_utterance" in info:
                said.append((round(float(e.data.time), 2), info["parent_utterance"]["text"]))
        assert said, "親が名前を言わなかった"
        assert said[0][1] == ("ぼおるだね" if tgt == "toy1" else "こっぷだね")
        assert 3.0 <= said[0][0] <= 3.2           # 3秒見続けたら言う
    finally:
        e.close()


def test_parent_vanish_really_hides():
    w = _labeling_world(solo_presentation=True, follow_gaze=True, follow_dist=0.3, vanish_utterance=True,
                        utterances={"toy1": "ぼおる", "toy2": "こっぷ"})
    e = world_eye.build_env(SCENE2, None, seed=2, world_extra=w, log_dir=None)
    try:
        e.reset(seed=2)
        texts = []
        hidden_at_nai = None
        for _ in range(1200):
            _o, _r, _t, _tr, info = e.step(None)
            if "parent_utterance" in info:
                texts.append(info["parent_utterance"]["text"])
                if "ない" in texts[-1]:
                    hidden_at_nai = (e.visible("test_object1"), e.visible("test_object2"))
                    break
        assert texts[0] in ("ぼおる", "こっぷ") and texts[-1] == texts[0] + "ないね", texts
        assert hidden_at_nai == (False, False)
    finally:
        e.close()


def test_world_flag_switches_run_v2():
    from core_v2 import run_v2
    from core_v2.world_eye.switch import pop_world_flag, select
    argv = ["x.json", "--world", "eye", "--quick"]
    assert pop_world_flag(argv) == "eye" and argv == ["x.json", "--quick"]
    assert pop_world_flag(["--world=mimo"]) == "mimo"
    orig = (run_v2.build_env, run_v2.close_env)
    try:
        select("eye")
        env = run_v2.build_env(SCENE2, "ボールはどこ？", seed=0, log_dir=None)
        rows = run_v2.run_trial(type("NoBrain", (), {"reset": lambda self: None})(), env, seed=0, duration_s=4.5, use_brain=False)
        run_v2.close_env(env)
        assert [r["heard"] for r in rows if r["heard"]] == ["ボールはどこ？"]
        assert all(r["ang_ball"] is not None for r in rows)
    finally:
        run_v2.build_env, run_v2.close_env = orig
