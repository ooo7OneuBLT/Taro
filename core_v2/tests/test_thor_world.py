"""家の場面（AI2-THOR）の確かめ。本物の AI2-THOR はクラウドに無いので、偽物の家（world_thor/fake.py）で回す。

回す前の予想（2026-10-19）：
* reset で、目（三人目のカメラ）がエージェントの 0.3m 前・床から 0.6m・下へ10度・画角60度に置かれる
* 目を右へ 0.5（15度）向けると、カメラの向き y が 15度になり、画面の中身が左へ 112×tan15°/tan30° ≈ 52画素ずれる
  （目の入口の指令の写し command_shift の予想と同じ。ずれは画像の位相相関で測る）
* 環境を何歩進めても、画像を読まなければ AI2-THOR は呼ばない。読むと、目が動いた分だけカメラを1回動かす
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
    e = fake_env()
    yield e
    e.close()


def test_import_does_not_need_ai2thor():
    code = ("import sys, core_v2.world_thor, core_v2.world_thor.taro, core_v2.tools.thor_watch; "
            "assert 'ai2thor' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO)


def test_reset_places_infant_eye(env):
    action, kw = env.controller.calls[0]
    assert action == "AddThirdPartyCamera"
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
    assert put[0] == brought[0] + 50           # 5秒で戻す
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
