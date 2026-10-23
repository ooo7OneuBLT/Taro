"""段階9：目だけの場面の部屋と20語の個体、名づけの親、代わりの目、段1〜3の写し（クラウドで回る形。DINOv2 は使わない）。

回す前の予想（2026-10-18）：
* 部屋の環境には、その部屋の語×覚える個体の体があり、reset では全部隠れている。3つ並べると3つとも画像に写る
  （隠す前後で変わる画素がそれぞれ 300 以上）
* 物の中心をカメラへ写した場所は、隠す前後で変わった画素の重心から 15画素以内（物の形は中心の左右で非対称なことがある）
* 太郎の目を物へ向けておくと、親は 0.5秒で鳴らし（音のある物）、1.5秒で名前を言い、1秒ごとに向きの番号を1つずらす
* 代わりの目は、画像に写っている物の数だけ荷札を名簿の側（kind）へ出す
"""

import os
import random

import numpy as np
import pytest

os.environ.setdefault("TARO_EYE_LOG", "off")

from core_v2 import world_eye  # noqa: E402,F401
from core_v2.tools import growth  # noqa: E402
from core_v2.world_eye import instances as ins  # noqa: E402
from core_v2.world_eye import noun_scene as ns  # noqa: E402
from core_v2.world_eye.eye_backend import project  # noqa: E402


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(growth, "LEDGER", str(tmp_path / "育ち帳.json"))
    monkeypatch.setattr(growth, "SNAP_DIR", str(tmp_path / "育ち"))
    return tmp_path


@pytest.fixture
def env(tmp_path):
    ins.use_objects("made")
    words = [w for w in ins.WORDS if ins.ROOM_OF[w] == "遊び"]
    e = ns.make_env("遊び", "made", log_dir=str(tmp_path / "log"), words=words, insts=ins.TRAIN_INST)
    e.reset(seed=0)
    e.set_eye(0.0, ns.EL0, cause="test")
    yield e
    e.close()


def test_room_has_instances_hidden_then_three_visible(env):
    assert len(env.n9) == 10 * len(ins.TRAIN_INST)
    assert not any(env.visible(n) for n in env.n9)
    placed = ns.place(env, [("わんわん", 0, 4), ("ぶうぶう", 1, 4), ("くま", 2, 4)])
    assert [p.word for p in placed] == ["わんわん", "ぶうぶう", "くま"]
    for p in placed:
        assert env.n9[p.name]["word"] == p.word and env.n9[p.name]["k"] == p.k
        a = env.render_eye().astype(int)
        env.hide(p.name)
        b = env.render_eye().astype(int)
        env.show(p.name)
        changed = np.abs(a - b).sum(axis=2) > 0
        assert changed.sum() >= 300, (p.word, int(changed.sum()))
        ys, xs = np.nonzero(changed)
        cx, cy = project(env, p.name)
        assert abs(cx - xs.mean()) < 15 and abs(cy - ys.mean()) < 15, (p.word, cx, cy, xs.mean(), ys.mean())


def test_object_sound_reaches_next_step_and_is_logged(env, tmp_path):
    from core_v2.world_eye import sounds as snd
    env.object_sound("n9_dog_0", snd.synth("dog", 0, seed=1))
    _, _, _, _, info = env.step()
    assert info.get("object_sound") is not None and len(info["object_sound"]) > 1000
    _, _, _, _, info = env.step()
    assert "object_sound" not in info
    ep = sorted(d for d in os.listdir(tmp_path / "log") if d.startswith("ep"))[-1]
    rows = open(tmp_path / "log" / ep / "events.csv", encoding="utf-8").read()
    assert ",sound,n9_dog_0," in rows


def test_parent_rings_names_and_turns_what_taro_looks_at(env):
    placed = ns.place(env, [("わんわん", 0, 4), ("ぶうぶう", 1, 4), ("くま", 2, 4)])
    dog = placed[0]
    env.set_eye(dog.az, dog.el, cause="test")        # 太郎の目を犬へ（目は物の中心をちょうど見る）
    par = ns.NounParent(random.Random(0), p_name=1.0, p_sound=1.0)
    said, rang, views = [], [], []
    for t in range(25):
        did = par.tick(env, placed)
        views.append(env.n9_view[dog.name])
        if "said" in did:
            said.append(t)
        if "rang" in did:
            rang.append(t)
        for _ in range(10):
            env.step()
    assert rang == [5] and said == [15]
    assert par.stats["turns"] == 2 and views[9] != views[11] and views[19] != views[21]
    assert ns.looking_at(env, placed) is dog


def test_table_eye_sends_one_tag_per_visible_object(env):
    from core_v2.brain import Brain
    from core_v2.world_eye.eye_backend import install_eye
    brain = Brain()
    info = install_eye(brain, "table", env_ref=lambda: env, objects="made")
    assert info["eye"] == "table"
    ns.place(env, [("わんわん", 0, 4), ("ぶうぶう", 1, 4), ("くま", 2, 4)])
    from core_v2.boxes.base import ENV
    out = brain.boxes["eyein"].step({ENV: env.render_eye()})
    objs = out["kind"]
    assert len(objs) == 3 + 1                          # 3つの物＋目の動きの地図
    assert all(float(o[0].norm()) > 0.5 for o in objs[:3])
    assert float(objs[-1][0].norm()) == 0.0


def test_growth_phases_1_2_3_and_the_embodied_probe(ledger, tmp_path):
    from core_v2.tools import grow_early, grow_names, noun_body
    r1 = grow_early.grow_phase1(seed=0)
    assert r1["phase"] == 1 and "がおお" in r1["conditions"]["known_words"]
    r2 = grow_early.grow_phase2(seed=0)
    assert r2["parent"] == r1["name"] and growth.month_of(r2) > 23
    assert growth.start_point(3, seed=0, world="eye")["name"] == r2["name"]
    g = grow_names.Grower(0, "made", "table", None, log_root=str(tmp_path / "g"), start="auto")
    assert g.start_name() == r2["name"]
    g.scene()
    g.close()
    assert g.n_scenes == 1 and g.n_tick == int((ns.SCENE_S + ns.GAP_S) / ns.TICK_S)
    r3 = growth.save(g.brain.boxes, 3, "段3_test", end=True, parent=r2["name"],
                     conditions={"world": "eye", "eye": "table", "seed": 0, "objects": "made"})
    assert growth.latest(3, seed=0, world="eye")["name"] == r3["name"]
    assert growth.month_of(r3) == growth.month_of(r2)
    t = noun_body.Tester(0, "made", "table", None, log_root=str(tmp_path / "t"), start=r3["name"])
    p = noun_body.probe(t, 0, ["わんわん"], 1, image_trials=0)
    t.close()
    assert set(p["A"]) == {"わんわん"} and set(p["C"]) == {"わんわん"}
    assert [r["test"] for r in t.rows] == ["A", "B4", "B2", "C"]
    s = noun_body.judge(p)
    assert s["A_pass"] is None and s["A_words"] == 1
