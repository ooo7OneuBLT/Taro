# -*- coding: utf-8 -*-
"""段階6（スミスとユーの再現）の場面を作る：6つの物を2つずつ左右に出す。

【2026-10-17新設・段階6】土台は段階2bの「ボール左」の場面（12ヶ月・座位・体を留める・
輻輳OFF・目を動かすのは core_v2 だけ）。舞台XMLを6物体入り（make_sy6_xml.py）に替えるので
qpos の並びが変わる。make_stage2_scene.py と同じく**関節を名前で突き合わせて**姿勢を写す。
* 左右の置き場所＝toy1・toy2 の定位置（視線の正面から左右に ANGLE_DEG 度、上へ ELEV_DEG 度、
  距離 0.30m）。toy1・toy2 の物そのものは、予定表があるあいだ退避する（toy_env._hold_pairs）
* 妨害物（toy3）の点滅・移動は使わない。親は何も言わない（語は実験ごとに world.word_test で渡す）
* 物の形と色は G/scripts/smith_yu_schedule.py の OBJECTS（world.pairs.objects に書く）。
  予定表（world.pairs.schedule）は空。実験ごとに world_extra で渡す

使い方:  python run/scene_tools/make_sy_scene.py
"""
import copy
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "G", "scripts") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "G", "scripts"))

from run.scene_tools import scene_io                          # noqa: E402
from run.scene_tools.make_stage2_scene import _qpos_by_joint  # noqa: E402
from run.scene_tools.make_sy6_xml import main as make_xml     # noqa: E402
from run.trainer import close_env                             # noqa: E402
from smith_yu_schedule import pairs_world                     # noqa: E402

BASE = "段階2b_3物体_ボール左_2026-10-15"
XML = "MIMo/mimoEnv/assets/benchmarkv2_scene_sy6.xml"
NAME = "段階6_スミスとユー_6物体_2026-10-17"
ANGLE_DEG = 10.0      # 左右の物を視線の正面から振る角度（片側）。段階2bのボールとコップは8度
ELEV_DEG = 5.0        # 上への角度（段階2bはボール10度・コップ4度。左右をそろえる）
DIST = 0.30


def make_scene(name=NAME, angle_deg=ANGLE_DEG, elev_deg=ELEV_DEG):
    base = scene_io.load(BASE)
    sc = copy.deepcopy(base)
    sc.pop("_path", None)
    sc["name"] = name
    sc["note"] = ("段階6（スミスとユーの再現）。6つの物を2つずつ左右に出す（world.pairs）。"
                  "左右の置き場所＝toy1・toy2 の定位置。土台: " + BASE + "。"
                  "作り方: run/scene_tools/make_sy_scene.py")
    w = sc["world"]
    w["xml"] = XML
    w["toy"] = dict(w["toy"], elev_deg=float(elev_deg), dist=DIST)
    w["toy2"] = dict(w["toy2"], angle_deg=float(angle_deg), elev_deg=float(elev_deg), dist=DIST)
    w["toy3"] = dict(w["toy3"], enabled=False)
    w["toy3"].pop("flash_move", None)
    w["toy4"] = dict(w["toy4"], enabled=False)
    w["word_test"] = {"schedule": []}
    w["pairs"] = pairs_world()

    old_env, _ = scene_io.build(base, vision=False)
    old_q = np.array(old_env.unwrapped.data.qpos, dtype=float)
    old_map = _qpos_by_joint(old_env)
    close_env(old_env)
    tmp = copy.deepcopy(sc)
    tmp["state"] = None
    new_env, _ = scene_io.build(tmp, vision=False)
    new_q = np.array(new_env.unwrapped.data.qpos, dtype=float)
    new_map = _qpos_by_joint(new_env)
    close_env(new_env)
    copied = missing = 0
    for key, (a, bb) in new_map.items():
        if key in old_map:
            oa, ob = old_map[key]
            assert ob - oa == bb - a, key
            new_q[a:bb] = old_q[oa:ob]
            copied += 1
        else:
            missing += 1                      # sy_obj0〜5（新しく入った物）
    print(f"[sy] 関節を名前で写した: {copied} 個／新規 {missing} 個")
    sc["state"] = {"qpos": [float(x) for x in new_q]}

    env, hands = scene_io.build(sc, vision=True)
    u = env.unwrapped
    u._set_anchor()
    u._place(u._toy_qadr, u._rest_pos)
    u._place(u._obj2_qadr, u._rest_pos2)
    import mujoco
    mujoco.mj_forward(u.model, u.data)
    for nm in ("test_object1", "test_object2"):
        print(f"[sy]   {nm}（左右の置き場所）: 視線から {u._gaze_angle_to(nm):.1f} 度")
    saved, drift = scene_io.save(sc, env=env, hands=hands, settle_seconds=3.0)
    close_env(env)
    print(f"[sy] 保存: {saved['name']}  安定確認: {drift}")
    return saved


if __name__ == "__main__":
    make_xml()
    make_scene()
