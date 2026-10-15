# -*- coding: utf-8 -*-
"""段階2の合格テスト用の場面（ボール・コップ・光って動く物）を作る。

【2026-10-15新設・段階2】仕様書 doc/設計/2026-10-15_段階2_仕様書.md §4-1。
土台は現行の座位場面「土台_座位12ヶ月_壁あり_胡坐_8択_消失発話_2026-09-15」
（12ヶ月・壁あり・胡坐）。舞台XMLを test_object3 入りのものに替えるため
qpos の並びが変わる（8択用の test_object5〜11 が消え、3・4 が入る）。そこで
**関節を名前（body名＋関節名）で突き合わせて**姿勢を写し、作り直した環境で
おもちゃを今の姿勢の視線の正面に置き直してから、scene_io.save で指紋ごと保存する。

置き方（ボールとコップの左右だけ入れ替えた2枚を作る）:
  ・ボール左：toy1=ボール（左）、toy2=コップ（右）、toy3=妨害物（上）
  ・ボール右：同じで、ボールとコップの左右だけ逆
  妨害物は toy3 の既存の振り分け（上下軸）で正面の上に置き、左右に揺らしながら点滅させる。
語：world.word_test に「ボールはどこ？」（4秒）を入れてある。「コップはどこ？」は
  実験ファイルで world.word_test を上書きする。

使い方:  python run/scene_tools/make_stage2_scene.py
"""
import copy
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from run.scene_tools import scene_io                       # noqa: E402
from run.scene_tools.make_f4toys_xml import main as make_xml  # noqa: E402
from run.trainer import close_env                          # noqa: E402

BASE = "土台_座位12ヶ月_壁あり_胡坐_8択_消失発話_2026-09-15"
XML = "MIMo/mimoEnv/assets/benchmarkv2_scene_f4toys.xml"
ANGLE_DEG = 12.0          # ボールとコップを視線の正面から左右に振る角度（片側）
TOY3_UP_DEG = 12.0        # 妨害物を正面から上へ振る角度
DIST = 0.30               # 目から物まで[m]（土台と同じ）
FLASH = {"amp_m": 0.03, "hz": 0.5, "blink_hz": 2.0}   # 3cm＝約5.7度の左右往復・2Hz点滅
NAMES = {+1: "段階2_3物体_ボール左_2026-10-15", -1: "段階2_3物体_ボール右_2026-10-15"}


def _qpos_by_joint(env):
    """{(body名, 関節名): qposの切れ端} を返す。test_object の freejoint は名前が無いので body名で区別する。"""
    m, d = env.unwrapped.model, env.unwrapped.data
    out = {}
    for j in range(m.njnt):
        a = int(m.jnt_qposadr[j])
        b = j + 1 < m.njnt and int(m.jnt_qposadr[j + 1]) or m.nq
        key = (m.body(int(m.jnt_bodyid[j])).name, m.joint(j).name)
        out[key] = (a, b)
    return out


def make_scene(side):
    base = scene_io.load(BASE)
    sc = copy.deepcopy(base)
    sc.pop("_path", None)
    sc["name"] = NAMES[side]
    sc["note"] = ("段階2の合格テスト（語を聞いて物を探す）。ボール=toy1・コップ=toy2・"
                  "光って動く妨害物=toy3。土台: " + BASE + "。"
                  "作り方: run/scene_tools/make_stage2_scene.py")
    w = sc["world"]
    w["xml"] = XML
    w.pop("present_slots", None)              # 8択用の差し出しスロット（新XMLには無い）
    w["toy"] = dict(w["toy"], shape="sphere", radius=0.03, dist=DIST,
                    rgba=[0.85, 0.15, 0.15, 1.0])
    w["toy"].pop("vanish_gap_before_sec", None)
    w["toy2"] = {"enabled": True, "shape": "cylinder", "radius": 0.03,
                 "rgba": [0.95, 0.95, 0.95, 1.0], "dist": DIST,
                 "angle_deg": ANGLE_DEG * side, "elev_deg": 0.0}
    w["toy3"] = {"enabled": True, "shape": "box", "radius": 0.025,
                 "rgba": [0.2, 0.4, 0.9, 1.0], "dist": DIST,
                 "angle_deg": TOY3_UP_DEG, "elev_deg": 0.0,
                 "flash_move": dict(FLASH), "rest_from_state": True}
    w["parent_labeling"] = dict(w["parent_labeling"], enabled=False)
    w["parent_intervene"] = False             # 見失っても親が差し出し直さない（物が視線へ寄ってくると測れない）
    w["word_test"] = {"schedule": [{"t": 4.0, "word": "ボールはどこ？"}]}
    b = sc["body"]
    b["vergence"] = False                     # 仕様書 §2-3：輻輳はOFF
    b["orienting_version"] = 2
    b["orienting_external_only"] = True       # 目を動かすのは core_v2 の orienteye だけ

    # --- 姿勢を名前で写す -------------------------------------------------
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
            missing += 1                      # test_object3/4（新しく入った物）
    print(f"[stage2] 関節を名前で写した: {copied} 個／新規 {missing} 個")
    sc["state"] = {"qpos": [float(x) for x in new_q]}

    # --- 作り直して、今の姿勢の視線の正面へ物を置き直し、保存 -------------------
    env, hands = scene_io.build(sc, vision=True)
    u = env.unwrapped
    u._set_anchor()
    u._place(u._toy_qadr, u._rest_pos)
    u._place(u._obj2_qadr, u._rest_pos2)
    u._place(u._obj3_qadr, u._rest_pos3)
    import mujoco
    mujoco.mj_forward(u.model, u.data)
    for nm in ("test_object1", "test_object2", "test_object3"):
        print(f"[stage2]   {nm}: 視線から {u._gaze_angle_to(nm):.1f} 度")
    saved, drift = scene_io.save(sc, env=env, hands=hands, settle_seconds=3.0)
    close_env(env)
    print(f"[stage2] 保存: {saved['name']}  安定確認: {drift}")
    return saved


if __name__ == "__main__":
    make_xml()
    for side in (+1, -1):
        make_scene(side)
