"""目だけの場面を描いて並べ、速さを測る（脳は回さない。DINOv2 も使わない）。

  python -m core_v2.tools.eye_world_check [--out G/logs/目だけの場面/確かめ]

出すもの（out フォルダ）：
* 場面の一覧.png … 段階2bの左・右、段階6（2つずつ）、MIMo の左目の画像（同じ場面、G/data/目のテスト/）、
                   一覧の物（くつ・おさら・おわん・かばん・がおお・ばす・でんしゃ・とけい・コップ）を机の上に並べた所、親の体と机
* 速さ.json      … 1枚描くのにかかる時間、環境の1歩（描かない）、脳の1回ぶん（10歩＋1枚）

回す前の予想（2026-10-18、クラウドの CPU・osmesa）：1枚 約16ミリ秒（最初の1枚は約0.3秒）、描かない1歩 0.1ミリ秒未満、
脳の1回ぶんの環境 約17ミリ秒。MIMo の体ありでは環境だけで1回 約40ミリ秒（toy_env の記録：1歩 4.3ミリ秒×10歩＋描画）。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np

os.environ.setdefault("TARO_EYE_LOG", "off")

from .. import world_eye  # noqa: E402
from ..world_eye import scene as scene_mod  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
S2L, S2R, S6 = ("段階2b_3物体_ボール左_2026-10-15", "段階2b_3物体_ボール右_2026-10-15",
                "段階6_スミスとユー_6物体_2026-10-17")
CATALOGUE = ["present_shoe", "present_cup", "present_plate", "present_bowl", "present_bag", "present_lion",
             "present_bus", "present_train", "present_clock"]


def label(img, text):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    ImageDraw.Draw(im).text((4, 4), text, fill=(255, 255, 255))
    return np.asarray(im)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("G", "logs", "目だけの場面", "確かめ"))
    a = ap.parse_args(argv)
    out = a.out if os.path.isabs(a.out) else os.path.join(REPO, a.out)
    os.makedirs(out, exist_ok=True)
    tiles, speed = [], {}

    for sc in (S2L, S2R):
        e = world_eye.build_env(sc, None, seed=0, silent=True, log_dir=None)
        e.reset(seed=0)
        tiles.append(label(e.render_eye(), sc[:9] + ("L" if sc == S2L else "R")))
        e.close()
    mimo = sorted(glob.glob(os.path.join(REPO, "G", "data", "目のテスト", f"{S2L}_seed0_t00.npz")))
    if mimo:
        tiles.append(label(np.load(mimo[0])["img_l"], "MIMo 2b L (other gaze)"))
    e = world_eye.build_env(S6, None, seed=0, log_dir=None,
                            world_extra={"pairs": {"schedule": [{"t0": 0.0, "t1": 9.0, "left": 0, "right": 1}]}})
    e.reset(seed=0)
    e.step(None)
    tiles.append(label(e.render_eye(), "stage6 pair 0|1"))
    e.close()

    # 一覧の物：机の上に3つずつ並べ、親の体も出す。目を少し下へ向ける
    e = world_eye.build_env(S2L, None, seed=0, silent=True, log_dir=None,
                            world_extra={"eye_world": {"table": True, "parent_figure": True}})
    e.reset(seed=0)
    for nm in ("test_object1", "test_object2", "test_object3"):
        e.hide(nm)
    for k in range(3):
        for i, nm in enumerate(CATALOGUE[3 * k:3 * k + 3]):
            y = -0.115 + 0.115 * i
            pos = scene_mod.EYE_MID + np.array([-0.30, y, -0.06])
            pos[2] = 0.33 + 0.035
            e.show(nm, pos)
        e.set_eye(0.0, -12.0)
        tiles.append(label(e.render_eye(), "catalogue " + ",".join(n[8:] for n in CATALOGUE[3 * k:3 * k + 3])))
        for nm in CATALOGUE[3 * k:3 * k + 3]:
            e.hide(nm)
    e.set_eye(0.0, 15.0)
    tiles.append(label(e.render_eye(), "parent & table"))

    # 速さ
    t = time.perf_counter()
    for _ in range(50):
        e.render_eye()
    speed["render_ms"] = round((time.perf_counter() - t) / 50 * 1000, 2)
    t = time.perf_counter()
    for _ in range(1000):
        e.step(None)
    speed["step_ms_no_render"] = round((time.perf_counter() - t), 4)
    t = time.perf_counter()
    for _ in range(30):
        for _k in range(10):
            obs, *_ = e.step(None)
        obs["eye_left"]
    speed["brain_tick_env_ms"] = round((time.perf_counter() - t) / 30 * 1000, 2)
    speed["px"] = e.px
    speed["MUJOCO_GL"] = os.environ.get("MUJOCO_GL")
    e.close()

    from PIL import Image
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[i:i + 4], axis=1) for i in range(0, len(tiles), 4)]
    Image.fromarray(np.concatenate(rows, axis=0)).save(os.path.join(out, "場面の一覧.png"))
    with open(os.path.join(out, "速さ.json"), "w", encoding="utf-8") as f:
        json.dump(speed, f, ensure_ascii=False, indent=1)
    print(json.dumps(speed, ensure_ascii=False), "→", out)
    return speed


if __name__ == "__main__":
    main()
