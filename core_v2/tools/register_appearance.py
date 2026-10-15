"""意味の辞書の登録道具（段階2・工程3）：「語 → その物の見た目の荷札」を1件ずつ記録する。

仕様書 §3 は「場面の最初に各物を正面に見せて、目の入口が出した荷札を1回記録する」だが、
環境には1つの物だけを正面に置く手軽な道が無い（物はリセットのたびに視線の正面を基準に
置き直される）。そこで、場面をそのまま回し、目の入口の切り出しのうち**その物の画素位置に
重なるもの**（カメラの投影で求めた位置から15画素以内）を FRAMES コマぶん集めて平均し、
固定の行列で荷札にする（雑音は足さない。雑音は意味の辞書が引くたびに足す）。

使い方：
  python -m core_v2.tools.register_appearance [場面名] [出力先]
既定：場面＝段階2_3物体_ボール左_2026-10-15、出力＝G/models/semantic_table.pt
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

from .. import run_v2
from ..tagging import PROJECTION_SEED, appearance_tag

WORDS = {"ぼおる": "test_object1", "こっぷ": "test_object2"}
FRAMES = 10
SKIP_TICKS = 5          # 最初の0.5秒は目立ちの地図が落ち着かないので飛ばす
MATCH_PX = 15.0


def project(u, body):
    """物の中心を左目の画像（224画素）へ投影する。"""
    cid = int(u.model.camera("eye_left").id)
    eye = u.data.cam_xpos[cid]
    R = u.data.cam_xmat[cid].reshape(3, 3)
    v = u.data.body(body).xpos - eye
    x, y, z = v @ R[:, 0], v @ R[:, 1], -(v @ R[:, 2])
    f = 112.0 / np.tan(np.radians(u.model.cam_fovy[cid]) / 2)
    return 112 + f * x / z, 112 - f * y / z


def collect(eyein, env, frames=FRAMES, seed=0):
    """場面を回して、物ごとの見た目（384）を集める。{body: [見た目, ...]}。"""
    u = env.unwrapped
    env.reset(seed=seed)
    eyein.reset()
    act = run_v2.zero_action(env)
    got = {b: [] for b in run_v2.OBJECTS.values()}
    others = []
    for i in range(SKIP_TICKS + frames):
        for _ in range(run_v2.K):
            obs, *_ = env.step(act)
        eyein.see(obs["eye_left"])
        if i < SKIP_TICKS:
            continue
        pos = {b: project(u, b) for b in got}
        for d in eyein.last_dets:
            dist = {b: np.hypot(d["pos"][0] - p[0], d["pos"][1] - p[1]) for b, p in pos.items()}
            b = min(dist, key=dist.get)
            (got[b] if dist[b] < MATCH_PX else others).append(np.asarray(d["appearance"], float))
    return got, others


def main(scene="段階2_3物体_ボール左_2026-10-15", out=None):
    from ..boxes.eyein import Eyein
    out = out or os.path.join(run_v2.REPO, "G", "models", "semantic_table.pt")
    run_v2.seed_all(0)
    eyein = Eyein()
    eyein.setup()
    env = run_v2.build_env(scene, None, seed=0)
    try:
        got, others = collect(eyein, env)
    finally:
        run_v2.close_env(env)
    table, counts = {}, {}
    for w, body in WORDS.items():
        if len(got[body]) < FRAMES // 2:
            raise RuntimeError(f"{w}（{body}）が {len(got[body])} コマしか切り出せなかった")
        table[w] = appearance_tag(np.mean(got[body], axis=0))
        counts[w] = len(got[body])
    # 確かめ：登録した荷札と、1コマごとの見た目の荷札の似かた
    print("登録した荷札と、各物の1コマごとの荷札の cos（平均）")
    for w, tag in table.items():
        row = {k: round(float(np.mean([float(appearance_tag(a) @ tag) for a in got[b]])), 3)
               for k, b in run_v2.OBJECTS.items() if got[b]}
        row["その他"] = round(float(np.mean([float(appearance_tag(a) @ tag) for a in others])), 3) if others else None
        print(f"  {w}: {row}")
    meta = {"scene": scene, "frames": counts, "projection_seed": PROJECTION_SEED,
            "code": None}
    try:
        from ..save import code_version
        meta["code"] = code_version()
    except Exception:  # noqa: BLE001
        pass
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"table": table, "meta": meta}, out)
    print(f"保存: {out}  {meta}")
    return out


if __name__ == "__main__":
    main(*sys.argv[1:])
