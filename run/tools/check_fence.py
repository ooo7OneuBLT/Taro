# -*- coding: utf-8 -*-
"""★環境の物（柵）と太郎の体の大きさが合っているかを測る。

【なぜ要るか、2026-07-30】柵の寸法（±31cm × ±16cm）は**新生児に合わせて**
決めたもので、「体を育てながら学習する」実験を始めたときに見直されなかった。
実測すると4ヶ月の体は柵の左右幅の★97%を占め（余裕★8mm）、
★目と頭が柵の柱に接触したまま学習していた。
⇒ 落とし穴チェックリスト 項84／人間模倣からの逸脱リスト（見えない柵）

使い方:  .venv/Scripts/python.exe run/tools/check_fence.py
"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8")
_R = r"C:\claude\AI\Taro"
for p in ("E/scripts", "D/scripts", "taro_core/src/body", "taro_core/src/brain",
          "taro_core/src/senses", "taro_core/src/wrapper"):
    sys.path.insert(0, os.path.join(_R, p))
os.chdir(_R)

import numpy as np, mujoco
import e_scene

HX, HY, H, T = 0.31, 0.16, 0.225, 0.024
IN_X, IN_Y = (HX - T / 2) * 2, (HY - T / 2) * 2

for age in (0.0, 2.0, 4.0):
    sc = e_scene.load("新生児_仰向け_柵あり")
    sc["body"]["age_months"] = age
    sc["fingerprint"] = None
    env, _sc = e_scene.build(sc, seed=0, verbose=False)
    u = env.unwrapped
    m, d = u.model, u.data
    mujoco.mj_forward(m, d)

    # ★体＝worldbody(0) の直下にぶら下がる「太郎の根」から下の body 全部
    #   柵・床は worldbody 直付けの geom なので body_id==0 になる
    body_geoms = [i for i in range(m.ngeom) if m.geom_bodyid[i] != 0 and 'test_object' not in (m.body(m.geom_bodyid[i]).name or '')]
    pos = d.geom_xpos[body_geoms]
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    bx, by, bz = hi - lo

    print(f"--- 月齢 {age:.1f} ヶ月   （体のgeom {len(body_geoms)}個）")
    print(f"  体の広がり  頭足 {bx*100:5.1f}cm  左右 {by*100:5.1f}cm  高さ {bz*100:5.1f}cm")
    print(f"  体の範囲    x {lo[0]*100:+6.1f}〜{hi[0]*100:+6.1f}cm   y {lo[1]*100:+6.1f}〜{hi[1]*100:+6.1f}cm")
    print(f"  柵の内側    x {-IN_X/2*100:+6.1f}〜{IN_X/2*100:+6.1f}cm   y {-IN_Y/2*100:+6.1f}〜{IN_Y/2*100:+6.1f}cm  高さ {H*100:.1f}cm")
    print(f"  ★余裕      頭足 {(IN_X-bx)*100:+5.1f}cm   左右 {(IN_Y-by)*100:+5.1f}cm")
    print(f"  ★体が占める 頭足 {bx/IN_X*100:3.0f}%   左右 {by/IN_Y*100:3.0f}%")
    hit = {}
    for c in range(d.ncon):
        n1 = m.geom(d.contact[c].geom1).name or f"#{d.contact[c].geom1}"
        n2 = m.geom(d.contact[c].geom2).name or f"#{d.contact[c].geom2}"
        if "fence" in n1 or "fence" in n2:
            part = n2 if "fence" in n1 else n1
            hit[part] = hit.get(part, 0) + 1
    print(f"  ★柵に当たっている体の部位 {len(hit)}種 {dict(sorted(hit.items()))}")
    print()
    env.close()
