# -*- coding: utf-8 -*-
"""環境の物（柵）と太郎の体の大きさが合っているかを測る。

【なぜ要るか、2026-07-30】柵の寸法（±31cm × ±16cm）は**新生児に合わせて**
決めたもので、「体を育てながら学習する」実験を始めたときに見直されなかった。
実測すると4ヶ月の体は柵の左右幅の97%を占め（余裕8mm）、
目と頭が柵の柱に接触したまま学習していた。
⇒ 落とし穴チェックリスト 項84／人間模倣からの逸脱リスト（見えない柵）

使い方:
    .venv/Scripts/python.exe run/tools/check_fence.py              # 0〜4ヶ月を0.25刻み
    .venv/Scripts/python.exe run/tools/check_fence.py 0 2 4        # 月齢を指定
    .venv/Scripts/python.exe run/tools/check_fence.py --scene 新生児_仰向け_柵なし

条件Cの学習ステップも併記する（18000回で0→4ヶ月なので step = 月齢/4×18000）。
  ⇒「何回目から柵に当たり始めるか」を、指標が崩れ始めた回と突き合わせられる。
"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8")
_R = r"C:\claude\AI\Taro"
for p in ("run/scene_tools", "D/scripts", "taro_core/src/body", "taro_core/src/brain",
          "taro_core/src/senses", "taro_core/src/wrapper"):
    sys.path.insert(0, os.path.join(_R, p))
os.chdir(_R)

import numpy as np, mujoco
import e_scene

HX, HY, H, T = 0.31, 0.16, 0.225, 0.024
IN_X, IN_Y = (HX - T / 2) * 2, (HY - T / 2) * 2

argv = sys.argv[1:]
scene = "新生児_仰向け_柵あり"
if "--scene" in argv:
    i = argv.index("--scene")
    scene = argv[i + 1]
    argv = argv[:i] + argv[i + 2:]
ages = [float(a) for a in argv] if argv else [i * 0.25 for i in range(17)]

print(f"シーン: {scene}")
print(f"柵の内側: 頭足 {IN_X*100:.1f}cm / 左右 {IN_Y*100:.1f}cm / 高さ {H*100:.1f}cm")
print()
print(f"{'月齢':>5}{'C条件の回数':>11}{'体(頭足)':>10}{'体(左右)':>10}"
      f"{'余裕(頭足)':>11}{'余裕(左右)':>11}{'当たる部位':>12}  内訳")
print("-" * 100)

for age in ages:
    sc = e_scene.load(scene)
    sc["body"]["age_months"] = age
    sc["fingerprint"] = None
    env, _sc = e_scene.build(sc, seed=0, verbose=False)
    u = env.unwrapped
    m, d = u.model, u.data
    mujoco.mj_forward(m, d)

    # 体の geom だけ集める。柵・床は worldbody 直付け（body_id==0）なので落ちる。
    #   注意：test_object は body を持つが (3.5, 3.0) に置かれた環境物なので除く
    #     （これを入れると体の広がりが376cmになる。2026-07-30 に実際に踏んだ）
    body_geoms = [i for i in range(m.ngeom)
                  if m.geom_bodyid[i] != 0
                  and "test_object" not in (m.body(m.geom_bodyid[i]).name or "")]
    pos = d.geom_xpos[body_geoms]
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    bx, by, bz = hi - lo

    hit = {}
    for c in range(d.ncon):
        n1 = m.geom(d.contact[c].geom1).name or f"#{d.contact[c].geom1}"
        n2 = m.geom(d.contact[c].geom2).name or f"#{d.contact[c].geom2}"
        if "fence" in n1 or "fence" in n2:
            part = (n2 if "fence" in n1 else n1).replace("geom:", "")
            hit[part] = hit.get(part, 0) + 1
    step = int(age / 4.0 * 18000)
    mark = "" if hit else "  "
    print(f"{age:>5.2f}{step:>11,}{bx*100:>9.1f}cm{by*100:>9.1f}cm"
          f"{(IN_X-bx)*100:>+10.1f}cm{(IN_Y-by)*100:>+10.1f}cm"
          f"{mark}{len(hit):>9}種  {', '.join(sorted(hit)) if hit else '—'}")
    env.close()
