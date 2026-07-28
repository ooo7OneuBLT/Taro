"""柵の内側に太郎が収まっているかを月齢ごとに測る。

【なぜ要るか、2026-07-28】体年齢を4ヶ月に上げたところ、Viewerでの目視で
「仰向けになっていない」とユーザーから報告があった。ベビーサークルの柵は
**新生児（0ヶ月）の体に合わせて作った**もので、月齢を上げると体が大きくなるため
柵に当たっている可能性がある。

    柵の内寸  長辺 ±0.31m（頭〜足）／ 短辺 ±0.16m（左右）
    0ヶ月の身長 約49cm ／ 4ヶ月 約68cm（素のMIMo）

＝「まず数値で状態を出す」（落とし穴チェックリスト項60）。目視の報告だけで動かない。

使い方:
    E_AGES=0,4 .venv/Scripts/python.exe E/scripts/e_fence_fit_check.py
"""
import os
import sys
import warnings
import contextlib
import io

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
for _p in [os.path.join(_ROOT, "D", "scripts"), os.path.join(_ROOT, "MIMo"),
           os.path.join(_ROOT, "taro_core"),
           os.path.join(_ROOT, "taro_core", "src", "body"), _HERE]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

AGES = [float(x) for x in os.environ.get("E_AGES", "0,4").split(",")]
SETTLE_SEC = float(os.environ.get("E_SETTLE", "3.0"))


def probe(age):
    from e_toy_env import (ToySupineEnv, infant_vision_params,
                           FENCE_HALF_X, FENCE_HALF_Y, FENCE_HEIGHT)
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env

    kw = body_kwargs_from_env(age, verbose=False)
    kw["flexion"] = True
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        env = ToySupineEnv(actuation_model=MuscleModel,
                           vision_params=infant_vision_params(acuity_age=max(age, 0.5)),
                           age=age, toy=True, vor=True, orient=False, **kw)
        env.reset(seed=0)
    u = env.unwrapped
    m, d = u.model, u.data
    dt = float(m.opt.timestep) * int(u.frame_skip)
    a = np.zeros(env.action_space.shape[0], dtype=np.float32)
    for _ in range(int(SETTLE_SEC / dt)):
        env.step(a)

    # 太郎の体だけの広がり（柵・床・おもちゃを除く）
    skip = ("floor", "wall", "fence", "object", "toy", "target", "world")
    ids = [i for i in range(1, m.nbody)
           if not any(k in m.body(i).name for k in skip)]
    pts = np.array([d.xipos[i] for i in ids])
    xmin, xmax = float(pts[:, 0].min()), float(pts[:, 0].max())
    ymin, ymax = float(pts[:, 1].min()), float(pts[:, 1].max())
    zmin, zmax = float(pts[:, 2].min()), float(pts[:, 2].max())

    # 体幹の向き（仰向けかどうか）。hip の回転行列から体の上方向を取る
    hip_bid = int(m.body("hip").id)
    R = np.array(d.xmat[hip_bid]).reshape(3, 3)
    # MIMoの体幹ローカル軸のうち、腹側がどちらを向くか＝仰向けの判定に使う
    up_world = np.array([0.0, 0.0, 1.0])
    axes = {"x": R[:, 0], "y": R[:, 1], "z": R[:, 2]}
    tilts = {k: float(np.degrees(np.arccos(np.clip(np.dot(v, up_world), -1, 1))))
             for k, v in axes.items()}

    # 柵との接触
    contacts = 0
    depth = 0.0
    for c in range(d.ncon):
        con = d.contact[c]
        n1 = m.geom(con.geom1).name or ""
        n2 = m.geom(con.geom2).name or ""
        if "fence" in n1 or "fence" in n2:
            contacts += 1
            depth = min(depth, float(con.dist))
    env.close()
    return dict(age=age, x=(xmin, xmax), y=(ymin, ymax), z=(zmin, zmax),
                tilts=tilts, contacts=contacts, depth=depth,
                fx=FENCE_HALF_X, fy=FENCE_HALF_Y, fh=FENCE_HEIGHT)


def main():
    print("=" * 74)
    print(f" 柵の内側に太郎が収まっているか（{SETTLE_SEC:g}秒落ち着かせてから測定）")
    print("=" * 74)
    outs = []
    for age in AGES:
        print(f"  [{age:g}ヶ月...]")
        outs.append(probe(age))

    f = outs[0]
    print(f"\n柵の内寸： 長辺 ±{f['fx']*100:.0f}cm（頭〜足）"
          f"  短辺 ±{f['fy']*100:.0f}cm（左右）  高さ {f['fh']*100:.0f}cm")

    print("\n" + "=" * 74)
    print("1. 太郎の体の広がり（体節重心の範囲）と柵との余裕")
    print("-" * 74)
    print(f"{'月齢':>5}{'長辺の範囲[cm]':>20}{'長さ':>8}{'余裕':>9}"
          f"{'短辺の範囲[cm]':>20}{'幅':>8}{'余裕':>9}")
    for o in outs:
        xlen = (o["x"][1] - o["x"][0]) * 100
        ylen = (o["y"][1] - o["y"][0]) * 100
        mx = min(o["fx"] - abs(o["x"][0]), o["fx"] - abs(o["x"][1])) * 100
        my = min(o["fy"] - abs(o["y"][0]), o["fy"] - abs(o["y"][1])) * 100
        print(f"{o['age']:>5g}"
              f"{o['x'][0]*100:>9.1f}〜{o['x'][1]*100:>6.1f}{xlen:>8.1f}{mx:>+9.1f}"
              f"{o['y'][0]*100:>9.1f}〜{o['y'][1]*100:>6.1f}{ylen:>8.1f}{my:>+9.1f}")
    print("  （余裕が負なら柵の外にはみ出している）")

    print("\n2. 柵との接触")
    print("-" * 74)
    print(f"{'月齢':>5}{'接触の数':>12}{'めり込み[mm]':>16}")
    for o in outs:
        print(f"{o['age']:>5g}{o['contacts']:>12}{o['depth']*1000:>16.2f}")

    print("\n3. 体幹の向き（仰向けかどうか）")
    print("-" * 74)
    print("   hip のローカル軸が真上（+Z）から何度ずれているか")
    print(f"{'月齢':>5}{'x軸':>10}{'y軸':>10}{'z軸':>10}{'高さ[cm]':>12}")
    for o in outs:
        t = o["tilts"]
        print(f"{o['age']:>5g}{t['x']:>10.1f}{t['y']:>10.1f}{t['z']:>10.1f}"
              f"{o['z'][0]*100:>7.1f}〜{o['z'][1]*100:>4.1f}")
    print("  ★同じ姿勢なら月齢が違っても角度はほぼ同じになるはず。")
    print("    大きくずれていたら、体が柵に当たって姿勢が変わっている")


if __name__ == "__main__":
    main()
