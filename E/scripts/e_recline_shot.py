"""リクライニングの姿勢を真横から撮って並べる（Viewerを操作せずに姿勢を確かめる）。

【なぜ要るか、2026-07-28】リクライニング環境を作ったが数値では
「体幹 -22.6度」「骨盤が x=-1.0m へ飛ぶ」といった値が出て、
**体がどこにいるのか把握できない**。Viewerで見るのが確実だが、
カメラ操作が要り、条件を並べて比べられない。
→ 真横から固定カメラで撮って横に並べる。

使い方:
    E_RECLINES=0,45,70 .venv/Scripts/python.exe E/scripts/e_recline_shot.py
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

import numpy as np      # noqa: E402
import mujoco           # noqa: E402
import matplotlib       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib import font_manager  # noqa: E402

for _f in ("Meiryo", "Yu Gothic", "MS Gothic", "IPAexGothic"):
    if any(_f in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

AGE = float(os.environ.get("E_AGE", "4.0"))
RECLINES = [float(x) for x in os.environ.get("E_RECLINES", "0,45,70").split(",")]
SETTLE = float(os.environ.get("E_SETTLE", "3.0"))
W, H = 640, 480
OUT = os.path.join(_ROOT, "E", "figures", "リクライニングの姿勢_2026-07-28.png")


def shot(rec):
    from e_toy_env import ToySupineEnv, infant_vision_params
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env

    kw = body_kwargs_from_env(AGE, verbose=False)
    kw["flexion"] = True
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        env = ToySupineEnv(actuation_model=MuscleModel,
                           vision_params=infant_vision_params(acuity_age=AGE),
                           age=AGE, toy=True, vor=True, orient=False,
                           recline_deg=rec, **kw)
        env.reset(seed=0)
    u = env.unwrapped
    m, d = u.model, u.data
    dt = float(m.opt.timestep) * int(u.frame_skip)
    a = np.zeros(env.action_space.shape[0], dtype=np.float32)
    for _ in range(int(SETTLE / dt)):
        env.step(a)

    def bpos(n):
        return np.array(d.xpos[int(m.body(n).id)], dtype=float)

    hip, head = bpos("hip"), bpos("head")
    v = head - hip
    trunk = float(np.degrees(np.arctan2(v[2], np.linalg.norm(v[:2]))))

    # 真横から撮る（太郎の体軸は x 方向なので、y 方向から見る）
    # ⚠️MIMoの視覚は小さい画像（64x64程度）なので、モデルの offscreen バッファも
    #   小さく設定されている。大きく撮るには先に広げておく必要がある
    #   （そうしないと "Error: offscreen framebuffer is too small" で落ちる）。
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), W)
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), H)
    ren = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    center = 0.5 * (hip + head)
    cam.lookat[:] = [float(center[0]), 0.0, float(max(center[2], 0.1))]
    cam.distance = 1.1
    cam.azimuth = 90.0     # y軸方向から見る＝真横
    cam.elevation = -5.0
    ren.update_scene(d, cam)
    img = ren.render().copy()
    ren.close()
    env.close()
    return dict(rec=rec, img=img, trunk=trunk,
                hip=hip, head=head)


def main():
    print("=" * 70)
    print(" リクライニングの姿勢を真横から撮る")
    print("=" * 70)
    outs = []
    for rec in RECLINES:
        print(f"  [{rec:g}度...]")
        outs.append(shot(rec))

    n = len(outs)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.4), squeeze=False)
    for j, o in enumerate(outs):
        ax = axes[0, j]
        ax.imshow(o["img"])
        ax.set_title(f"設定 {o['rec']:.0f}度   実際の体幹 {o['trunk']:+.1f}度\n"
                     f"骨盤(x={o['hip'][0]:+.2f}, z={o['hip'][2]:+.2f})  "
                     f"頭(x={o['head'][0]:+.2f}, z={o['head'][2]:+.2f})",
                     fontsize=10)
        ax.axis("off")
    fig.suptitle("リクライニング環境の姿勢（真横から・3秒落ち着かせた後）",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=120, bbox_inches="tight")
    print(f"\n  保存: {os.path.abspath(OUT)}")
    for o in outs:
        print(f"    {o['rec']:>4.0f}度 → 体幹 {o['trunk']:+6.1f}度")


if __name__ == "__main__":
    main()
