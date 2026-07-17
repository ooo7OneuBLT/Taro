"""reset()直後・何もstepする前の初期状態を1枚レンダリングして目視確認する。"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import mujoco
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from d_beta_sitting_env import BetaSittingEnv

OUT = os.path.abspath(os.path.join(_HERE, os.pardir, "logs", "video"))
W, H = 640, 480


def main():
    env = BetaSittingEnv()
    env.reset(seed=0)   # ここまでしかしない＝一切step前

    hip = env.unwrapped.data.body("hip").xpos.copy()
    mimo_loc = env.unwrapped.data.body("mimo_location").xpos.copy()
    # 床(floor geom)のz位置と、体の各部位の最下点(z最小)を見て、浮いているか判定。
    zmin = 1e9
    for i in range(env.unwrapped.model.nbody):
        name = mujoco.mj_id2name(env.unwrapped.model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        if name.startswith("beta_") or name in ("world",):
            continue
        zmin = min(zmin, float(env.unwrapped.data.body(i).xpos[2]))
    print(f"hip位置 = {np.round(hip, 3)}")
    print(f"mimo_location位置 = {np.round(mimo_loc, 3)}")
    print(f"アルファの体の中で最も低いbodyのz座標 = {zmin:.3f}（0に近いほど床に近い/離れているほど浮いている）")

    env.unwrapped.model.vis.global_.offwidth = W
    env.unwrapped.model.vis.global_.offheight = H
    ren = mujoco.Renderer(env.unwrapped.model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 1.8
    cam.elevation = -10
    cam.azimuth = 90
    cam.lookat = [0.15, 0.0, 0.2]
    ren.update_scene(env.unwrapped.data, camera=cam)
    img = ren.render()

    os.makedirs(OUT, exist_ok=True)
    png = os.path.join(OUT, "beta_sitting_initial.png")
    cv2.imwrite(png, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print("png:", png)


if __name__ == "__main__":
    main()
