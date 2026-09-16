"""座位アルファ＋左右ベータを、実際の行動を入れながら第三者視点で撮る（目で確認）。"""
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
    env.reset(seed=0)
    na = env.action_space.shape[0]
    rng = np.random.default_rng(1)

    env.unwrapped.model.vis.global_.offwidth = W
    env.unwrapped.model.vis.global_.offheight = H
    ren = mujoco.Renderer(env.unwrapped.model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 1.8
    cam.elevation = -15
    cam.azimuth = 90
    cam.lookat = [0.15, 0.0, 0.3]

    frames = []
    N = 150
    for i in range(N):
        act = rng.uniform(-0.6, 0.6, na).astype(np.float32)  # アルファは運動性喃語(ランダム)
        y = 0.35 * np.sin(2 * np.pi * i / N)                  # ベータは左右往復
        env.unwrapped.set_beta_target([0.3, y, 0.35])
        env.step(act)
        ren.update_scene(env.unwrapped.data, camera=cam)
        frames.append(ren.render())

    os.makedirs(OUT, exist_ok=True)
    mp4 = os.path.join(OUT, "beta_sitting_thirdperson.mp4")
    vw = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), 20, (W, H))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    idxs = [0, N // 3, 2 * N // 3, N - 1]
    sheet = np.concatenate([frames[k] for k in idxs], axis=1)
    png = os.path.join(OUT, "beta_sitting_sheet.png")
    cv2.imwrite(png, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print("mp4:", mp4)
    print("png:", png)


if __name__ == "__main__":
    main()
