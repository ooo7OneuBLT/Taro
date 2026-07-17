"""
ベータ（操り人形）を実際に動かして第三者視点で撮る。数字でなく目で見る（見た目が壊れて
いないか・どう動くか）。失敗しても映像を残す。

注意：今のベータは全関節削除＝1個の剛体なので「脚で歩く」ことはできない（滑って移動する
だけ）。左右に動かして、egomotionで意味のある「相手が左右に動く」映像を作る。
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import mujoco
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from d_beta_puppet import build_puppet_only_model

OUT = os.path.abspath(os.path.join(_HERE, os.pardir, "logs", "video"))
W, H = 640, 480


def main():
    model = build_puppet_only_model()
    data = mujoco.MjData(model)
    ax = [model.actuator(f"puppet_{a}").id for a in "xyz"]

    ren = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 2.2
    cam.elevation = -12
    cam.azimuth = 90          # 前方から見る＝左右(y)移動が画面の横移動に見える

    # 落ち着かせ
    for _ in range(100):
        mujoco.mj_step(model, data)

    frames = []
    N = 120
    for i in range(N):
        phase = i / N
        # 左右(y)に往復＋前後(x)に少し＋上下(z)に小さくバウンド（歩行"風"の動き）
        y = 0.45 * np.sin(2 * np.pi * phase)
        x = 0.10 * np.sin(4 * np.pi * phase)
        z = 0.03 * abs(np.sin(4 * np.pi * phase))
        data.ctrl[ax] = [x, y, z]
        for _ in range(6):
            mujoco.mj_step(model, data)
        cam.lookat = data.body("beta_mimo_location").xpos.copy()
        ren.update_scene(data, camera=cam)
        frames.append(ren.render())

    os.makedirs(OUT, exist_ok=True)
    mp4 = os.path.join(OUT, "beta_puppet_thirdperson.mp4")
    vw = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), 20, (W, H))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()

    # 見た目確認用の静止画シート（4フレーム横並び）＝画像で直接目視できる
    idxs = [0, N // 4, N // 2, 3 * N // 4]
    sheet = np.concatenate([frames[k] for k in idxs], axis=1)
    png = os.path.join(OUT, "beta_puppet_sheet.png")
    cv2.imwrite(png, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    print("mp4:", mp4)
    print("png:", png)


if __name__ == "__main__":
    main()
