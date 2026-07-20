"""【太郎が"実際に受け取っている"視覚を見る】視覚obsそのものを画像・動画に出す。

【なぜ要るか】
既存の一人称動画（`d_c5_motor_quality.py run_eyeview`）は**眼球カメラを生で描画**したもので、
**太郎の脳に入る映像ではない**。実際の入力は、その後に
  ①解像度 128×128 に落とし  ②Mayer et al.(1995)の月齢別視力からMTFを作って高周波を落とす
という処理を経ている（新生児の視力は成人の約1/8）。**この差を見ないと「太郎に見えているか」は判断できない**。

【出すもの】横に3つ並べた比較画像
  [第三者視点]  [生の眼球カメラ]  [★太郎の実入力(左目)] [★太郎の実入力(右目)]
                  ↑acuityなし        ↑acuity適用済み＝脳が受け取るもの

⚠️視覚obsは `VISION_MIN_DT`(0.1sim秒=10Hz) で間引いてキャッシュされるので、動画では
   同じ画が数コマ続く。**これは実際に太郎が受け取っている更新頻度そのもの**なので直さない。

使い方:
  python e_eye_input.py [n_ticks] [seed]
  環境変数 E_TOY_OBJ=0 / E_FENCE=0 で「貧しい環境」にできる
出力: E/logs/video/  （PNG数枚 + mp4）
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "D", "scripts"))
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "C", "scripts"))
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "taro_core"))
import paths  # noqa: E402
paths.setup_brain_path()
sys.path.insert(0, paths.MIMO_DIR)

import mimoEnv  # noqa: F401,E402
import mujoco  # noqa: E402
import torch  # noqa: E402
import cv2  # noqa: E402
import d_c5_motor_quality as mq  # noqa: E402

PANEL = 240          # 各パネルの表示サイズ[px]
OUT_DIR = os.path.join(_HERE, os.pardir, "logs", "video")


def label(img, text):
    """パネル上端に説明を焼き込む（どれが何か分からなくなるのを防ぐ）。"""
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                cv2.LINE_AA)
    return img


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    os.makedirs(OUT_DIR, exist_ok=True)
    env, brain, fusion, emb_proj, cereb, n_act = mq.build("off", age=0)
    policy = mq.make_policy(brain, fusion, emb_proj, cereb, n_act, babble=True)
    raw = env.unwrapped
    m, d = raw.model, raw.data
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), 640)
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), 480)

    third_ren = mujoco.Renderer(m, height=480, width=640)
    third_cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(m, third_cam)
    third_cam.distance *= 0.32
    third_cam.elevation = -35.0
    # 生の眼球カメラ（acuityなし）＝比較対象
    raw_ren = mujoco.Renderer(m, height=128, width=128)
    raw_cam = mujoco.MjvCamera(); raw_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    raw_cam.fixedcamid = int(m.camera("eye_left").id)

    torch.manual_seed(seed); np.random.seed(seed)
    obs, _ = env.reset(seed=seed)
    hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)
    frames = []
    SUB = 5      # 物理5ステップ(50ms)に1コマ＝視覚の更新(10Hz)とほぼ同じ刻み
    ctrl = None; step_i = 0
    tag = f"e1eyeinput_{'poor' if not raw._toy and not raw._fence else 'rich'}_seed{seed}"
    print(f"\n録画開始（{n}tick／おもちゃ={'有' if raw._toy else '無'}／柵={'有' if raw._fence else '無'}）")

    for tick in range(n):
        for k in range(mq.K):
            if k % mq.CTRL_M == 0:
                a, hidden = policy(obs, prev_a, hidden)
                ctrl = mq.rescale_action(a, env.action_space); prev_a = a
            obs, r, term, trunc, info = env.step(ctrl)
            if step_i % SUB == 0:
                third_cam.lookat[:] = d.body("upper_body").xpos
                third_ren.update_scene(d, camera=third_cam)
                third = cv2.resize(third_ren.render().copy(), (PANEL, PANEL))
                raw_ren.update_scene(d, camera=raw_cam)
                eye_raw = cv2.resize(raw_ren.render().copy(), (PANEL, PANEL),
                                     interpolation=cv2.INTER_NEAREST)
                # ★太郎が実際に受け取っている視覚obs（acuity適用済み・キャッシュ込み）
                vis = raw.get_vision_obs()
                inl = cv2.resize(np.asarray(vis["eye_left"]).copy(), (PANEL, PANEL),
                                 interpolation=cv2.INTER_NEAREST)
                inr = cv2.resize(np.asarray(vis["eye_right"]).copy(), (PANEL, PANEL),
                                 interpolation=cv2.INTER_NEAREST)
                row = np.hstack([label(third, "3rd person"),
                                 label(eye_raw, "eye cam (raw)"),
                                 label(inl, "TARO INPUT L (acuity)"),
                                 label(inr, "TARO INPUT R (acuity)")])
                frames.append(row)
            step_i += 1
            if term or trunc:
                obs, _ = env.reset(); hidden = brain.init_motor_hidden()
                prev_a = torch.zeros(n_act); break

    # 代表フレームをPNGで（動画を開かなくても即見られるように）
    idx = np.linspace(0, len(frames) - 1, min(5, len(frames))).astype(int)
    for i, fi in enumerate(idx):
        p = os.path.join(OUT_DIR, f"{tag}_{i:02d}.png")
        cv2.imwrite(p, cv2.cvtColor(frames[fi], cv2.COLOR_RGB2BGR))
        print("PNG:", p)
    h, w = frames[0].shape[:2]
    pv = os.path.join(OUT_DIR, f"{tag}.mp4")
    vw = cv2.VideoWriter(pv, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    print("動画:", pv, f"（{len(frames)}コマ）")
    env.close()


if __name__ == "__main__":
    main()
