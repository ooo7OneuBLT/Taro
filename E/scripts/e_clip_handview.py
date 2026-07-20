"""【目視用】手が視野に入った瞬間だけを切り出して動画にする。

【なぜ要るか】
hand regard は 1〜3% しか起きない。等倍速のViewerで眺めていると **1分に1〜2回**しか
起きず、見逃す。そこで「手が視野に入った前後」だけを集めて短い動画にまとめ、
**数値が本当に行動として起きているか**を目で確かめられるようにする。
（Viewerでの通し観察と併用する。片方だけにしない：Viewerは"普段どうしているか"、
  この動画は"起きた瞬間に何をしていたか"を見るためのもの）

【出すもの】各イベントについて横に3つ並べたコマ
    [第三者視点]  [太郎の実入力(左目)]  [太郎の実入力(右目)]
  ・イベント前後 PRE/POST tick を含める（どう手を持ってきたかが見える）
  ・コマの左上にイベント番号と、そのtickで手が視野内かどうかを焼き込む

使い方:
  python e_clip_handview.py [n_ticks] [seed]
  環境変数 C5_CKPT で見るモデルを指定
出力: E/logs/video/e1clip_*.mp4 （＋代表コマのPNG）
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch

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
import cv2  # noqa: E402
import d_c5_motor_quality as mq  # noqa: E402
import e_toy_env as te  # noqa: E402
from e_hand_in_view import hand_in_view  # noqa: E402

PANEL = 240
PRE, POST = 3, 3        # イベントの前後何tickを含めるか
OUT_DIR = os.path.join(_HERE, os.pardir, "logs", "video")


def label(img, text, color=(255, 255, 255)):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return img


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    os.makedirs(OUT_DIR, exist_ok=True)
    half_fov = te.VISION_FOVY / 2.0
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

    torch.manual_seed(seed); np.random.seed(seed)
    obs, _ = env.reset(seed=seed)
    hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)

    ring = []            # 直近 PRE tick ぶんのコマを保持（イベント前を含めるため）
    clips = []           # 切り出したコマ列
    n_event = 0
    remain = 0           # イベント後、あと何tick録るか

    def make_frame(inview, tag):
        third_cam.lookat[:] = d.body("upper_body").xpos
        third_ren.update_scene(d, camera=third_cam)
        third = cv2.resize(third_ren.render().copy(), (PANEL, PANEL))
        raw._vision_cache = None
        vis = raw.get_vision_obs()
        l = cv2.resize(np.asarray(vis["eye_left"]).copy(), (PANEL, PANEL),
                       interpolation=cv2.INTER_NEAREST)
        r = cv2.resize(np.asarray(vis["eye_right"]).copy(), (PANEL, PANEL),
                       interpolation=cv2.INTER_NEAREST)
        col = (80, 255, 80) if inview else (200, 200, 200)
        return np.hstack([label(third, tag, col),
                          label(l, "TARO INPUT L", col),
                          label(r, "TARO INPUT R", col)])

    print(f"\n{n} tick 走査して、手が視野に入った瞬間の前後を切り出します…")
    for t in range(n):
        a, hidden = policy(obs, prev_a, hidden)
        ctrl = mq.rescale_action(a, env.action_space); prev_a = a
        for k in range(mq.K):
            obs, r_, term, trunc, info = env.step(ctrl)
            if term or trunc:
                break
        inview = bool(hand_in_view(m, d))
        if inview and remain == 0:
            n_event += 1
            clips.extend(ring)            # イベント前のコマを先に足す
            remain = POST + 1
        if remain > 0:
            clips.append(make_frame(inview, f"#{n_event} t={t} {'HAND IN VIEW' if inview else '-'}"))
            remain -= 1
            if inview:
                remain = POST             # 続いている間は延長
        else:
            ring.append(make_frame(inview, f"t={t}"))
            if len(ring) > PRE:
                ring.pop(0)
        if term or trunc:
            obs, _ = env.reset()
            hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)
            ring.clear()

    if not clips:
        print("手が視野に入る場面がありませんでした")
        env.close(); return
    ck = os.path.basename(os.environ.get("C5_CKPT", "model")).replace(".pt", "")
    tag = f"e1clip_{ck}_seed{seed}"
    h, w = clips[0].shape[:2]
    pv = os.path.join(OUT_DIR, f"{tag}.mp4")
    vw = cv2.VideoWriter(pv, cv2.VideoWriter_fourcc(*"mp4v"), 4, (w, h))   # 4fps＝じっくり見る
    for f in clips:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    for i, fi in enumerate(np.linspace(0, len(clips) - 1, min(4, len(clips))).astype(int)):
        p = os.path.join(OUT_DIR, f"{tag}_{i:02d}.png")
        cv2.imwrite(p, cv2.cvtColor(clips[fi], cv2.COLOR_RGB2BGR))
    print(f"  イベント {n_event} 回 / {n} tick（{100*n_event/n:.2f}%）")
    print(f"  動画: {pv}（{len(clips)}コマ・4fps）")
    print(f"  PNG: {tag}_00〜.png")
    env.close()


if __name__ == "__main__":
    main()
