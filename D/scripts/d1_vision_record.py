"""
太郎の「目」に、動いている養育者がどう映るかを記録する（グラウンディング用）。

【なぜ・2026-07-16】触覚だけの路線が壁に当たり、視覚に戻ってきた。まず数値の前に、
**太郎の目に相手がどう見えているか**を目で確かめる（[[feedback-watch-dont-just-measure]]：
指標の前に必ず動画で見る）。描画バグ修正後（`d1_carer_vision_env.get_vision_obs` が生APIで
眼球を直接描画）の、正しい一人称視点で記録する。

【何を映すか】仰向けの太郎の目は上を向く＝**顔の上**に来た養育者だけが見える（実測済み：
胸を狙うと空だけ・顔の上だと視界の24%が赤カプセル）。そこで養育者を顔の上で
  ①左右にゆっくり動かす（太郎の視野を横切る）
  ②近づく→遠ざかる（顔に迫る＝あやし/カンガルーケア）
の2つの分かりやすい動きをさせ、太郎の**左目の映像**を動画にする。

使い方: python d1_vision_record.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "C"))
import paths
paths.setup_brain_path()
sys.path.insert(0, paths.MIMO_DIR)

import numpy as np
import cv2
import gymnasium as gym
import mimoEnv  # noqa
from gymnasium.envs.registration import register

from d1_carer_vision_env import CarerVisionEnv, lean_vision_params
from d_supine_env import infant_touch_params

register(id="CarerVisRec-v0", entry_point="d1_carer_vision_env:CarerVisionEnv", max_episode_steps=100000)

FACE_X = 0.37     # 太郎の顔の真上あたり（実測：左目 [0.374,-0.028,0.135]）
RES = 128         # 記録用の解像度（学習は64だが、見る用に少し上げる）


def carer_target(t):
    """養育者を顔の上で動かす。t=経過sim秒。前半=左右、後半=近づく/遠ざかる。"""
    if t < 10.0:                                  # ①左右に横切る
        y = 0.14 * np.sin(2 * np.pi * t / 5.0)
        return [FACE_X, y, 0.32]
    tt = t - 10.0                                 # ②近づく→遠ざかる
    z = 0.42 - 0.16 * (0.5 + 0.5 * np.sin(2 * np.pi * tt / 5.0))
    return [FACE_X, 0.0, z]


def main():
    env = gym.make("CarerVisRec-v0", vision_params=lean_vision_params(RES),
                   touch_params=infant_touch_params(2.0))
    obs, _ = env.reset(seed=0)
    u = env.unwrapped
    a = np.zeros(u.action_space.shape[0])          # 太郎は何もしない（相手の動きだけを見る）

    frames = []
    t = 0.0
    for i in range(20):                            # 20 sim秒
        for k in range(100):
            u.set_hand_target(carer_target(t))
            obs, r, te, tr, info = env.step(a)
            t += u.dt
            if k % 4 == 0:
                frames.append(obs["eye_left"].copy())

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                       "logs", "video", "taro_eye_view.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # 見やすいよう4倍に拡大（最近傍＝画素をそのまま大きく）
    big = [cv2.resize(f, (RES * 4, RES * 4), interpolation=cv2.INTER_NEAREST) for f in frames]
    h, w, _ = big[0].shape
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), (1.0 / u.dt) / 4, (w, h))
    for f in big:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    env.close()
    print(f"太郎の左目の映像を記録: {os.path.normpath(out)}  （{len(frames)}コマ）")
    print("前半10秒＝養育者が左右に横切る / 後半10秒＝近づく→遠ざかる")


if __name__ == "__main__":
    main()
