"""
★egomotionデモ：自分の動き"だけ"で視界がどれだけ動くかを、両視点動画で見比べる。

【なぜ】腕運動版は視界がほぼ動かず実験が空振りと判明（ユーザーが動画で発見）。土台も不安定
（太郎が沈む・養育者が定位置1.4から降下してくる）だった。ここでは土台を安定化し、養育者を
固定して、自己運動の種類ごとに「視界の動き」と「相手が画面に残るか」を純粋に見せる。

【土台の安定化】各デモの頭で
  ・太郎を初期姿勢(init_position)にリセット＋速度ゼロ
  ・養育者を最初から定位置へワープ（降下の混入を断つ）
してから、指定の自己運動を命令する。養育者は毎ステップ位置制御で固定。

【見せる自己運動】①静止 ②腕 ③頭yaw(左右) ④頭nod(縦)。第三者視点+一人称の横並び動画と
コマ並べ画像を保存。ラベルは英字のみ（動画ライブラリが日本語非対応のため）。

使い方: python d_ego_demo.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "C"))
import paths; paths.setup_brain_path(); sys.path.insert(0, paths.MIMO_DIR)

import numpy as np
import cv2
import mujoco
import gymnasium as gym
import mimoEnv  # noqa
from gymnasium.envs.registration import register
from d1_carer_vision_env import CarerVisionEnv, lean_vision_params
from d_supine_env import infant_touch_params
import d_vision_egomotion as E

register(id="CarerEgoDemo-v0", entry_point="d1_carer_vision_env:CarerVisionEnv", max_episode_steps=100000)

RES = 32
FACE_X = 0.37
Z_FIX = 0.55    # 養育者の高さ。低いと真上のカプセルが顔(目)を隠すので少し上げる
SIZE = 0.10
K = 8
SUB = 25
HOME_Z = 1.40   # _HAND_HOME z（養育者スライド関節の原点）

# 動かす関節
ARMS = list(range(14, 72))
HEAD_YAW = [5]     # head_swivel（左右）
HEAD_NOD = [6]     # head_tilt（縦）


def third_person(ren, data):
    # 仰向けの太郎の顔・目は真上を向く→かなり見下ろす角度にしないと目と頭の向きが見えない。
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.30, 0.0, 0.16]; cam.distance = 0.82; cam.azimuth = 95.0; cam.elevation = -68.0
    ren.update_scene(data, camera=cam); return ren.render()


def reset_scene(u):
    """太郎を初期姿勢へ、養育者を定位置(FACE_X,0,Z_FIX)へワープ。降下も沈みも断つ。"""
    u.data.qpos[:] = u.init_position
    u.data.qvel[:] = 0.0
    for nm, world in (("carer_x", FACE_X - 0.10), ("carer_y", 0.0), ("carer_z", Z_FIX - HOME_Z)):
        adr = u.model.joint(nm).qposadr[0]
        u.data.qpos[adr] = world
    mujoco.mj_forward(u.model, u.data)


def demo(movable, scale, label):
    env = gym.make("CarerEgoDemo-v0", vision_params=lean_vision_params(RES),
                   touch_params=infant_touch_params(2.0), hand_size=SIZE, render_mode="rgb_array")
    env.reset(seed=0); u = env.unwrapped; na = u.action_space.shape[0]
    ren = mujoco.Renderer(u.model, 240, 240)
    reset_scene(u)
    # 数ステップ落ち着かせる（養育者を固定位置で保持）
    for _ in range(20):
        u.set_hand_target([FACE_X, 0.0, Z_FIX]); env.step(np.zeros(na, np.float32))
    drift = np.zeros(na, np.float32)
    if movable is not None:
        rng = np.random.default_rng(2)
        d = rng.standard_normal(len(movable)); d /= np.linalg.norm(d) + 1e-9
        drift[movable] = d.astype(np.float32)
    eyes, tps, reds = [], [], []
    for t in range(K):
        ctrl = (scale * (t + 1) / K * drift).astype(np.float32)
        for _ in range(SUB):
            u.set_hand_target([FACE_X, 0.0, Z_FIX]); obs, *_ = env.step(ctrl)
        eye = obs["eye_left"]
        eyes.append(cv2.resize(eye, (240, 240), interpolation=cv2.INTER_NEAREST))
        tps.append(third_person(ren, u.data))
        reds.append(E._red_pixels(eye))
    eyes_small = [cv2.resize(e, (RES, RES)) for e in eyes]
    chg = float(np.mean(np.abs(np.diff(np.stack(eyes_small).astype(np.float32), axis=0))))
    ren.close(); env.close()
    return eyes, tps, reds, chg


def save(label, eyes, tps, out_mp4, out_png):
    frames = []
    for j, (tp, eye) in enumerate(zip(tps, eyes)):
        c = np.zeros((240, 488, 3), np.uint8); c[:, :240] = tp; c[:, 248:] = eye
        cv2.putText(c, "3rd person", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(c, "Taro eye", (254, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(c, f"{label}  {j+1}/{K}", (6, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        frames.append(c)
    vw = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), 6, (488, 240))
    for f in frames:
        for _ in range(3):
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    # コンタクトシート（全コマ横並び）
    sheet = np.concatenate(frames, axis=0)
    cv2.imwrite(out_png, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def main():
    vdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "logs", "video")
    os.makedirs(vdir, exist_ok=True)
    cases = [("STILL", None, 0.0), ("ARMS", ARMS, 0.6),
             ("HEAD-YAW(L-R)", HEAD_YAW, 0.5), ("HEAD-NOD(updown)", HEAD_NOD, 0.5)]
    print("養育者は固定。自己運動だけで視界がどれだけ動くか（size0.10, z0.38固定）\n")
    print(f"{'motion':18s} {'eye-change':>10s} {'red min-max':>12s}")
    for label, mv, sc in cases:
        eyes, tps, reds, chg = demo(mv, sc, label)
        tag = label.split("(")[0].lower().replace("-", "_")
        save(label, eyes, tps, os.path.join(vdir, f"demo_{tag}.mp4"),
             os.path.join(vdir, f"demo_{tag}_sheet.png"))
        print(f"{label:18s} {chg:10.2f} {min(reds):5d}-{max(reds):<5d}   "
              f"-> demo_{tag}.mp4 / _sheet.png", flush=True)
    print("\n横並び画像(コンタクトシート)を見比べて、視界が動く&相手が残る動きを選ぶ。")


if __name__ == "__main__":
    main()
