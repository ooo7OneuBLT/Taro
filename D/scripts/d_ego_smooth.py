"""なめらかな連続動画（1ステップ刻み）で、頭yawによるegomotionを両視点GIFにする。

10ステップ間引きでなく、数ステップ刻みで記録＝普通の動画のように見える。養育者は固定、
太郎が頭を左右にサイン波で振る。第三者視点＋一人称を横並びにしてGIF保存。
"""
import os, sys, warnings, math
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "taro_core"))
import paths; paths.setup_brain_path(); sys.path.insert(0, paths.MIMO_DIR)
import numpy as np, cv2, mujoco, gymnasium as gym, mimoEnv  # noqa
from gymnasium.envs.registration import register
from d1_carer_vision_env import CarerVisionEnv, lean_vision_params
from d_supine_env import infant_touch_params
import d_vision_egomotion as E
import d_ego_demo as D

register(id="CarerEgoSmooth-v0", entry_point="d1_carer_vision_env:CarerVisionEnv", max_episode_steps=100000)

N = 150          # 総ステップ
STRIDE = 2       # 何ステップごとに1コマ記録（1なら完全連続だが重い）
AMP = 0.6
PERIOD = 75      # 1往復のステップ数


def main():
    env = gym.make("CarerEgoSmooth-v0", vision_params=lean_vision_params(D.RES),
                   touch_params=infant_touch_params(2.0), hand_size=0.10, render_mode="rgb_array")
    env.reset(seed=0); u = env.unwrapped; na = u.action_space.shape[0]
    ren = mujoco.Renderer(u.model, 240, 240)
    D.reset_scene(u)
    for _ in range(20):
        u.set_hand_target([D.FACE_X, 0.0, D.Z_FIX]); env.step(np.zeros(na, np.float32))
    frames = []
    for s in range(N):
        ctrl = np.zeros(na, np.float32)
        ctrl[D.HEAD_YAW[0]] = AMP * math.sin(2 * math.pi * s / PERIOD)
        u.set_hand_target([D.FACE_X, 0.0, D.Z_FIX]); obs, *_ = env.step(ctrl)
        if s % STRIDE == 0:
            eye = cv2.resize(obs["eye_left"], (240, 240), interpolation=cv2.INTER_NEAREST)
            tp = D.third_person(ren, u.data)
            c = np.zeros((240, 488, 3), np.uint8); c[:, :240] = tp; c[:, 248:] = eye
            cv2.putText(c, "3rd person", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(c, "Taro eye", (254, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            frames.append(c)
    ren.close(); env.close()
    vdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "logs", "video")
    os.makedirs(vdir, exist_ok=True)
    mp4 = os.path.join(vdir, "ego_smooth_headyaw.mp4")
    gif = os.path.join(vdir, "ego_smooth_headyaw.gif")
    vw = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), 20, (488, 240))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    try:
        import imageio
        imageio.mimsave(gif, [f for f in frames], duration=0.05)
        print(f"GIF保存: {gif}", flush=True)
    except Exception as e:
        from PIL import Image
        Image.fromarray(frames[0]).save(gif, save_all=True,
            append_images=[Image.fromarray(f) for f in frames[1:]], duration=50, loop=0)
        print(f"GIF保存(PIL): {gif}", flush=True)
    print(f"MP4保存: {mp4}  総{len(frames)}コマ（{N}ステップ/{STRIDE}刻み）", flush=True)


if __name__ == "__main__":
    main()
