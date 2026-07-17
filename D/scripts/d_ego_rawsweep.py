"""
★クリップに区切る前の、生の連続映像。親の左右移動＋太郎の頭yawを同時にゆっくりスイープし、
「親が視界から消える瞬間」を自分の目で確認するためのもの。数値や推測より先に、まず見る。

3パターンを続けて記録：
  A) 親だけ左右にスイープ（頭は動かさない）＝親の移動幅だけでの見え方
  B) 頭だけ左右にyaw（親は中央固定）＝egomotionだけでの見え方
  C) 親の左右移動＋頭yawを"同じ向き"に重ねる＝最悪ケース（視界から消えやすいはず）
各コマに現在の親のy位置・頭yaw角度・赤画素数を表示。両視点(第三者+一人称)。
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
import d_ego_leftright as L

register(id="EgoRawSweep-v0", entry_point="d1_carer_vision_env:CarerVisionEnv", max_episode_steps=100000)

N = 240          # 総ステップ（スイープ1往復）
STRIDE = 3       # 記録間引き


def sweep(mode, label):
    env = gym.make("EgoRawSweep-v0", vision_params=lean_vision_params(L.RES),
                   touch_params=infant_touch_params(2.0), hand_size=L.CARER_SIZE, render_mode="rgb_array")
    env.reset(seed=0); u = env.unwrapped; na = u.action_space.shape[0]
    ren = mujoco.Renderer(u.model, 240, 240)
    L.place(u, 0.0)
    for _ in range(20):
        u.set_hand_target([L.FACE_X, 0.0, L.Z_FIX]); env.step(np.zeros(na, np.float32))
    frames = []
    for s in range(N):
        phase = math.sin(2 * math.pi * s / N)     # -1→+1→-1 と1往復
        y = L.Y_SPAN * phase if mode in ("A", "C") else 0.0
        yaw = 0.0
        ctrl = np.zeros(na, np.float32)
        if mode in ("B", "C"):
            yaw = 0.40 * phase                     # 親と"同じ位相"＝最悪ケース(C)
            ctrl[L.HEAD_YAW] = yaw
        u.set_hand_target([L.FACE_X, y, L.Z_FIX]); obs, *_ = env.step(ctrl)
        if s % STRIDE == 0:
            red = E._red_pixels(obs["eye_left"])
            eye = cv2.resize(obs["eye_left"], (240, 240), interpolation=cv2.INTER_NEAREST)
            tp = D.third_person(ren, u.data)
            c = np.zeros((240, 488, 3), np.uint8); c[:, :240] = tp; c[:, 248:] = eye
            cv2.putText(c, "3rd person", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(c, "Taro eye", (254, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(c, f"{label}  y={y:+.3f} yaw={yaw:+.2f} red={red}",
                        (6, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1)
            if red < L.RED_MIN:
                cv2.putText(c, "OUT OF VIEW", (254, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 60, 60), 2)
            frames.append(c)
    ren.close(); env.close()
    return frames


def save(frames, tag, vdir):
    mp4 = os.path.join(vdir, f"rawsweep_{tag}.mp4")
    vw = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), 20, (488, 240))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    gif = os.path.join(vdir, f"rawsweep_{tag}.gif")
    try:
        import imageio
        imageio.mimsave(gif, frames, duration=0.05)
    except Exception:
        from PIL import Image
        Image.fromarray(frames[0]).save(gif, save_all=True,
            append_images=[Image.fromarray(f) for f in frames[1:]], duration=50, loop=0)
    return os.path.abspath(mp4), os.path.abspath(gif)


def main():
    vdir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "logs", "video"))
    os.makedirs(vdir, exist_ok=True)
    cases = [("A", "PARENT-ONLY sweep"), ("B", "HEAD-YAW-ONLY sweep"), ("C", "WORST-CASE combined")]
    for mode, label in cases:
        frames = sweep(mode, label)
        out_of_view = sum(1 for f in frames)  # placeholder; real count below
        mp4, gif = save(frames, mode, vdir)
        print(f"[{mode}] {label}")
        print(f"  mp4: {mp4}")
        print(f"  gif: {gif}", flush=True)


if __name__ == "__main__":
    main()
