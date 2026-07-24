"""egomotion左右実験（d_ego_leftright.py）を**ライブViewer**で見る版。

【なぜ作るか】
`d_ego_leftright.py` は判定器にかける数値実験で、動画(mp4/gif)は保存するが
生成後に見るだけ＝ライブでは見られない。[[feedback-both-view-videos]] の方針
「結果の確認は原則Viewer(等倍速ライブ)」に沿って、同じシナリオ（親=左右移動／
太郎=頭yaw自己運動）を `mujoco.viewer.launch_passive` でライブ再生できるようにする。

【中身は d_ego_leftright.py と同一のシナリオ】
定数（FOVY・CARER_SIZE・Y_SPAN・YAW_LO/HI 等）を re-export して**値をコピーしない**
＝本番実験と食い違う設定でうっかり別物を見てしまう事故を防ぐ。

【操作】
 スペース=一時停止、`.`/`,`=速く/遅く、`0`=等倍、`1`〜`4`=次のケースへ即切替。
 カメラは MuJoCo ビューア標準の 里 `[` / `]` キーで切替可能（三人称 ⇔ eye_left/eye_right）。
 [[feedback-both-view-videos]]：**第三者と一人称の両方を見る**（カメラ切替で両方確認）。

使い方: python d_ego_leftright_view.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 os.pardir, os.pardir, "taro_core"))
import paths  # noqa: E402
paths.setup_brain_path()
sys.path.insert(0, paths.MIMO_DIR)

import time  # noqa: E402
import numpy as np  # noqa: E402
import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import gymnasium as gym  # noqa: E402
import mimoEnv  # noqa: F401,E402
from gymnasium.envs.registration import register  # noqa: E402
from d1_carer_vision_env import lean_vision_params  # noqa: E402
from d_supine_env import infant_touch_params  # noqa: E402
# 本番実験の定数をそのまま使う（食い違い防止）
import d_ego_leftright as LR  # noqa: E402

if not any(e.id == "EgoLR-v0" for e in gym.envs.registry.values()):
    register(id="EgoLR-v0", entry_point="d1_carer_vision_env:CarerVisionEnv",
              max_episode_steps=100000)

CASES = [
    (False, False, "STILL / LEFT"),
    (False, True,  "STILL / RIGHT"),
    (True,  False, "SELFMOVE(head-yaw) / LEFT"),
    (True,  True,  "SELFMOVE(head-yaw) / RIGHT"),
]


def main():
    env = gym.make("EgoLR-v0", vision_params=lean_vision_params(LR.RES, fovy=LR.FOVY),
                    touch_params=infant_touch_params(2.0), hand_size=LR.CARER_SIZE,
                    render_mode="rgb_array")
    env.reset(seed=0)
    u = env.unwrapped
    m, d = u.model, u.data
    na = u.action_space.shape[0]

    case_idx = [0]
    speed = [1.0]

    def _key_cb(keycode):
        try:
            ch = chr(keycode)
        except ValueError:
            return
        if ch in ".>":
            speed[0] = min(speed[0] * 2 if speed[0] > 0 else 64.0, 64.0)
        elif ch in ",<":
            speed[0] = max(speed[0] / 2 if speed[0] > 0 else 1.0, 0.0625)
        elif ch == "0":
            speed[0] = 1.0
        elif ch in "mM":
            speed[0] = 0.0
        elif ch in "1234":
            case_idx[0] = int(ch) - 1
        else:
            return
        print(f"  [switch] speed=x{speed[0]:.4g}" if speed[0] > 0 else "  [switch] MAX",
              flush=True)

    print("\negomotion左右実験（本番と同一設定）をライブ再生します。")
    print(f"  解像度{LR.RES} fovy{LR.FOVY} 親サイズ{LR.CARER_SIZE} 左右幅±{LR.Y_SPAN} "
          f"頭yaw{LR.YAW_LO}-{LR.YAW_HI}")
    print("  1〜4キーでケース切替／スペース=一時停止／.=速く ,=遅く／"
          "カメラは[ ]キーで三人称⇔eye_leftを切替")
    for i, (sm, rt, lab) in enumerate(CASES):
        print(f"    {i+1}: {lab}")

    dt_env = m.opt.timestep * u.frame_skip
    SYNC_DT = 1.0 / 60.0

    with mujoco.viewer.launch_passive(m, d, key_callback=_key_cb) as viewer:
        t_wall = time.perf_counter()
        t_draw = 0.0
        cur_case = -1
        ys = ctrl = None
        yaw_dir = yaw_amp = 0.0
        t_in_clip = 0

        while viewer.is_running():
            if case_idx[0] != cur_case or t_in_clip >= LR.K * LR.SUB:
                cur_case = case_idx[0]
                sm, rt, lab = CASES[cur_case]
                ys_pts = (np.linspace(-LR.Y_SPAN, LR.Y_SPAN, LR.K) if rt
                          else np.linspace(LR.Y_SPAN, -LR.Y_SPAN, LR.K))
                yaw_dir = 1.0
                yaw_amp = (LR.YAW_LO + LR.YAW_HI) / 2.0
                LR.place(u, ys_pts[0])
                for _ in range(15):
                    u.set_hand_target([LR.FACE_X, ys_pts[0], LR.Z_FIX])
                    env.step(np.zeros(na, np.float32))
                t_in_clip = 0
                print(f"  --- 再生中: {lab} ---", flush=True)

            sm, rt, lab = CASES[cur_case]
            ys_pts = (np.linspace(-LR.Y_SPAN, LR.Y_SPAN, LR.K) if rt
                      else np.linspace(LR.Y_SPAN, -LR.Y_SPAN, LR.K))
            t_coarse = min(t_in_clip // LR.SUB, LR.K - 1)
            ctrl = np.zeros(na, np.float32)
            if sm:
                ctrl[LR.HEAD_YAW] = yaw_amp * (t_coarse + 1) / LR.K * yaw_dir
            u.set_hand_target([LR.FACE_X, ys_pts[t_coarse], LR.Z_FIX])
            env.step(ctrl)
            t_in_clip += 1

            now = time.perf_counter()
            if now - t_draw >= SYNC_DT:
                viewer.sync()
                t_draw = now
            if speed[0] > 0:
                t_wall += dt_env / speed[0]
                lag = t_wall - time.perf_counter()
                if lag > 0:
                    time.sleep(lag)
                elif lag < -0.25:
                    t_wall = time.perf_counter()
            else:
                t_wall = now

    env.close()


if __name__ == "__main__":
    main()
