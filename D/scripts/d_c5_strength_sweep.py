"""taro-C5 実験(a)：筋力（アクチュエータ最大トルク）を数段階に弱めて、運動の質が
乳児的（低〜中振幅・低速・低ジャーク）に近づくかを、再学習なしで測る＋コンタクトシート化。

「筋力を弱める」＝ m.actuator_gear（力の倍率）を factor 倍する（＝同じ神経指令でも出る
トルクが小さい＝乳児の弱い筋肉）。方策・行動空間・脳は一切触らない＝変数は筋力1つだけ。

使い方: python d_c5_strength_sweep.py [ticks]   （既定20ティック/条件）
出力: D/logs/video/c5_strength_sheet.png（各筋力での第三者視点コマ）＋数値テーブル
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import mujoco
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from d_c5_motor_quality import build, make_policy, actuated_dofs, JerkMeter, K
from test_phase8_motor_learning import rescale_action

OUT = os.path.abspath(os.path.join(_HERE, os.pardir, "logs", "video"))
FACTORS = [1.0, 0.5, 0.25, 0.1]
N_FRAMES = 4   # コンタクトシートに並べる第三者コマ数/条件


def run():
    ticks = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    env, brain, fusion, emb_proj, cereb, n_act = build("off")
    m, d = env.unwrapped.model, env.unwrapped.data
    dofs = actuated_dofs(m)
    dt_env = m.opt.timestep * env.unwrapped.frame_skip
    mimo_act = [i for i in range(m.nu) if not m.actuator(i).name.startswith("beta_")]
    orig_gear = m.actuator_gear[:, 0].copy()

    # レンダラ（第三者視点）
    m.vis.global_.offwidth = 480; m.vis.global_.offheight = 360
    ren = mujoco.Renderer(m, height=360, width=480)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 1.6; cam.elevation = -18; cam.azimuth = 90; cam.lookat = [0.0, 0.0, 0.15]

    rows = []
    print(f"\n{'筋力':>6} | {'mean|jerk|':>10} {'境界':>8} {'内部':>8} | "
          f"{'関節速度 平均max':>14} {'最大':>7} | {'|行動|':>6}")
    print("-" * 78)
    for f in FACTORS:
        m.actuator_gear[mimo_act, 0] = orig_gear[mimo_act] * f
        obs, _ = env.reset(seed=0)
        policy = make_policy(brain, fusion, emb_proj, cereb, n_act, babble=False)
        hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)
        meter = JerkMeter(dofs, dt_env)
        qvel_max, act_abs = [], []
        frame_at = set(int(x) for x in np.linspace(0, ticks - 1, N_FRAMES))
        frames = []
        for tick in range(ticks):
            a, hidden = policy(obs, prev_a, hidden)
            act_abs.append(float(np.abs(a.numpy()).mean()))
            ctrl = rescale_action(a, env.action_space)
            for k in range(K):
                obs, r, te, tr, info = env.step(ctrl)
                meter.observe(d.qacc.copy(), is_boundary=(k == 0))
                qvel_max.append(float(np.abs(d.qvel[6:]).max()))
                if te or tr:
                    obs, _ = env.reset(); hidden = brain.init_motor_hidden(); break
            prev_a = a
            if tick in frame_at:
                ren.update_scene(d, camera=cam); frames.append(ren.render())
        s = meter.summary()
        print(f"{f:>6.2f} | {s['mean']:>10.0f} {s['boundary']:>8.0f} {s['interior']:>8.0f} | "
              f"{np.mean(qvel_max):>14.2f} {np.max(qvel_max):>7.1f} | {np.mean(act_abs):>6.3f}")
        # 行頭に筋力ラベルを載せてコマを横並び
        labeled = []
        for i, fr in enumerate(frames):
            fr = fr.copy()
            if i == 0:
                cv2.putText(fr, f"strength x{f}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 50, 50), 2)
            labeled.append(fr)
        rows.append(np.concatenate(labeled, axis=1))

    m.actuator_gear[mimo_act, 0] = orig_gear[mimo_act]  # 後始末
    os.makedirs(OUT, exist_ok=True)
    sheet = np.concatenate(rows, axis=0)
    path = os.path.join(OUT, "c5_strength_sheet.png")
    cv2.imwrite(path, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"\nコンタクトシート: {path}")
    print("（各行が筋力1.0/0.5/0.25/0.1倍。左から時間経過。振幅＝手足の振れ幅を見る）")
    print("\n参考：乳児の関節速度はおよそ1〜3 rad/s。65 rad/s級は物理的に異常。")


if __name__ == "__main__":
    run()
