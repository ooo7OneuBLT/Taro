"""taro-C5 診断：「力が大きすぎる」仮説を、アクチュエータ強度・体の質量・方策の指令の大きさ・
到達する関節速度から検証する（学習なし・数値のみ）。"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from d_c5_motor_quality import build, make_policy, K
from test_phase8_motor_learning import rescale_action

env, brain, fusion, emb_proj, cereb, n_act = build("off")
m, d = env.unwrapped.model, env.unwrapped.data

# ── ① アクチュエータの強さ（最大トルク）──
gear = m.actuator_gear[:, 0].copy()
frange = m.actuator_forcerange.copy()
mimo_act = [i for i in range(m.nu) if not m.actuator(i).name.startswith("beta_")]
maxtorque = np.abs(gear[mimo_act])
print("=== ① アクチュエータ強度（最大トルク Nm 相当）===")
print(f"  アクチュエータ数={len(mimo_act)}  最大トルク: 最小={maxtorque.min():.2f} "
      f"中央={np.median(maxtorque):.2f} 最大={maxtorque.max():.2f} 合計={maxtorque.sum():.1f}")
# 代表的な関節をいくつか名前つきで
for nm_key in ("head", "left_shoulder", "left_elbow", "right_hip1", "left_knee"):
    for i in mimo_act:
        if nm_key in m.actuator(i).name:
            print(f"    {m.actuator(i).name:28s} 最大トルク={abs(gear[i]):.2f}")
            break

# ── ② 体の質量 ──
total_mass = m.body_mass.sum()
print("\n=== ② 体の質量 ===")
print(f"  全身の合計質量 = {total_mass:.2f} kg")
# MIMo標準は乳児想定。参考：新生児~3.5kg / 1歳~9kg / 大人~62kg
for bn in ("head", "upper_body", "left_upper_arm", "left_lower_leg"):
    try:
        print(f"    {bn:16s} = {m.body(bn).mass[0]:.3f} kg")
    except Exception:
        pass

# ── ③ 方策が実際どれだけ強く押すか＋到達する関節速度 ──
obs, _ = env.reset(seed=0)
policy = make_policy(brain, fusion, emb_proj, cereb, n_act, babble=False)
hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)
act_abs, torque_frac, qvel_max = [], [], []
for tick in range(15):
    a, hidden = policy(obs, prev_a, hidden)
    an = a.numpy()
    act_abs.append(np.abs(an).mean())
    # |action| は[-1,1]。1.0で最大トルク。=最大トルクの何割を使っているか
    torque_frac.append(np.abs(an).mean())
    ctrl = rescale_action(a, env.action_space)
    for _ in range(K):
        obs, r, te, tr, info = env.step(ctrl)
        qvel_max.append(float(np.abs(d.qvel[6:]).max()))  # 関節角速度の最大（体の自由関節qvel[:6]除く）
        if te or tr:
            obs, _ = env.reset(); hidden = brain.init_motor_hidden(); break
    prev_a = a

print("\n=== ③ 方策の指令の大きさ・到達する速度 ===")
print(f"  |行動|の平均 = {np.mean(act_abs):.3f}  （0=無力, 1=最大トルク。＝最大筋力の何割を常用しているか）")
print(f"  関節角速度の最大の平均 = {np.mean(qvel_max):.2f} rad/s  "
      f"（参考：1 rad/s≒57°/s。数十rad/s＝1秒に何回転もする異常な速さ）")
print(f"  関節角速度の最大の最大 = {np.max(qvel_max):.2f} rad/s")

print("\n=== まとめ ===")
print(f"太郎は最大筋力の約{np.mean(act_abs)*100:.0f}%を毎秒フルに使い、体重{total_mass:.1f}kgを"
      f"最大{np.max(qvel_max):.0f} rad/sで振り回している。")
