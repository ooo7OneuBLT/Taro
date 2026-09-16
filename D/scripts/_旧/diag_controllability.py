"""【放棄・2026-07-15】この系統は仮説が否定されたため使っていない。記録として残す。

当時の仮説：Dでmarginが伸びない真因は「体が行動でどれだけ動かせるか（可制御性）」の差である。
そこで「行動による固有感覚の差 ÷ 行動ゼロの受動ドリフト」を代理指標にして測った。

否定した根拠（実測）：Cの比 1.17 < Dの比 1.30 なのに、marginはCが圧勝(+51 vs +11)。
＝この代理指標は目的の量をまるで測れておらず、**判断を積極的に誤らせた**。

本当の原因（同日判明）：
  ①margin +11 の正体は **C_REPLAY(睡眠リプレイ)が既定OFFだった**こと（Cも同条件なら+11）。
  ②D0が学べなかったのは **借りた環境が3〜500stepで太郎の人生を打ち切っていた**こと。
  ③自己モデルの成否を分けるのは **体が行動で実際にどれだけ動けるか**（立位Cは転倒して手足が
    自由になり+51、座位D0は腰を世界に溶接され腕も畳まれて学べず）。仮説の"方向"は近かったが、
    上の代理指標では捉えられていなかった。

教訓：代理指標は「既知の正解ケース」で先に較正する（検証の落とし穴チェックリスト 項6）。
現在の後継：d_supine_check.py / d_supine_touch_truth.py（接触ペアと基準線で直接測る）
"""
"""
診断：two_agent環境でAの固有感覚が「行動で決まるか(可制御か)」を、学習なしで測る。
margin低下の真因が (A)proprio内容(qpos只) か (B)体の不安定/落下 かを切り分ける。

手順：同一初期状態から K=100 を
  - 行動a1 / 行動a2(別) で回し、最終proprioの「行動による差」
  - 行動ゼロで回した「受動ドリフト」
を比較。行動差 >> ドリフト なら可制御(=proprioで測れる)。
併せて基部(free joint)の高さ推移で落下を確認。qpos/qvel別に評価。
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, mujoco
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from two_agent_env import TwoAgentMIMo

env = TwoAgentMIMo(sep=3.0)
m, d = env.model, env.data
lo = np.array([m.actuator_ctrlrange[i, 0] for i in env.aid]); hi = np.array([m.actuator_ctrlrange[i, 1] for i in env.aid])
b_frozen = np.zeros(env.nb)
K = 100

nonfree = [j for j in env.a_joints if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
free = [j for j in env.a_joints if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]

def rescale(a01):
    return lo + (a01 + 1.0) / 2.0 * (hi - lo)

def snapshot():
    return d.qpos.copy(), d.qvel.copy(), d.act.copy() if d.act.size else None, d.time

def restore(s):
    d.qpos[:] = s[0]; d.qvel[:] = s[1]
    if s[2] is not None: d.act[:] = s[2]
    d.time = s[3]; mujoco.mj_forward(m, d)

def roll(a01, s0):
    restore(s0)
    ctrl = rescale(a01)
    for _ in range(K):
        for k, ai in enumerate(env.aid): d.ctrl[ai] = np.clip(ctrl[k], lo[k], hi[k])
        for k, bi in enumerate(env.bid): d.ctrl[bi] = b_frozen[k]
        mujoco.mj_step(m, d)
    qpos = np.array([d.qpos[m.jnt_qposadr[j]] for j in nonfree])
    qvel = np.array([d.qvel[m.jnt_dofadr[j]] for j in nonfree])
    basez = d.qpos[m.jnt_qposadr[free[0]] + 2] if free else np.nan  # free joint z
    return qpos, qvel, basez

env.reset()
np.random.seed(0)
s0 = snapshot()
# 初期基部高さ
basez0 = d.qpos[m.jnt_qposadr[free[0]] + 2] if free else np.nan

diffs_qpos, diffs_qvel, drift_qpos, drift_qvel, basez_end = [], [], [], [], []
for _ in range(12):
    a1 = np.random.uniform(-1, 1, env.na); a2 = np.random.uniform(-1, 1, env.na)
    q1, v1, bz = roll(a1, s0); q2, v2, _ = roll(a2, s0)
    q0, v0, _ = roll(np.zeros(env.na), s0)  # 受動(行動ゼロ)
    diffs_qpos.append(np.linalg.norm(q1 - q2)); diffs_qvel.append(np.linalg.norm(v1 - v2))
    drift_qpos.append(np.linalg.norm(q1 - q0)); drift_qvel.append(np.linalg.norm(v1 - v0))
    basez_end.append(bz)

print("=== 可制御性診断 (two_agent solo, K=100) ===")
print(f"基部高さ: 初期 z={basez0:.3f} → K=100後 平均 z={np.mean(basez_end):.3f}  (大きく下がる=落下)")
print(f"[qpos] 行動による差(a1 vs a2) = {np.mean(diffs_qpos):.4f}   受動ドリフト(a1 vs 0) = {np.mean(drift_qpos):.4f}")
print(f"        比(行動差/ドリフト) = {np.mean(diffs_qpos)/max(np.mean(drift_qpos),1e-9):.2f}   (>>1なら可制御)")
print(f"[qvel] 行動による差(a1 vs a2) = {np.mean(diffs_qvel):.4f}   受動ドリフト(a1 vs 0) = {np.mean(drift_qvel):.4f}")
print(f"        比(行動差/ドリフト) = {np.mean(diffs_qvel)/max(np.mean(drift_qvel),1e-9):.2f}")
print(f"qpos次元={len(nonfree)}  qvel次元={len(nonfree)}")
