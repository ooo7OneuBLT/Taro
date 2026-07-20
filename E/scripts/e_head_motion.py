"""太郎の「頭の動き」が新生児として妥当かを実測する。

【なぜ測るか】2026-07-20 Viewerでの目視でユーザーが指摘：
  「赤ちゃんってこんなに頭を振り回すもんなのかな？」
文献を当たると、新生児の頭は**振り回さない**：
  ・安静時は**頭を片側に向けたまま**。中線に保てるのは「興味ある物を見ている時」か「泣く時」だけ
  ・重力に抗する首の筋力は**これから数週かけて育つ**（＝新生児にはまだ無い）
  ・全身運動(writhing)は「引き締まった外観・比較的遅い速度・**限られた振幅**」
  ・頭回転の可動域は2-3ヶ月で約70度(=1.22 rad)。新生児はさらに小さい
太郎はここから外れている疑いがある（C5の残課題＝関節速度5.6 rad/s vs 乳児域1〜3 rad/s）。
**勘で首を弱める前に、まず測る**（落とし穴チェックの筋）。

【測ること】
  ①首の関節ごとの角度範囲（可動限界のどれだけを使っているか）
  ②首の関節の角速度（|ω| の平均・最大・p95）→ 乳児域1〜3 rad/s と比較
  ③頭が中線から外れている角度（＝片側に向いているか、振り回しているか）
  ④頭のワールド姿勢の変化速度

【条件】既定は素の仰向け環境（柵・おもちゃなし）＝C5の運動品質そのものを見る。
E_TOY=1 を付ければ柵・おもちゃ入りでも測れる（環境差の切り分け用）。

使い方: python e_head_motion.py [n_ticks] [age]
  ※モデルは C5_CKPT、体年齢は引数（既定0=新生児）
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
import d_c5_motor_quality as mq  # noqa: E402  （環境構築・方策を再利用＝車輪の再発明をしない）

# 文献の目安
LIT_ROM_DEG = 70.0      # 頭回転の可動域（2-3ヶ月）。新生児はこれより小さい
INFANT_SPEED = (1.0, 3.0)   # 乳児の関節角速度の目安[rad/s]（C5で使った基準）


def main():
    n_ticks = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    age = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    mq.AGE = age

    env, brain, fusion, emb_proj, cereb, n_act = mq.build("off", age=age)
    policy = mq.make_policy(brain, fusion, emb_proj, cereb, n_act, babble=True)
    m, d = env.unwrapped.model, env.unwrapped.data

    # 首まわりの関節を名前で拾う（MIMoの首関節）
    neck = []
    for j in range(m.njnt):
        nm = m.joint(j).name
        if "head" in nm or "neck" in nm:
            neck.append(j)
    print(f"neck joints found: {[m.joint(j).name for j in neck]}")
    if not neck:
        print("!! no neck joints found - check joint names")
        return

    qadr = [int(m.jnt_qposadr[j]) for j in neck]
    dadr = [int(m.jnt_dofadr[j]) for j in neck]
    rng = [m.jnt_range[j].copy() for j in neck]

    obs, _ = env.reset(seed=0)
    hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)
    ang, vel, head_z = [], [], []
    all_qvel, body_pos = [], []
    for t in range(n_ticks):
        a, hidden = policy(obs, prev_a, hidden)
        ctrl = mq.rescale_action(a, env.action_space); prev_a = a
        for k in range(mq.K):
            obs, r, te, tr, info = env.step(ctrl)
            ang.append([float(d.qpos[i]) for i in qadr])
            vel.append([float(d.qvel[i]) for i in dadr])
            head_z.append(d.body("head").xmat.reshape(3, 3)[:, 2].copy())  # 頭の上向きベクトル
            all_qvel.append(d.qvel[6:].copy())          # freejoint(6)を除く全関節速度
            body_pos.append(d.body("upper_body").xpos.copy())
            if te or tr:
                break
        if te or tr:
            obs, _ = env.reset(); hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)

    ang = np.asarray(ang); vel = np.asarray(vel); head_z = np.asarray(head_z)
    # 全関節（首以外も含む）の速度と体の移動＝四肢の暴れと移動の指標
    allv = np.abs(np.asarray(all_qvel))
    bp = np.asarray(body_pos)
    print(f"\n[whole body]  all-joint |w| mean={allv.mean():.2f} p95={np.percentile(allv,95):.2f} "
          f"max={allv.max():.2f} rad/s")
    print(f"[whole body]  torso drift: x range={bp[:,0].max()-bp[:,0].min():.3f} "
          f"y range={bp[:,1].max()-bp[:,1].min():.3f} m  "
          f"max dist from start={np.linalg.norm(bp[:,:2]-bp[0,:2],axis=1).max():.3f} m")
    print(f"\n=== head motion (age={age:.0f}mo, {len(ang)} ticks) ===")
    print(f"{'joint':22s} {'range(deg)':>12s} {'used(deg)':>10s} {'use%':>6s} "
          f"{'|w|mean':>8s} {'|w|p95':>8s} {'|w|max':>8s}")
    for i, j in enumerate(neck):
        lo, hi = np.degrees(rng[i])
        used = np.degrees(ang[:, i].max() - ang[:, i].min())
        full = max(hi - lo, 1e-6)
        w = np.abs(vel[:, i])
        print(f"{m.joint(j).name:22s} [{lo:+6.1f},{hi:+6.1f}] {used:10.1f} "
              f"{used/full*100:5.1f}% {w.mean():8.2f} {np.percentile(w,95):8.2f} {w.max():8.2f}")

    wall = np.abs(vel)
    print(f"\nall neck joints : |w| mean={wall.mean():.2f} p95={np.percentile(wall,95):.2f} "
          f"max={wall.max():.2f} rad/s")
    print(f"infant reference: {INFANT_SPEED[0]:.0f}-{INFANT_SPEED[1]:.0f} rad/s")

    # 頭の向きが「片側に向いたまま」か「振り回している」か
    # 頭の上向きベクトルが世界のzからどれだけ倒れているか＝頭の傾き
    tilt = np.degrees(np.arccos(np.clip(head_z[:, 2], -1, 1)))
    print(f"\nhead tilt from vertical: mean={tilt.mean():.1f} deg  "
          f"min={tilt.min():.1f}  max={tilt.max():.1f}  std={tilt.std():.1f}")
    # 変化の速さ（連続フレーム間の角度差）
    dtilt = np.abs(np.diff(tilt))
    print(f"tilt change per physics step: mean={dtilt.mean():.3f} deg  max={dtilt.max():.3f} deg")

    print("\n=== verdict ===")
    fast = wall.mean() > INFANT_SPEED[1]
    wide = any(np.degrees(ang[:, i].max() - ang[:, i].min()) > LIT_ROM_DEG
               for i in range(len(neck)))
    print(f"  neck speed above infant range (>3 rad/s) : {'YES (non-human)' if fast else 'no'}")
    print(f"  rotation exceeds 70 deg (2-3mo ROM)      : {'YES (non-human)' if wide else 'no'}")
    print("  NOTE: newborns keep the head turned to ONE side at rest; large, fast,")
    print("        repeated head swings are not typical newborn behaviour.")
    env.close()


if __name__ == "__main__":
    main()
