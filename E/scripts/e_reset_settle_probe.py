"""リセット直後に首が勝手に64度回る原因を切り分ける。

【なぜ】視線誘導反射を測ろうとしたら、どの条件でも
「おもちゃが視界に入っていた時間 0.0%」（ズレ角 94〜111度・視野の半角は30度）になった。
ユーザーが Viewer で見たときは視界に入っていたのに、スクリプトでは入らない。

差は**物理を回すかどうか**。Viewer は物理を止めた状態から始めるので跳ねない。
`e_head_omega_check.py` の実測：

```
時刻     首の角度[度]
0.00s     3.22   -27.64   -0.46
0.40s    64.11     1.88   -4.90    ← 0.4秒で首が64度回る
```

これを直さないと、視線に関わる実験は何ひとつ成立しない。

【候補】
  A. 屈筋トーンのバネ（減衰なしの qpos_spring）が体を弾いている
  B. 重力で頭が転がる（新生児の首の筋力は lift_ratio=1.0 ＝ ぎりぎり支えられない）
  C. 生理的屈曲の jnt_range（伸展の限界）に初期姿勢が食い込んでいて弾かれる
  D. 初期姿勢が床から浮いていて落下の衝撃で回る

【測り方】同じ環境で条件だけ変え、首3関節の角度と視線のズレ角を6秒追う。
"""

# 注意：古い方式（2026-07-30 に整理）。新しい実験は `run/main.py` を通す。
#   【経緯】目標Eの実験スクリプトが118本あり、うち66本が**独立に環境を組み立てていた**。
#     そのため「学習は関節モード（90関節を独立に駆動＝逸脱リスト 逸脱5）、
#     測定とViewerは筋肉モード（拮抗筋2本/関節）」という**別の体で動く**事故が起きた
#     （ユーザーの目視「視線誘導反射の実験の時とは動きが全然違う」で発覚。
#      実測で動きが人間の新生児の約3.3倍速かった）。
#   【設計と移行計画】`E/docs/実行基盤_設計.md`
#   注意：このファイルは**記録として残す**（削除しない方針）。
#     中の測り方は再利用できるので、プラグインへ移すときの元にする。
import os, sys, warnings
warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
for p in [os.path.join(_ROOT, "D", "scripts"), os.path.join(_ROOT, "MIMo"),
          os.path.join(_ROOT, "taro_core"),
          os.path.join(_ROOT, "taro_core", "src", "body"),
          _HERE]:
    if p not in sys.path:
        sys.path.insert(0, p)
try: sys.stdout.reconfigure(errors="replace")
except Exception: pass

import numpy as np
import mujoco

SEC = 6.0
SAMPLE_AT = [0.0, 0.2, 0.5, 1.0, 2.0, 4.0, 6.0]
HALF_FOV = 30.0


def gaze_angle(m, d, cam_id, toy_bid):
    cpos = d.cam_xpos[cam_id]
    fwd = -d.cam_xmat[cam_id].reshape(3, 3)[:, 2]
    v = d.xpos[toy_bid] - cpos
    n = np.linalg.norm(v)
    if n < 1e-9:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(fwd, v / n), -1, 1))))


def run_case(env, dt, n_act, neck, cam_id, toy_bid, tone_k, gravity_on):
    m, d = env.unwrapped.model, env.unwrapped.data
    saved_g = np.array(m.opt.gravity, dtype=float).copy()
    saved_k = {j: float(m.jnt_stiffness[j]) for j in tone_k}
    env.reset(seed=0)
    m.opt.gravity[:] = saved_g if gravity_on else 0.0
    for j, k in tone_k.items():
        m.jnt_stiffness[j] = k

    n = int(SEC / dt)
    picks = {round(t / dt): t for t in SAMPLE_AT}
    out = {0.0: (np.degrees([d.qpos[a] for a in neck["qadr"]]).copy(),
                 gaze_angle(m, d, cam_id, toy_bid))}
    a = np.zeros(n_act, dtype=np.float32)
    for step in range(1, n + 1):
        env.step(a)
        if step in picks:
            out[picks[step]] = (np.degrees([d.qpos[q] for q in neck["qadr"]]).copy(),
                                gaze_angle(m, d, cam_id, toy_bid))
    m.opt.gravity[:] = saved_g
    for j, k in saved_k.items():
        m.jnt_stiffness[j] = k
    return out


def main():
    from e_toy_env import ToySupineEnv, infant_vision_params
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env

    kw = body_kwargs_from_env(0.0, verbose=False)
    kw["flexion"] = True
    env = ToySupineEnv(actuation_model=MuscleModel,
                       vision_params=infant_vision_params(),
                       age=0.0, toy=True, vor=True, orient=False, **kw)
    m, d = env.unwrapped.model, env.unwrapped.data
    env.reset(seed=0)
    dt = float(m.opt.timestep) * int(env.unwrapped.frame_skip)
    n_act = env.action_space.shape[0]

    neck = {"name": [], "qadr": []}
    for j in range(m.njnt):
        nm = m.joint(j).name
        if any(k in nm for k in ("head", "neck")):
            neck["name"].append(nm.replace("robot:", ""))
            neck["qadr"].append(int(m.jnt_qposadr[j]))
    cam_id = next(c for c in range(m.ncam) if "eye_left" in (m.camera(c).name or ""))
    toy_bid = int(m.body("test_object1").id)

    # 屈筋トーンのバネが入っている関節（stiffness > 0）
    tone_joints = [j for j in range(m.njnt) if float(m.jnt_stiffness[j]) > 0.0]
    tone_full = {j: float(m.jnt_stiffness[j]) for j in tone_joints}
    tone_zero = {j: 0.0 for j in tone_joints}
    tone_weak = {j: v * 0.25 for j, v in tone_full.items()}

    print("=== リセット直後に首が回る原因の切り分け ===")
    print(f"  action=0（完全脱力）・{SEC:.0f}秒・視野の半角 {HALF_FOV:.0f}度")
    print(f"  首の関節: {neck['name']}")
    print(f"  屈筋トーンのバネが入っている関節: {len(tone_joints)}個\n")

    cases = [
        ("現状（トーン k=0.2・重力あり）", tone_full, True),
        ("トーン弱め（k=0.05）",           tone_weak, True),
        ("トーンOFF（k=0）",               tone_zero, True),
        ("重力なし（トーンは現状）",       tone_full, False),
    ]

    for label, tk, grav in cases:
        r = run_case(env, dt, n_act, neck, cam_id, toy_bid, tk, grav)
        print(f"--- {label} ---")
        hdr = "".join(f"{nm[:12]:>13}" for nm in neck["name"])
        print(f"{'t[s]':>6}{hdr}{'視線のズレ':>13}")
        for t in SAMPLE_AT:
            angs, g = r[t]
            row = "".join(f"{v:>13.2f}" for v in angs)
            mark = "" if g < HALF_FOV else "  視界の外"
            print(f"{t:>6.1f}{row}{g:>13.1f}{mark}")
        a0 = r[0.0][0]
        a_end = r[SEC][0]
        print(f"  首の総変化量 {np.abs(a_end - a0).sum():.1f}度"
              f"  ／ 視線のズレ {r[0.0][1]:.1f}度 → {r[SEC][1]:.1f}度\n")

    env.close()
    print("=== 読み方 ===")
    print("  トーンを弱める/切ると収まる → 原因は屈筋トーンのバネ（減衰が無い）")
    print("  重力を切ると収まる           → 首の筋力不足（lift_ratio=1.0 は境界値）")
    print("  どれでも回る                 → 初期姿勢そのものが不安定（落下・食い込み）")


if __name__ == "__main__":
    main()
