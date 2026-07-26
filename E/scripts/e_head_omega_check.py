"""VORが読んでいる「頭の角速度」が本当に頭の回転なのかを検証する（＝計測器の検証）。

【なぜ】
  ・ユーザーの目視（Viewer）：「頭はまったく動いてない」
  ・VORの内部値           ：頭の角速度 平均 1.98 rad/s（＝113度/秒）、最大 10.9 rad/s
  真っ向から食い違う。どちらかが誤り。

【突き合わせる3つの量】
  ①data.cvel[head][:3]        …VORが今つかっている値（MuJoCoのcom基準の空間速度の回転部分）
  ②xquat の数値微分から出す角速度 …頭の姿勢の変化を直接見る＝これが「本当の回転」
  ③首の関節角度の変化            …見た目の「頭が動く」に一番近い量

  ①と②が一致 → 頭は本当に激しく回っている（目視した時点では既に収まっていた）
  ①と②が食い違う → VORの入力そのものが壊れている
"""
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

SEC = 6.0
PRINT_EVERY = 20


def quat_to_omega(q_prev, q_now, dt):
    """2つの姿勢クォータニオンから角速度[rad/s]（ワールド基準）を出す。"""
    q0 = np.asarray(q_prev, dtype=float)
    q1 = np.asarray(q_now, dtype=float)
    # 相対回転 dq = q1 * conj(q0)
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1
    cw, cx, cy, cz = w0, -x0, -y0, -z0
    dw = w1*cw - x1*cx - y1*cy - z1*cz
    dx = w1*cx + x1*cw + y1*cz - z1*cy
    dy = w1*cy - x1*cz + y1*cw + z1*cx
    dz = w1*cz + x1*cy - y1*cx + z1*cw
    v = np.array([dx, dy, dz], dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros(3)
    angle = 2.0 * float(np.arctan2(n, abs(dw)))
    if angle > np.pi:
        angle -= 2 * np.pi
    return (v / n) * (angle / dt)


def main():
    from e_toy_env import ToySupineEnv, infant_vision_params
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env

    age = 0.0
    kw = body_kwargs_from_env(age, verbose=False)
    # ★VORを切って測る。VOR自身が眼を動かして頭に反作用を与える可能性を排除する。
    env = ToySupineEnv(actuation_model=MuscleModel,
                       vision_params=infant_vision_params(),
                       age=age, vor=False, orient=False, **kw)
    m, d = env.unwrapped.model, env.unwrapped.data
    env.reset(seed=0)
    dt = float(m.opt.timestep) * int(env.unwrapped.frame_skip)
    n_act = env.action_space.shape[0]
    hid = int(m.body("head").id)

    neck = [j for j in range(m.njnt)
            if any(k in m.joint(j).name for k in ("head", "neck"))]
    print("首まわりの関節:", [m.joint(j).name for j in neck])
    print(f"dt = {dt}\n")

    a = np.zeros(n_act, dtype=np.float32)
    n = int(SEC / dt)
    q_prev = np.array(d.xquat[hid], dtype=float).copy()
    neck_prev = np.array([d.qpos[int(m.jnt_qposadr[j])] for j in neck], dtype=float)

    print(f"{'t[s]':>7}{'①cvel':>11}{'②姿勢微分':>13}{'③首の関節':>12}"
          f"    {'首の角度[deg]'}")
    print("-" * 78)
    rows = []
    for step in range(n):
        env.step(a)
        q_now = np.array(d.xquat[hid], dtype=float).copy()
        w_cvel = np.array(d.cvel[hid][:3], dtype=float)
        w_quat = quat_to_omega(q_prev, q_now, dt)
        neck_now = np.array([d.qpos[int(m.jnt_qposadr[j])] for j in neck], dtype=float)
        w_neck = float(np.linalg.norm((neck_now - neck_prev) / dt))
        q_prev = q_now
        neck_prev = neck_now
        rows.append((step * dt, float(np.linalg.norm(w_cvel)),
                     float(np.linalg.norm(w_quat)), w_neck))
        if step % PRINT_EVERY == 0:
            angs = "  ".join(f"{np.degrees(v):7.2f}" for v in neck_now)
            print(f"{step*dt:>7.2f}{np.linalg.norm(w_cvel):>11.4f}"
                  f"{np.linalg.norm(w_quat):>13.4f}{w_neck:>12.4f}    {angs}")

    arr = np.array(rows)
    print("\n=== まとめ（単位 rad/s。1 rad/s = 57.3 度/秒）===")
    for i, lbl in [(1, "①cvel[head]（VORが使っている値）"),
                   (2, "②姿勢の数値微分（本当の回転）"),
                   (3, "③首の関節角度の変化")]:
        print(f"  {lbl:<34} 平均 {arr[:,i].mean():>8.4f}  最大 {arr[:,i].max():>8.4f}")

    late = arr[arr[:, 0] >= 3.0]
    print("\n  --- 3秒以降だけ（初期の跳ねを除く）---")
    for i, lbl in [(1, "①cvel[head]"), (2, "②姿勢の数値微分"), (3, "③首の関節")]:
        print(f"  {lbl:<34} 平均 {late[:,i].mean():>8.4f}  最大 {late[:,i].max():>8.4f}")

    r = np.corrcoef(arr[:, 1], arr[:, 2])[0, 1]
    print(f"\n  ①と②の相関 : {r:+.4f}")
    ratio = arr[:, 1].mean() / max(arr[:, 2].mean(), 1e-9)
    print(f"  ①÷② の比  : {ratio:.2f} 倍")
    print("\n  比が 1 に近い → cvel は正しい。頭は本当に動いている。")
    print("  比が大きい     → cvel は頭の回転ではない別の量を含んでいる＝VORの入力が壊れている。")

    env.close()


if __name__ == "__main__":
    main()
