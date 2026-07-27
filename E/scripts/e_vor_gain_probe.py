"""VOR（前庭動眼反射）の実効利得を測り直す。

【なぜ測り直すか】2026-07-27 に VOR の指令を
    kv * (目標角速度 − 今の眼球角速度)   ← 眼球速度のフィードバック
  → kv * 目標角速度                      ← 三ニューロン弓に近い直通路
に変えた。理由は、前者が「眼球の速度をゼロに保つ制御」になり、**視線誘導反射の
サッケードを完全に打ち消していた**ため（実測：同じ指令で VOR OFF なら1秒で
25.6度動くのに、VOR ON では -1.4度）。人間の VOR は半規管 → 前庭神経核 →
外眼筋運動核 → 筋 の三シナプスで、経路にフィードバックループを含まない。

⚠️過去の Kv スイープ表（e_vor.py 冒頭）は、反射が筋肉モデルに対応していなかった
時期の測定なので無効。今回が最初のまともな較正になる。

【目標】暗所のVOR利得は 1〜4ヶ月児で 1.03 ± 0.014（成人 0.59 ± 0.03）。

【測り方】首を正弦波で振って頭を回し、そのとき眼球がどれだけ逆に回るかを測る。
    実効利得 = −(眼球の角速度) / (頭の角速度)   ← 最小二乗の傾きで求める
太郎は脱力（首以外は action=0）。おもちゃも柵も出さない。
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
for p in [os.path.join(_ROOT, "D", "scripts"), os.path.join(_ROOT, "MIMo"),
          os.path.join(_ROOT, "taro_core"),
          os.path.join(_ROOT, "taro_core", "src", "body"),
          os.path.join(_ROOT, "taro_core", "src", "brain"), _HERE]:
    if p not in sys.path:
        sys.path.insert(0, p)
try: sys.stdout.reconfigure(errors="replace")
except Exception: pass

import numpy as np

KVS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
HEAD_HZ = 0.5          # 首を振る周期[Hz]
# ★頭を振る強さ。最初 0.6 で測ったら頭の角速度が 1.45 rad/s（83度/秒）になり、
#   指令が 59〜95% のtickで飽和して Kv を変えても利得が動かなかった。
#   実際の新生児の頭は **0.32〜0.41 rad/s（18〜23度/秒）**（過去の実測）。
#   人間のVOR測定でも正弦波回転のピーク速度は 20〜60度/秒 が典型。
HEAD_AMPS = [0.05, 0.10, 0.20]
SECONDS = 8.0
TARGET_GAIN = 1.03     # 1〜4ヶ月児の実測（暗所）


def main():
    from e_toy_env import ToySupineEnv, infant_vision_params
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env
    from spinal_cord.cpg import write_joint_command
    import e_vor as VORMOD

    kw = body_kwargs_from_env(0.0, verbose=False)
    kw["flexion"] = True
    env = ToySupineEnv(actuation_model=MuscleModel,
                       vision_params=infant_vision_params(),
                       age=0.0, toy=False, vor=True, orient=False, **kw)
    u = env.unwrapped
    m, d = u.model, u.data
    env.reset(seed=0)
    dt = float(m.opt.timestep) * int(u.frame_skip)
    nu, dim = int(m.nu), env.action_space.shape[0]
    n = int(SECONDS / dt)

    neck_i = next(i for i in range(nu) if m.actuator(i).name == "act:head_swivel")
    head_bid = int(m.body("head").id)
    # 水平方向の眼球ユニットだけを見る（首の swivel と対応する軸）
    units = [x for x in u._vor.units
             if "horizontal" in m.actuator(x["aid"]).name]

    print("=== VOR の実効利得を測り直す ===")
    print(f"  首を {HEAD_HZ} Hz で振る／{SECONDS} 秒")
    print(f"  目標 {TARGET_GAIN}（1〜4ヶ月児・暗所。Finocchio 1991）\n")
    print(f"{'振幅':>6}{'Kv':>7}{'実効利得':>11}{'頭の角速度':>14}"
          f"{'眼球の角速度':>14}{'飽和':>9}")

    best = None
    for HEAD_AMP in HEAD_AMPS:
      for kv in KVS:
          u._vor.kv = float(kv)
          env.reset(seed=0)
          u._vor.reset()
          hs, es, sat = [], [], 0
          for i in range(n):
              a = np.zeros(dim, dtype=np.float32)
              cmd = HEAD_AMP * np.sin(2 * np.pi * HEAD_HZ * i * dt)
              write_joint_command(a, neck_i, cmd, nu, co_activation=0.0)
              env.step(a)
              w_world = np.array(d.cvel[head_bid][:3], dtype=float)
              for x in units:
                  R = np.array(d.xmat[x["bid"]], dtype=float).reshape(3, 3)
                  hs.append(float(np.dot(R.T @ w_world, x["axis"])))
                  es.append(float(d.qvel[x["dofadr"]]))
                  if abs(kv * (-u._vor.gain) * hs[-1]) >= 1.0:
                      sat += 1
          hs, es = np.array(hs), np.array(es)
          # 実効利得＝ −眼球角速度 を 頭角速度 で回帰した傾き
          den = float((hs * hs).sum())
          g = float(-(hs * es).sum() / den) if den > 1e-12 else float("nan")
          hw = np.abs(hs).mean()
          print(f"{HEAD_AMP:>6.2f}{kv:>7.1f}{g:>11.3f}"
                f"{hw:>9.3f}r/s({np.degrees(hw):>4.0f}度/s)"
                f"{np.abs(es).mean():>12.3f}r/s{sat/max(len(hs),1)*100:>8.0f}%")
          if best is None or abs(g - TARGET_GAIN) < abs(best[2] - TARGET_GAIN):
              best = (HEAD_AMP, kv, g)

    env.close()
    print(f"\n  目標 {TARGET_GAIN} に最も近いのは Kv={best[0]}（利得 {best[1]:.3f}）")
    print("\n=== 読み方 ===")
    print("  利得が Kv を上げても頭打ちになる → 眼筋の力か飽和が効いている")
    print("  飽和した割合が高い               → 指令が [-1,1] に収まっていない")


if __name__ == "__main__":
    main()
