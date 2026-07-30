"""VORの中身を1ステップずつ覗いて、なぜ眼が可動域の端まで押されるかを特定する。

【前提＝e_eye_drift_probe.py で判明したこと】
  左目 vertical：VOR OFF なら 31.59 度のまま**微動だにしない**。
                 VOR ON にすると 1 秒で下限 -47 度に張り付く。重力を切っても同じ。
  → 重力でもバネでもない。**VORの出す指令そのもの**が眼を押している。

【何を出すか】各ステップの
  w_head_raw : data.cvel[head] の角速度（生）
  w_sensed   : 三半規管を通した後（高域通過）
  w_axis     : 眼球の当該軸へ投影した値
  w_des      : -gain × w_axis （＝VORの目標角速度）
  w_eye      : 今の眼球角速度
  ctrl       : 実際に書き込んだ制御入力
"""

# ⚠️★古い方式（2026-07-30 に整理）。新しい実験は `run/main.py` を通す。
#   【経緯】目標Eの実験スクリプトが118本あり、うち66本が**独立に環境を組み立てていた**。
#     そのため「学習は関節モード（90関節を独立に駆動＝逸脱リスト 逸脱5）、
#     測定とViewerは筋肉モード（拮抗筋2本/関節）」という**別の体で動く**事故が起きた
#     （ユーザーの目視「視線誘導反射の実験の時とは動きが全然違う」で発覚。
#      実測で動きが人間の新生児の約3.3倍速かった）。
#   【設計と移行計画】`E/docs/実行基盤_設計.md`
#   ⚠️このファイルは**記録として残す**（削除しない方針）。
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

WATCH = "robot:left_eye_vertical"   # VOR OFF では完全に静止していた関節
SEC = 2.0
PRINT_EVERY = 10                    # 何ステップおきに表示するか


def main():
    from e_toy_env import ToySupineEnv, infant_vision_params
    from e_vor import VOR
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env

    age = 0.0
    kw = body_kwargs_from_env(age, verbose=False)
    env = ToySupineEnv(actuation_model=MuscleModel,
                       vision_params=infant_vision_params(),
                       age=age, vor=True, orient=False, **kw)
    m, d = env.unwrapped.model, env.unwrapped.data
    env.reset(seed=0)
    dt = float(m.opt.timestep) * int(env.unwrapped.frame_skip)
    n_act = env.action_space.shape[0]

    vor = env.unwrapped._vor
    unit = None
    for u in vor.units:
        if m.joint(u["jid"]).name == WATCH:
            unit = u
            break
    if unit is None:
        print("対象の関節がVORの管理下にない:", WATCH)
        print("  VORが見ている関節:", [m.joint(u["jid"]).name for u in vor.units])
        env.close(); return

    aname = m.actuator(unit["aid"]).name
    print(f"監視する関節 : {WATCH}")
    print(f"対応アクチュエータ : {aname}")
    print(f"関節軸 (ローカル) : {unit['axis']}")
    print(f"可動域 : {np.degrees(unit['jlo']):.1f} 〜 {np.degrees(unit['jhi']):.1f} deg")
    print(f"制御入力の範囲 : {unit['lo']:.2f} 〜 {unit['hi']:.2f}")
    print(f"gain={vor.gain}  kv={vor.kv}  kp_ocr={vor.kp_ocr}")
    print(f"三半規管 : {'あり' if vor.canals is not None else 'なし'}"
          f"  時定数={getattr(vor.canals, 'tau', None)}")
    print(f"耳石器   : {'あり' if vor.otolith is not None else 'なし'}")
    print(f"OCRの対象か（アクチュエータ名にtorsionalを含むか）: {'torsional' in aname}")

    # VOR を包んで中間値を記録する
    rows = []

    a = np.zeros(n_act, dtype=np.float32)
    n = int(SEC / dt)
    print(f"\n{'t[s]':>7}{'w_head_raw':>12}{'w_sensed':>11}{'w_axis':>10}"
          f"{'w_des':>10}{'w_eye':>10}{'ctrl':>9}{'角度[deg]':>11}")
    print("-" * 82)

    for step in range(n):
        # step の前に、その時点での VOR の内部量を自分で再現して記録する
        w_world = vor._head_omega_world(d)
        w_raw_norm = float(np.linalg.norm(w_world))
        # ※ canals.update は step 内でも呼ばれるので、ここでは状態を壊さないよう
        #   コピーで計算する
        if vor.canals is not None:
            base = vor.canals.baseline.copy()
            alpha = 1.0 - float(np.exp(-dt / vor.canals.tau))
            base2 = base + alpha * (w_world - base)
            w_sensed = w_world - base2
        else:
            w_sensed = w_world
        R = np.array(d.xmat[unit["bid"]], dtype=float).reshape(3, 3)
        w_axis = float(np.dot(R.T @ w_sensed, unit["axis"]))
        w_des = -vor.gain * w_axis
        w_eye = float(d.qvel[unit["dofadr"]])
        ctrl_pred = float(np.clip(vor.kv * (w_des - w_eye), unit["lo"], unit["hi"]))
        ang = float(np.degrees(d.qpos[unit["qposadr"]]))

        if step % PRINT_EVERY == 0:
            print(f"{step*dt:>7.2f}{w_raw_norm:>12.5f}{float(np.linalg.norm(w_sensed)):>11.5f}"
                  f"{w_axis:>10.5f}{w_des:>10.5f}{w_eye:>10.5f}"
                  f"{ctrl_pred:>9.3f}{ang:>11.2f}")
        rows.append((step * dt, w_raw_norm, w_axis, w_des, w_eye, ctrl_pred, ang))
        env.step(a)

    arr = np.array(rows)
    print("\n=== まとめ ===")
    print(f"  頭の角速度（生）  平均 {arr[:,1].mean():.5f}  最大 {arr[:,1].max():.5f} rad/s")
    print(f"  当該軸への投影    平均 {arr[:,2].mean():+.5f}  絶対値の平均 {np.abs(arr[:,2]).mean():.5f}")
    print(f"  VORの目標角速度   平均 {arr[:,3].mean():+.5f}")
    print(f"  実際の制御入力    平均 {arr[:,5].mean():+.3f}  "
          f"（+{(arr[:,5]>0).mean()*100:.0f}% / -{(arr[:,5]<0).mean()*100:.0f}%）")
    print(f"  角度              {arr[0,6]:.2f} → {arr[-1,6]:.2f} deg")
    print("\n  ★ 頭の角速度がほぼ0なのに制御入力が片側に偏っていたら、")
    print("     VORは『頭を打ち消す』以外の何かで眼を押している。")

    env.close()


if __name__ == "__main__":
    main()
