"""リセット直後に何が起きているかを、実験とまったく同じ条件で等倍速で見る。

【なぜ作ったか】ユーザー「僕が今見ているViewerではその症状を再現できないんだけど」。
そのとおりで、姿勢編集パネル（`e_pose_editor.py`）は
  ・保存した位置（顔から離れた場所）におもちゃを置く
  ・物理を止めた状態から始める
ので、**リセット直後の弾かれ方が起きない**。

一方、実験スクリプトは `env.reset()` の直後から物理を回すため、
  ・おもちゃが顔に 15.6mm めり込む → 拘束反力 1961 Nm
  ・0.4秒で首が 64度回る → 視線がおもちゃから 100度ずれる
となっていた。その様子を目で見るためのビューア。

【使い方】
    python E/scripts/e_reset_replay_viewer.py

  パネルで条件を切り替えて「もう一度最初から」を押すと、その条件で再生し直す。
  等倍速で再生する（実時間と同じ速さ）。

【環境変数】
    E_TOY_DELAY     何秒待っておもちゃを運ぶか（0 で従来＝最初から置く）
    E_TOY_APPROACH  運ぶのにかける秒数
    E_SPEED         再生速度。0.25 で4分の1のスロー再生（既定 1.0）
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
import os, sys, time, warnings
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
import mujoco.viewer
import tkinter as tk

HALF_FOV = 30.0
EPISODE_SEC = 8.0
SPEED = float(os.environ.get("E_SPEED", "1.0"))


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
    zero = np.zeros(n_act, dtype=np.float32)

    toy_bid = int(m.body("test_object1").id)
    head_bid = int(m.body("head").id)
    cam_id = next(c for c in range(m.ncam) if "eye_left" in (m.camera(c).name or ""))
    import e_toy_env as TE

    # ---------------- パネル ----------------
    win = tk.Tk()
    win.title("リセット直後に何が起きているか（等倍速）")
    win.geometry("470x430+30+10")
    win.attributes("-topmost", True)

    tk.Label(win, text="条件", font=("", 11, "bold")).pack(pady=(10, 2))
    delay_var = tk.DoubleVar(value=float(TE.TOY_APPEAR_DELAY))
    f = tk.Frame(win); f.pack(fill="x", padx=14)
    tk.Label(f, text="待つ秒数", width=12, anchor="w").pack(side="left")
    tk.Scale(f, from_=0.0, to=4.0, resolution=0.5, orient="horizontal",
             variable=delay_var, length=280).pack(side="left")

    appr_var = tk.DoubleVar(value=float(TE.TOY_APPROACH_SEC))
    f = tk.Frame(win); f.pack(fill="x", padx=14)
    tk.Label(f, text="運ぶ秒数", width=12, anchor="w").pack(side="left")
    tk.Scale(f, from_=0.1, to=2.0, resolution=0.1, orient="horizontal",
             variable=appr_var, length=280).pack(side="left")

    dist_var = tk.DoubleVar(value=float(env.unwrapped._toy_dist))
    f = tk.Frame(win); f.pack(fill="x", padx=14)
    tk.Label(f, text="顔からの距離[m]", width=12, anchor="w").pack(side="left")
    tk.Scale(f, from_=0.06, to=0.20, resolution=0.005, orient="horizontal",
             variable=dist_var, length=280).pack(side="left")
    tk.Label(win, text="（頭の表面+おもちゃ = 8.3cm 未満だと必ずめり込む／腕は18.6cm）",
             fg="#666", font=("", 8)).pack()

    speed_var = tk.DoubleVar(value=SPEED)
    f = tk.Frame(win); f.pack(fill="x", padx=14)
    tk.Label(f, text="再生速度", width=12, anchor="w").pack(side="left")
    tk.Scale(f, from_=0.1, to=2.0, resolution=0.1, orient="horizontal",
             variable=speed_var, length=280).pack(side="left")

    loop_var = tk.BooleanVar(value=True)
    tk.Checkbutton(win, text=f"{EPISODE_SEC:.0f}秒たったら自動でやり直す",
                   variable=loop_var).pack(pady=(4, 2))

    info = tk.Label(win, text="", font=("Consolas", 10), justify="left")
    info.pack(pady=(6, 4))
    pen_label = tk.Label(win, text="", font=("Consolas", 10), justify="left")
    pen_label.pack()

    _restart = [True]

    def restart():
        _restart[0] = True

    tk.Button(win, text="もう一度最初から", command=restart, width=20,
              bg="#37a", fg="white").pack(pady=8)
    tk.Label(win, text="ビューア: 左ドラッグ=回転 / 右ドラッグ=平行移動 / スクロール=ズーム",
             fg="#666", font=("", 8)).pack(pady=4)

    # ---------------- 再生 ----------------
    print("\n等倍速で再生します。パネルの条件を変えて「もう一度最初から」を押してください。\n",
          flush=True)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        t_sim = 0.0
        wall0 = time.time()
        while viewer.is_running():
            try:
                if not win.winfo_exists():
                    break
            except Exception:
                break

            if _restart[0]:
                TE.TOY_APPEAR_DELAY = float(delay_var.get())
                TE.TOY_APPROACH_SEC = float(appr_var.get())
                env.unwrapped._toy_dist = float(dist_var.get())
                env.reset(seed=0)
                t_sim = 0.0
                wall0 = time.time()
                _restart[0] = False

            env.step(zero)
            t_sim += dt

            # めり込み（おもちゃ×太郎。床は除く）
            worst, worst_name = 0.0, ""
            for i in range(d.ncon):
                c = d.contact[i]
                if c.dist >= -1e-5:
                    continue
                b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
                if toy_bid not in (b1, b2):
                    continue
                other = b2 if b1 == toy_bid else b1
                nm = m.body(other).name
                if nm == "floor":
                    continue
                if -c.dist * 1000 > worst:
                    worst, worst_name = -c.dist * 1000, nm

            cpos = d.cam_xpos[cam_id]
            fwd = -d.cam_xmat[cam_id].reshape(3, 3)[:, 2]
            v = d.xpos[toy_bid] - cpos
            nv = float(np.linalg.norm(v))
            gz = float(np.degrees(np.arccos(np.clip(np.dot(fwd, v / nv), -1, 1)))) \
                if nv > 1e-9 else float("nan")
            w = float(np.linalg.norm(d.cvel[head_bid][:3]))
            pending = getattr(env.unwrapped, "_toy_pending", False)
            phase = ("おもちゃは遠くで待機中" if pending and
                     not getattr(env.unwrapped, "_toy_arriving", False)
                     else "親が運んでいる" if pending else "設置ずみ")

            info.config(text=f"経過 {t_sim:5.2f} 秒   {phase}\n"
                             f"視線のズレ {gz:6.1f}°  "
                             f"{'視界の中' if gz < HALF_FOV else '視界の外'}\n"
                             f"頭の角速度 {w:6.3f} rad/s"
                             f"{'   弾かれています' if w > 1.0 else ''}",
                        fg="#070" if gz < HALF_FOV else "#a00")
            if worst > 0.01:
                pen_label.config(text=f"めり込み {worst:.2f} mm（{worst_name}）",
                                 fg="#a00")
            else:
                pen_label.config(text="めり込みなし", fg="#070")

            if loop_var.get() and t_sim >= EPISODE_SEC:
                _restart[0] = True

            try:
                win.update()
            except tk.TclError:
                break
            viewer.sync()

            # 等倍速に合わせて待つ
            sp = max(0.05, float(speed_var.get()))
            target = wall0 + t_sim / sp
            lag = target - time.time()
            if lag > 0:
                time.sleep(min(lag, 0.05))

    env.close()
    try:
        win.destroy()
    except Exception:
        pass


if __name__ == "__main__":
    main()
