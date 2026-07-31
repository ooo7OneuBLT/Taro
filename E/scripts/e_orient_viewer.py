"""視線誘導反射の動きを等倍速で見る。サッケードの間隔・閾値をその場で変えて比べる。

【なぜ】数値では「間隔を 0.2秒 → 1.3秒 にすると、おもちゃが見えていた割合が
19% → 100% になる」と出た。それを目で確かめるためのビューア。

【使い方】
    python E/scripts/e_orient_viewer.py

  ・「サッケードの間隔」を 0.2 にすると、太郎が自分でおもちゃを視界から追い出す
  ・0.9〜1.3 にすると、おもちゃが視界に留まる
  ・「おもちゃを揺らす」を切ると、外界の動きが無い状態での挙動が見える

【画面の見方】
  一人称視点   太郎の左目に映っている映像。測定器が検出した画素は緑
  発火         サッケードを撃った瞬間に赤く光る
  中心からのずれ 0.0＝ど真ん中、1.0＝画面の端。反射が仕事をしていれば減る

【環境変数】
    E_SPEED   再生速度（既定 1.0＝等倍）
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
import e_visibility as VIS

SHAKE_HZ = 2.5
SHAKE_AMP = 0.015
EPISODE_SEC = 20.0
SPEED = float(os.environ.get("E_SPEED", "1.0"))


def main():
    os.environ.setdefault("E_ORIENT_V", "2")
    os.environ.setdefault("E_TOY_MODE", "hold")
    os.environ.setdefault("E_FLEXION", "1")
    from e_toy_env import ToySupineEnv, infant_vision_params
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env
    import e_orienting_v2 as OR
    import e_toy_env as TE

    kw = body_kwargs_from_env(0.0, verbose=False)
    kw["flexion"] = True
    env = ToySupineEnv(actuation_model=MuscleModel,
                       vision_params=infant_vision_params(),
                       age=0.0, toy=True, vor=True, orient=True, **kw)
    u = env.unwrapped
    m, d = u.model, u.data
    env.reset(seed=0)
    dt = float(m.opt.timestep) * int(u.frame_skip)
    zero = np.zeros(env.action_space.shape[0], dtype=np.float32)
    toy_bid = int(m.body("test_object1").id)
    toy_jid = next(j for j in range(m.njnt)
                   if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                   and m.body(m.jnt_bodyid[j]).name == "test_object1")
    toy_qadr = int(m.jnt_qposadr[toy_jid])
    toy_dof = int(m.jnt_dofadr[toy_jid])
    reflex = u._orienting

    # ---------------- パネル ----------------
    win = tk.Tk()
    win.title("視線誘導反射を見る（等倍速）")
    _h = min(760, win.winfo_screenheight() - 90)
    win.geometry(f"470x{_h}+30+10")
    win.attributes("-topmost", True)

    tk.Label(win, text="サッケードの設定", font=("", 11, "bold")).pack(pady=(10, 2))

    lat_var = tk.DoubleVar(value=0.2)
    f = tk.Frame(win); f.pack(fill="x", padx=12)
    tk.Label(f, text="間隔[秒]", width=11, anchor="w").pack(side="left")
    tk.Scale(f, from_=0.1, to=1.5, resolution=0.1, orient="horizontal",
             variable=lat_var, length=290).pack(side="left")
    tk.Label(win, text="0.2＝今の設定／0.5〜0.9＝新生児の実測（Aslin & Salapatek 1975）",
             fg="#666", font=("", 8)).pack()

    thr_var = tk.DoubleVar(value=0.02)
    f = tk.Frame(win); f.pack(fill="x", padx=12)
    tk.Label(f, text="発火の閾値", width=11, anchor="w").pack(side="left")
    tk.Scale(f, from_=0.0, to=0.5, resolution=0.01, orient="horizontal",
             variable=thr_var, length=290).pack(side="left")
    tk.Label(win, text="0.02＝今の設定（低すぎて常に発火）／0.25前後で動きを選べる",
             fg="#666", font=("", 8)).pack()

    reflex_var = tk.BooleanVar(value=True)
    tk.Checkbutton(win, text="視線誘導反射を効かせる", variable=reflex_var).pack(anchor="w", padx=20)
    shake_var = tk.BooleanVar(value=True)
    tk.Checkbutton(win, text="おもちゃを小さく激しく揺らす", variable=shake_var).pack(anchor="w", padx=20)
    loop_var = tk.BooleanVar(value=True)
    tk.Checkbutton(win, text=f"{EPISODE_SEC:.0f}秒たったらやり直す", variable=loop_var).pack(anchor="w", padx=20)

    speed_var = tk.DoubleVar(value=SPEED)
    f = tk.Frame(win); f.pack(fill="x", padx=12)
    tk.Label(f, text="再生速度", width=11, anchor="w").pack(side="left")
    tk.Scale(f, from_=0.1, to=2.0, resolution=0.1, orient="horizontal",
             variable=speed_var, length=290).pack(side="left")

    tk.Label(win, text="太郎の目に映っているもの（検出画素は緑）",
             font=("", 10, "bold")).pack(pady=(6, 2))
    eye_canvas = tk.Label(win)
    eye_canvas.pack()
    _imgtk = [None]

    info = tk.Label(win, text="", font=("Consolas", 10), justify="left")
    info.pack(pady=(6, 2))
    fire = tk.Label(win, text="", font=("", 14, "bold"), fg="#a00")
    fire.pack()

    _restart = [True]
    tk.Button(win, text="もう一度最初から", command=lambda: _restart.__setitem__(0, True),
              width=20, bg="#37a", fg="white").pack(pady=8)
    tk.Label(win, text="ビューア: 左ドラッグ=回転 / 右ドラッグ=平行移動 / スクロール=ズーム",
             fg="#666", font=("", 8)).pack()

    # ---------------- 再生 ----------------
    print("\n等倍速で再生します。「間隔」を 0.2 と 0.9 で見比べてください。\n", flush=True)
    with mujoco.viewer.launch_passive(m, d) as viewer:
        t_sim, wall0 = 0.0, time.time()
        base = None
        prev_sacc = 0
        fire_until = -1.0
        devs, seens = [], []
        while viewer.is_running():
            try:
                if not win.winfo_exists():
                    break
            except Exception:
                break

            OR.SACCADE_LATENCY = float(lat_var.get())
            OR.SACCADE_MIN_STRENGTH = float(thr_var.get())
            u._orienting = reflex if reflex_var.get() else None

            if _restart[0]:
                env.reset(seed=0)
                reflex.reset()
                t_sim, wall0 = 0.0, time.time()
                base, prev_sacc = None, 0
                devs, seens = [], []
                _restart[0] = False

            # おもちゃが設置されたら基準位置を覚え、以後は揺らす
            if base is None and not getattr(u, "_toy_pending", True):
                base = np.array(u._rest_pos, dtype=float)
            if base is not None:
                off = (SHAKE_AMP * np.sin(2 * np.pi * SHAKE_HZ * t_sim)
                       if shake_var.get() else 0.0)
                u._rest_pos = base + np.array([0.0, off, 0.0])
                d.qpos[toy_qadr:toy_qadr + 3] = u._rest_pos
                d.qvel[toy_dof:toy_dof + 6] = 0.0

            env.step(zero)
            t_sim += dt

            if reflex.n_saccades > prev_sacc:
                prev_sacc = reflex.n_saccades
                fire_until = t_sim + 0.25
            fire.config(text="● 撃った" if t_sim < fire_until else "")

            u._vision_t = None
            u._vision_cache = None
            imgs = u.get_vision_obs()
            img = imgs.get("eye_left") if isinstance(imgs, dict) else None
            dev, npix, seen = float("nan"), 0, False
            if img is not None:
                v = VIS.visible_in_image(img)
                npix, seen = v["n_pixels"], v["seen"]
                if seen:
                    dev = float(np.hypot(v["cx"], v["cy"]))
                    devs.append(dev)
                seens.append(1.0 if seen else 0.0)
                try:
                    from PIL import Image, ImageTk
                    arr = np.asarray(img)
                    if arr.dtype != np.uint8:
                        arr = np.clip(arr, 0, 255).astype(np.uint8)
                    arr = arr.copy()
                    arr[VIS.red_mask(arr)] = [0, 255, 0]
                    _imgtk[0] = ImageTk.PhotoImage(
                        Image.fromarray(arr).resize((192, 192), Image.NEAREST))
                    eye_canvas.config(image=_imgtk[0])
                except Exception:
                    pass

            seen_pct = (float(np.mean(seens)) * 100) if seens else 0.0
            dev_avg = (float(np.mean(devs)) if devs else float("nan"))
            info.config(
                text=f"経過 {t_sim:5.1f}秒   撃った数 {reflex.n_saccades:3d}\n"
                     f"反応の強さ {reflex.strength:6.3f}（閾値 {thr_var.get():.2f}）\n"
                     f"見えている {'はい' if seen else 'いいえ'}"
                     f"（{npix:4d}画素）  ここまでの割合 {seen_pct:5.1f}%\n"
                     f"中心からのずれ {dev:5.2f}   平均 {dev_avg:5.2f}"
                     f"（0=真ん中／反射OFFで0.48）",
                fg="#070" if seen else "#a00")

            if loop_var.get() and t_sim >= EPISODE_SEC:
                _restart[0] = True

            try:
                win.update()
            except tk.TclError:
                break
            viewer.sync()
            sp = max(0.05, float(speed_var.get()))
            lag = wall0 + t_sim / sp - time.time()
            if lag > 0:
                time.sleep(min(lag, 0.05))

    env.close()
    try:
        win.destroy()
    except Exception:
        pass


if __name__ == "__main__":
    main()
