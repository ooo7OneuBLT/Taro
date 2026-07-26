"""姿勢とおもちゃ位置の編集ツール（MuJoCo Viewer + スライダーのパネル）。

【なぜ作るか】MuJoCo の Viewer は Unity のような XYZ 矢印（ギズモ）を持たず、
Ctrl+右ドラッグは**カメラ平面内にしか動かせない**（奥行きが変えられない）。
また数値がターミナルにしか出ないので、見ながら調整できない。
→ スライダーで各軸を独立に動かし、距離を画面に出し、保存できるパネルを別窓で出す。

【できること】
  ・おもちゃの位置を X / Y / Z のスライダーで動かす
  ・目・肩・手からの距離をリアルタイム表示
  ・関節の角度（股・膝・肘）をスライダーで動かして姿勢を探す
  ・「保存」でその設定を JSON に書き出す

使い方:
    .venv/Scripts/python.exe E/scripts/e_pose_editor.py
環境変数:
    E_FLEXION=1   生理的屈曲（伸展側の壁）をON
    E_VOR=0       前庭動眼反射をOFF
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
for p in [os.path.join(_ROOT, "D", "scripts"), os.path.join(_ROOT, "MIMo"),
          os.path.join(_ROOT, "taro_core"), _HERE]:
    if p not in sys.path:
        sys.path.insert(0, p)
try: sys.stdout.reconfigure(errors="replace")
except Exception: pass

import numpy as np
import mujoco
import mujoco.viewer
import tkinter as tk

ARM_REACH = 0.158
# 編集する関節（左右まとめて動かす）
POSE_JOINTS = [("hip1", "股（前後）"), ("hip2", "股（開き）"),
               ("knee", "膝"),
               ("shoulder_horizontal", "肩（前後）"),
               ("shoulder_ad_ab", "肩（開き）"),
               ("shoulder_rotation", "肩（ひねり）"),
               ("elbow", "肘")]
SAVE_PATH = os.path.join(_HERE, os.pardir, "docs", "pose_editor_saved.json")


def main():
    from e_toy_env import ToySupineEnv, infant_vision_params
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env

    kw = body_kwargs_from_env(0.0, verbose=True)
    env = ToySupineEnv(actuation_model=MuscleModel,
                       vision_params=infant_vision_params(),
                       age=0.0, toy=True, orient=False, **kw)
    m, d = env.unwrapped.model, env.unwrapped.data
    env.reset(seed=0)
    env.unwrapped._anchor = None          # 吊り紐を無効化（勝手に戻らないように）

    n_act = env.action_space.shape[0]
    zero = np.zeros(n_act, dtype=np.float32)

    toy_bid = m.body("test_object1").id
    toy_jid = m.body_jntadr[toy_bid]
    toy_qadr = int(m.jnt_qposadr[toy_jid])
    toy_dof = int(m.jnt_dofadr[toy_jid])
    toy_pos0 = d.qpos[toy_qadr:toy_qadr + 3].copy()

    eye_bid = m.body("left_eye").id
    sh_bid = m.body("right_upper_arm").id
    hand_bid = m.body("right_hand").id
    # 視線方向を測るためのカメラ（MuJoCoのカメラは自分の -Z 方向を見る）
    cam_id = None
    for c in range(m.ncam):
        if "eye" in (m.camera(c).name or ""):
            cam_id = c
            break
    HALF_FOV = 30.0   # 視野角60度の半分。この角度を超えると視界の外

    # ★前回保存した設定があれば読み込む（作業をやり直さなくて済むように）
    saved = None
    if os.path.exists(SAVE_PATH):
        try:
            with open(SAVE_PATH, encoding="utf-8") as fp:
                saved = json.load(fp)
            print(f"[load] 前回の保存を読み込みました: {SAVE_PATH}", flush=True)
        except Exception as e:
            print(f"[load] 読み込み失敗: {e}", flush=True)

    # 編集対象の関節（左右ペア）を集める
    joints = []
    for base, jp in POSE_JOINTS:
        pair = []
        for side in ("right_", "left_"):
            try:
                j = m.joint("robot:" + side + base)
            except Exception:
                continue
            pair.append((int(j.id), int(m.jnt_qposadr[j.id])))
        if pair:
            lo, hi = np.degrees(m.jnt_range[pair[0][0]])
            cur = float(np.degrees(d.qpos[pair[0][1]]))
            if saved and base in saved.get("joints", {}):
                cur = float(saved["joints"][base])     # 保存値を初期値にする
            joints.append(dict(base=base, jp=jp, pair=pair,
                               lo=float(lo), hi=float(hi), init=cur))
    if saved and "toy_pos" in saved:
        toy_pos0 = np.array(saved["toy_pos"], dtype=float)

    # ---------------- パネル ----------------
    root = tk.Tk()
    root.title("太郎 姿勢／おもちゃ 編集パネル")
    root.geometry("460x760+40+20")
    root.attributes("-topmost", True)

    state = {"hold_pose": tk.BooleanVar(value=True),
             "shake": tk.BooleanVar(value=False),
             "freeze": tk.BooleanVar(value=True),   # ★既定で物理を止める
             "toy": list(toy_pos0)}

    tk.Label(root, text="おもちゃの位置 [m]", font=("", 11, "bold")).pack(pady=(10, 2))
    toy_vars = []
    for i, (axis, lo, hi) in enumerate([("X（頭↔足）", -0.1, 0.5),
                                         ("Y（左↔右）", -0.3, 0.3),
                                         ("Z（低↔高）", 0.0, 0.5)]):
        f = tk.Frame(root); f.pack(fill="x", padx=14)
        tk.Label(f, text=axis, width=11, anchor="w").pack(side="left")
        v = tk.DoubleVar(value=float(toy_pos0[i]))
        tk.Scale(f, from_=lo, to=hi, resolution=0.005, orient="horizontal",
                 variable=v, length=280, showvalue=True).pack(side="left")
        toy_vars.append(v)

    dist_label = tk.Label(root, text="", font=("Consolas", 10), justify="left")
    dist_label.pack(pady=(6, 10))

    tk.Label(root, text="関節の角度 [度]（左右まとめて）",
             font=("", 11, "bold")).pack(pady=(4, 2))
    tk.Checkbutton(root, text="★物理演算を止める（姿勢編集モード。重力も衝突も無し）",
                   variable=state["freeze"], fg="#a30").pack()
    tk.Checkbutton(root, text="この角度で固定する（外すと物理に任せる）",
                   variable=state["hold_pose"]).pack()
    joint_vars = []
    for jd in joints:
        f = tk.Frame(root); f.pack(fill="x", padx=14)
        tk.Label(f, text=jd["jp"], width=11, anchor="w").pack(side="left")
        v = tk.DoubleVar(value=jd["init"])
        tk.Scale(f, from_=jd["lo"], to=jd["hi"], resolution=1, orient="horizontal",
                 variable=v, length=280, showvalue=True).pack(side="left")
        joint_vars.append(v)

    tk.Checkbutton(root, text="おもちゃを小さく激しく揺らす（1.5cm / 2.5Hz）",
                   variable=state["shake"]).pack(pady=(8, 2))

    msg = tk.Label(root, text="", fg="#0a7", font=("", 9))
    msg.pack()

    def save():
        data = {"toy_pos": [float(v.get()) for v in toy_vars],
                "joints": {jd["base"]: float(v.get())
                           for jd, v in zip(joints, joint_vars)}}
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        with open(SAVE_PATH, "w", encoding="utf-8") as fp:
            json.dump(data, fp, ensure_ascii=False, indent=2)
        msg.config(text=f"保存しました → {os.path.relpath(SAVE_PATH, _ROOT)}")
        print("[saved]", json.dumps(data, ensure_ascii=False), flush=True)

    def reset_pose():
        for jd, v in zip(joints, joint_vars):
            v.set(jd["init"])
        for i in range(3):
            toy_vars[i].set(float(toy_pos0[i]))
        msg.config(text="初期値に戻しました")

    bf = tk.Frame(root); bf.pack(pady=10)
    tk.Button(bf, text="この設定を保存", command=save, width=16,
              bg="#2a7", fg="white").pack(side="left", padx=6)
    tk.Button(bf, text="初期値に戻す", command=reset_pose, width=14).pack(side="left")

    tk.Label(root, text="ビューア側: 左ドラッグ=回転 / 右ドラッグ=平行移動 / スクロール=ズーム",
             fg="#666", font=("", 8)).pack(side="bottom", pady=6)

    # ---------------- メインループ ----------------
    dt = float(m.opt.timestep) * int(env.unwrapped.frame_skip)
    t = 0.0
    tick = 0
    # 体の根元（胴体の自由関節）の初期位置を覚えておく＝編集中に流れないように
    root_qadr = None
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and m.body(m.jnt_bodyid[j]).name != "test_object1":
            root_qadr = int(m.jnt_qposadr[j])
            break
    root_qpos0 = d.qpos[root_qadr:root_qadr + 7].copy() if root_qadr is not None else None

    print("\nパネルとビューアを開きました。", flush=True)
    print("★既定で『物理演算を止める』がONです（重力も衝突も効かないので、",
          flush=True)
    print("  関節を動かしても飛んでいきません）。物理を見たいときはチェックを外してください。\n",
          flush=True)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            freeze = state["freeze"].get()

            # おもちゃの位置をスライダー値に固定
            pos = np.array([v.get() for v in toy_vars], dtype=float)
            if state["shake"].get():
                pos = pos + np.array([0.0,
                                      0.015 * np.sin(2 * np.pi * 2.5 * t), 0.0])
            d.qpos[toy_qadr:toy_qadr + 3] = pos
            d.qvel[toy_dof:toy_dof + 6] = 0.0

            # 関節をスライダー値に固定
            if state["hold_pose"] .get() or freeze:
                for jd, v in zip(joints, joint_vars):
                    ang = np.radians(v.get())
                    for jid, qadr in jd["pair"]:
                        d.qpos[qadr] = ang
                        d.qvel[m.jnt_dofadr[jid]] = 0.0

            if freeze:
                # ★物理を回さない。関節角から体の位置を計算するだけ（純粋な運動学）。
                #   衝突も重力も効かないので、膝を曲げても柵にぶつかって飛ばない。
                if root_qpos0 is not None:
                    d.qpos[root_qadr:root_qadr + 7] = root_qpos0   # 体が流れないよう固定
                d.qvel[:] = 0.0
                d.qacc[:] = 0.0
                mujoco.mj_forward(m, d)
            else:
                env.step(zero)
            t += dt
            tick += 1

            if tick % 5 == 0:
                toy = d.xpos[toy_bid]
                de = np.linalg.norm(toy - d.xpos[eye_bid]) * 100
                ds = np.linalg.norm(toy - d.xpos[sh_bid]) * 100
                dh = np.linalg.norm(toy - d.xpos[hand_bid]) * 100
                # ★視線とおもちゃのなす角＝視界に入っているかの判定
                gaze_txt = ""
                if cam_id is not None:
                    cpos = d.cam_xpos[cam_id]
                    fwd = -d.cam_xmat[cam_id].reshape(3, 3)[:, 2]   # カメラは -Z を見る
                    v = toy - cpos
                    nv = np.linalg.norm(v)
                    if nv > 1e-9:
                        ang = float(np.degrees(np.arccos(
                            np.clip(np.dot(fwd, v / nv), -1, 1))))
                        inside = ang < HALF_FOV
                        gaze_txt = (f"\n視線からのズレ  {ang:6.1f}°  "
                                    f"{'視界の中' if inside else '★視界の外'}"
                                    f"（限界 {HALF_FOV:.0f}°）")
                        dist_label.config(fg="#070" if inside else "#a00")
                dist_label.config(
                    text=f"目 → おもちゃ   {de:6.1f} cm\n"
                         f"肩 → おもちゃ   {ds:6.1f} cm  （腕の {ds/(ARM_REACH*100)*100:3.0f}%）\n"
                         f"手 → おもちゃ   {dh:6.1f} cm" + gaze_txt)
                try:
                    root.update()
                except tk.TclError:
                    break
            viewer.sync()

    env.close()
    try:
        root.destroy()
    except Exception:
        pass


if __name__ == "__main__":
    main()
