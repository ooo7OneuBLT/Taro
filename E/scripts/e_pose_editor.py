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

# ⚠️★古い方式（2026-07-30 に整理）。新しい実験は `run/main.py` を通す。
#   【経緯】目標Eの実験スクリプトが118本あり、うち66本が**独立に環境を組み立てていた**。
#     そのため「学習は関節モード（90関節を独立に駆動＝逸脱リスト 逸脱5）、
#     測定とViewerは筋肉モード（拮抗筋2本/関節）」という**別の体で動く**事故が起きた
#     （ユーザーの目視「視線誘導反射の実験の時とは動きが全然違う」で発覚。
#      実測で動きが人間の新生児の約3.3倍速かった）。
#   【設計と移行計画】`E/docs/実行基盤_設計.md`
#   ⚠️このファイルは**記録として残す**（削除しない方針）。
#     中の測り方は再利用できるので、プラグインへ移すときの元にする。
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

# ★腕のリーチ [m]。⚠️2026-07-26 修正：以前は 0.158（上腕7.9 + 前腕8.5cm）としていたが、
#   **手そのものの長さ（指先までの2.2cm）が抜けていた**ため、届く距離を短く見積もり、
#   パネルに「腕の275%」のような誤った割合を表示していた（ユーザー指摘「絶対届くはずなのに」）。
#   実測（`right_upper_arm`→`right_lower_arm`→`right_hand` の距離＋手のgeomの広がり）。
ARM_REACH = 0.186
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
    # ★orient=True で反射を生成しておき、実際に効かせるかはパネルのトグルで切り替える
    env = ToySupineEnv(actuation_model=MuscleModel,
                       vision_params=infant_vision_params(),
                       age=0.0, toy=True, orient=True, **kw)
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
    if saved and "toy_half_size" in saved:
        _g = int(m.body("test_object1").geomadr[0])
        _s = float(saved["toy_half_size"])
        m.geom_size[_g] = [_s, _s, _s]

    # ---------------- パネル ----------------
    win = tk.Tk()
    win.title("太郎 姿勢／おもちゃ 編集パネル")
    # 画面より高い窓は下が切れて保存ボタンにも届かなくなる。画面に収まる高さにして、
    # 中身は縦スクロールできるようにする（項目が増えても収まる）。
    _h = min(980, win.winfo_screenheight() - 90)
    win.geometry(f"500x{_h}+30+10")
    win.attributes("-topmost", True)

    _outer = tk.Frame(win); _outer.pack(fill="both", expand=True)
    _cv = tk.Canvas(_outer, highlightthickness=0)
    _sb = tk.Scrollbar(_outer, orient="vertical", command=_cv.yview)
    _cv.configure(yscrollcommand=_sb.set)
    _sb.pack(side="right", fill="y")
    _cv.pack(side="left", fill="both", expand=True)
    root = tk.Frame(_cv)          # 以降の部品はすべてこの中に置く
    _wid = _cv.create_window((0, 0), window=root, anchor="nw")
    root.bind("<Configure>", lambda e: _cv.configure(scrollregion=_cv.bbox("all")))
    _cv.bind("<Configure>", lambda e: _cv.itemconfigure(_wid, width=e.width))
    _cv.bind_all("<MouseWheel>",
                 lambda e: _cv.yview_scroll(int(-e.delta / 120), "units"))

    # E_FREEZE=0 で物理ONの状態から始める（姿勢が重力で崩れないかを見るとき）
    _freeze0 = os.environ.get("E_FREEZE", "1") == "1"
    state = {"hold_pose": tk.BooleanVar(value=_freeze0),
             "shake": tk.BooleanVar(value=False),
             "freeze": tk.BooleanVar(value=_freeze0),
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

    # おもちゃの大きさ（geom の half-size）
    f = tk.Frame(root); f.pack(fill="x", padx=14)
    tk.Label(f, text="大きさ(半辺)", width=11, anchor="w").pack(side="left")
    toy_gadr = int(m.body("test_object1").geomadr[0])
    size0 = float(m.geom_size[toy_gadr][0])
    size_var = tk.DoubleVar(value=size0)
    tk.Scale(f, from_=0.005, to=0.05, resolution=0.0025, orient="horizontal",
             variable=size_var, length=280, showvalue=True).pack(side="left")

    dist_label = tk.Label(root, text="", font=("Consolas", 10), justify="left")
    dist_label.pack(pady=(6, 4))

    # ★一人称視点（太郎の目に映る映像）。視線がどこを向いているかを直接見る。
    #   第三者視点だけだと「おもちゃが見えているつもり」が起きる（2026-07-26）。
    tk.Label(root, text="太郎の目に映っているもの（左目）",
             font=("", 10, "bold")).pack(pady=(4, 2))
    eye_canvas = tk.Label(root)
    eye_canvas.pack()
    _eye_imgtk = [None]     # GCで消えないよう保持する

    tk.Label(root, text="関節の角度 [度]（左右まとめて）",
             font=("", 11, "bold")).pack(pady=(4, 2))
    def _on_freeze_toggle(*_):
        # ★物理を回すときは「角度で固定する」を自動で外す。
        #   両方ONだと、毎ステップ関節角と速度を強制 → 物理が反力を出す →
        #   また強制、を繰り返してエネルギーが溜まり**太郎が暴れる**
        #   （ユーザーの目視「すごい物理を進めると太郎が暴れる」で発覚）。
        if not state["freeze"].get():
            state["hold_pose"].set(False)

    tk.Checkbutton(root, text="★物理演算を止める（姿勢編集モード。重力も衝突も無し）",
                   variable=state["freeze"], fg="#a30",
                   command=_on_freeze_toggle).pack()
    tk.Checkbutton(root, text="この角度で固定する（★物理ONのまま使うと暴れます）",
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

    # ★2026-07-26 追加：問題を切り替えて目視で確認するためのトグル。
    #   ここまでの測定で見つかった3つの問題を、Viewer で直接見られるようにする。
    tk.Label(root, text="★問題の切り替え（目視で確認する）",
             font=("", 10, "bold"), fg="#a30").pack(pady=(8, 2))
    state["babble"] = tk.BooleanVar(value=False)
    state["reflex"] = tk.BooleanVar(value=False)
    tk.Checkbutton(root, text="自発運動を流す（問題1：眼球が振り回される）",
                   variable=state["babble"]).pack(anchor="w", padx=24)
    tk.Checkbutton(root, text="視線誘導反射を効かせる（問題2：逆効果の疑い）",
                   variable=state["reflex"]).pack(anchor="w", padx=24)
    # ★問題3：屈筋トーンのバネに減衰がなく、手足が振動して頭が揺れる疑い
    #   （数値では頭部角速度が 0.011 → 0.094 rad/s と8倍になる。目視は未確認）
    state["tone"] = tk.BooleanVar(value=True)
    tk.Checkbutton(root, text="屈筋トーンのバネを効かせる（問題3：振動の疑い）",
                   variable=state["tone"]).pack(anchor="w", padx=24)
    head_label = tk.Label(root, text="", font=("Consolas", 9), fg="#a30")
    head_label.pack()

    # ★2026-07-26：めり込みの表示。物理を止めていると「おもちゃが顔にめり込んでいる」
    #   ことが見た目では分からず、物理を回した瞬間に巨大な反力で弾き飛ばされる。
    #   実測：おもちゃが頭に 15.6mm・右目に 13.4mm めり込み、拘束反力 1461+502 Nm。
    #   首の筋力 0.066 Nm の2万倍。0.4秒で首が64度回っていた。
    tk.Label(root, text="めり込み（物理を回すとここで弾かれる）",
             font=("", 10, "bold")).pack(pady=(8, 2))
    pen_label = tk.Label(root, text="", font=("Consolas", 9), justify="left")
    pen_label.pack()

    # ★2026-07-26：測定器（e_visibility）の判定をその場で見せる。
    #   「見えている」を角度だけで測っていて、柵の向こうのおもちゃを
    #   「視界内100%」と数えていた（落とし穴 項51）。目で確かめられるようにする。
    tk.Label(root, text="測定器の判定（3つを突き合わせる）",
             font=("", 10, "bold")).pack(pady=(8, 2))
    judge_label = tk.Label(root, text="", font=("Consolas", 9), justify="left")
    judge_label.pack()
    mask_var = tk.BooleanVar(value=True)
    tk.Checkbutton(root, text="一人称視点に、検出した画素を緑で重ねる",
                   variable=mask_var).pack()

    msg = tk.Label(root, text="", fg="#0a7", font=("", 9))
    msg.pack()

    def save():
        data = {"toy_pos": [float(v.get()) for v in toy_vars],
                "toy_half_size": float(size_var.get()),
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

    def restore_supine():
        """体を仰向けの初期位置に戻す（ビューア側の Reset で転がったときの復帰用）。

        ビューアの Reset（Backspace）は MuJoCo が持つ XML の初期値に戻すだけなので、
        そこから物理が回っておもちゃや柵にぶつかり、仰向けでなくなることがある。
        こちらは「胴体の位置と向き」「スライダーの関節角」「速度ゼロ」をまとめて戻す。
        """
        if root_qpos0 is not None:
            d.qpos[root_qadr:root_qadr + 7] = root_qpos0
        for jd, v in zip(joints, joint_vars):
            ang = np.radians(v.get())
            for jid, qadr in jd["pair"]:
                d.qpos[qadr] = ang
        # おもちゃもスライダーの位置へ戻す（体の下敷きになったまま復帰しないため）
        d.qpos[toy_qadr:toy_qadr + 3] = [v.get() for v in toy_vars]
        d.qvel[:] = 0.0
        d.qacc[:] = 0.0
        mujoco.mj_forward(m, d)
        msg.config(text="仰向けに戻しました（速度もゼロにしました）")

    bf = tk.Frame(root); bf.pack(pady=10)
    tk.Button(bf, text="この設定を保存", command=save, width=14,
              bg="#2a7", fg="white").pack(side="left", padx=4)
    tk.Button(bf, text="仰向けに戻す", command=restore_supine, width=13,
              bg="#37a", fg="white").pack(side="left", padx=4)
    tk.Button(bf, text="初期値に戻す", command=reset_pose, width=12).pack(side="left")

    tk.Label(root, text="ビューア側: 左ドラッグ=回転 / 右ドラッグ=平行移動 / スクロール=ズーム",
             fg="#666", font=("", 8)).pack(side="bottom", pady=6)

    # ---------------- メインループ ----------------
    dt = float(m.opt.timestep) * int(env.unwrapped.frame_skip)
    t = 0.0
    tick = 0
    # 自発運動の生成器（トグルで使う）
    sys.path.insert(0, os.path.join(_ROOT, "taro_core", "src", "brain", "spinal_cord"))
    from cpg import ColoredNoiseGenerator
    _gen = [ColoredNoiseGenerator(n_act, seed=0)]
    _act = [np.zeros(n_act, dtype=np.float32)]
    _saved_reflex = [env.unwrapped._orienting]
    # 屈筋トーンのバネを切り替えるために、今の stiffness を覚えておく
    _tone_joints = [j for j in range(m.njnt) if float(m.jnt_stiffness[j]) > 0.0]
    _tone_k = {j: float(m.jnt_stiffness[j]) for j in _tone_joints}
    _tone_on = [True]
    _head_bid = m.body("head").id
    _head_w = []
    # 体の根元（胴体の自由関節）の初期位置を覚えておく＝編集中に流れないように
    root_qadr = None
    for j in range(m.njnt):
        # ⚠️★【2026-07-29 修正】「おもちゃ以外の自由関節」で探すと、MIMo本家の
        #   シーンに元からある **test_object2**（使っていない球）を掴む。
        #   太郎の体は body 名 "mimo_location"（関節名は "mimo_orientation"）。
        #   落とし穴 項67 の再発。名指しで取ること。
        if (m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                and (m.body(int(m.jnt_bodyid[j])).name or "") == "mimo_location"):
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
            # ★TOY_MODE="hold"（親が手に持っている）は step のたびに `_rest_pos` へ
            #   位置を書き戻すので、ここで qpos を書いても**上書きされて効かない**
            #   （ユーザーの目視「物理をONにするとおもちゃがスライダーで変更できない」）。
            #   スライダーの値を `_rest_pos` 側にも入れて、親が持つ位置ごと動かす。
            env.unwrapped._rest_pos = pos.copy()
            # おもちゃの大きさをスライダーに追従させる
            s = float(size_var.get())
            if abs(float(m.geom_size[toy_gadr][0]) - s) > 1e-9:
                m.geom_size[toy_gadr] = [s, s, s]
                # 質量と慣性も大きさに合わせる（密度一定＝体積比で変える）
                dens = float(m.body_mass[toy_bid]) / max((2 * size0) ** 3, 1e-12)
                m.body_mass[toy_bid] = dens * (2 * s) ** 3

            # 関節をスライダー値に固定
            if state["hold_pose"] .get() or freeze:
                for jd, v in zip(joints, joint_vars):
                    ang = np.radians(v.get())
                    for jid, qadr in jd["pair"]:
                        d.qpos[qadr] = ang
                        d.qvel[m.jnt_dofadr[jid]] = 0.0

            # ★問題3の切り替え：屈筋トーンのバネを効かせるかどうか
            want_tone = state["tone"].get()
            if want_tone != _tone_on[0]:
                for jid in _tone_joints:
                    m.jnt_stiffness[jid] = _tone_k[jid] if want_tone else 0.0
                _tone_on[0] = want_tone

            # ★問題の切り替え：反射を効かせるかどうか（環境側の apply を止める）
            rf = env.unwrapped._orienting
            if rf is not None:
                env.unwrapped._orienting = rf if state["reflex"].get() else None
                _saved_reflex[0] = rf
            elif state["reflex"].get() and _saved_reflex[0] is not None:
                env.unwrapped._orienting = _saved_reflex[0]

            if freeze:
                # ★物理を回さない。関節角から体の位置を計算するだけ（純粋な運動学）。
                #   衝突も重力も効かないので、膝を曲げても柵にぶつかって飛ばない。
                if root_qpos0 is not None:
                    d.qpos[root_qadr:root_qadr + 7] = root_qpos0   # 体が流れないよう固定
                d.qvel[:] = 0.0
                d.qacc[:] = 0.0
                mujoco.mj_forward(m, d)
            else:
                # ★問題1を見るためのトグル：自発運動を流すか
                if state["babble"].get():
                    if tick % 10 == 0:
                        _act[0] = np.clip(0.5 + 0.174 * _gen[0].sample(0.7), 0.0, 1.0
                                          ).astype(np.float32)
                    env.step(_act[0])
                else:
                    env.step(zero)
            t += dt
            tick += 1
            # 頭部角速度を記録（問題3を数値でも見る）
            _head_w.append(float(np.linalg.norm(d.cvel[_head_bid][:3])))
            if len(_head_w) > 300:
                _head_w.pop(0)

            if tick % 5 == 0:
                # パネルの窓を閉じたあとにラベルを触ると TclError で落ちる
                # （閉じるたびに赤いエラーが出ていた）。先に生存を確かめて抜ける。
                try:
                    if not win.winfo_exists():
                        break
                except Exception:
                    break
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
                if _head_w:
                    hw = np.array(_head_w)
                    warn = ""
                    if not freeze and state["hold_pose"].get():
                        warn = "\n★「この角度で固定する」がONのまま物理を回しています＝暴れます"
                    head_label.config(
                        text=f"頭の角速度  平均 {hw.mean():.3f}  最大 {hw.max():.3f} rad/s"
                             f"（直近3秒。落ち着いていれば 0.01〜0.03 が目安）{warn}")

                # ★太郎の体とおもちゃのめり込みを列挙する（床どうしは除く）
                pen = []
                for ci in range(d.ncon):
                    c = d.contact[ci]
                    if c.dist >= -1e-4:          # 0.1mm 未満は無視
                        continue
                    b1 = int(m.geom_bodyid[c.geom1])
                    b2 = int(m.geom_bodyid[c.geom2])
                    n1, n2 = m.body(b1).name, m.body(b2).name
                    if toy_bid not in (b1, b2):  # おもちゃが絡むものだけ見る
                        continue
                    other = n2 if b1 == toy_bid else n1
                    pen.append((other, -c.dist * 1000))
                if pen:
                    pen.sort(key=lambda x: -x[1])
                    txt = "★おもちゃがめり込んでいる：\n" + "\n".join(
                        f"   {nm:<16} {mm:6.2f} mm" for nm, mm in pen[:4])
                    pen_label.config(text=txt, fg="#a00")
                else:
                    pen_label.config(text="めり込みなし（この位置なら弾かれません）",
                                     fg="#070")
                # 一人称視点を更新（重いので5tickに1回＝約20Hz相当より粗く）
                if tick % 20 == 0:
                    try:
                        # ★視覚は VISION_MIN_DT(0.1秒) のキャッシュを持つ。
                        #   物理を止めていると data.time が進まないので
                        #   **キャッシュが永久に切れず、最初の1枚が出続ける**
                        #   （ユーザーの目視「最初は緑になってたのに途中から
                        #     緑にならなくなった」で発覚）。毎回捨てて描き直す。
                        env.unwrapped._vision_t = None
                        env.unwrapped._vision_cache = None
                        imgs = env.unwrapped.get_vision_obs()
                        if isinstance(imgs, dict) and "eye_left" in imgs:
                            from PIL import Image, ImageTk
                            import e_visibility as VIS
                            arr = np.asarray(imgs["eye_left"])
                            if arr.dtype != np.uint8:
                                arr = np.clip(arr, 0, 255).astype(np.uint8)

                            # ★測定器の3つの判定をその場で出す
                            rep = VIS.report(m, d, toy_bid, arr)
                            j = f"①角度   {rep['angle']:5.1f}°  " \
                                f"{'視野内' if rep['in_fov'] else '★視野外'}\n"
                            j += ("②光線   遮蔽なし\n" if rep["ray_ok"]
                                  else f"②光線   ★{rep['ray_hit']}に遮られている\n")
                            if rep["pix_seen"]:
                                j += (f"③画像   {rep['n_pixels']:4d}画素"
                                      f"（画面の{rep['frac']*100:.1f}%）  "
                                      f"中心からのずれ {np.hypot(rep['cx'], rep['cy']):.2f}")
                            else:
                                j += "③画像   ★映っていない"
                            judge_label.config(
                                text=j,
                                fg="#070" if rep.get("pix_seen") else "#a00")

                            if mask_var.get():
                                # 検出した画素を緑で塗って重ねる＝測定器が
                                # 「どこをおもちゃと認識しているか」を目で見る
                                msk = VIS.red_mask(arr)
                                arr = arr.copy()
                                arr[msk] = [0, 255, 0]
                            im = Image.fromarray(arr).resize((192, 192),
                                                              Image.NEAREST)
                            _eye_imgtk[0] = ImageTk.PhotoImage(im)
                            eye_canvas.config(image=_eye_imgtk[0])
                    except Exception as e:
                        judge_label.config(text=f"（判定できない: {e}）", fg="#a60")
                try:
                    win.update()
                except tk.TclError:
                    break
            viewer.sync()

    env.close()
    try:
        win.destroy()
    except Exception:
        pass


if __name__ == "__main__":
    main()
