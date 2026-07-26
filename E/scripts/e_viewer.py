"""★太郎を見るための統一ビューア。おもちゃ・姿勢・反射・測定器・再生を1枚にまとめる。

【なぜ作ったか】ビューアが3つに分かれていて、それぞれ別のパネルを持っていた。
ユーザーの要望（2026-07-26）：
> 編集設定は毎回出すようにできない？
> それかすべての編集ウィンドウを統一して、それぞれを表示したり、畳んだりできるようにするとか

【統合したもの】
    e_pose_editor.py          姿勢・おもちゃの編集
    e_reset_replay_viewer.py  リセット直後を等倍速で見る
    e_orient_viewer.py        視線誘導反射の観察
（元の3つは記録として残す。今後はこのファイルを使う）

【使い方】
    python E/scripts/e_viewer.py

  見出しをクリックすると、その区画を開いたり畳んだりできる。
  設定は「この設定を保存」で `E/docs/viewer_saved.json` に残り、次回自動で読み込む。

【区画】
    おもちゃ    位置・大きさ・揺らす
    姿勢        関節の角度・物理のON/OFF・仰向けに戻す
    反射        視線誘導（間隔・閾値）・前庭動眼反射・屈筋トーン
    測定器      3つの判定・一人称視点（★検出画素を緑で表示）
    再生        速度・やり直し・自発運動

【環境変数】
    E_SPEED       再生速度（既定 1.0＝等倍）
    E_FREEZE      1 で物理を止めた状態から始める（既定 1）
    E_TOY_MODE    hold（親が持つ・既定）／tether（吊る）／free（落ちる）
    E_TOY_DELAY   おもちゃが登場するまでの秒数（既定 1.0）
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
for p in [os.path.join(_ROOT, "D", "scripts"), os.path.join(_ROOT, "MIMo"),
          os.path.join(_ROOT, "taro_core"),
          os.path.join(_ROOT, "taro_core", "src", "body"),
          os.path.join(_ROOT, "taro_core", "src", "brain"),
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

SAVE_PATH = os.path.join(_HERE, os.pardir, "docs", "viewer_saved.json")
OLD_SAVE = os.path.join(_HERE, os.pardir, "docs", "pose_editor_saved.json")

ARM_REACH = 0.186          # 肩から手先まで[m]（上腕+前腕+手）
HALF_FOV = 30.0            # 視野の半角[度]（fovy=60）
SHAKE_HZ = 2.5             # 「小さく激しく」＝2.5Hz
SHAKE_AMP = 0.015          # 1.5cm
EPISODE_SEC = 20.0

POSE_JOINTS = [
    ("hip1", "股（前後）"), ("hip2", "股（開き）"), ("knee", "ひざ"),
    ("shoulder_horizontal", "肩（前後）"), ("shoulder_ad_ab", "肩（開き）"),
    ("shoulder_rotation", "肩（ひねり）"), ("elbow", "ひじ"),
]


class Section:
    """クリックで開閉できる区画。畳めば縦に収まる。"""

    def __init__(self, parent, title, opened=True):
        self.title = title
        self.opened = bool(opened)
        self.head = tk.Button(parent, anchor="w", relief="flat",
                              bg="#dde3ee", font=("", 10, "bold"),
                              command=self.toggle)
        self.head.pack(fill="x", padx=6, pady=(8, 0))
        self.body = tk.Frame(parent)
        if self.opened:
            self._show()
        self._label()

    def _show(self):
        # ★`after=self.head` が必須。pack_forget したフレームをそのまま pack すると
        #   **元の位置ではなく一番下に付く**ので、閉じて開くと区画が末尾へ飛び、
        #   見た目には「展開できない」ように見える（ユーザーの目視 2026-07-26）。
        self.body.pack(fill="x", padx=4, after=self.head)

    def _label(self):
        self.head.config(text=("▼ " if self.opened else "▶ ") + self.title)

    def toggle(self):
        self.opened = not self.opened
        if self.opened:
            self._show()
        else:
            self.body.pack_forget()
        self._label()


def slider(parent, label, var, lo, hi, res, width=11, length=280, note=None,
           command=None):
    f = tk.Frame(parent); f.pack(fill="x", padx=10)
    tk.Label(f, text=label, width=width, anchor="w").pack(side="left")
    tk.Scale(f, from_=lo, to=hi, resolution=res, orient="horizontal",
             variable=var, length=length, command=command).pack(side="left")
    if note:
        tk.Label(parent, text=note, fg="#666", font=("", 8)).pack(anchor="w", padx=14)


def main():
    from e_toy_env import ToySupineEnv, infant_vision_params
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env
    import e_orienting_v2 as OR
    import e_toy_env as TE

    kw = body_kwargs_from_env(0.0, verbose=True)
    env = ToySupineEnv(actuation_model=MuscleModel,
                       vision_params=infant_vision_params(),
                       age=0.0, toy=True, vor=True, orient=True, **kw)
    u = env.unwrapped
    m, d = u.model, u.data
    env.reset(seed=0)
    dt = float(m.opt.timestep) * int(u.frame_skip)
    n_act = env.action_space.shape[0]
    zero = np.zeros(n_act, dtype=np.float32)

    toy_bid = int(m.body("test_object1").id)
    toy_jid = int(m.body_jntadr[toy_bid])
    toy_qadr = int(m.jnt_qposadr[toy_jid])
    toy_dof = int(m.jnt_dofadr[toy_jid])
    toy_gadr = int(m.body("test_object1").geomadr[0])
    head_bid = int(m.body("head").id)
    eye_bid = int(m.body("left_eye").id)
    sh_bid = int(m.body("right_upper_arm").id)
    hand_bid = int(m.body("right_hand").id)
    cam_id = next((c for c in range(m.ncam) if "eye" in (m.camera(c).name or "")), None)

    reflex = u._orienting
    vor = u._vor
    tone_joints = [j for j in range(m.njnt) if float(m.jnt_stiffness[j]) > 0.0]
    tone_k = {j: float(m.jnt_stiffness[j]) for j in tone_joints}

    # 首の3軸（前後の傾き／左右のひねり／左右の傾き）
    neck_ids = {}
    for j in range(m.njnt):
        nm = m.joint(j).name.split(":")[-1]
        if nm in ("head_tilt", "head_swivel", "head_tilt_side"):
            neck_ids[nm] = j
    neck_damp0 = {k: float(m.dof_damping[int(m.jnt_dofadr[v])])
                  for k, v in neck_ids.items()}
    # 首の軸まわりの頭の慣性（平行軸の定理）＝臨界減衰の計算に使う
    _hb = int(m.body("head").id)
    _neck_arm = float(np.linalg.norm(
        np.array(d.xpos[_hb]) - np.array(d.xpos[int(m.body_parentid[_hb])])))
    NECK_I = float(m.body_inertia[_hb][0]) + float(m.body_mass[_hb]) * _neck_arm ** 2

    # 体の根元（胴体の自由関節）＝「仰向けに戻す」で使う
    root_qadr = None
    for j in range(m.njnt):
        if (m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                and m.body(m.jnt_bodyid[j]).name != "test_object1"):
            root_qadr = int(m.jnt_qposadr[j])
            break
    root_qpos0 = (d.qpos[root_qadr:root_qadr + 7].copy()
                  if root_qadr is not None else None)

    # ---- 前回の保存を読む（旧ファイルからも引き継ぐ）--------------------
    saved = None
    for path in (SAVE_PATH, OLD_SAVE):
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as fp:
                    saved = json.load(fp)
                print(f"[load] 保存を読み込みました: {path}", flush=True)
                break
            except Exception as e:
                print(f"[load] 読み込み失敗 {path}: {e}", flush=True)

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
                cur = float(saved["joints"][base])
            joints.append(dict(base=base, jp=jp, pair=pair,
                               lo=float(lo), hi=float(hi), init=cur))

    toy_pos0 = d.qpos[toy_qadr:toy_qadr + 3].copy()
    if saved and "toy_pos" in saved:
        toy_pos0 = np.array(saved["toy_pos"], dtype=float)
    size0 = float(m.geom_size[toy_gadr][0])
    if saved and "toy_half_size" in saved:
        size0 = float(saved["toy_half_size"])
        m.geom_size[toy_gadr] = [size0, size0, size0]

    # ================= パネル =================
    win = tk.Tk()
    win.title("太郎ビューア（統一版）")
    _h = min(1000, win.winfo_screenheight() - 80)
    win.geometry(f"520x{_h}+20+10")
    win.attributes("-topmost", True)

    outer = tk.Frame(win); outer.pack(fill="both", expand=True)
    cv = tk.Canvas(outer, highlightthickness=0)
    sb = tk.Scrollbar(outer, orient="vertical", command=cv.yview)
    cv.configure(yscrollcommand=sb.set)
    sb.pack(side="right", fill="y")
    cv.pack(side="left", fill="both", expand=True)
    root = tk.Frame(cv)
    wid = cv.create_window((0, 0), window=root, anchor="nw")
    root.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
    cv.bind("<Configure>", lambda e: cv.itemconfigure(wid, width=e.width))
    cv.bind_all("<MouseWheel>", lambda e: cv.yview_scroll(int(-e.delta / 120), "units"))

    op = (saved or {}).get("open", {})
    freeze0 = os.environ.get("E_FREEZE", "1") == "1"

    # ---- 区画1：おもちゃ ------------------------------------------------
    sec_toy = Section(root, "おもちゃ", op.get("toy", True))
    # ★スライダーを動かしたら自動で「固定する」に切り替える。
    #   そうしないと、既定（環境まかせ）のときスライダーが効かず、
    #   「動かせない」ように見える（ユーザーの目視 2026-07-26）。
    _ui_ready = [False]
    _syncing = [False]

    def on_toy_slider(*_):
        if _ui_ready[0] and not _syncing[0]:
            follow_var.set(True)

    toy_vars = []
    for i, (ax, lo, hi) in enumerate([("X（頭↔足）", -0.1, 0.5),
                                      ("Y（左↔右）", -0.3, 0.3),
                                      ("Z（低↔高）", 0.0, 0.5)]):
        v = tk.DoubleVar(value=float(toy_pos0[i]))
        slider(sec_toy.body, ax, v, lo, hi, 0.005, command=on_toy_slider)
        toy_vars.append(v)
    size_var = tk.DoubleVar(value=size0)
    slider(sec_toy.body, "大きさ(半辺)", size_var, 0.005, 0.05, 0.0025)
    shake_var = tk.BooleanVar(value=bool((saved or {}).get("shake", True)))
    tk.Checkbutton(sec_toy.body, text=f"小さく激しく揺らす（{SHAKE_HZ}Hz・{SHAKE_AMP*100:.1f}cm）",
                   variable=shake_var).pack(anchor="w", padx=14)
    # ★既定は「環境まかせ」＝環境が視線の正面に置く。
    #   保存された位置は**あなたが手で置いたもの**で、視線の正面とは限らない
    #   （実際 X=0.155 は目 X≈0.20 より頭側＝視線の後ろだった。2026-07-26）。
    #   反射を試すときは環境まかせにしないと、条件が変わって比べられない。
    follow_var = tk.BooleanVar(value=bool((saved or {}).get("follow", False)))
    tk.Checkbutton(sec_toy.body,
                   text="★スライダーの位置に固定する（外すと環境が視線の正面に置く）",
                   variable=follow_var).pack(anchor="w", padx=14)
    tk.Label(sec_toy.body,
             text=f"持たせ方 {TE.TOY_MODE} ／ 登場 {TE.TOY_APPEAR_DELAY:.1f}秒後に"
                  f"{TE.TOY_APPROACH_SEC:.1f}秒かけて{TE.TOY_APPROACH_FROM}から",
             fg="#666", font=("", 8)).pack(anchor="w", padx=14)

    # ---- ★あなたが「親」をやる ------------------------------------------
    #   人間の親は、赤ちゃんの顔の向きを見ておもちゃをその前に持っていく。
    #   太郎にはそれが無いので、首が動くとおもちゃが視界から外れたままになる。
    #   自動化する前に**手で試して、必要な介入の性質を掴む**のが目的。
    #   ⚠️瞬間移動はしない（ワープは随伴性の学習を壊す。_apply_tether の注記と同じ）。
    tk.Label(sec_toy.body, text="★あなたが「親」をやる",
             font=("", 10, "bold")).pack(anchor="w", padx=14, pady=(8, 0))
    tk.Label(sec_toy.body, justify="left", fg="#555", font=("", 8),
             text="人間の親は赤ちゃんの顔の向きに合わせておもちゃを見せる。\n"
                  "太郎にはそれが無いので、首が動くと視界から外れたままになる。\n"
                  "⚠️手動は「何が必要か」を掴むためのもの。数値の比較には使えない。"
             ).pack(anchor="w", padx=18)
    carry_sec = tk.DoubleVar(value=1.0)
    slider(sec_toy.body, "運ぶ秒数", carry_sec, 0.2, 3.0, 0.1, width=9, length=250,
           note="瞬間移動させない（随伴性を壊すため）")
    _parent = [None]     # 手動で運んでいる最中の状態
    parent_log = []      # 押した時刻の記録

    def bring_to_face():
        """今の視線の正面へ、指定秒かけておもちゃを運ぶ（＝親が持っていく）。"""
        cid = int(m.camera("eye_left").id)
        eyes = []
        for nm2 in ("eye_left", "eye_right"):
            try:
                eyes.append(np.array(d.cam_xpos[int(m.camera(nm2).id)], dtype=float))
            except Exception:
                pass
        origin = np.mean(eyes, axis=0) if eyes else np.array(d.cam_xpos[cid], dtype=float)
        fwd = -np.array(d.cam_xmat[cid], dtype=float).reshape(3, 3)[:, 2]
        dist = float(getattr(u, "_toy_dist", 0.086))
        goal = origin + fwd * dist
        start = np.array(d.xpos[toy_bid], dtype=float)
        _parent[0] = {"t": 0.0, "from": start, "to": goal,
                      "sec": max(0.05, float(carry_sec.get()))}
        follow_var.set(True)     # 手動モードに切り替える
        parent_log.append(None)   # 時刻はループ側で埋める
        msg.config(text=f"親が顔の前へ運んでいます（{len(parent_log)}回目）")

    tk.Button(sec_toy.body, text="★顔の前へ持っていく", command=bring_to_face,
              width=22, bg="#a63", fg="white").pack(pady=4)
    parent_label = tk.Label(sec_toy.body, text="", font=("Consolas", 9),
                            justify="left")
    parent_label.pack(anchor="w", padx=18)

    # ---- 区画2：姿勢 ----------------------------------------------------
    sec_pose = Section(root, "姿勢", op.get("pose", True))
    st_freeze = tk.BooleanVar(value=freeze0)
    st_hold = tk.BooleanVar(value=freeze0)

    def on_freeze(*_):
        # 物理を回すときは角度の固定を外す（両方ONだと反力と喧嘩して暴れる）
        if not st_freeze.get():
            st_hold.set(False)

    tk.Checkbutton(sec_pose.body, text="★物理演算を止める（編集モード。重力も衝突も無し）",
                   variable=st_freeze, fg="#a30", command=on_freeze).pack(anchor="w", padx=14)
    tk.Checkbutton(sec_pose.body, text="この角度で固定する（★物理ONのまま使うと暴れます）",
                   variable=st_hold).pack(anchor="w", padx=14)
    joint_vars = []
    for jd in joints:
        v = tk.DoubleVar(value=jd["init"])
        slider(sec_pose.body, jd["jp"], v, jd["lo"], jd["hi"], 1)
        joint_vars.append(v)

    # ---- 区画3：反射 ----------------------------------------------------
    sec_ref = Section(root, "反射", op.get("reflex", True))
    st_orient = tk.BooleanVar(value=bool((saved or {}).get("orient", False)))
    tk.Checkbutton(sec_ref.body, text="視線誘導反射（動くものに目・首を向ける）",
                   variable=st_orient).pack(anchor="w", padx=14)
    lat_var = tk.DoubleVar(value=float((saved or {}).get("latency", OR.SACCADE_LATENCY)))
    slider(sec_ref.body, "  間隔[秒]", lat_var, 0.1, 1.5, 0.1,
           note="0.2＝旧設定（撃ちすぎて視界から追い出す）／0.5〜0.9＝新生児の実測")
    thr_var = tk.DoubleVar(value=float((saved or {}).get("threshold", OR.SACCADE_MIN_STRENGTH)))
    slider(sec_ref.body, "  発火の閾値", thr_var, 0.0, 0.5, 0.01,
           note="0.02＝旧設定（低すぎて常に発火）／0.25前後で動きを選べる")
    st_vor = tk.BooleanVar(value=True)
    tk.Checkbutton(sec_ref.body, text="前庭動眼反射（頭の動きを眼で打ち消す）",
                   variable=st_vor).pack(anchor="w", padx=14)
    st_tone = tk.BooleanVar(value=True)
    tk.Checkbutton(sec_ref.body, text=f"屈筋トーンのバネ（四肢 {len(tone_joints)}関節）",
                   variable=st_tone).pack(anchor="w", padx=14)

    # ---- 区画3b：首のバネ（★調整中。値が決まったら core に実装する）----------
    sec_neck = Section(root, "首のバネ（調整中）", op.get("neck", True))
    tk.Label(sec_neck.body, justify="left", fg="#555", font=("", 8),
             text="首にはバネが無く（stiffness=0）、重力で60秒かけて60度倒れ続ける。\n"
                  "正常な新生児でも頸部に軽い抵抗はある（過緊張は正常児の0.7%だけ\n"
                  "／Amiel-Tison 1977）。死後標本の剛性は屈曲0.175・伸展0.40 Nm/rad\n"
                  "（筋を含まない下限値／Luck 2008）。生体の実測値は存在しない。"
             ).pack(anchor="w", padx=14, pady=(2, 4))
    st_neck = tk.BooleanVar(value=bool((saved or {}).get("neck_on", False)))
    tk.Checkbutton(sec_neck.body, text="★首にバネを効かせる",
                   variable=st_neck, fg="#a30").pack(anchor="w", padx=14)
    nk_k = tk.DoubleVar(value=float((saved or {}).get("neck_k", 0.2)))
    slider(sec_neck.body, "剛性[Nm/rad]", nk_k, 0.0, 0.6, 0.01,
           note="死後標本の下限 0.175〜0.40／四肢のトーンは 0.2")
    nk_c = tk.DoubleVar(value=float((saved or {}).get("neck_c", 0.0265)))
    slider(sec_neck.body, "減衰", nk_c, 0.0, 0.2, 0.005,
           note="既定 0.0265（元からある値）。臨界減衰は下に表示")
    nk_t = tk.DoubleVar(value=float((saved or {}).get("neck_target", -25.0)))
    slider(sec_neck.body, "目標角[度]", nk_t, -60.0, 30.0, 1.0,
           note="首を引き寄せる角度。★文献に無いので仮決め。-25度＝今の初期姿勢あたり")
    nk_all = tk.BooleanVar(value=False)
    tk.Checkbutton(sec_neck.body, text="3軸すべてに効かせる（外すと前後の傾きだけ）",
                   variable=nk_all).pack(anchor="w", padx=14)
    neck_label = tk.Label(sec_neck.body, text="", font=("Consolas", 9), justify="left")
    neck_label.pack(anchor="w", padx=14)

    _drop = [None]      # 「頭を持ち上げて離す」テストの状態

    def drop_head():
        """頭を持ち上げて離す。戻り方（速さ・行き過ぎ・振動）を見る。
        人間の頸部の評価（他動運動に対する抵抗）に相当する操作。"""
        for j in neck_ids.values():
            d.qpos[int(m.jnt_qposadr[j])] = np.radians(40.0)   # 顎を上げた位置へ
            d.qvel[int(m.jnt_dofadr[j])] = 0.0
            break
        mujoco.mj_forward(m, d)
        _drop[0] = {"t0": None, "peak": -1e9, "settled": None}
        msg.config(text="頭を40度に持ち上げて離しました。戻り方を見てください")

    tk.Button(sec_neck.body, text="頭を持ち上げて離す", command=drop_head,
              width=20).pack(pady=4)

    # ---- 区画4：測定器 --------------------------------------------------
    sec_mes = Section(root, "測定器", op.get("measure", True))
    tk.Label(sec_mes.body, text="太郎の目に映っているもの（左目・検出画素は緑）",
             font=("", 9, "bold")).pack(pady=(4, 2))
    eye_canvas = tk.Label(sec_mes.body)
    eye_canvas.pack()
    _imgtk = [None]
    mask_var = tk.BooleanVar(value=True)
    tk.Checkbutton(sec_mes.body, text="検出した画素を緑で重ねる",
                   variable=mask_var).pack(anchor="w", padx=14)
    judge_label = tk.Label(sec_mes.body, text="", font=("Consolas", 9), justify="left")
    judge_label.pack(anchor="w", padx=14)
    dist_label = tk.Label(sec_mes.body, text="", font=("Consolas", 9), justify="left")
    dist_label.pack(anchor="w", padx=14)
    pen_label = tk.Label(sec_mes.body, text="", font=("Consolas", 9), justify="left")
    pen_label.pack(anchor="w", padx=14)
    head_label = tk.Label(sec_mes.body, text="", font=("Consolas", 9),
                          justify="left", fg="#a30")
    head_label.pack(anchor="w", padx=14)

    # ---- 区画5：再生 ----------------------------------------------------
    sec_run = Section(root, "再生", op.get("run", True))
    speed_var = tk.DoubleVar(value=float(os.environ.get("E_SPEED", "1.0")))
    slider(sec_run.body, "速度", speed_var, 0.1, 2.0, 0.1,
           note="1.0＝等倍速。0.25＝4分の1のスロー")
    st_babble = tk.BooleanVar(value=False)
    tk.Checkbutton(sec_run.body, text="自発運動を流す（もがき運動）",
                   variable=st_babble).pack(anchor="w", padx=14)
    st_loop = tk.BooleanVar(value=False)
    tk.Checkbutton(sec_run.body, text=f"{EPISODE_SEC:.0f}秒たったらやり直す",
                   variable=st_loop).pack(anchor="w", padx=14)
    run_label = tk.Label(sec_run.body, text="", font=("Consolas", 9), justify="left")
    run_label.pack(anchor="w", padx=14)

    # ---- ボタン ---------------------------------------------------------
    msg = tk.Label(root, text="", fg="#0a7", font=("", 9))
    msg.pack(pady=(6, 0))
    bf = tk.Frame(root); bf.pack(pady=8)
    _restart = [True]

    def save():
        data = {"toy_pos": [float(v.get()) for v in toy_vars],
                "toy_half_size": float(size_var.get()),
                "joints": {jd["base"]: float(v.get())
                           for jd, v in zip(joints, joint_vars)},
                "shake": bool(shake_var.get()), "follow": bool(follow_var.get()),
                "orient": bool(st_orient.get()),
                "latency": float(lat_var.get()), "threshold": float(thr_var.get()),
                "neck_on": bool(st_neck.get()), "neck_k": float(nk_k.get()),
                "neck_c": float(nk_c.get()), "neck_target": float(nk_t.get()),
                "open": {"toy": sec_toy.opened, "pose": sec_pose.opened,
                         "reflex": sec_ref.opened, "neck": sec_neck.opened,
                         "measure": sec_mes.opened, "run": sec_run.opened}}
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        with open(SAVE_PATH, "w", encoding="utf-8") as fp:
            json.dump(data, fp, ensure_ascii=False, indent=2)
        msg.config(text=f"保存しました → {os.path.relpath(SAVE_PATH, _ROOT)}")

    def restore_supine():
        if root_qpos0 is not None:
            d.qpos[root_qadr:root_qadr + 7] = root_qpos0
        for jd, v in zip(joints, joint_vars):
            ang = np.radians(v.get())
            for jid, qadr in jd["pair"]:
                d.qpos[qadr] = ang
        d.qpos[toy_qadr:toy_qadr + 3] = [v.get() for v in toy_vars]
        d.qvel[:] = 0.0
        d.qacc[:] = 0.0
        mujoco.mj_forward(m, d)
        msg.config(text="仰向けに戻しました")

    tk.Button(bf, text="この設定を保存", command=save, width=13,
              bg="#2a7", fg="white").pack(side="left", padx=3)
    tk.Button(bf, text="仰向けに戻す", command=restore_supine, width=12,
              bg="#37a", fg="white").pack(side="left", padx=3)
    tk.Button(bf, text="最初からやり直す", command=lambda: _restart.__setitem__(0, True),
              width=14).pack(side="left", padx=3)
    tk.Label(root, text="ビューア: 左ドラッグ=回転 / 右ドラッグ=平行移動 / スクロール=ズーム",
             fg="#666", font=("", 8)).pack(pady=4)

    # ================= 再生 =================
    print("\nパネルとビューアを開きました。見出しをクリックで区画を開閉できます。\n", flush=True)
    _ui_ready[0] = True      # ここから先のスライダー操作は「人が動かした」とみなす
    from spinal_cord.cpg import ColoredNoiseGenerator
    gen = [ColoredNoiseGenerator(n_act, seed=0)]
    act = [zero.copy()]
    tone_on = [True]
    head_w, devs, seens, neck_hist = [], [], [], []
    prev_sacc = [0]
    fire_until = [-1.0]

    with mujoco.viewer.launch_passive(m, d) as viewer:
        t_sim, wall0, tick = 0.0, time.time(), 0
        toy_base = [None]
        while viewer.is_running():
            try:
                if not win.winfo_exists():
                    break
            except Exception:
                break

            if _restart[0]:
                env.reset(seed=0)
                reflex.reset()
                t_sim, wall0, tick = 0.0, time.time(), 0
                toy_base[0] = None
                head_w.clear(); devs.clear(); seens.clear(); neck_hist.clear()
                parent_log.clear(); _parent[0] = None
                prev_sacc[0] = 0
                _restart[0] = False

            freeze = st_freeze.get()
            OR.SACCADE_LATENCY = float(lat_var.get())
            OR.SACCADE_MIN_STRENGTH = float(thr_var.get())
            u._orienting = reflex if st_orient.get() else None
            u._vor = vor if st_vor.get() else None
            want_tone = st_tone.get()
            if want_tone != tone_on[0]:
                for j in tone_joints:
                    m.jnt_stiffness[j] = tone_k[j] if want_tone else 0.0
                tone_on[0] = want_tone

            # ★首のバネ（調整中）。スライダーの値をそのまま反映する
            _on = st_neck.get()
            _tgt = np.radians(float(nk_t.get()))
            for nm, j in neck_ids.items():
                if nm != "head_tilt" and not nk_all.get():
                    m.jnt_stiffness[j] = 0.0
                    m.dof_damping[int(m.jnt_dofadr[j])] = neck_damp0[nm]
                    continue
                m.jnt_stiffness[j] = float(nk_k.get()) if _on else 0.0
                m.qpos_spring[int(m.jnt_qposadr[j])] = _tgt if _on else 0.0
                m.dof_damping[int(m.jnt_dofadr[j])] = (float(nk_c.get()) if _on
                                                       else neck_damp0[nm])

            # おもちゃ。★揺らしは「固定するか」と独立に効かせる。
            #   固定ON  … スライダーの位置＋揺れ
            #   固定OFF … 環境が置いた位置（視線の正面）＋揺れ
            wob = np.array([0.0, SHAKE_AMP * np.sin(2 * np.pi * SHAKE_HZ * t_sim), 0.0]) \
                if shake_var.get() else np.zeros(3)

            # ★親が顔の前へ運んでいる最中（ボタンで発動）。終わるとスライダーに引き継ぐ
            if _parent[0] is not None:
                if parent_log and parent_log[-1] is None:
                    parent_log[-1] = round(t_sim, 2)   # 押された時刻を確定
                _parent[0]["t"] += dt
                frac = min(1.0, _parent[0]["t"] / _parent[0]["sec"])
                pos = (_parent[0]["from"]
                       + (_parent[0]["to"] - _parent[0]["from"]) * frac) + wob
                d.qpos[toy_qadr:toy_qadr + 3] = pos
                d.qvel[toy_dof:toy_dof + 6] = 0.0
                u._rest_pos = pos.copy()
                if frac >= 1.0:
                    _syncing[0] = True
                    for i2 in range(3):
                        toy_vars[i2].set(round(float(_parent[0]["to"][i2]), 3))
                    _syncing[0] = False
                    _parent[0] = None
            elif follow_var.get():
                pos = np.array([v.get() for v in toy_vars], dtype=float) + wob
                d.qpos[toy_qadr:toy_qadr + 3] = pos
                d.qvel[toy_dof:toy_dof + 6] = 0.0
                u._rest_pos = pos.copy()      # hold モードはここを見るので同じ値を入れる
            else:
                # 環境が置いた基準位置を一度だけ覚え、以後はそこを中心に揺らす
                if toy_base[0] is None and not getattr(u, "_toy_pending", True) \
                        and getattr(u, "_rest_pos", None) is not None:
                    toy_base[0] = np.array(u._rest_pos, dtype=float)
                if toy_base[0] is not None:
                    pos = toy_base[0] + wob
                    u._rest_pos = pos.copy()
                    d.qpos[toy_qadr:toy_qadr + 3] = pos
                    d.qvel[toy_dof:toy_dof + 6] = 0.0
                    # 環境が置いた位置をスライダーに映す（そこから微調整できる）
                    if tick % 20 == 0:
                        _syncing[0] = True
                        for i in range(3):
                            if abs(toy_vars[i].get() - toy_base[0][i]) > 1e-4:
                                toy_vars[i].set(round(float(toy_base[0][i]), 3))
                        _syncing[0] = False
            s = float(size_var.get())
            if abs(float(m.geom_size[toy_gadr][0]) - s) > 1e-9:
                m.geom_size[toy_gadr] = [s, s, s]

            # 姿勢
            if st_hold.get() or freeze:
                for jd, v in zip(joints, joint_vars):
                    ang = np.radians(v.get())
                    for jid, qadr in jd["pair"]:
                        d.qpos[qadr] = ang
                        d.qvel[int(m.jnt_dofadr[jid])] = 0.0

            if freeze:
                if root_qpos0 is not None:
                    d.qpos[root_qadr:root_qadr + 7] = root_qpos0
                d.qvel[:] = 0.0
                d.qacc[:] = 0.0
                mujoco.mj_forward(m, d)
            else:
                if st_babble.get():
                    if tick % 10 == 0:
                        act[0] = np.clip(0.5 + 0.174 * gen[0].sample(0.7), 0.0, 1.0
                                         ).astype(np.float32)
                    env.step(act[0])
                else:
                    env.step(zero)
            t_sim += dt
            tick += 1
            head_w.append(float(np.linalg.norm(d.cvel[head_bid][:3])))
            if len(head_w) > 300:
                head_w.pop(0)
            if reflex.n_saccades > prev_sacc[0]:
                prev_sacc[0] = reflex.n_saccades
                fire_until[0] = t_sim + 0.25

            if tick % 5 == 0:
                # --- 測定器 ---
                u._vision_t = None          # 時間が止まるとキャッシュが切れないので毎回捨てる
                u._vision_cache = None
                img = None
                try:
                    imgs = u.get_vision_obs()
                    img = imgs.get("eye_left") if isinstance(imgs, dict) else None
                except Exception:
                    pass
                rep = VIS.report(m, d, toy_bid, img)
                dev = float("nan")
                if rep.get("pix_seen"):
                    dev = float(np.hypot(rep["cx"], rep["cy"]))
                    devs.append(dev)
                seens.append(1.0 if rep.get("pix_seen") else 0.0)

                j = f"①角度   {rep['angle']:5.1f}°  {'視野内' if rep['in_fov'] else '★視野外'}\n"
                j += ("②光線   遮蔽なし\n" if rep["ray_ok"]
                      else f"②光線   ★{rep['ray_hit']}に遮られている\n")
                if rep.get("pix_seen"):
                    j += (f"③画像   {rep['n_pixels']:4d}画素  "
                          f"中心からのずれ {dev:.2f}（平均 {np.mean(devs):.2f}）")
                else:
                    j += "③画像   ★映っていない"
                judge_label.config(text=j, fg="#070" if rep.get("pix_seen") else "#a00")

                toy = d.xpos[toy_bid]
                de = np.linalg.norm(toy - d.xpos[eye_bid]) * 100
                ds = np.linalg.norm(toy - d.xpos[sh_bid]) * 100
                dh = np.linalg.norm(toy - d.xpos[hand_bid]) * 100
                dist_label.config(
                    text=f"目→おもちゃ {de:5.1f}cm  肩→ {ds:5.1f}cm"
                         f"（腕の{ds/(ARM_REACH*100)*100:3.0f}%）  手→ {dh:5.1f}cm")

                pen = []
                for ci in range(d.ncon):
                    c = d.contact[ci]
                    if c.dist >= -1e-4:
                        continue
                    b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
                    if toy_bid not in (b1, b2):
                        continue
                    other = b2 if b1 == toy_bid else b1
                    pen.append((m.body(other).name, -c.dist * 1000))
                # ★親としての介入の記録（何回・どれくらいの間隔で持っていったか）
                _pl = [x for x in parent_log if x is not None]
                if _pl:
                    gaps = [_pl[i] - _pl[i - 1] for i in range(1, len(_pl))]
                    avg = (sum(gaps) / len(gaps)) if gaps else float("nan")
                    parent_label.config(
                        text=f"親の介入 {len(_pl)}回"
                             f"（平均 {avg:.1f}秒おき）"
                             f"{'　運搬中' if _parent[0] is not None else ''}")
                else:
                    parent_label.config(
                        text="まだ介入していません（おもちゃが見えなくなったら押す）")

                if pen:
                    pen.sort(key=lambda x: -x[1])
                    pen_label.config(
                        text="★めり込み " + " ".join(f"{n}{v:.1f}mm" for n, v in pen[:3]),
                        fg="#a00")
                else:
                    pen_label.config(text="めり込みなし", fg="#070")

                hw = np.array(head_w) if head_w else np.zeros(1)
                warn = ""
                if not freeze and st_hold.get():
                    warn = "\n★「この角度で固定する」がONのまま物理を回しています＝暴れます"
                head_label.config(
                    text=f"頭の角速度 平均 {hw.mean():.3f} 最大 {hw.max():.3f} rad/s{warn}")

                # --- 首のバネの様子 ---
                jt = neck_ids.get("head_tilt")
                if jt is not None:
                    ang = float(np.degrees(d.qpos[int(m.jnt_qposadr[jt])]))
                    vel = float(np.degrees(d.qvel[int(m.jnt_dofadr[jt])]))
                    neck_hist.append(ang)
                    if len(neck_hist) > 400:
                        neck_hist.pop(0)
                    k = max(float(nk_k.get()), 1e-9)
                    c_crit = 2.0 * np.sqrt(k * NECK_I)
                    c_now = float(nk_c.get())
                    ratio = c_now / c_crit if c_crit > 1e-12 else float("inf")
                    kind = ("★振動する（減衰不足）" if ratio < 0.7
                            else "ちょうど良い" if ratio < 1.6 else "戻りが遅い（減衰過多）")
                    recent = np.array(neck_hist[-100:]) if len(neck_hist) > 20 else np.zeros(1)
                    swing = float(recent.max() - recent.min())
                    txt = (f"首の角度 {ang:7.2f}度（目標 {nk_t.get():5.1f}）  "
                           f"速さ {vel:6.2f}度/s\n"
                           f"直近2秒の振れ幅 {swing:5.2f}度  "
                           f"{'落ち着いている' if swing < 1.0 else '★まだ動いている'}\n"
                           f"臨界減衰 {c_crit:.4f}／今 {c_now:.4f}（{ratio:.2f}倍）{kind}")
                    if _drop[0] is not None:
                        if _drop[0]["t0"] is None:
                            _drop[0]["t0"] = t_sim
                        el = t_sim - _drop[0]["t0"]
                        _drop[0]["peak"] = max(_drop[0]["peak"], -ang)
                        if _drop[0]["settled"] is None and swing < 1.0 and el > 0.5:
                            _drop[0]["settled"] = el
                        st = (f"{_drop[0]['settled']:.1f}秒で落ち着いた"
                              if _drop[0]["settled"] else f"{el:.1f}秒経過")
                        txt += f"\n★持ち上げテスト：{st}  行き過ぎ {-_drop[0]['peak']:.1f}度"
                    neck_label.config(
                        text=txt, fg="#a00" if ratio < 0.7 and st_neck.get() else "#333")

                seen_pct = float(np.mean(seens)) * 100 if seens else 0.0
                fired = "● 撃った" if t_sim < fire_until[0] else ""
                # ★おもちゃの状態。最初の1.5秒は「遠くで待機中」＝視界に無いのが正常
                if getattr(u, "_toy_pending", False):
                    phase = ("おもちゃは遠くで待機中"
                             if not getattr(u, "_toy_arriving", False)
                             else "親が運んでいる")
                elif follow_var.get():
                    phase = "スライダーの位置に固定中"
                else:
                    phase = "設置ずみ（環境が視線の正面に置いた）"
                run_label.config(
                    text=f"{phase}\n"
                         f"経過 {t_sim:5.1f}秒  サッケード {reflex.n_saccades:3d}発  {fired}\n"
                         f"反応の強さ {reflex.strength:6.3f}（閾値 {thr_var.get():.2f}）\n"
                         f"見えていた割合 {seen_pct:5.1f}%  "
                         f"ずれ平均 {np.mean(devs) if devs else float('nan'):5.2f}"
                         f"（反射OFFで0.48）")

                if img is not None and mask_var.get() is not None:
                    try:
                        from PIL import Image, ImageTk
                        arr = np.asarray(img)
                        if arr.dtype != np.uint8:
                            arr = np.clip(arr, 0, 255).astype(np.uint8)
                        arr = arr.copy()
                        if mask_var.get():
                            arr[VIS.red_mask(arr)] = [0, 255, 0]
                        _imgtk[0] = ImageTk.PhotoImage(
                            Image.fromarray(arr).resize((192, 192), Image.NEAREST))
                        eye_canvas.config(image=_imgtk[0])
                    except Exception:
                        pass

                try:
                    win.update()
                except tk.TclError:
                    break

            if st_loop.get() and t_sim >= EPISODE_SEC:
                _restart[0] = True

            viewer.sync()
            if not freeze:
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
