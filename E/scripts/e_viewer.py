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

    # ★【2026-07-28】測定スクリプト（e_orient_converge_test.py）と条件を揃える。
    #   揃えないと「Viewerで見ている太郎と、測っている太郎が別物」になる
    #   （2026-07-25 に実際に起きた問題＝身体の設定が散らばる）。
    #     体年齢4ヶ月  … この反射を使うリーチングの月齢に揃えた
    #     視力も4ヶ月  … 体と揃える（1ヶ月の4.3倍）
    #     頭を抑える    … 人間の乳児実験と同じ条件（Hunter & Richards 2003）
    from e_head_hold import CaregiverHands
    _AGE = float(os.environ.get("E_AGE", "4.0"))
    _HEAD_HOLD = os.environ.get("E_HEAD_HOLD", "1") == "1"

    # ★リクライニングの角度[度]。0=仰向け。モデル構築時に決まるので実行中は変えられない
    _RECLINE = float(os.environ.get("E_RECLINE", "0"))
    _EYE_REST_V = float(os.environ.get("E_EYE_REST_V", "0"))
    kw = body_kwargs_from_env(_AGE, verbose=True)
    env = ToySupineEnv(actuation_model=MuscleModel,
                       vision_params=infant_vision_params(acuity_age=_AGE),
                       age=_AGE, toy=True, vor=True, orient=True,
                       recline_deg=_RECLINE, **kw)
    u = env.unwrapped
    m, d = u.model, u.data
    env.reset(seed=0)
    hands = CaregiverHands(m, d)
    # ★支える角度。指定しなければ「今の角度」で支える。
    #   リクライニングでは顎を引かせないと視線が上を向いてしまうので、
    #   前後（head_tilt）だけ指定できるようにした（2026-07-28）。
    _HOLD_TILT = os.environ.get("E_HOLD_TILT")
    _hold_tgt = ({"head_tilt": float(_HOLD_TILT)} if _HOLD_TILT else None)
    if _HEAD_HOLD:
        hands.hold(target=_hold_tgt, verbose=True)
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
    # ⚠️リセット直後のおもちゃは**退避位置 [3,3,0.05]**（登場を1秒遅らせる仕組み）。
    #   そのままスライダーの初期値にすると、範囲（X:-0.1〜0.5 / Y:-0.3〜0.3）の外なので
    #   少し動かした瞬間に範囲内へクランプされ、**おもちゃが突然どこかへ飛ぶ**
    #   （ユーザーの目視「スライダーで位置を変えるとどこからか出てきた」2026-07-27）。
    #   環境が置くはずの位置（視線の正面）を初期値にする。
    if float(np.max(np.abs(toy_pos0[:2]))) > 1.0:
        try:
            u._set_anchor()
            toy_pos0 = np.array(u._rest_pos, dtype=float)
        except Exception:
            toy_pos0 = np.array([0.28, 0.0, 0.18], dtype=float)
    if saved and "toy_pos" in saved:
        toy_pos0 = np.array(saved["toy_pos"], dtype=float)
    # ★プリセットが持つおもちゃの位置（E_TOY_POS）は保存値より優先する。
    #   ＝「この環境ではここに置く」という条件の一部なので（2026-07-28）。
    if os.environ.get("E_TOY_POS"):
        try:
            toy_pos0 = np.array([float(x) for x in
                                 os.environ["E_TOY_POS"].split(",")], dtype=float)
        except Exception:
            pass
    size0 = float(m.geom_size[toy_gadr][0])
    # ⚠️E_TOY_RADIUS を明示したときは保存値で上書きしない。
    #   上書きすると、環境変数で指定した大きさが黙って無視される
    #   （実際 E_TOY_RADIUS=0.010 が保存値 0.02 に置き換わっていた）。
    if saved and "toy_half_size" in saved and "E_TOY_RADIUS" not in os.environ:
        size0 = float(saved["toy_half_size"])
        m.geom_size[toy_gadr] = [size0, size0, size0]

    # ================= パネル =================
    # ★【2026-07-28 レイアウト改訂】1列（幅520px）で6区画を縦に積んでいたため
    #   画面からはみ出して縦に長すぎた（ユーザーの指摘）。**2列**に変え、
    #   目の映像も左右そろえて出す。
    win = tk.Tk()
    win.title("太郎ビューア（統一版）")
    _h = min(980, win.winfo_screenheight() - 80)
    _w = min(1080, win.winfo_screenwidth() - 60)
    win.geometry(f"{_w}x{_h}+20+10")
    # ⚠️★最前面に固定しない（2026-07-28、ユーザーの要望）。他の窓を見るたびに
    #   ビューアが邪魔になるため。MuJoCoの3D窓とパネルは別窓なので、必要なら
    #   タスクバーから前面に出せる。

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

    # ★2列。左＝環境と条件（いじる物）／右＝見る物（映像・数値）
    #
    # ⚠️★【2026-07-28 修正】最初 `pack` + `pack_propagate(False)` + `width=520` で
    #   組んだところ、**幅だけ固定されて高さが潰れ**、列の中身が1つも表示されなかった
    #   （ユーザーの報告「一番下のボタン以外表示されない」）。
    #   pack_propagate(False) は「子に合わせてリサイズしない」＝高さも指定しないと
    #   既定の小さい値のままになる。
    #   → `grid` + `columnconfigure(weight=1, uniform=...)` で幅を等分し、
    #     高さは中身に任せる（伝播させる）。
    cols = tk.Frame(root); cols.pack(fill="both", expand=True)
    cols.columnconfigure(0, weight=1, uniform="col")
    cols.columnconfigure(1, weight=1, uniform="col")
    colL = tk.Frame(cols); colL.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
    colR = tk.Frame(cols); colR.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

    op = (saved or {}).get("open", {})
    # ★2026-07-27：既定を「物理を動かす」に変えた（それまでは止まって起動）。
    #   反射が目を動かすのは env.step() の中なので、物理を止めていると
    #   **方向は正しく計算されているのにサッケードが1発も撃たれない**
    #   （ユーザーの目視：強さ0.406＞閾値、ずれ0.95 なのに 0発。2026-07-27）。
    #   Viewer の主目的は反射の観察なので、止めたい人が自分でONにする形にする。
    freeze0 = os.environ.get("E_FREEZE", "0") == "1"

    # ---- 区画0：環境（プリセット）★2026-07-28 新設 -----------------------
    #
    # 【なぜ要るか】ユーザーの要望「環境をいくつかバージョン用意して、プルダウンで選んで
    # 最初からやり直しを押したらその環境で始まる」。実験ごとに条件（月齢・柵・頭の支え）を
    # 手で合わせるのは間違いのもとで、実際に「Viewerで見ている太郎と測っている太郎が
    # 別物」という問題が起きていた（2026-07-25、体型補正が環境ごとにバラバラだった件）。
    #
    # ⚠️**月齢だけはモデルを作るときに決まる**（geomの寸法・質量が変わる）ので、
    #   実行中に変えられない。月齢が変わるプリセットを選んだときは**プロセスを
    #   作り直す**（環境変数を設定して自分を起動し直す）。それ以外は即反映。
    sec_env = Section(colL, "環境（プリセット）", op.get("env", True))

    # プリセット定義。age だけが「作り直しが要る」項目。
    PRESETS = {
        "視線誘導反射の測定（4ヶ月・仰向け・頭を抑える）": dict(
            age=4.0, head_hold=True, fence=False, orient=True, toy_radius=0.0056,
            recline=0.0, hold_tilt=None, eye_rest_v=0.0,
            note="e_orient_converge_test.py と同じ条件。仰向け・眼球は正中位。"
                 "人間の乳児実験の作法（Hunter & Richards 2003）"),
        "新生児（0ヶ月・支えなし・柵あり）": dict(
            age=0.0, head_hold=False, fence=True, orient=True, toy_radius=0.0056,
            note="2026-07-27 までの条件。過去の測定と比べるとき用"),
        "4ヶ月・支えなし・柵なし": dict(
            age=4.0, head_hold=False, fence=False, orient=True, toy_radius=0.0056,
            note="頭の支えが効いているかを見るための対照"),
        "自由に動く（4ヶ月・柵あり・反射なし）": dict(
            age=4.0, head_hold=False, fence=True, orient=False, toy_radius=0.0056,
            note="自発運動の観察用。柵に当たるかもここで見る"),
        "★リーチング（リクライニング60度・顎を引く）": dict(
            age=4.0, head_hold=True, fence=False, orient=True, toy_radius=0.0056,
            recline=60.0, hold_tilt=60.0, eye_rest_v=-15.0,
            toy_pos=(-0.048, -0.115, 0.113),
            note="体幹64.5度・首60度（顎を引く）・眼球-15度。"
                 "おもちゃは目から17.8cm・視線から3.7度・腕の94%＝見えて届く。"
                 "リーチはこの姿勢が有利（Carvalho 2007／Savelsbergh 1994）"),
        "リクライニング45度": dict(
            age=4.0, head_hold=True, fence=False, orient=True, toy_radius=0.0056,
            recline=45.0, hold_tilt=45.0, eye_rest_v=-15.0,
            note="実測で体幹30.4度。より寝た姿勢"),
        "リクライニング30度": dict(
            age=4.0, head_hold=True, fence=False, orient=True, toy_radius=0.0056,
            recline=30.0, hold_tilt=30.0, eye_rest_v=-10.0,
            note="実測で体幹23.8度。ほぼ仰向けに近い"),
    }
    preset_names = list(PRESETS)
    # 今の起動条件に最も近いプリセットを初期選択にする
    # ⚠️★リクライニング角も判定に入れる（2026-07-28）。入れていなかったため、
    #   70度で起動しているのに「0度のプリセット」が選ばれ、「最初からやり直し」を
    #   押すと条件が違うと判定されて**体の作り直しが走り、窓が閉じた**
    #   （ユーザーの報告「最初からにすると落ちる」）。
    def _matches(v):
        """起動時の条件とプリセットが一致するか。★姿勢一式で判定する（2026-07-28）。

        ⚠️リクライニング角だけを見ていたため、「顎を引いた60度」で起動しても
        「4ヶ月・柵なし」（仰向け）が選ばれていた（ユーザーの報告）。
        姿勢の設定を環境変数に散らしていたのが原因なので、プリセット側に
        姿勢一式を持たせ、判定もそれで行う。
        """
        _t = v.get("hold_tilt")
        _c = (float(_HOLD_TILT) if _HOLD_TILT else None)
        if (_t is None) != (_c is None):
            return False
        if _t is not None and abs(float(_t) - _c) > 1e-9:
            return False
        return (abs(v["age"] - _AGE) < 1e-9
                and v["head_hold"] == _HEAD_HOLD
                and abs(float(v.get("recline", 0.0)) - _RECLINE) < 1e-9
                and abs(float(v.get("eye_rest_v", 0.0)) - _EYE_REST_V) < 1e-9)

    _cur = next((k for k, v in PRESETS.items() if _matches(v)), preset_names[0])
    preset_var = tk.StringVar(value=_cur)
    _pf = tk.Frame(sec_env.body); _pf.pack(fill="x", padx=10, pady=(4, 0))
    tk.Label(_pf, text="環境", width=6, anchor="w").pack(side="left")
    tk.OptionMenu(_pf, preset_var, *preset_names).pack(side="left", fill="x", expand=True)
    preset_note = tk.Label(sec_env.body, text=PRESETS[_cur]["note"],
                           fg="#666", font=("", 8), wraplength=460, justify="left")
    preset_note.pack(anchor="w", padx=14)
    tk.Label(sec_env.body,
             text="★「最初からやり直し」を押すと、選んだ環境で始まります\n"
                  "　 月齢が変わるときは体を作り直すので数十秒かかります",
             fg="#a30", font=("", 8), justify="left").pack(anchor="w", padx=14)

    # 個別のスイッチ（プリセットを選ぶと連動して変わる）
    st_fence = tk.BooleanVar(value=True)
    tk.Checkbutton(sec_env.body, text="柵（ベビーサークル）を有効にする",
                   variable=st_fence).pack(anchor="w", padx=14)
    st_hold_head = tk.BooleanVar(value=_HEAD_HOLD)
    tk.Checkbutton(sec_env.body, text="実験者が頭を抑える（人間の乳児実験と同じ）",
                   variable=st_hold_head, fg="#06a").pack(anchor="w", padx=14)
    env_label = tk.Label(sec_env.body, text="", font=("Consolas", 9),
                         justify="left", fg="#444")
    env_label.pack(anchor="w", padx=14)

    def on_preset(*_a):
        p = PRESETS[preset_var.get()]
        preset_note.config(text=p["note"])
        st_fence.set(bool(p["fence"]))
        st_hold_head.set(bool(p["head_hold"]))
    preset_var.trace_add("write", on_preset)

    # 柵のgeom（実行時にON/OFFする）。名前は e_toy_env の `fence_post_{i}`。
    fence_gids = [g for g in range(m.ngeom) if "fence" in (m.geom(g).name or "")]
    fence_rgba0 = {g: m.geom_rgba[g].copy() for g in fence_gids}
    fence_con0 = {g: (int(m.geom_contype[g]), int(m.geom_conaffinity[g]))
                  for g in fence_gids}
    fence_on = [True]

    def apply_fence(want):
        """柵を実行時に消す／戻す。★geomを消せないので当たり判定と色で表現する。"""
        # ⚠️★環境側のフラグも合わせる。おもちゃの置き場所は `_set_anchor` が
        #   「柵の内側」にクランプするので、ここを切らないと**柵を消しても
        #   おもちゃだけ柵の枠に押し込まれる**（2026-07-28 に発覚）。
        u._fence = bool(want)
        for g in fence_gids:
            if want:
                m.geom_rgba[g] = fence_rgba0[g]
                m.geom_contype[g], m.geom_conaffinity[g] = fence_con0[g]
            else:
                _c = fence_rgba0[g].copy(); _c[3] = 0.0     # 透明＝見えない
                m.geom_rgba[g] = _c
                m.geom_contype[g] = 0                        # 当たらない
                m.geom_conaffinity[g] = 0
        fence_on[0] = bool(want)

    # ---- 区画1：おもちゃ ------------------------------------------------
    sec_toy = Section(colL, "おもちゃ", op.get("toy", True))
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
        # ★距離はスライダーの値を使う（2026-07-28。それまで環境の固定値だった）
        dist = float(dist_var.get())
        u._toy_dist = dist          # 環境側（_set_anchor）にも反映
        goal = origin + fwd * dist
        # ★柵の内側にとどめる。人間の親は柵の外に手を出さない。
        #   首が横を向いていると「視線の正面」が柵の外になり、
        #   おもちゃが柵に遮られて見えなくなる（ユーザーの報告 2026-07-27）。
        # ⚠️★【2026-07-28 修正】**柵があるときだけ**にした。柵は新生児の体に合わせた
        #   寸法なので、4ヶ月の体では顔の前が枠の外になり、おもちゃが胸元へ落ちていた
        #   （ユーザーの目視「顔の前にもっていくを押しても視界に来ないで胸元に来る」）。
        if fence_on[0]:
            mgn = 0.03
            goal[0] = float(np.clip(goal[0], -TE.FENCE_HALF_X + mgn, TE.FENCE_HALF_X - mgn))
            goal[1] = float(np.clip(goal[1], -TE.FENCE_HALF_Y + mgn, TE.FENCE_HALF_Y - mgn))
        goal[2] = float(max(goal[2], 0.04))
        start = np.array(d.xpos[toy_bid], dtype=float)
        _parent[0] = {"t": 0.0, "from": start, "to": goal,
                      "sec": max(0.05, float(carry_sec.get()))}
        follow_var.set(True)     # 手動モードに切り替える
        parent_log.append(None)   # 時刻はループ側で埋める
        msg.config(text=f"親が顔の前へ運んでいます（{len(parent_log)}回目）")

    # ★【2026-07-28 新設】目からおもちゃまでの距離。
    #
    # 【なぜ要るか】従来 `TOY_DISTANCE = 0.086`（8.6cm）の固定値で、これは
    # **0ヶ月の腕の長さ（18.6cm）を基準に決めた暫定値**だった。しかも e_toy_env.py の
    # コメントには実測結果として「8.6cmは柵に当たる／15cmならめり込まず腕で届く」と
    # 書いてあるのに、既定値が更新されていなかった。体年齢を上げると腕が伸びるので、
    # 距離も見直す必要がある。
    #
    # ⚠️★近すぎると**輻輳（寄り目）**が要る。太郎は両目に同じ指令を出す実装
    # （Hering の等神経支配の法則）で、輻輳は実装していない。
    #     距離8.6cm・瞳孔間4.5cm → 必要な寄り目 約29度
    #     距離15cm              → 約17度
    #     距離30cm              → 約8.6度
    # ＝近いほど「両目で同じものを見られない」状態になる。下に必要角を表示する。
    dist_var = tk.DoubleVar(value=float((saved or {}).get(
        "toy_dist", getattr(u, "_toy_dist", 0.086))))
    slider(sec_toy.body, "目からの距離[m]", dist_var, 0.05, 0.40, 0.005,
           note="★「顔の前へ持っていく」を押すとこの距離に置く")
    vergence_label = tk.Label(sec_toy.body, text="", font=("Consolas", 9),
                              justify="left", fg="#06a")
    vergence_label.pack(anchor="w", padx=14)

    # 瞳孔間距離（両目のカメラの間隔）を実測しておく＝輻輳角の計算に使う
    try:
        _el = np.array(d.cam_xpos[int(m.camera("eye_left").id)], dtype=float)
        _er = np.array(d.cam_xpos[int(m.camera("eye_right").id)], dtype=float)
        IPD = float(np.linalg.norm(_el - _er))
    except Exception:
        IPD = 0.045

    tk.Button(sec_toy.body, text="★顔の前へ持っていく", command=bring_to_face,
              width=22, bg="#a63", fg="white").pack(pady=4)
    parent_label = tk.Label(sec_toy.body, text="", font=("Consolas", 9),
                            justify="left")
    parent_label.pack(anchor="w", padx=18)

    # ---- 区画2：姿勢 ----------------------------------------------------
    sec_pose = Section(colL, "姿勢", op.get("pose", True))
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
    sec_ref = Section(colL, "反射", op.get("reflex", True))
    st_orient = tk.BooleanVar(value=bool((saved or {}).get("orient", False)))
    tk.Checkbutton(sec_ref.body, text="視線誘導反射（動くものに目・首を向ける）",
                   variable=st_orient).pack(anchor="w", padx=14)
    lat_var = tk.DoubleVar(value=float((saved or {}).get("latency", OR.SACCADE_LATENCY)))
    slider(sec_ref.body, "  間隔[秒]", lat_var, 0.1, 1.5, 0.1,
           note="0.2＝旧設定（撃ちすぎて視界から追い出す）／0.5〜0.9＝新生児の実測")
    thr_var = tk.DoubleVar(value=float((saved or {}).get("threshold", OR.SACCADE_MIN_STRENGTH)))
    # ⚠️範囲を 0〜0.5 から 0〜0.06 に狭めた。実測で決めた値は 0.015 で、
    #   0.25 のような値にすると**動きの強さが届かず一度も撃たない**。
    slider(sec_ref.body, "  発火の閾値", thr_var, 0.0, 0.06, 0.0025,
           note="0.02＝旧設定（低すぎて常に発火）／0.25前後で動きを選べる")
    st_vor = tk.BooleanVar(value=True)
    tk.Checkbutton(sec_ref.body, text="前庭動眼反射（頭の動きを眼で打ち消す）",
                   variable=st_vor).pack(anchor="w", padx=14)
    st_tone = tk.BooleanVar(value=True)
    tk.Checkbutton(sec_ref.body, text=f"屈筋トーンのバネ（四肢 {len(tone_joints)}関節）",
                   variable=st_tone).pack(anchor="w", padx=14)

    # ---- 区画3b：首のバネ（★調整中。値が決まったら core に実装する）----------
    sec_neck = Section(colL, "首のバネ（調整中）", op.get("neck", True))
    tk.Label(sec_neck.body, justify="left", fg="#555", font=("", 8),
             text="首にはバネが無く（stiffness=0）、重力で60秒かけて60度倒れ続ける。\n"
                  "正常な新生児でも頸部に軽い抵抗はある（過緊張は正常児の0.7%だけ\n"
                  "／Amiel-Tison 1977）。死後標本の剛性は屈曲0.175・伸展0.40 Nm/rad\n"
                  "（筋を含まない下限値／Luck 2008）。生体の実測値は存在しない。"
             ).pack(anchor="w", padx=14, pady=(2, 4))
    # ★2026-07-27修正：初期値は**core が実際に設定した値**から読む。
    #   保存ファイルの古い値（バネOFF・剛性0.2・目標-25度）を初期値にしていたため、
    #   起動しただけで core の実装（剛性0.40・目標-45度）が上書きされて消えていた。
    #   ユーザーの報告「首のバネ OFF」で発覚。
    _jt0 = neck_ids.get("head_tilt")
    _k0 = float(m.jnt_stiffness[_jt0]) if _jt0 is not None else 0.0
    _t0 = (float(np.degrees(m.qpos_spring[int(m.jnt_qposadr[_jt0])]))
           if _jt0 is not None else 0.0)
    _c0 = (float(m.dof_damping[int(m.jnt_dofadr[_jt0])]) if _jt0 is not None else 0.0265)
    tk.Label(sec_neck.body, fg="#070", font=("", 8),
             text=f"起動時に core が設定した値：剛性{_k0:.2f} 目標{_t0:+.0f}度 減衰{_c0:.4f}"
             ).pack(anchor="w", padx=14)
    st_neck = tk.BooleanVar(value=(_k0 > 0.0))
    tk.Checkbutton(sec_neck.body, text="★首にバネを効かせる（外すと core の設定を消します）",
                   variable=st_neck, fg="#a30").pack(anchor="w", padx=14)
    nk_k = tk.DoubleVar(value=(_k0 if _k0 > 0 else 0.4))
    slider(sec_neck.body, "剛性[Nm/rad]", nk_k, 0.0, 0.6, 0.01,
           note="死後標本の下限 0.175〜0.40／四肢のトーンは 0.2")
    nk_c = tk.DoubleVar(value=_c0)
    slider(sec_neck.body, "減衰", nk_c, 0.0, 0.2, 0.005,
           note="臨界減衰（振動しない最小値）は下に表示")
    # ★目標角の範囲と既定を姿勢に合わせる（2026-07-28）。
    #   仰向けは -45度（重力を織り込んだ実効値）だが、リクライニングでは
    #   顎を引く側（正）にしないと視線が上を向いてしまう。
    _nk_lo, _nk_hi = (-60.0, 30.0) if _RECLINE <= 0 else (-30.0, 70.0)
    _nk_def = -45.0 if _RECLINE <= 0 else 30.0
    nk_t = tk.DoubleVar(value=(_t0 if _k0 > 0 else _nk_def))
    slider(sec_neck.body, "目標角[度]", nk_t, _nk_lo, _nk_hi, 1.0,
           note=("★重力を織り込んだ実効値。-45度で実際は-25度あたりに落ち着く"
                 if _RECLINE <= 0 else
                 "★正が前屈（顎を引く）。リクライニングでは顎を引かないと視線が上を向く"))
    # ★★リクライニングでは既定でON。
    #   首のバネは前後（head_tilt）だけに入れる設計だったが、これは**仰向け前提**。
    #   体を起こすと頭の重さが左右方向にも効くので、前後だけだと
    #   **頭が横を向いて倒れる**（実測：視線が真横 y=0.83。2026-07-28）。
    #   ユーザーの目視「さっきはなんか首が変だった」の原因。
    nk_all = tk.BooleanVar(value=(_RECLINE > 0))
    tk.Checkbutton(sec_neck.body,
                   text="3軸すべてに効かせる（外すと前後の傾きだけ）"
                        "★リクライニングでは必要",
                   variable=nk_all, fg=("#a30" if _RECLINE > 0 else "black")
                   ).pack(anchor="w", padx=14)
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
    sec_mes = Section(colR, "測定器", op.get("measure", True))
    # ★【2026-07-28】右目も出す（ユーザーの要望）。それまで左目だけだった。
    #   両眼を並べると「片方にしか映っていない」ことに気づける。
    #   ⚠️太郎の反射は**両眼の画像を使う**（左目だけ使っていた旧実装は 2026-07-26 に修正済み）
    #     ので、片目だけ見ていると反射が見ているものと食い違う。
    tk.Label(sec_mes.body, text="太郎の目に映っているもの（検出画素は緑）",
             font=("", 9, "bold")).pack(pady=(4, 2))
    eye_row = tk.Frame(sec_mes.body); eye_row.pack()
    _eye_frames, eye_canvases = {}, {}
    for _side, _lb in (("eye_left", "左目"), ("eye_right", "右目")):
        _f = tk.Frame(eye_row); _f.pack(side="left", padx=4)
        tk.Label(_f, text=_lb, font=("", 8)).pack()
        _c = tk.Label(_f); _c.pack()
        _eye_frames[_side] = _f
        eye_canvases[_side] = _c
    _imgtk = {"eye_left": None, "eye_right": None}
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
    sec_run = Section(colR, "再生", op.get("run", True))
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
                # ★2026-07-28 追加。距離・環境の条件も保存する。
                #   これが無いと「保存した位置」を再現しても距離の設定が失われる。
                "toy_dist": float(dist_var.get()),
                "age": float(_AGE), "head_hold": bool(st_hold_head.get()),
                "fence": bool(st_fence.get()), "preset": preset_var.get(),
                "joints": {jd["base"]: float(v.get())
                           for jd, v in zip(joints, joint_vars)},
                "shake": bool(shake_var.get()), "follow": bool(follow_var.get()),
                "orient": bool(st_orient.get()),
                "latency": float(lat_var.get()), "threshold": float(thr_var.get()),
                "neck_on": bool(st_neck.get()), "neck_k": float(nk_k.get()),
                "neck_c": float(nk_c.get()), "neck_target": float(nk_t.get()),
                # ★【2026-07-28 追加】スイッチの状態も残す。
                #   これが無いと、保存された位置を測るときに**別の姿勢の太郎**を
                #   測ってしまう（実際、首のバネ30度で調整した設定を
                #   「実験者が60度で支える」条件で測って、まったく違う値が出た）。
                "head_hold": bool(st_hold_head.get()),
                "hold_tilt": (float(_HOLD_TILT) if _HOLD_TILT else None),
                "neck_all_axes": bool(nk_all.get()),
                "freeze": bool(st_freeze.get()), "pose_hold": bool(st_hold.get()),
                "vor": bool(st_vor.get()), "tone": bool(st_tone.get()),
                "eye_rest_v": float(_EYE_REST_V),
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

    # ★「今どうなっているか」をまとめてクリップボードへ（ユーザーの提案 2026-07-27）。
    #   설定だけでなく**測定値も**入れる。「こうしたらこうなった」を正確に伝えるため。
    _snapshot = [""]

    def copy_state():
        try:
            win.clipboard_clear()
            win.clipboard_append(_snapshot[0])
            win.update()
            msg.config(text="設定と測定値をコピーしました（貼り付けて共有できます）")
        except Exception as e:
            msg.config(text=f"コピーに失敗: {e}")

    tk.Button(bf, text="現在の設定をコピー", command=copy_state, width=16,
              bg="#657", fg="white").pack(side="left", padx=3)
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
        # ★2026-07-27：初期カメラを太郎の顔に寄せる。
        #   これが無いと MuJoCo の既定カメラ（シーン全体を引きで映す）になり、
        #   柵の外から見下ろす画になる。顔の前にある直径2cmのおもちゃは
        #   小さすぎて柵に隠れ、**「おもちゃが出てこない」ように見えた**
        #   （ユーザーの目視 2026-07-27。実際には正しい位置にあった）。
        try:
            # ★真上寄りから見下ろす。仰向けの太郎と、顔の前のおもちゃが
            #   両方いちどに入る角度（実測で選んだ）。
            _look = np.array(d.body("head").xpos, dtype=float)
            try:
                _toy = np.array(d.body("test_object1").xpos, dtype=float)
                if np.max(np.abs(_toy[:2])) < 1.0:      # 退避位置でなければ中点を見る
                    _look = (_look + _toy) * 0.5
            except Exception:
                pass
            viewer.cam.lookat[:] = _look
            viewer.cam.distance = 0.42
            viewer.cam.elevation = -62.0
            viewer.cam.azimuth = 180.0
        except Exception:
            pass
        t_sim, wall0, tick = 0.0, time.time(), 0
        toy_base = [None]
        while viewer.is_running():
            try:
                if not win.winfo_exists():
                    break
            except Exception:
                break

            if _restart[0]:
                # ★★【2026-07-28】プリセットで**月齢が変わる**なら、体そのものを
                #   作り直す必要がある（geomの寸法・質量はモデル構築時に決まるので
                #   実行中には変えられない）。環境変数を設定して自分を起動し直す。
                _p = PRESETS[preset_var.get()]
                # ★体の角度もモデル構築時に決まるので作り直しが要る
                # ★姿勢一式（リクライニング角・顎の角度・眼球の基準）は
                #   モデル構築時／リセット時に決まるので、変わるなら作り直す。
                _p_tilt = _p.get("hold_tilt")
                _cur_tilt = (float(_HOLD_TILT) if _HOLD_TILT else None)
                _tilt_changed = (
                    (_p_tilt is None) != (_cur_tilt is None)
                    or (_p_tilt is not None and _cur_tilt is not None
                        and abs(float(_p_tilt) - _cur_tilt) > 1e-9))
                if (abs(float(_p["age"]) - _AGE) > 1e-9
                        or abs(float(_p.get("recline", 0.0)) - _RECLINE) > 1e-9
                        or abs(float(_p.get("eye_rest_v", 0.0)) - _EYE_REST_V) > 1e-9
                        or _tilt_changed):
                    msg.config(text="★条件が変わったので体を作り直します。"
                                    "新しい窓が開いたら、この窓は閉じます")
                    win.update_idletasks()
                    win.after(0, lambda: None)
                    _envv = dict(os.environ)
                    _envv["E_AGE"] = str(_p["age"])
                    _envv["E_HEAD_HOLD"] = "1" if _p["head_hold"] else "0"
                    _envv["E_FENCE"] = "1" if _p["fence"] else "0"
                    _envv["E_TOY_RADIUS"] = str(_p["toy_radius"])
                    _envv["E_RECLINE"] = str(_p.get("recline", 0.0))
                    _envv["E_EYE_REST_V"] = str(_p.get("eye_rest_v", 0.0))
                    if _p.get("hold_tilt") is None:
                        _envv.pop("E_HOLD_TILT", None)
                    else:
                        _envv["E_HOLD_TILT"] = str(_p["hold_tilt"])
                    if _p.get("toy_pos"):
                        _envv["E_TOY_POS"] = ",".join(str(v) for v in _p["toy_pos"])
                    # ★★【2026-07-28 修正】`os.execve` をやめた。
                    #   Windows には本当の exec が無く、Python は「新プロセスを作って
                    #   自分は終了する」動作になる。その結果、新プロセスが親のコンソールを
                    #   失って管理から外れ、**見た目にはアプリが消えた**ように見えた
                    #   （ユーザーの報告「最初からやり直すを押すとアプリが消える」）。
                    #   → 独立したプロセスとして起動し、自分は普通に閉じる。
                    import subprocess
                    _flags = 0
                    if os.name == "nt":
                        # 新しいコンソールを開く＝完全に独立したプロセスになる。
                        # ⚠️DETACHED_PROCESS だとログが見えず、失敗しても気づけない
                        _flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
                    try:
                        subprocess.Popen(
                            [sys.executable, os.path.abspath(__file__)],
                            env=_envv, cwd=os.getcwd(), creationflags=_flags)
                        msg.config(text="新しい条件で立ち上げています。"
                                        "別の窓が開くまで少し待ってください")
                        win.update_idletasks()
                    except Exception as _e:
                        msg.config(text=f"⚠️起動に失敗しました: {_e}")
                        _restart[0] = False
                        continue
                    try:
                        env.close()
                    except Exception:
                        pass
                    try:
                        win.destroy()
                    except Exception:
                        pass
                    return

                env.reset(seed=0)
                # ★やり直しのたびに支え直す（目標角は「支え始めた時点の角度」なので、
                #   リセット後の姿勢で取り直す必要がある）
                if st_hold_head.get():
                    hands.hold(target=_hold_tgt)
                else:
                    hands.release()
                apply_fence(st_fence.get())
                reflex.reset()
                t_sim, wall0, tick = 0.0, time.time(), 0
                toy_base[0] = None
                head_w.clear(); devs.clear(); seens.clear(); neck_hist.clear()
                parent_log.clear(); _parent[0] = None
                # ★おもちゃも初期状態に戻す。戻さないと「顔の前へ持っていく」で
                #   記録された位置（首が横を向いていれば体の横＝柵の外）が残り、
                #   やり直しても変な所にあるままになる（ユーザーの報告 2026-07-27）。
                follow_var.set(False)          # 環境まかせに戻す
                _syncing[0] = True
                for _i in range(3):
                    toy_vars[_i].set(round(float(toy_pos0[_i]), 3))
                _syncing[0] = False
                prev_sacc[0] = 0
                _restart[0] = False

            freeze = st_freeze.get()
            # ★物理を止めていると反射の指令が実行されない（step の中にあるため）
            if freeze and st_orient.get() and tick % 60 == 0:
                msg.config(text="⚠️物理を止めています。反射は目を動かせません"
                                "（『物理を止める』を外してください）")
            OR.SACCADE_LATENCY = float(lat_var.get())
            OR.SACCADE_MIN_STRENGTH = float(thr_var.get())
            u._orienting = reflex if st_orient.get() else None
            u._vor = vor if st_vor.get() else None
            want_tone = st_tone.get()
            if want_tone != tone_on[0]:
                for j in tone_joints:
                    m.jnt_stiffness[j] = tone_k[j] if want_tone else 0.0
                tone_on[0] = want_tone

            # ★★【2026-07-28 修正】頭を抑えているあいだは、このスライダーを**適用しない**。
            #   実験者の手も首のバネも同じ `jnt_stiffness` を使うので、毎tickここで
            #   スライダーの値（既定0.4）を書くと、`hands.hold()` が入れた強さ200が
            #   **上書きされて消えていた**（ユーザーの目視「首が結構動いてる」で発覚）。
            #   ＝落とし穴チェックリスト項62「設定した値が本当に体に届いているか」の再発。
            if not st_hold_head.get():
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

            # ★環境のスイッチ（柵・実験者の手）を実行中でも反映する
            if bool(st_fence.get()) != fence_on[0]:
                apply_fence(st_fence.get())
            if bool(st_hold_head.get()) != bool(hands.holding):
                if st_hold_head.get():
                    hands.hold(target=_hold_tgt)
                else:
                    hands.release()

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
            # ⚠️★【2026-07-28 修正】`freeze`（物理を止める）でスライダーの角度を
            #   書き戻していたため、**止めた瞬間に姿勢が作りかけの値へ飛んでいた**
            #   （ユーザーの要望「現状維持のまま物理を止めてほしい」）。
            #   スライダーで姿勢を作りたいときは「姿勢を固定」（st_hold）を使う。
            #   止めるだけなら今の姿勢をそのまま保つ。
            if st_hold.get():
                for jd, v in zip(joints, joint_vars):
                    ang = np.radians(v.get())
                    for jid, qadr in jd["pair"]:
                        d.qpos[qadr] = ang
                        d.qvel[int(m.jnt_dofadr[jid])] = 0.0

            # ★物理を止めているあいだは step() が呼ばれず、おもちゃを運ぶ処理も
            #   動かない。待たずに定位置へ置く（2026-07-27）。
            if freeze and getattr(u, "_toy_pending", False):
                try:
                    u.place_toy_now()
                    toy_base[0] = np.array(u._rest_pos, dtype=float)
                    _syncing[0] = True
                    for _i in range(3):
                        toy_vars[_i].set(round(float(u._rest_pos[_i]), 3))
                    _syncing[0] = False
                except Exception:
                    pass

            if freeze:
                # ⚠️★体の根元（位置・向き）も戻さない。止めた時点の姿勢を保つ。
                #   `root_qpos0` はリセット直後の値なので、書き戻すと
                #   リクライニングで座っていた体が仰向けの初期位置へ飛ぶ。
                #   「仰向けに戻す」ボタンを使いたいときだけ明示的に戻す。
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
                imgs = {}
                try:
                    imgs = u.get_vision_obs() or {}
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

                # ★距離と輻輳（寄り目）の必要角を出す（2026-07-28）。
                #   実際の目からおもちゃまでの距離も測って、設定値と比べられるようにする。
                try:
                    _eyes = [np.array(d.cam_xpos[int(m.camera(_n).id)], dtype=float)
                             for _n in ("eye_left", "eye_right")]
                    _org = np.mean(_eyes, axis=0)
                    _real = float(np.linalg.norm(
                        np.array(d.xpos[toy_bid], dtype=float) - _org))
                    _verg = 2.0 * np.degrees(np.arctan2(IPD / 2.0, max(_real, 1e-4)))
                    vergence_label.config(
                        text=f"実際の距離 {_real*100:5.1f}cm（設定 {dist_var.get()*100:.1f}cm）\n"
                             f"両目で見るのに要る寄り目 {_verg:5.1f}度"
                             f"（瞳孔間 {IPD*100:.1f}cm）★太郎は寄り目ができない")
                except Exception:
                    pass

                # ★環境の状態＝「今どの条件で見ているか」を常に出す（2026-07-28）。
                #   Viewer と測定で条件が食い違っていた事故を防ぐため。
                _off = hands.offsets() if hands.holding else {}
                _offmax = max((abs(v) for v in _off.values()), default=0.0)
                env_label.config(
                    text=f"体年齢 {_AGE:g}ヶ月（視力も同じ）  "
                         f"柵 {'あり' if fence_on[0] else 'なし'}  "
                         f"リクライニング {_RECLINE:.0f}度\n"
                         f"顎 {(_HOLD_TILT + '度で支える') if _HOLD_TILT else '指定なし'}"
                         f"　眼球の基準 {_EYE_REST_V:+.0f}度\n"
                         f"実験者の手 {'抑えている' if hands.holding else 'なし'}"
                         + (f"（頭のずれ {_offmax:.2f}度）" if hands.holding else ""))

                # ★左右の目をそれぞれ描く（2026-07-28）
                try:
                    from PIL import Image, ImageTk
                    for _side, _cv in eye_canvases.items():
                        _im = imgs.get(_side) if isinstance(imgs, dict) else None
                        if _im is None:
                            continue
                        arr = np.asarray(_im)
                        if arr.dtype != np.uint8:
                            arr = np.clip(arr, 0, 255).astype(np.uint8)
                        arr = arr.copy()
                        if mask_var.get():
                            arr[VIS.red_mask(arr)] = [0, 255, 0]
                        _imgtk[_side] = ImageTk.PhotoImage(
                            Image.fromarray(arr).resize((176, 176), Image.NEAREST))
                        _cv.config(image=_imgtk[_side])
                except Exception:
                    pass

                # ★コピー用のまとめ（「現在の設定をコピー」ボタンが使う）
                _toy = [f"{v.get():.3f}" for v in toy_vars]
                _pl2 = [x for x in parent_log if x is not None]
                _gaps = [_pl2[i] - _pl2[i - 1] for i in range(1, len(_pl2))]
                _navg = (sum(_gaps) / len(_gaps)) if _gaps else float("nan")
                _nk = neck_ids.get("head_tilt")
                _nkang = (float(np.degrees(d.qpos[int(m.jnt_qposadr[_nk])]))
                          if _nk is not None else float("nan"))
                _snapshot[0] = (
                    "【設定】\n"
                    f"おもちゃ  X={_toy[0]} Y={_toy[1]} Z={_toy[2]}  "
                    f"大きさ{size_var.get():.3f}  "
                    f"揺らす:{'ON' if shake_var.get() else 'OFF'}  "
                    f"位置を固定:{'ON' if follow_var.get() else 'OFF（環境まかせ）'}\n"
                    f"姿勢      物理:{'止めている' if freeze else '動かしている'}  "
                    f"角度を固定:{'ON' if st_hold.get() else 'OFF'}\n"
                    f"反射      視線誘導:{'ON' if st_orient.get() else 'OFF'}"
                    f"（間隔{lat_var.get():.2f}秒 閾値{thr_var.get():.2f}）  "
                    f"VOR:{'ON' if st_vor.get() else 'OFF'}  "
                    f"屈筋トーン:{'ON' if st_tone.get() else 'OFF'}\n"
                    f"首のバネ  {'ON' if st_neck.get() else 'OFF'}  "
                    f"剛性{nk_k.get():.2f}  減衰{nk_c.get():.4f}  "
                    f"目標{nk_t.get():+.0f}度  "
                    f"{'3軸' if nk_all.get() else '前後だけ'}\n"
                    f"再生      速度{speed_var.get():.1f}倍  "
                    f"自発運動:{'ON' if st_babble.get() else 'OFF'}\n"
                    "\n【今の状態】\n"
                    f"経過 {t_sim:.1f}秒  サッケード {reflex.n_saccades}発\n"
                    f"反応の強さ {reflex.strength:.3f}（閾値 {thr_var.get():.2f}）\n"
                    f"①角度 {rep['angle']:.1f}度 "
                    f"{'視野内' if rep['in_fov'] else '★視野外'}  "
                    f"②光線 {'遮蔽なし' if rep['ray_ok'] else '★' + str(rep['ray_hit']) + 'に遮られている'}  "
                    f"③画像 "
                    + (f"{rep['n_pixels']}画素 中心からのずれ {dev:.2f}"
                       if rep.get("pix_seen") else "★映っていない") + "\n"
                    f"見えていた割合 {seen_pct:.1f}%  "
                    f"ずれ平均 {(np.mean(devs) if devs else float('nan')):.2f}"
                    f"（反射OFFで0.48）\n"
                    f"首の角度 {_nkang:.1f}度  頭の角速度 平均{hw.mean():.3f}\n"
                    f"目→おもちゃ {de:.1f}cm  肩→ {ds:.1f}cm"
                    f"（腕の{ds/(ARM_REACH*100)*100:.0f}%）\n"
                    f"親の介入 {len(_pl2)}回"
                    + (f"（平均 {_navg:.1f}秒おき）" if _gaps else "")
                )

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
