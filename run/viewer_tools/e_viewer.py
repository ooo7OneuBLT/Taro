"""太郎を見るための統一ビューア。おもちゃ・姿勢・反射・測定器・再生を1枚にまとめる。

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
    python run/viewer_tools/e_viewer.py

  見出しをクリックすると、その区画を開いたり畳んだりできる。
  設定は「この設定を保存」で `E/docs/viewer_saved.json` に残り、次回自動で読み込む。

【区画】
    おもちゃ    位置・大きさ・揺らす
    姿勢        関節の角度・物理のON/OFF・仰向けに戻す
    反射        視線誘導（間隔・閾値）・前庭動眼反射・屈筋トーン
    測定器      3つの判定・一人称視点（検出画素を緑で表示）
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
# 注意：【2026-07-31】wrapper と senses を足した。無いと `hybrid_env` が読めない。
#   以前は `from run.taro_setup import Taro` が**副作用で**パスを足していたので
#   たまたま通っていた。taro_setup を通らない経路（内臓つき環境を先に作る等）で
#   ModuleNotFoundError になる。読み込みの順番に依存させない。
# 【なぜ、2026-08-07】run/viewer_tools/ へ移設した際に追加。e_visibility（この
#   ファイル冒頭でimport）・e_orienting_v2・e_toy_env・e_head_hold・e_body_config
#   （いずれも関数内での遅延import）は E/scripts 直下に残る（目標Eの本能実装群。
#   今回の移設のスコープ外）。以前は _HERE が E/scripts を指していたため
#   sys.path に _HERE を足すだけで暗黙に読めていたが、_HERE が run/viewer_tools に
#   変わった今はこれが効かない。run/scene_tools/e_scene.py が2026-08-05に
#   実際に踏んだのと同じ罠（設計：作業記録（非公開）
#   2026-08-07_runSystem移設_統合版.md 2.3節）。
for p in [os.path.join(_ROOT, "D", "scripts"), os.path.join(_ROOT, "MIMo"),
          os.path.join(_ROOT, "taro_core"),
          os.path.join(_ROOT, "taro_core", "src", "body"),
          os.path.join(_ROOT, "taro_core", "src", "brain"),
          os.path.join(_ROOT, "taro_core", "src", "wrapper"),
          os.path.join(_ROOT, "taro_core", "src", "senses"),
          os.path.join(_ROOT, "run", "scene_tools"),
          os.path.join(_ROOT, "E", "scripts"),
          _ROOT,
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

# 【なぜ、2026-08-07】以前は _HERE（E/scripts）の1つ上を"E"と仮定して
#   os.path.join(_HERE, os.pardir, "docs", ...) と書いていた。run/viewer_tools/ へ
#   移設すると _HERE の1つ上は "run" になり、この式は run/docs/viewer_saved.json
#   という存在しないパスを黙って見に行く（エラーは出ない。保存・復元が静かに
#   壊れる典型例）。保存先そのものは E/docs/ のまま動かさない（現状これを使うのは
#   目標Eの編集セッションだけで、保存内容はEの実験に紐づくデータのため）。
#   _ROOT 基準の絶対パスに直すことで、置き場所が変わっても壊れないようにした。
SAVE_PATH = os.path.join(_ROOT, "E", "docs", "viewer_saved.json")
OLD_SAVE = os.path.join(_ROOT, "E", "docs", "pose_editor_saved.json")

ARM_REACH = 0.186          # 肩から手先まで[m]（上腕+前腕+手）
HALF_FOV = 30.0            # 視野の半角[度]（fovy=60）
SHAKE_HZ = 2.5             # 「小さく激しく」＝2.5Hz
SHAKE_AMP = 0.015          # 1.5cm
EPISODE_SEC = 20.0

# ============================================================================
# 姿勢を作るための関節（2026-07-29 に全身へ拡張）
# ----------------------------------------------------------------------------
# 【なぜ広げたか】ユーザーの指摘「今のViewerは一部の関節の回転しかできない。
# 首の位置を変えたり腕の角度を変えたりしたい」。
# それまでは7つ（股2・ひざ・肩3・ひじ）だけで、しかも**左右が必ず同じ角度**だった。
#
# 太郎の関節は全部で96個あるが、姿勢を作るのに要るのは43個
# （首3・体幹8・腕8・手首6・股6・ひざ2・足首6・つま先4）。
# 指44個と眼球6個は除く：指は細かすぎ、眼球は反射が動かすので人が決めるものではない。
#
# 各項目 = (関節名の末尾, 日本語名)
# 左右がある関節は `right_` / `left_` を付けて探す（PAIRED=True のグループ）。
POSE_GROUPS = [
    # (グループ名, 左右のペアがあるか, [(関節名, 日本語名), ...], 既定で開くか)
    ("首", False, [
        ("head_tilt", "首（前後・うなずき）"),
        ("head_swivel", "首（左右ふり）"),
        ("head_tilt_side", "首（左右かたむき）"),
    ], True),
    ("肩・うで", True, [
        ("shoulder_horizontal", "肩（前後）"),
        ("shoulder_ad_ab", "肩（開き）"),
        ("shoulder_rotation", "肩（ひねり）"),
        ("elbow", "ひじ"),
    ], True),
    ("体幹", False, [
        ("hip_bend1", "腰（前後の曲げ・下）"),
        ("hip_bend2", "腰（前後の曲げ・上）"),
        ("hip_lean1", "腰（左右たおし・下）"),
        ("hip_lean2", "腰（左右たおし・上）"),
        ("hip_rot1", "腰（ひねり・下）"),
        ("hip_rot2", "腰（ひねり・上）"),
        ("chest_lean", "胸（左右たおし）"),
        ("chest_rot", "胸（ひねり）"),
    ], False),
    ("手首", True, [
        ("hand1", "手首1"), ("hand2", "手首2"), ("hand3", "手首3"),
    ], False),
    ("股・あし", True, [
        ("hip1", "股（前後）"), ("hip2", "股（開き）"), ("hip3", "股（ひねり）"),
        ("knee", "ひざ"),
    ], True),
    ("足首・つま先", True, [
        ("foot1", "足首1"), ("foot2", "足首2"), ("foot3", "足首3"),
        ("toes", "つま先"), ("big_toe", "親ゆび"),
    ], False),
]
# 旧定義（7関節・左右同時）。互換のために名前だけ残す
POSE_JOINTS = [(nm, jp) for _, _, items, _ in POSE_GROUPS for nm, jp in items]


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
        # `after=self.head` が必須。pack_forget したフレームをそのまま pack すると
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

    # 【2026-07-28】測定スクリプト（e_orient_converge_test.py）と条件を揃える。
    #   揃えないと「Viewerで見ている太郎と、測っている太郎が別物」になる
    #   （2026-07-25 に実際に起きた問題＝身体の設定が散らばる）。
    #     体年齢4ヶ月  … この反射を使うリーチングの月齢に揃えた
    #     視力も4ヶ月  … 体と揃える（1ヶ月の4.3倍）
    #     頭を抑える    … 人間の乳児実験と同じ条件（Hunter & Richards 2003）
    from e_head_hold import CaregiverHands
    import e_scene

    # ---- 駆動モード（筋肉／関節）2026-08-10 新設 --------------------------
    #   【なぜ】run.type=edit（このファイル）は taro.actuation を一切見ておらず、
    #   常に筋肉モードで体を組み立てていた（監査：作業記録（非公開）
    #   2026-08-10_run系システムとViewerの型バグ横断監査.md「中2」）。
    #   run/plugins/common/scene.py 57〜66行目と同じ語彙判定にする
    #   （joint/spring/springdamper/torque → 関節、それ以外→筋肉、不明値はエラー）。
    #   E_ACTUATION 未指定（既定）は "muscle" ＝今までと完全に同じ挙動。
    def _resolve_actuation(varname="E_ACTUATION"):
        mode = str(os.environ.get(varname, "muscle")).lower()
        if mode in ("joint", "spring", "springdamper", "torque"):
            from mimoActuation.actuation import SpringDamperModel
            print("注意[actuation] 関節モード（90関節を独立に駆動）＝逸脱リスト 逸脱5 の"
                  "逸脱を選んでいます。人間の新生児は拮抗筋を同時に力ませる[Tier1]",
                  flush=True)
            return SpringDamperModel, "joint"
        if mode in ("muscle", "muscles", ""):
            return MuscleModel, "muscle"
        raise ValueError(f"{varname} が不明: {mode}（muscle か joint）")

    _ACT_MODEL, _ACTUATION_MODE = _resolve_actuation()

    # ========================================================================
    # シーン方式（2026-07-29 新設）— E_SCENE=名前 でシーンから始める
    # ------------------------------------------------------------------------
    # 【なぜ】環境の条件が4か所（コードの定数／環境変数／プリセット／保存ファイル）に
    # 散らばっており、Viewer で見ている太郎と測定している太郎が食い違っていた。
    # シーンを使うと**環境を組み立てるのは `e_scene.build` だけ**になるので、
    # Viewer と測定が構造的に同じ環境になる。設計は `E/docs/シーン方式_設計.md`。
    # ========================================================================
    _scene = None
    _scene_name = os.environ.get("E_SCENE")
    if _scene_name:
        _scene = e_scene.load(_scene_name)
        # 月齢だけは E_AGE で上書きできる（2026-07-31）。
        #   【なぜ要るか】シーンは「新生児（0ヶ月）」で作ってあるが、
        #   体を育てる実験の学習済みモデルは**終わりの月齢（4ヶ月）の体**で学んでいる。
        #   シーンの月齢のまま脳を読むと「4ヶ月の脳が新生児の体を動かす」ことになり、
        #   見たいものと違う状態を見ることになる（2026-07-31 に実際に起きた）。
        #   注意：上書きしたらシーンの指紋とは一致しないので照合を外す。
        if os.environ.get("E_AGE"):
            _ov = float(os.environ["E_AGE"])
            if abs(_ov - float(_scene["body"]["age_months"])) > 1e-9:
                print(f"[scene] 月齢を上書き: "
                      f"{_scene['body']['age_months']} → {_ov} ヶ月"
                      f"（E_AGE の指定。指紋の照合は外します）", flush=True)
                _scene["body"]["age_months"] = _ov
                _scene["fingerprint"] = None
        _AGE = float(_scene["body"]["age_months"])
        _RECLINE = float(_scene["world"]["recline_deg"])
        _EYE_REST_V = float(_scene["body"]["eye_rest_vertical_deg"])
        _hh = _scene["setup"].get("head_hold") or {}
        _HEAD_HOLD = bool(_hh)
        _tgt = _hh.get("target_deg") or None
        _HOLD_TILT = (str(_tgt["head_tilt"]) if _tgt and "head_tilt" in _tgt else None)
        _hold_tgt = _tgt
        print(f"[scene] 「{_scene['name']}」から始めます", flush=True)
        if _scene.get("note"):
            print(f"        {_scene['note']}", flush=True)
        _scene_for_build = _scene
        if (_scene.get("setup") or {}).get("limb_tone"):
            import copy as _copy_lt
            _scene_for_build = _copy_lt.deepcopy(_scene)
            _scene_for_build["setup"]["limb_tone"] = None
            # 【なぜ、2026-08-07・重大バグ修正】e_scene.build() は scene.setup.limb_tone を
            #   無条件に物理へ適用するため、Viewer のチェックボックス（st_tone_limb、下記）が
            #   OFF でも常にONになる混線バグがあった（監査報告
            #   作業記録（非公開） 2-1）。
            #   ここで初回構築時は limb_tone を意図的に外し、「四肢の筋緊張を物理へ
            #   適用する経路」を下の limb_tone_apply()/limb_tone_release()（チェックボックス
            #   と連動する経路）だけに一本化する。build() は apply_state の後に
            #   mj_forward するだけで物理ステップは進めないため、limb_tone を外しても
            #   返ってきた直後の姿勢（qpos）はシーンの state と完全に一致する
            #   （run/scene_tools/e_scene.py の build()・_apply_limb_tone() で確認済み）。
        env, hands = e_scene.build(_scene_for_build, orient=True, vor=True, seed=0,
                                   verbose=True, actuation_model=_ACT_MODEL)
        u = env.unwrapped
        m, d = u.model, u.data
        if hands is None:
            hands = CaregiverHands(m, d)
        # 注意：【なぜ、2026-08-07・重大バグ修正の副作用対策】ここで verify() を
        #   呼ぶのをやめ、on_limb_tone() の初回呼び出し（msg 生成後）のあとへ
        #   移した。上の修正1aで limb_tone を外した直後のこの時点では、
        #   四肢の筋緊張がまだ物理へ入っていない（意図的な一時状態）ため、
        #   ここで照合すると「バネの入った関節数 記録=43 / 今=29」のような
        #   **偽陽性の食い違い警告**が limb_tone を使う全シーンで毎回出てしまう
        #   （実測で確認済み）。on_limb_tone() が実行された後なら物理は
        #   保存時と同じ状態に戻っているので、そちらで照合する。
    else:
        # ---- 従来の起動（環境変数で条件を指定する）--------------------------
        # 注意：こちらは順次たたむ予定。新しい実験はシーンを使うこと。
        _AGE = float(os.environ.get("E_AGE", "4.0"))
        _HEAD_HOLD = os.environ.get("E_HEAD_HOLD", "1") == "1"
        # リクライニングの角度[度]。0=仰向け。モデル構築時に決まるので実行中は変えられない
        _RECLINE = float(os.environ.get("E_RECLINE", "0"))
        _EYE_REST_V = float(os.environ.get("E_EYE_REST_V", "0"))
        kw = body_kwargs_from_env(_AGE, verbose=True)
        env = ToySupineEnv(actuation_model=_ACT_MODEL,
                           vision_params=infant_vision_params(acuity_age=_AGE),
                           age=_AGE, toy=True, vor=True, orient=True,
                           recline_deg=_RECLINE, **kw)
        u = env.unwrapped
        m, d = u.model, u.data
        env.reset(seed=0)
        hands = CaregiverHands(m, d)
        # 支える角度。指定しなければ「今の角度」で支える。
        #   リクライニングでは顎を引かせないと視線が上を向いてしまうので、
        #   前後（head_tilt）だけ指定できるようにした（2026-07-28）。
        _HOLD_TILT = os.environ.get("E_HOLD_TILT")
        _hold_tgt = ({"head_tilt": float(_HOLD_TILT)} if _HOLD_TILT else None)
        if _HEAD_HOLD:
            hands.hold(target=_hold_tgt, verbose=True)
    dt = float(m.opt.timestep) * int(u.frame_skip)
    n_act = env.action_space.shape[0]
    zero = np.zeros(n_act, dtype=np.float32)

    # ---- 四肢の筋力スライダー（2026-08-12 新設）------------------------------
    #
    # 【なぜ】0ヶ月児の四肢の筋力が弱すぎて腕が物理的に上がらない問題が判明した
    # （実測：肩の外転筋に最大指令を20秒入れ続けても補正係数1.0では25.9度しか
    # 上がらない。仕様 作業記録（非公開）
    # 2026-08-12_Viewer筋力スライダーと筋緊張バグ修正.md 依頼1）。
    #
    # 【実体】`MuscleModel._update_torque()`（MIMo/mimoActuation/muscle.py
    # 306〜322行目）は毎物理stepで self.fmax を読んでトルクを計算する
    # （self.fmaxはコンストラクタで1回だけ計算されるのではなく、
    # _update_torqueが呼ばれるたびに現在の値を参照する。実際に読んで確認済み）。
    # つまり set_fmax()（同ファイル403〜409行目、既存のセッター）で書き換えれば
    # 次の物理stepから即座に反映される。actuator_gearを外から書き換える方式は
    # `_apply_torque()`（324〜331行目）が毎stepそれを上書きするため機能しない。
    #
    # 【対象＝四肢のみ】既存の apply_limb_inversion_fix
    # （taro_core/src/body/infant_limbs.py 189〜194行目）が対象を決める判定
    # （act:head*・act:chest*で始まるアクチュエータを除外＝四肢）をそのまま
    # 踏襲する（検証の落とし穴チェックリスト項30「同じ式が複数箇所にあったら
    # 置き場所が間違っている」の考え方に沿い、同じ判定式を使う。関数自体は
    # 「除外して残りを使う」という単純な形なので、ここでも同じ条件式をそのまま
    # 書く。新しく別の基準を作っていない）。
    am = u.actuation_model
    _LIMB_AIDS = [i for i in range(m.nu)
                 if not (m.actuator(i).name or "").startswith(("act:head", "act:chest"))]
    # actuation=joint（SpringDamperModel）には fmax 属性が無いことを確認済み
    #   （MIMo/mimoActuation/actuation.py に fmax の記述なし）。
    # 環境変数キャリブレーションが無いと fmax がスカラー（float）のことがある
    #   （muscle.py 259〜269行目）。この場合は関節ごとに調整できないので、
    #   既存の scale_actuator_strength と同じ流儀で「何もしない」を選び、
    #   スライダー側に効果が無いことを明示する（黙って無視しない）。
    _fmax_base = None
    if hasattr(am, "fmax"):
        _f0 = np.asarray(am.fmax, dtype=float)
        if _f0.ndim > 0:
            _fmax_base = _f0.copy()          # Viewer起動直後・未操作時点の基準値
    _fs_available = _fmax_base is not None
    _fs_n_act = int(getattr(am, "n_actuators", len(_fmax_base) // 2)) if _fs_available else 0

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
    # 注意：【2026-07-29 修正】以前は「おもちゃ（test_object1）以外の自由関節」で
    #   探していたが、このモデルには**使っていない予備の物体 test_object2**があり、
    #   そちらを先に拾っていた（位置 3.5, 3.0, 0.05）。
    #   ＝「仰向けに戻す」はずっと**体ではなく予備の物体**を動かしていた。
    #   落とし穴 項67「体を動かす自由関節は、思っている body に無い」の再発。
    root_qadr = None
    for j in range(m.njnt):
        if (m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                and (m.body(int(m.jnt_bodyid[j])).name or "") == TE.ROOT_BODY):
            root_qadr = int(m.jnt_qposadr[j])
            break
    if root_qadr is None:
        print(f"[viewer] 注意体（body='{TE.ROOT_BODY}'）の自由関節が見つからない。"
              "「仰向けに戻す」は姿勢だけ戻します", flush=True)
    root_qpos0 = (d.qpos[root_qadr:root_qadr + 7].copy()
                  if root_qadr is not None else None)

    # ---- 前回の保存を読む（旧ファイルからも引き継ぐ）--------------------
    # 注意：シーンから始めたときは旧保存を読まない（2026-07-29）。
    #   読むと、シーンの姿勢で立ち上げたのにスライダーの初期値だけ旧ファイルの
    #   値になり、姿勢保持がONの瞬間に**別の姿勢へ書き換わる**。
    #   ＝「散らばった設定が競合する」問題そのものなので、入口を1つに保つ。
    saved = None
    for path in ([] if _scene is not None else (SAVE_PATH, OLD_SAVE)):
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as fp:
                    saved = json.load(fp)
                print(f"[load] 保存を読み込みました: {path}", flush=True)
                break
            except Exception as e:
                print(f"[load] 読み込み失敗 {path}: {e}", flush=True)

    # ---- 姿勢を作る関節を集める（2026-07-29 に全身へ拡張）------------------
    #   左右がある関節は**別々のスライダー**にする。以前は必ず同じ角度に
    #     なっていたので、片手だけ口に持っていくような姿勢が作れなかった。
    def _find_joint(name):
        """関節を名前で探す。見つからなければ None（モデルによって無い関節がある）。"""
        for pre in ("robot:", ""):
            try:
                j = m.joint(pre + name)
                return int(j.id), int(m.jnt_qposadr[j.id])
            except Exception:
                continue
        return None

    joints = []
    for gname, paired, items, opened in POSE_GROUPS:
        for base, jp in items:
            cands = ([(f"right_{base}", "右"), (f"left_{base}", "左")]
                     if paired else [(base, "")])
            for jname, side in cands:
                got = _find_joint(jname)
                if got is None:
                    continue
                jid, qadr = got
                lo, hi = np.degrees(m.jnt_range[jid])
                if abs(hi - lo) < 1e-9:      # 可動域ゼロ＝動かせない関節は出さない
                    continue
                joints.append(dict(
                    base=jname, key=jname, group=gname, side=side,
                    jp=(f"{side}{jp}" if side else jp),
                    pair=[(jid, qadr)],
                    lo=float(lo), hi=float(hi),
                    init=float(np.degrees(d.qpos[qadr])),
                    is_neck=(gname == "首"),
                    twin=(f"left_{base}" if side == "右" else
                          (f"right_{base}" if side == "左" else None))))
    print(f"[pose] 姿勢を作れる関節 {len(joints)} 個", flush=True)

    toy_pos0 = d.qpos[toy_qadr:toy_qadr + 3].copy()
    # 注意：リセット直後のおもちゃは**退避位置 [3,3,0.05]**（登場を1秒遅らせる仕組み）。
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
    # プリセットが持つおもちゃの位置（E_TOY_POS）は保存値より優先する。
    #   ＝「この環境ではここに置く」という条件の一部なので（2026-07-28）。
    if os.environ.get("E_TOY_POS"):
        try:
            toy_pos0 = np.array([float(x) for x in
                                 os.environ["E_TOY_POS"].split(",")], dtype=float)
        except Exception:
            pass
    size0 = float(m.geom_size[toy_gadr][0])
    # 注意：E_TOY_RADIUS を明示したときは保存値で上書きしない。
    #   上書きすると、環境変数で指定した大きさが黙って無視される
    #   （実際 E_TOY_RADIUS=0.010 が保存値 0.02 に置き換わっていた）。
    if saved and "toy_half_size" in saved and "E_TOY_RADIUS" not in os.environ:
        size0 = float(saved["toy_half_size"])
        m.geom_size[toy_gadr] = [size0, size0, size0]

    # ================= パネル =================
    # 【2026-07-28 レイアウト改訂】1列（幅520px）で6区画を縦に積んでいたため
    #   画面からはみ出して縦に長すぎた（ユーザーの指摘）。**2列**に変え、
    #   目の映像も左右そろえて出す。
    win = tk.Tk()
    win.title("太郎ビューア（統一版）")
    _h = min(980, win.winfo_screenheight() - 80)
    _w = min(1080, win.winfo_screenwidth() - 60)
    win.geometry(f"{_w}x{_h}+20+10")
    # 注意：最前面に固定しない（2026-07-28、ユーザーの要望）。他の窓を見るたびに
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

    # 2列。左＝環境と条件（いじる物）／右＝見る物（映像・数値）
    #
    # 注意：【2026-07-28 修正】最初 `pack` + `pack_propagate(False)` + `width=520` で
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
    # 2026-07-27：既定を「物理を動かす」に変えた（それまでは止まって起動）。
    #   反射が目を動かすのは env.step() の中なので、物理を止めていると
    #   **方向は正しく計算されているのにサッケードが1発も撃たれない**
    #   （ユーザーの目視：強さ0.406＞閾値、ずれ0.95 なのに 0発。2026-07-27）。
    #   Viewer の主目的は反射の観察なので、止めたい人が自分でONにする形にする。
    freeze0 = os.environ.get("E_FREEZE", "0") == "1"

    # ---- 区画0：環境（プリセット）2026-07-28 新設 -----------------------
    #
    # 【なぜ要るか】ユーザーの要望「環境をいくつかバージョン用意して、プルダウンで選んで
    # 最初からやり直しを押したらその環境で始まる」。実験ごとに条件（月齢・柵・頭の支え）を
    # 手で合わせるのは間違いのもとで、実際に「Viewerで見ている太郎と測っている太郎が
    # 別物」という問題が起きていた（2026-07-25、体型補正が環境ごとにバラバラだった件）。
    #
    # 注意：**月齢だけはモデルを作るときに決まる**（geomの寸法・質量が変わる）ので、
    #   実行中に変えられない。月齢が変わるプリセットを選んだときは**プロセスを
    #   作り直す**（環境変数を設定して自分を起動し直す）。それ以外は即反映。
    sec_env = Section(colL, "シーン（環境）", op.get("env", True))

    # ------------------------------------------------------------------
    # シーン一覧（2026-07-29）。旧 PRESETS を置き換えたもの。
    #   プリセットは Viewer の中にしか無く、測定スクリプトからは使えなかった。
    #   シーンはファイルなので、Viewer で作ったものをそのまま測定が読める。
    # ------------------------------------------------------------------
    scene_names = e_scene.list_scenes()
    _init_scene = (_scene["name"] if _scene else
                   (scene_names[0] if scene_names else ""))
    scene_var = tk.StringVar(value=_init_scene)
    scene_name_var = tk.StringVar(value=(_init_scene or "新しいシーン"))
    note_var = tk.StringVar(value=(_scene.get("note", "") if _scene else ""))

    _pf = tk.Frame(sec_env.body); _pf.pack(fill="x", padx=10, pady=(4, 0))
    tk.Label(_pf, text="シーン", width=6, anchor="w").pack(side="left")
    _scene_menu = tk.OptionMenu(_pf, scene_var,
                                *(scene_names or ["（シーンがまだ無い）"]))
    _scene_menu.pack(side="left", fill="x", expand=True)
    scene_note = tk.Label(sec_env.body, text="", fg="#666", font=("", 8),
                          wraplength=460, justify="left")
    scene_note.pack(anchor="w", padx=14)
    tk.Label(sec_env.body,
             text="「最初からやり直す」を押すと、選んだシーンで始まります\n"
                  "　 別のシーンに切り替えるときは体を作り直すので数十秒かかります",
             fg="#a30", font=("", 8), justify="left").pack(anchor="w", padx=14)

    def _refresh_scene_menu():
        names = e_scene.list_scenes()
        mnu = _scene_menu["menu"]
        mnu.delete(0, "end")
        for n in names:
            mnu.add_command(label=n, command=tk._setit(scene_var, n))

    def on_scene_pick(*_a):
        """選んだシーンが**成立しているか**を、始める前に見せる。

        注意：「姿勢が崩れる」「おもちゃが体に隠れて見えない」は、
          実際に走らせてからでは気づきにくい（2026-07-28 に両方とも起きた）。
          保存時に記録した指紋と安定確認の結果をここに出す。
        """
        try:
            sc = e_scene.load(scene_var.get())
        except Exception:
            scene_note.config(text="", fg="#666")
            return
        fp = sc.get("fingerprint") or {}
        se = fp.get("settle") or {}
        lines = []
        if sc.get("note"):
            lines.append(sc["note"])
        lines.append(f"{sc['body']['age_months']:g}ヶ月 / "
                     f"リクライニング{sc['world']['recline_deg']:g}度 / "
                     f"柵{'あり' if sc['world']['fence'] else 'なし'} / "
                     f"頭を支える{'あり' if sc['setup'].get('head_hold') else 'なし'}")
        warn = False
        if se:
            lines.append(se.get("summary", ""))
            warn = warn or not se.get("ok", True)
        if fp.get("toy_visible_left") is not None:
            lines.append("おもちゃは見えています" if fp["toy_visible_left"]
                         else "注意おもちゃが体に隠れて見えません")
            warn = warn or not fp["toy_visible_left"]
        if fp.get("toy_reach_ratio") is not None:
            r = float(fp["toy_reach_ratio"])
            lines.append(f"おもちゃは肩から腕の{r*100:.0f}%"
                         + ("（届く）" if r <= 1.0 else "（届かない）"))
            warn = warn or r > 1.0
        scene_note.config(text="\n".join(t for t in lines if t),
                          fg=("#a30" if warn else "#666"))
    scene_var.trace_add("write", on_scene_pick)

    # 旧プリセット（削除予定）。シーンへ移し終えたら消す。
    # 注意：2026-07-29 現在、下の PRESETS はどこからも参照していない。
    #   条件の由来をたどれるようにするためだけに残している。
    _OLD_PRESETS = {
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
        "リーチング（リクライニング60度・顎を引く）": dict(
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
    # 個別のスイッチ（シーンを選ぶと連動して変わる）
    st_fence = tk.BooleanVar(value=True)
    tk.Checkbutton(sec_env.body, text="柵（ベビーサークル）を有効にする",
                   variable=st_fence).pack(anchor="w", padx=14)
    st_hold_head = tk.BooleanVar(value=_HEAD_HOLD)
    tk.Checkbutton(sec_env.body, text="実験者が頭を抑える（人間の乳児実験と同じ）",
                   variable=st_hold_head, fg="#06a").pack(anchor="w", padx=14)
    env_label = tk.Label(sec_env.body, text="", font=("Consolas", 9),
                         justify="left", fg="#444")
    env_label.pack(anchor="w", padx=14)

    # シーンを選んだら、そのシーンの説明とスイッチを反映する
    #   注意：柵と頭の支えは**実行中に切り替えられる**ので即反映してよい。
    #     月齢・リクライニング角・眼球の基準角はモデル構築時に決まるので、
    #     「最初からやり直す」でプロセスごと作り直す。
    def on_scene_switches(*_a):
        try:
            sc = e_scene.load(scene_var.get())
        except Exception:
            return
        st_fence.set(bool(sc["world"]["fence"]))
        st_hold_head.set(bool(sc["setup"].get("head_hold")))
        scene_name_var.set(sc["name"])
        note_var.set(sc.get("note", ""))
    scene_var.trace_add("write", on_scene_switches)
    on_scene_pick()          # 起動時の説明を出す
    if _scene is not None:
        st_fence.set(bool(_scene["world"]["fence"]))

    # 柵のgeom（実行時にON/OFFする）。名前は e_toy_env の `fence_post_{i}`。
    fence_gids = [g for g in range(m.ngeom) if "fence" in (m.geom(g).name or "")]
    fence_rgba0 = {g: m.geom_rgba[g].copy() for g in fence_gids}
    fence_con0 = {g: (int(m.geom_contype[g]), int(m.geom_conaffinity[g]))
                  for g in fence_gids}
    fence_on = [True]

    def apply_fence(want):
        """柵を実行時に消す／戻す。geomを消せないので当たり判定と色で表現する。"""
        # 注意：環境側のフラグも合わせる。おもちゃの置き場所は `_set_anchor` が
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
    # スライダーを動かしたら自動で「固定する」に切り替える。
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
    # 既定は「環境まかせ」＝環境が視線の正面に置く。
    #   保存された位置は**あなたが手で置いたもの**で、視線の正面とは限らない
    #   （実際 X=0.155 は目 X≈0.20 より頭側＝視線の後ろだった。2026-07-26）。
    #   反射を試すときは環境まかせにしないと、条件が変わって比べられない。
    follow_var = tk.BooleanVar(value=bool((saved or {}).get("follow", False)))
    tk.Checkbutton(sec_toy.body,
                   text="スライダーの位置に固定する（外すと環境が視線の正面に置く）",
                   variable=follow_var).pack(anchor="w", padx=14)
    tk.Label(sec_toy.body,
             text=f"持たせ方 {TE.TOY_MODE} ／ 登場 {TE.TOY_APPEAR_DELAY:.1f}秒後に"
                  f"{TE.TOY_APPROACH_SEC:.1f}秒かけて{TE.TOY_APPROACH_FROM}から",
             fg="#666", font=("", 8)).pack(anchor="w", padx=14)

    # ---- あなたが「親」をやる ------------------------------------------
    #   人間の親は、赤ちゃんの顔の向きを見ておもちゃをその前に持っていく。
    #   太郎にはそれが無いので、首が動くとおもちゃが視界から外れたままになる。
    #   自動化する前に**手で試して、必要な介入の性質を掴む**のが目的。
    #   注意：瞬間移動はしない（ワープは随伴性の学習を壊す。_apply_tether の注記と同じ）。
    tk.Label(sec_toy.body, text="あなたが「親」をやる",
             font=("", 10, "bold")).pack(anchor="w", padx=14, pady=(8, 0))
    tk.Label(sec_toy.body, justify="left", fg="#555", font=("", 8),
             text="人間の親は赤ちゃんの顔の向きに合わせておもちゃを見せる。\n"
                  "太郎にはそれが無いので、首が動くと視界から外れたままになる。\n"
                  "注意手動は「何が必要か」を掴むためのもの。数値の比較には使えない。"
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
        # 距離はスライダーの値を使う（2026-07-28。それまで環境の固定値だった）
        dist = float(dist_var.get())
        u._toy_dist = dist          # 環境側（_set_anchor）にも反映
        goal = origin + fwd * dist
        # 柵の内側にとどめる。人間の親は柵の外に手を出さない。
        #   首が横を向いていると「視線の正面」が柵の外になり、
        #   おもちゃが柵に遮られて見えなくなる（ユーザーの報告 2026-07-27）。
        # 注意：【2026-07-28 修正】**柵があるときだけ**にした。柵は新生児の体に合わせた
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

    # 【2026-07-28 新設】目からおもちゃまでの距離。
    #
    # 【なぜ要るか】従来 `TOY_DISTANCE = 0.086`（8.6cm）の固定値で、これは
    # **0ヶ月の腕の長さ（18.6cm）を基準に決めた暫定値**だった。しかも e_toy_env.py の
    # コメントには実測結果として「8.6cmは柵に当たる／15cmならめり込まず腕で届く」と
    # 書いてあるのに、既定値が更新されていなかった。体年齢を上げると腕が伸びるので、
    # 距離も見直す必要がある。
    #
    # 注意：近すぎると**輻輳（寄り目）**が要る。太郎は両目に同じ指令を出す実装
    # （Hering の等神経支配の法則）で、輻輳は実装していない。
    #     距離8.6cm・瞳孔間4.5cm → 必要な寄り目 約29度
    #     距離15cm              → 約17度
    #     距離30cm              → 約8.6度
    # ＝近いほど「両目で同じものを見られない」状態になる。下に必要角を表示する。
    dist_var = tk.DoubleVar(value=float((saved or {}).get(
        "toy_dist", getattr(u, "_toy_dist", 0.086))))
    slider(sec_toy.body, "目からの距離[m]", dist_var, 0.05, 0.40, 0.005,
           note="「顔の前へ持っていく」を押すとこの距離に置く")
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

    tk.Button(sec_toy.body, text="顔の前へ持っていく", command=bring_to_face,
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

    tk.Checkbutton(sec_pose.body, text="物理演算を止める（編集モード。重力も衝突も無し）",
                   variable=st_freeze, fg="#a30", command=on_freeze).pack(anchor="w", padx=14)
    tk.Checkbutton(sec_pose.body, text="この角度で固定する（物理ONのまま使うと暴れます）",
                   variable=st_hold).pack(anchor="w", padx=14)

    # ---- 左右対称スイッチ ------------------------------------------------
    #   既定はOFF（左右を別々に動かせる）。以前は必ず同じ角度になっていて、
    #     片手だけ口へ持っていくような**非対称の姿勢が作れなかった**。
    st_sym = tk.BooleanVar(value=False)
    tk.Checkbutton(sec_pose.body, text="左右を対称に動かす（片方を動かすともう片方も同じ角度）",
                   variable=st_sym, fg="#06a").pack(anchor="w", padx=14)

    joint_vars = []
    _by_key = {}          # 関節名 → スライダー（左右連動に使う）
    _sym_busy = [False]   # 連動の再帰を防ぐ

    # ボタンは**スライダーより先**に置く（43個の下だと遠くて押しに行けない）
    _pf2 = tk.Frame(sec_pose.body); _pf2.pack(fill="x", padx=14, pady=(4, 2))
    tk.Button(_pf2, text="いまの姿勢を取り込む",
              command=lambda: pose_from_body(), width=20).pack(side="left", padx=2)
    tk.Button(_pf2, text="右→左に写す",
              command=lambda: pose_mirror_rl(), width=13).pack(side="left", padx=2)
    tk.Label(sec_pose.body,
             text="首を動かすと、実験者の手が支える目標角も一緒に動きます\n"
                  "　 姿勢を作る手順： ①物理を止める ②角度を決める ③固定をON ④保存",
             fg="#666", font=("", 8), justify="left").pack(anchor="w", padx=14)

    # ---- 四肢の筋緊張（2026-07-29 新設）--------------------------------
    #
    # 【なぜ要るか】ユーザーの目視「手を体の前にやったけど、すぐ重力で下に降りちゃう」。
    # 太郎の四肢には筋緊張が無いので、脱力すると腕が真横に伸びきる。
    # 実測：腕を体の前にしても20秒で **18cm 落ちて**、手とおもちゃが24cmに広がる。
    # 人間の4ヶ月児は脱力しても肘が曲がり手が体の前にある（Dubowitz 1970 ほか）。
    #
    # 【使い方】姿勢を作る → このスイッチをON → その姿勢が「戻る先」になる。
    #   自発運動でバネより強い力が出れば腕は動き、力を抜けば戻る。
    # 注意：意思ではなく**身体の性質**。太郎の脳（方策）は通らない。
    _lt0 = ((_scene or {}).get("setup") or {}).get("limb_tone") or {}
    st_tone_limb = tk.BooleanVar(value=bool(_lt0))
    # 【なぜ、2026-08-08】旧「屈筋トーン」（st_tone、反射区画）と
    #   この「四肢の筋緊張」（st_tone_limb、姿勢区画）は別々のチェックボックスだったが、
    #   apply_limb_tone() が profile 引数（"newborn_flexor"/"reach_limb"）を受けるように
    #   なったので、目標角の種類を選ぶラジオボタンに統合した。
    #   _lt0 に target_deg があれば「新生児の既定姿勢（文献）」で保存されたシーン
    #   （もう1人の実装担当が作る run/scenes/新生児_*.json 想定）とみなし、
    #   無ければ従来どおり「今の姿勢を保つ」を初期モードにする。
    lt_mode = tk.StringVar(value=("newborn" if _lt0.get("target_deg") else "reach"))
    lt_hold = tk.DoubleVar(value=float(_lt0.get("hold_deg", 10.0)))
    _limb_saved = [None]

    def limb_tone_apply(*_a):
        """いまの姿勢、または新生児の既定姿勢を四肢の筋緊張の目標にする。"""
        from infant_limbs import apply_limb_tone, limb_tone_joints
        profile = "newborn_flexor" if lt_mode.get() == "newborn" else "reach_limb"
        if _limb_saved[0] is None:      # 最初の1回だけ元の値を控える（OFFで戻すため）
            # 【なぜ groups=("arm","leg") で集めるか】"newborn_flexor" は
            #   ("shoulder","elbow","hip_sagittal","knee")、"reach_limb" は
            #   ("arm","leg") を対象にする。"arm"+"leg" は両方の対象関節を
            #   包含する上位集合（"hip_sagittal"はhip1・hip2で、"leg"のhip
            #   グループhip1/hip2/hip3に含まれる）なので、モードを問わず
            #   この集合で保存しておけば、どちらのモードで適用してもOFFに
            #   戻すときに正しく復元できる。
            names = set(limb_tone_joints(m, groups=("arm", "leg")))
            sv = {}
            for j in range(m.njnt):
                nm = (m.joint(j).name or "").split(":")[-1]
                if nm in names:
                    adr, dof = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
                    sv[nm] = (float(m.jnt_stiffness[j]), float(m.qpos_spring[adr]),
                              float(m.dof_damping[dof]), j, adr, dof)
            _limb_saved[0] = sv
        r = apply_limb_tone(m, d, age=_AGE, profile=profile,
                            hold_deg=float(lt_hold.get()), verbose=True)
        _mode_jp = "新生児の既定姿勢（文献）" if profile == "newborn_flexor" else "今の姿勢"
        msg.config(text=f"四肢の筋緊張の目標を「{_mode_jp}」にしました"
                        f"（{r['n']}関節・許すずれ{lt_hold.get():g}度）", fg="#0a7")

    def limb_tone_release(*_a):
        sv = _limb_saved[0]
        if not sv:
            return
        for nm, (k, sp, dp, j, adr, dof) in sv.items():
            m.jnt_stiffness[j] = k
            m.qpos_spring[adr] = sp
            m.dof_damping[dof] = dp
        msg.config(text="四肢の筋緊張を切りました（腕は重力で落ちます）", fg="#0a7")

    def on_limb_tone(*_a):
        (limb_tone_apply if st_tone_limb.get() else limb_tone_release)()

    tk.Checkbutton(sec_pose.body,
                   text="四肢の筋緊張（脱力しても四肢が既定の姿勢に保たれる）",
                   variable=st_tone_limb, fg="#06a",
                   command=on_limb_tone).pack(anchor="w", padx=14)
    _ltf = tk.Frame(sec_pose.body); _ltf.pack(anchor="w", padx=28)
    tk.Radiobutton(_ltf, text="新生児の既定姿勢（文献）", variable=lt_mode,
                   value="newborn", command=on_limb_tone).pack(side="left")
    tk.Radiobutton(_ltf, text="今の姿勢を保つ", variable=lt_mode,
                   value="reach", command=on_limb_tone).pack(side="left")
    slider(sec_pose.body, "  許すずれ[度]", lt_hold, 2, 40, 1, width=13, length=200,
           note="　 小さいほど硬い。注意乳児の四肢の筋緊張の実測値は文献に存在しない"
                "（Tier3・感度分析の対象）")

    def limb_tone_capture_current():
        """このボタンは常に『今の姿勢』を目標にする（ラジオボタンの選択を上書きする）。
        押した瞬間の姿勢を目標に固定したい、という操作の意味そのものが
        「新生児の既定姿勢」モードとは両立しないため、押すと自動的に
        「今の姿勢を保つ」モードへ切り替える。
        """
        lt_mode.set("reach")
        st_tone_limb.set(True)
        on_limb_tone()

    tk.Button(sec_pose.body, text="いまの姿勢を筋緊張の目標にする",
              command=limb_tone_capture_current, width=28).pack(anchor="w", padx=14, pady=(0, 2))

    def _make_sym_hook(jd, var):
        def _hook(*_a):
            if not st_sym.get() or _sym_busy[0] or not jd.get("twin"):
                return
            tw = _by_key.get(jd["twin"])
            if tw is None:
                return
            _sym_busy[0] = True
            try:
                tw.set(var.get())
            finally:
                _sym_busy[0] = False
        return _hook

    # 部位ごとに折りたたみ（43個を一列に並べると縦に長すぎて操作できない）
    for gname, paired, items, opened in POSE_GROUPS:
        mine = [jd for jd in joints if jd["group"] == gname]
        if not mine:
            continue
        sub = Section(sec_pose.body, f"　{gname}（{len(mine)}）", opened)
        for jd in mine:
            v = tk.DoubleVar(value=jd["init"])
            slider(sub.body, jd["jp"], v, jd["lo"], jd["hi"], 1, width=16, length=200)
            joint_vars.append(v)
            _by_key[jd["key"]] = v
            jd["var"] = v
    for jd in joints:
        jd["var"].trace_add("write", _make_sym_hook(jd, jd["var"]))

    def pose_from_body():
        """いまの体の角度をスライダーへ取り込む。

        【なぜ要るか】物理を回して落ち着いた姿勢を土台にして手直ししたい。
        取り込みが無いと、スライダーは起動時の値のままなので「固定」を押した
        瞬間に**作りかけの姿勢へ飛ぶ**（2026-07-28 に踏んだ問題と同じ形）。
        """
        _sym_busy[0] = True
        try:
            for jd in joints:
                jd["var"].set(round(float(np.degrees(d.qpos[jd["pair"][0][1]])), 1))
        finally:
            _sym_busy[0] = False
        msg.config(text="いまの姿勢をスライダーに取り込みました", fg="#0a7")

    def pose_mirror_rl():
        """右の角度を左へ写す（左右対称の姿勢をすぐ作る）。"""
        _sym_busy[0] = True
        try:
            for jd in joints:
                if jd.get("side") == "右" and jd.get("twin") in _by_key:
                    _by_key[jd["twin"]].set(jd["var"].get())
        finally:
            _sym_busy[0] = False
        msg.config(text="右の角度を左へ写しました", fg="#0a7")


    # ---- 区画3：反射 ----------------------------------------------------
    sec_ref = Section(colL, "反射", op.get("reflex", True))
    st_orient = tk.BooleanVar(value=bool((saved or {}).get("orient", False)))
    tk.Checkbutton(sec_ref.body, text="視線誘導反射（動くものに目・首を向ける）",
                   variable=st_orient).pack(anchor="w", padx=14)
    lat_var = tk.DoubleVar(value=float((saved or {}).get("latency", OR.SACCADE_LATENCY)))
    slider(sec_ref.body, "  間隔[秒]", lat_var, 0.1, 1.5, 0.1,
           note="0.2＝旧設定（撃ちすぎて視界から追い出す）／0.5〜0.9＝新生児の実測")
    thr_var = tk.DoubleVar(value=float((saved or {}).get("threshold", OR.SACCADE_MIN_STRENGTH)))
    # 注意：範囲を 0〜0.5 から 0〜0.06 に狭めた。実測で決めた値は 0.015 で、
    #   0.25 のような値にすると**動きの強さが届かず一度も撃たない**。
    slider(sec_ref.body, "  発火の閾値", thr_var, 0.0, 0.06, 0.0025,
           note="0.02＝旧設定（低すぎて常に発火）／0.25前後で動きを選べる")
    st_vor = tk.BooleanVar(value=True)
    tk.Checkbutton(sec_ref.body, text="前庭動眼反射（頭の動きを眼で打ち消す）",
                   variable=st_vor).pack(anchor="w", padx=14)

    # ---- 区画3b：首のバネ（調整中。値が決まったら core に実装する）----------
    sec_neck = Section(colL, "首のバネ（調整中）", op.get("neck", True))
    tk.Label(sec_neck.body, justify="left", fg="#555", font=("", 8),
             text="首にはバネが無く（stiffness=0）、重力で60秒かけて60度倒れ続ける。\n"
                  "正常な新生児でも頸部に軽い抵抗はある（過緊張は正常児の0.7%だけ\n"
                  "／Amiel-Tison 1977）。死後標本の剛性は屈曲0.175・伸展0.40 Nm/rad\n"
                  "（筋を含まない下限値／Luck 2008）。生体の実測値は存在しない。"
             ).pack(anchor="w", padx=14, pady=(2, 4))
    # 2026-07-27修正：初期値は**core が実際に設定した値**から読む。
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
    tk.Checkbutton(sec_neck.body, text="首にバネを効かせる（外すと core の設定を消します）",
                   variable=st_neck, fg="#a30").pack(anchor="w", padx=14)
    nk_k = tk.DoubleVar(value=(_k0 if _k0 > 0 else 0.4))
    slider(sec_neck.body, "剛性[Nm/rad]", nk_k, 0.0, 0.6, 0.01,
           note="死後標本の下限 0.175〜0.40／四肢のトーンは 0.2")
    nk_c = tk.DoubleVar(value=_c0)
    slider(sec_neck.body, "減衰", nk_c, 0.0, 0.2, 0.005,
           note="臨界減衰（振動しない最小値）は下に表示")
    # 目標角の範囲と既定を姿勢に合わせる（2026-07-28）。
    #   仰向けは -45度（重力を織り込んだ実効値）だが、リクライニングでは
    #   顎を引く側（正）にしないと視線が上を向いてしまう。
    _nk_lo, _nk_hi = (-60.0, 30.0) if _RECLINE <= 0 else (-30.0, 70.0)
    _nk_def = -45.0 if _RECLINE <= 0 else 30.0
    nk_t = tk.DoubleVar(value=(_t0 if _k0 > 0 else _nk_def))
    slider(sec_neck.body, "目標角[度]", nk_t, _nk_lo, _nk_hi, 1.0,
           note=("重力を織り込んだ実効値。-45度で実際は-25度あたりに落ち着く"
                 if _RECLINE <= 0 else
                 "正が前屈（顎を引く）。リクライニングでは顎を引かないと視線が上を向く"))
    # リクライニングでは既定でON。
    #   首のバネは前後（head_tilt）だけに入れる設計だったが、これは**仰向け前提**。
    #   体を起こすと頭の重さが左右方向にも効くので、前後だけだと
    #   **頭が横を向いて倒れる**（実測：視線が真横 y=0.83。2026-07-28）。
    #   ユーザーの目視「さっきはなんか首が変だった」の原因。
    nk_all = tk.BooleanVar(value=(_RECLINE > 0))
    tk.Checkbutton(sec_neck.body,
                   text="3軸すべてに効かせる（外すと前後の傾きだけ）"
                        "リクライニングでは必要",
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
    # 【2026-07-28】右目も出す（ユーザーの要望）。それまで左目だけだった。
    #   両眼を並べると「片方にしか映っていない」ことに気づける。
    #   注意：太郎の反射は**両眼の画像を使う**（左目だけ使っていた旧実装は 2026-07-26 に修正済み）
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

    # 【なぜ、2026-08-07】駆動モード（探索ノイズの性質。white=白色ガウス／
    #   colored=色付きノイズ／colored+synergy=色付き+粗いシナジー）の表示・切替UIを、
    #   既定で畳まれている「脳」区画から、既定で開いているここ（再生区画・
    #   もがき運動チェックボックスの直後）へ移した（仕様：もがき運動の駆動モード
    #   統一とUI移設、2026-08-07）。もがき運動（下の _babble_noise() 相当）は
    #   「脳」区画の外にあり、UIがそちらにしか無いと実質使えなかった。
    #   NOISE_MODES / noise_mode / noise_label_dirty は「脳」区画・もがき運動の
    #   両方から共有される状態なので、ここ1箇所にだけ置く（重複させない）。
    # 【なぜ、2026-08-12】4つ目の駆動モード「reflex_common」（伸張反射＋揺らぐ
    #   振動子の共通駆動、taro_core/src/brain/spinal_cord/common_drive.py・
    #   stretch_reflex.py）を追加した。仕様：作業記録（非公開）
    #   2026-08-12_Viewer反射共通駆動モード追加.md。
    #   actuation=muscle が必須（moment_1/moment_2はMuscleModel構築時にしか
    #   計算されないため、run/config.pyのバリデーションと同じ制約）。
    NOISE_MODES = ["white", "colored", "colored+synergy", "reflex_common"]
    noise_mode = ["white"]          # 循環の現在地。書き換えるのは _set_noise_mode() だけ
    noise_label_dirty = [True]      # 画面ラベルの更新が要るか（別スレッドから直接
                                     # tkinterを触らないためのフラグ。下記「注意」参照）
    noise_mode_var = tk.StringVar(value=noise_mode[0])   # ラジオボタンの選択状態
    # 実際の適用処理（_set_noise_mode）は「脳」の準備のあとで定義するので、
    #   ここでは空のフックだけ置く（既存の _switch_hook と同じパターン）。
    _noise_ui_hook = [lambda *_: None]

    _nf = tk.Frame(sec_run.body); _nf.pack(anchor="w", padx=14, pady=(2, 0))
    tk.Label(_nf, text="駆動モード:").pack(side="left")
    _noise_radio_widgets = {}
    for _nv, _nlab in (("white", "白色"), ("colored", "色付き"),
                        ("colored+synergy", "色付き+シナジー"),
                        ("reflex_common", "反射+共通駆動")):
        _rb = tk.Radiobutton(_nf, text=_nlab, variable=noise_mode_var, value=_nv,
                             command=lambda: _noise_ui_hook[0](noise_mode_var.get()))
        _rb.pack(side="left")
        _noise_radio_widgets[_nv] = _rb
    # 【やること2・actuation=muscle必須のガード】actuation=joint のとき、
    #   ラジオボタン自体を選べなくする（黙って別モードで動くことを防ぐ第一段）。
    #   防御的な第二段は下の _set_noise_mode() 内にある。
    if _ACTUATION_MODE != "muscle":
        _noise_radio_widgets["reflex_common"].config(state=tk.DISABLED)
    noise_mode_label = tk.Label(
        sec_run.body, text="駆動モード: white（Nキー、またはボタンで切替）",
        font=("Consolas", 9), justify="left", fg="#666")
    noise_mode_label.pack(anchor="w", padx=14)

    # ---- 四肢の筋力調整（2026-08-12 新設）---------------------------------
    #
    # 【なぜ】もがき運動・脳・反射+共通駆動のどの駆動方式でも共通して効く
    #   身体側のパラメータ（fmax）なので、特定の駆動モードの区画ではなく
    #   「再生」区画の中に独立の折りたたみ小区画として置く（仕様やること3・
    #   UI整理）。既定では畳んでおく（常用するものではないため、開いたままだと
    #   区画が縦に伸びすぎる）。
    #
    # 【対数スケール】1.0〜200程度を対数的に振れるよう、スライダー変数自体は
    #   log10(倍率) を持たせ（0.0〜log10(200)）、表示・適用の直前に 10**value
    #   へ変換する（仕様の推奨どおり）。
    import math as _fs_math
    _FS_LOG_MAX = _fs_math.log10(200.0)
    fs_log_var = tk.DoubleVar(value=0.0)     # log10(1.0) = 0.0 ＝既定は等倍

    def _fs_current_x():
        return 10.0 ** float(fs_log_var.get())

    sec_fs = Section(sec_run.body, "四肢の筋力調整", False)
    if not _fs_available:
        tk.Label(sec_fs.body, fg="#a33", font=("", 8), justify="left",
                 text="注意fmaxが定数（キャリブレーションファイル無し）、または\n"
                      "actuation=joint のため、このスライダーは効きません"
                 ).pack(anchor="w", padx=14, pady=(0, 4))
    else:
        slider(sec_fs.body, "筋力倍率(対数)", fs_log_var, 0.0, _FS_LOG_MAX, 0.01,
               note="1.0〜200倍を対数目盛で調整（四肢のみ。首・体幹は変わりません）。"
                    "既定1.0倍＝何もしなければ挙動は変わりません")
        fs_value_label = tk.Label(sec_fs.body, font=("Consolas", 9), fg="#666",
                                  justify="left", text="倍率 1.00x")
        fs_value_label.pack(anchor="w", padx=14)

        def _fs_reset_default():
            fs_log_var.set(0.0)
            msg.config(text="四肢の筋力倍率を既定値(1.0倍)に戻しました", fg="#0a7")

        tk.Button(sec_fs.body, text="既定値に戻す(1.0倍)",
                  command=_fs_reset_default, width=20).pack(
            anchor="w", padx=14, pady=(2, 6))

    # ---- もがき運動パラメータ調整（2026-08-11 新設）----------------------
    #
    # 【なぜ】0ヶ月児の自発運動（もがき運動）が人間の乳児らしくない問題を追う際、
    #   1条件の測定に数分〜20分かかり非効率だった。パラメータをスライダーで
    #   動かしながら探索できるようにする（実装仕様 2026-08-11）。
    #
    # 【最重要・過去に3回踏んだ罠への対策】このファイルには「GUIの表示と
    #   実際の物理適用が別経路で、GUIを操作しても反映されない」バグが過去に
    #   3回見つかっている（監査報告 2026-08-07・2026-08-10）。ここで作る
    #   5変数は、下の `_babble_noise()` と駆動ループの `elif st_babble.get():`
    #   ブロックが**毎tick `.get()` で直接読む**（起動時に1回だけ読んで
    #   別の変数へコピーしない）。柵（st_fence）・実験者の手（st_hold_head）と
    #   同じ配線パターン。
    #
    # 初期値は、これまでハードコードされていた値と完全に一致させる
    # （std=0.174 / cocon=0.5 / syn_w=0.6 / K=10 / beta=0.7）。
    #   ＝「何も操作しなければ挙動は1ビットも変わらない」ことがこの一致で保証される。
    _BAB_DEFAULTS = dict(std=0.174, cocon=0.5, syn_w=0.6, k=10, beta=0.7)
    bab_std_var = tk.DoubleVar(value=_BAB_DEFAULTS["std"])
    bab_cocon_var = tk.DoubleVar(value=_BAB_DEFAULTS["cocon"])
    bab_synw_var = tk.DoubleVar(value=_BAB_DEFAULTS["syn_w"])
    bab_k_var = tk.IntVar(value=_BAB_DEFAULTS["k"])
    bab_beta_var = tk.DoubleVar(value=_BAB_DEFAULTS["beta"])

    sec_bab = Section(sec_run.body, "もがき運動パラメータ調整", True)
    slider(sec_bab.body, "①ゆらぎの大きさ", bab_std_var, 0.0, 0.6, 0.01,
           note="act = 共収縮の下駄 + ①×ノイズ。既定0.174")
    slider(sec_bab.body, "②共収縮の下駄", bab_cocon_var, 0.0, 1.0, 0.01,
           note="拮抗筋の同時収縮量。既定0.5")
    slider(sec_bab.body, "③シナジーの重み", bab_synw_var, 0.0, 1.0, 0.01,
           note="駆動モード「色付き+シナジー」のときだけ効く。既定0.6")
    slider(sec_bab.body, "④保持の長さK", bab_k_var, 1, 40, 1,
           note="この物理ステップ数ごとに1回、新しい行動を選び直す。既定10")
    slider(sec_bab.body, "⑤ノイズの色 beta", bab_beta_var, 0.0, 2.0, 0.01,
           note="駆動モード「色付き」「色付き+シナジー」のときだけ効く。既定0.7")

    def _bab_reset_defaults():
        bab_std_var.set(_BAB_DEFAULTS["std"])
        bab_cocon_var.set(_BAB_DEFAULTS["cocon"])
        bab_synw_var.set(_BAB_DEFAULTS["syn_w"])
        bab_k_var.set(_BAB_DEFAULTS["k"])
        bab_beta_var.set(_BAB_DEFAULTS["beta"])
        msg.config(text="もがき運動のパラメータを既定値に戻しました", fg="#0a7")

    tk.Button(sec_bab.body, text="既定値に戻す", command=_bab_reset_defaults,
              width=16).pack(anchor="w", padx=14, pady=(2, 6))

    # ---- 反射+共通駆動パラメータ調整（2026-08-12 新設）--------------------
    #
    # 【なぜ】仕様 作業記録（非公開）
    #   2026-08-12_Viewer反射共通駆動モード追加.md やること3。駆動モード
    #   「反射+共通駆動」を選んだときだけ効く（もがき運動が「①〜⑤」で
    #   white/colored/colored+synergyだけを制御するのと同じ切り分け）。
    #
    # 【対象関節数】グルーピング（none/per_limb/whole_body）に関わらず対象と
    #   なる関節の集合そのものは変わらない（グルーピングは「どうまとめるか」
    #   だけを変える）ので、ここで1回だけ実測してラベルに出す。関節名リストを
    #   手で数えない（検証の落とし穴チェックリスト項28「今日の事故＝28関節の
    #   はずが6関節」の再発防止。run.taro_setup._reflex_common_joint_indices を
    #   そのまま使う＝taro_core・run側の計算を重複させない）。
    if _ACTUATION_MODE == "muscle":
        from run.taro_setup import _reflex_common_joint_indices as _rc_joint_idx_fn
        _rc_idx_by_limb = _rc_joint_idx_fn(env)
        _rc_n_joints = sum(len(v) for v in _rc_idx_by_limb.values())
    else:
        _rc_n_joints = 0

    # k_sの既定値は run/config.py に設定キーが無く、常に
    #   spinal_cord/stretch_reflex.py の DEFAULT_K_S が使われる（仕様の
    #   注意どおり）。値を手で書き写さず import して使う。
    from spinal_cord.stretch_reflex import DEFAULT_K_S as _RC_DEFAULT_K_S

    # ρの既定値0.3とグルーピングの既定値の組み合わせに注意（仕様やること3）：
    #   grouping="none"だと各関節が1要素グループになりρが実質無効になる
    #   （RhythmicCommonDriveGroupのdocstring参照）。既定でρが目に見える効果を
    #   持つよう、grouping既定は"per_limb"を選んだ（実装担当の判断。作業記録に明記）。
    _RC_DEFAULTS = dict(rho=0.3, f0=0.5, a0=1.0, k_s=float(_RC_DEFAULT_K_S),
                        grouping="per_limb")
    rc_rho_var = tk.DoubleVar(value=_RC_DEFAULTS["rho"])
    rc_f0_var = tk.DoubleVar(value=_RC_DEFAULTS["f0"])
    rc_a0_var = tk.DoubleVar(value=_RC_DEFAULTS["a0"])
    rc_gain_var = tk.DoubleVar(value=_RC_DEFAULTS["k_s"])
    rc_grouping_var = tk.StringVar(value=_RC_DEFAULTS["grouping"])

    sec_rc = Section(sec_run.body, "反射+共通駆動パラメータ調整", True)
    if _ACTUATION_MODE != "muscle":
        tk.Label(sec_rc.body, fg="#a33", font=("", 8), justify="left",
                 text="注意actuation=muscle のときだけ使えます（今の体="
                      f"{_ACTUATION_MODE}）。ラジオボタンも選べません"
                 ).pack(anchor="w", padx=14, pady=(0, 4))
    slider(sec_rc.body, "ρ（共通の度合い）", rc_rho_var, 0.0, 1.0, 0.01,
           note="0=各関節が独立に揺れる／1=グループ内が同じ揺れを共有。既定0.3")
    slider(sec_rc.body, "振動子の周波数 f0[Hz]", rc_f0_var, 0.1, 2.0, 0.01,
           note="揺らぎの中心周波数。既定0.5Hz")
    slider(sec_rc.body, "振動子の振幅 A0", rc_a0_var, 0.0, 2.0, 0.01,
           note="揺らぎの中心振幅（無次元）。既定1.0")
    slider(sec_rc.body, "伸張反射のゲイン k_s", rc_gain_var, 0.0, 20.0, 0.1,
           note=f"筋が伸びたときの反射の強さ。既定{float(_RC_DEFAULT_K_S):g}"
                "（spinal_cord/stretch_reflex.pyのDEFAULT_K_S）")

    def _rc_invalidate(*_a):
        """グルーピングを変えたら、次に使うタイミングで組み立て直す。

        【なぜ】グルーピング（none/per_limb/whole_body）は関節indexの
        グループ分けそのもの（RhythmicCommonDriveGroupのトポロジー）を変える
        ため、rho/f0/A0/k_sのような「その場で属性を書き換えるだけ」の
        リアルタイム切替が効かない。作り直す必要があるので、いま持っている
        オブジェクトを捨てて次回 _ensure_reflex_common() で新しく作り直させる
        （過去に3回踏んだ「GUIの表示と物理適用が別経路」バグを避けるため、
        黙って古いグルーピングのまま動き続ける、を避ける）。
        """
        _reflex_common_holder[0] = None

    _rcgf = tk.Frame(sec_rc.body); _rcgf.pack(anchor="w", padx=14, pady=(2, 0))
    tk.Label(_rcgf, text="グルーピング:").pack(side="left")
    for _gv, _glab in (("none", "なし"), ("per_limb", "四肢ごと"),
                       ("whole_body", "全身")):
        tk.Radiobutton(_rcgf, text=_glab, variable=rc_grouping_var, value=_gv,
                       command=_rc_invalidate).pack(side="left")
    tk.Label(sec_rc.body, fg="#666", font=("", 8),
             text="none にすると各関節が1要素グループになり、ρは実質効かなくなります"
             ).pack(anchor="w", padx=14)

    rc_joint_label = tk.Label(
        sec_rc.body, font=("Consolas", 9), justify="left", fg="#666",
        text=f"対象関節数: {_rc_n_joints}（両腕・両脚の合計）")
    rc_joint_label.pack(anchor="w", padx=14, pady=(2, 0))

    def _rc_reset_defaults():
        rc_rho_var.set(_RC_DEFAULTS["rho"])
        rc_f0_var.set(_RC_DEFAULTS["f0"])
        rc_a0_var.set(_RC_DEFAULTS["a0"])
        rc_gain_var.set(_RC_DEFAULTS["k_s"])
        rc_grouping_var.set(_RC_DEFAULTS["grouping"])
        _rc_invalidate()
        msg.config(text="反射+共通駆動のパラメータを既定値に戻しました", fg="#0a7")

    tk.Button(sec_rc.body, text="既定値に戻す", command=_rc_reset_defaults,
              width=16).pack(anchor="w", padx=14, pady=(2, 6))

    st_loop = tk.BooleanVar(value=False)
    tk.Checkbutton(sec_run.body, text=f"{EPISODE_SEC:.0f}秒たったらやり直す",
                   variable=st_loop).pack(anchor="w", padx=14)
    run_label = tk.Label(sec_run.body, text="", font=("Consolas", 9), justify="left")
    run_label.pack(anchor="w", padx=14)

    # ---- 区画6：脳（学習したモデルで動かす）2026-07-31 新設 --------------
    # 【なぜ足したか】ユーザーの要望「すべてを編集ウィンドウ付きの Viewer に統合したい」。
    #   これまで「学習した太郎を見る」のは run/viewer.py にしかなく、
    #   編集パネルを使いながら学習後の動きを見ることができなかった。
    #   設計は E/docs/実行基盤_設計.md §7.5（第4段階）。
    # 注意：モデルを指定して起動したときは**開いた状態**にする。
    #   【なぜ、2026-07-31】既定で畳んでいたら、ユーザーが区画の存在に気づけず
    #   「チェックを入れていないので太郎が動かない」状態になった。
    #   見えないものは使えない。指定があるときは最初から見せる。
    sec_brain = Section(colR, "脳（学習したモデル）",
                        op.get("brain", bool(os.environ.get("E_VIEW_MODEL"))))
    brain_a_var = tk.StringVar(value=os.environ.get("E_VIEW_MODEL", ""))
    brain_b_var = tk.StringVar(value=os.environ.get("E_VIEW_MODEL_B", ""))
    st_brain = tk.BooleanVar(value=False)
    brain_which = tk.IntVar(value=0)          # 0=A / 1=B
    brain_std_var = tk.DoubleVar(value=float(os.environ.get("E_VIEW_STD", "0.174")))
    # 注意：駆動モードの切替UI（NOISE_MODES・noise_mode・noise_label_dirty・
    #   ラジオボタン）は2026-08-07に「再生」区画（もがき運動チェックボックスの
    #   近く）へ移設した。もがき運動側の探索ノイズも同じ noise_mode で駆動する
    #   ようになったため（仕様：もがき運動の駆動モード統一とUI移設）。

    tk.Label(sec_brain.body, justify="left", fg="#666", font=("", 8),
             text="学習したモデル(.pt)を読むと、自発運動の代わりに\n"
                  "その脳が行動を決めます。2つ入れると見比べられます。").pack(
        anchor="w", padx=14, pady=(0, 3))

    def _pick_model(var):
        """ファイル選択のダイアログ。手で打ち込むのも残す（長いパス対策）。"""
        from tkinter import filedialog
        p = filedialog.askopenfilename(
            title="学習したモデルを選ぶ", initialdir=os.path.join(_ROOT, "E", "logs"),
            filetypes=[("学習したモデル", "*.pt"), ("すべて", "*.*")])
        if p:
            var.set(os.path.relpath(p, _ROOT) if p.startswith(_ROOT) else p)

    for _lab, _var in (("脳A", brain_a_var), ("脳B（比べる用・任意）", brain_b_var)):
        _f = tk.Frame(sec_brain.body); _f.pack(fill="x", padx=14, pady=1)
        tk.Label(_f, text=_lab, width=17, anchor="w").pack(side="left")
        tk.Entry(_f, textvariable=_var, width=30).pack(side="left", fill="x", expand=True)
        tk.Button(_f, text="選ぶ", command=lambda v=_var: _pick_model(v)).pack(side="left")

    slider(sec_brain.body, "探索の揺らぎ", brain_std_var, 0.0, 0.5, 0.001,
           note="学習中の実効値≒0.174。0にすると迷いのない動きになる")
    _bf2 = tk.Frame(sec_brain.body); _bf2.pack(anchor="w", padx=14, pady=2)
    tk.Checkbutton(_bf2, text="この脳で動かす", variable=st_brain).pack(side="left")
    # 押したら必ず何か起きる。command を付けないと「切り替わったのか分からない」
    #   （2026-07-31：ユーザーの「Bのラジオを押しても表示が変わらない」）
    _switch_hook = [lambda: None]      # 中身は下（脳の準備のあと）で差し替える
    tk.Radiobutton(_bf2, text="A", variable=brain_which, value=0,
                   command=lambda: _switch_hook[0]()).pack(side="left")
    tk.Radiobutton(_bf2, text="B", variable=brain_which, value=1,
                   command=lambda: _switch_hook[0]()).pack(side="left")
    # 目標指向の探索（Goal Babbling）。run/viewer.py から移した（2026-07-31）
    #   注意：2026-07-30 の実測で**有害**と判明（おもちゃ接触−32%・persist 1000%）。
    #     さらに 2026-07-31 に「太郎の実装は Self-Prior ではなかった」と分かった
    #     （頻度分布を持たず一様ランダムに選ぶ）。作り直す予定の機能。
    #     見比べのために残す＝直したときに「前と何が違うか」を目で見るため。
    st_gb = tk.BooleanVar(value=False)
    gb_rate_var = tk.DoubleVar(value=0.5)
    tk.Checkbutton(sec_brain.body, variable=st_gb,
                   text="目標指向の探索を混ぜる（注意今の実装は有害と判明・作り直し予定）"
                   ).pack(anchor="w", padx=14)
    slider(sec_brain.body, "目標指向の割合", gb_rate_var, 0.0, 1.0, 0.05,
           note="学習ループとは違う近似。0.5＝半分の判断で過去の感覚を目標にする")
    brain_label = tk.Label(sec_brain.body, text="（まだ読み込んでいません）",
                           font=("Consolas", 9), justify="left", fg="#666")
    brain_label.pack(anchor="w", padx=14)
    # 駆動モードの表示・切替UIは「再生」区画へ移設済み（上記「注意」参照）。

    # ---- ボタン ---------------------------------------------------------
    msg = tk.Label(root, text="", fg="#0a7", font=("", 9), justify="left")
    msg.pack(pady=(6, 0))

    # 【なぜ、2026-08-07・重大バグ修正】上の修正1aで初回構築時は limb_tone を
    #   外したので、ここで一度だけ on_limb_tone() を呼び、チェックボックスの
    #   初期値（=シーンの元の設定）どおりに物理へ反映する。これにより
    #   _limb_saved[0] が「本当に脱力した状態（剛性0）」を正しく記憶できる
    #   （修正前は e_scene 側が既に適用済みの状態を誤って「元の状態」として
    #   記憶してしまい、チェックを外しても何も起きないバグの直接の原因だった）。
    # 注意：この呼び出しは `on_limb_tone()` の定義直後には置けない。
    #   `limb_tone_apply()`/`limb_tone_release()` は末尾で `msg.config(...)` を呼ぶが、
    #   `msg`（tk.Label）はこの位置（区画の組み立てが全部終わったあと）まで
    #   生成されていないため、定義直後に呼ぶと
    #   `NameError: free variable 'msg' referenced before assignment` で
    #   Viewer 起動時に必ず落ちる（実測で確認済み。scratchpad の最小再現で検証した）。
    #   msg 生成後・メインループ開始前のここなら安全。
    on_limb_tone()
    if _scene is not None:
        # 保存した状態と本当に同じ環境になったかを確かめる（止めはしない＝
        # Viewer は直すための道具なので、ずれていても開けた方が直せる）。
        # 【なぜここに移したか】上の e_scene.build() 直後ではなく、on_limb_tone()
        #   のあとで照合する（詳細はビルド直後のコメント参照）。
        e_scene.verify(_scene, env, strict=False, verbose=True)

    # ---- シーンとして保存する欄（2026-07-29 新設）------------------------
    #   ここで名前を付けて保存すると、測定スクリプトが E_SCENE=名前 で
    #   **まったく同じ状態**から始められる。
    sf = tk.Frame(root); sf.pack(fill="x", padx=12, pady=(6, 0))
    tk.Label(sf, text="シーン名", width=8, anchor="w").pack(side="left")
    tk.Entry(sf, textvariable=scene_name_var, width=30).pack(side="left", padx=(0, 6))
    tk.Label(sf, text="説明", width=4, anchor="w").pack(side="left")
    tk.Entry(sf, textvariable=note_var, width=44).pack(side="left", fill="x",
                                                       expand=True)

    bf = tk.Frame(root); bf.pack(pady=8)
    _restart = [True]

    # ========================================================================
    # シーンとして保存する（2026-07-29）
    # ------------------------------------------------------------------------
    # 【なぜ変えたか】旧 `viewer_saved.json` は**一部の項目しか保存していなかった**。
    #   実物には preset 名に「リクライニング60度」と書いてあるのに、
    #   リクライニング角そのものが入っていない。読む側が足りない値を環境変数から
    #   補うので、保存した状態と測定した状態が別物になっていた。
    # ⇒ シーンは「体・環境・実験前の設定・姿勢（qpos 丸ごと）・指紋」を全部持つ。
    # ========================================================================
    def _current_scene():
        """いまの Viewer の状態からシーンを組み立てる。"""
        import copy as _copy
        sc = (_copy.deepcopy(_scene) if _scene is not None
              else e_scene.default_scene("新しいシーン"))
        sc.pop("_path", None)
        sc["body"]["age_months"] = float(_AGE)
        sc["body"]["eye_rest_vertical_deg"] = float(_EYE_REST_V)
        sc["world"]["recline_deg"] = float(_RECLINE)
        sc["world"]["fence"] = bool(st_fence.get())
        sc["world"]["toy"]["radius"] = float(size_var.get())
        sc["world"]["toy"]["dist"] = float(dist_var.get())
        # 注意：首のバネと実験者の手は**同じ `jnt_stiffness` を使う**ので排他。
        #   頭を抑えているあいだ Viewer はスライダーを適用しない（落とし穴 項62）。
        if st_hold_head.get():
            sc["setup"]["head_hold"] = {"stiffness": float(hands.stiffness),
                                        "target_deg": (dict(_hold_tgt) if _hold_tgt
                                                       else None)}
            sc["setup"]["neck_tone"] = None
        else:
            sc["setup"]["head_hold"] = None
            sc["setup"]["neck_tone"] = ({"target_deg": float(nk_t.get()),
                                         "stiffness": float(nk_k.get())}
                                        if st_neck.get() else None)
        # 四肢の筋緊張。目標角は書かない＝**保存した姿勢が戻る先**になる
        #   （読み込み側は `apply_state` のあとに効かせるので、これで一致する）
        sc["setup"]["limb_tone"] = ({"hold_deg": float(lt_hold.get()),
                                     "groups": ["arm", "leg"]}
                                    if st_tone_limb.get() else None)
        # 【なぜ、2026-08-07】駆動モード（noise_mode）は「体そのものの設定」ではなく
        #   実験条件・観察条件なので、build() が使う setup 辞書の中には入れない
        #   （default_scene() のコメント「setup に無い項目は build() で使われない」の
        #   前提を壊さないため）。記録専用の追加キーとしてトップレベルに置く。
        #   load() 側の _merge() は base に無いキーもそのまま引き継ぐので、
        #   run/scene_tools/e_scene.py 側は一切変更しなくてよい。
        sc["noise_mode"] = str(noise_mode[0])
        return sc

    def save_scene():
        name = (scene_name_var.get() or "").strip()
        if not name:
            msg.config(text="シーンの名前を入れてください", fg="#a30")
            return
        try:
            sc = _current_scene()
            sc["note"] = (note_var.get() or "").strip()
            saved, drift = e_scene.save(
                sc, name=name, env=env,
                hands=(hands if st_hold_head.get() else None),
                settle_seconds=3.0, verbose=True)
        except Exception as e:
            msg.config(text=f"保存に失敗: {e}", fg="#a30")
            return
        ok = (drift or {}).get("ok", True)
        fp = saved.get("fingerprint") or {}
        seen = fp.get("toy_visible_left")
        parts = [f"保存しました → run/scenes/{name}.json"]
        if drift:
            parts.append(drift["summary"].replace("　", " "))
        if seen is not None:
            parts.append("おもちゃは見えています" if seen
                         else "注意おもちゃが体に隠れて見えません")
        msg.config(text="\n".join(parts), fg=("#0a7" if ok and seen else "#a30"))
        # 保存し直したときに一覧を最新にする
        try:
            _refresh_scene_menu()
        except Exception:
            pass

    def save():
        data = {"toy_pos": [float(v.get()) for v in toy_vars],
                "toy_half_size": float(size_var.get()),
                # 2026-07-28 追加。距離・環境の条件も保存する。
                #   これが無いと「保存した位置」を再現しても距離の設定が失われる。
                "toy_dist": float(dist_var.get()),
                "age": float(_AGE), "head_hold": bool(st_hold_head.get()),
                "fence": bool(st_fence.get()), "preset": scene_var.get(),
                "joints": {jd["key"]: float(jd["var"].get()) for jd in joints},
                "shake": bool(shake_var.get()), "follow": bool(follow_var.get()),
                "orient": bool(st_orient.get()),
                "latency": float(lat_var.get()), "threshold": float(thr_var.get()),
                "neck_on": bool(st_neck.get()), "neck_k": float(nk_k.get()),
                "neck_c": float(nk_c.get()), "neck_target": float(nk_t.get()),
                # 【2026-07-28 追加】スイッチの状態も残す。
                #   これが無いと、保存された位置を測るときに**別の姿勢の太郎**を
                #   測ってしまう（実際、首のバネ30度で調整した設定を
                #   「実験者が60度で支える」条件で測って、まったく違う値が出た）。
                "head_hold": bool(st_hold_head.get()),
                "hold_tilt": (float(_HOLD_TILT) if _HOLD_TILT else None),
                "neck_all_axes": bool(nk_all.get()),
                "freeze": bool(st_freeze.get()), "pose_hold": bool(st_hold.get()),
                "vor": bool(st_vor.get()),
                "eye_rest_v": float(_EYE_REST_V),
                "noise_mode": str(noise_mode[0]),
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
        for jd in joints:
            ang = np.radians(jd["var"].get())
            for jid, qadr in jd["pair"]:
                d.qpos[qadr] = ang
        d.qpos[toy_qadr:toy_qadr + 3] = [v.get() for v in toy_vars]
        d.qvel[:] = 0.0
        d.qacc[:] = 0.0
        mujoco.mj_forward(m, d)
        msg.config(text="仰向けに戻しました")

    # 「今どうなっているか」をまとめてクリップボードへ（ユーザーの提案 2026-07-27）。
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
    tk.Button(bf, text="旧形式で保存", command=save, width=11).pack(side="left", padx=3)
    tk.Button(bf, text="この状態をシーンとして保存", command=save_scene, width=24,
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
    # 【なぜ、2026-08-07】もがき運動（st_babble）の探索ノイズ生成器を、脳側の
    #   explore() と同じ CPG（taro_core/src/brain/spinal_cord/cpg.py）経由に
    #   統一する（仕様：もがき運動の駆動モード統一とUI移設）。以前はここで
    #   ColoredNoiseGenerator を固定初期化しており、Nキー/UIで選んだ noise_mode
    #   を一切見ていなかった（常に colored・beta=0.7固定）。
    # 参照：もがき運動系のツール（e_friction_probe.py 119行目・e_pose_editor.py
    #   400行目・e_reach_babble_check.py 92行目）はすべて beta=0.7, std=0.174 を
    #   使っており、その慣例はここでも変えない。syn_w はUIを持たないので脳側と
    #   同じ既定 0.6 を使う。
    from spinal_cord.cpg import CPG
    from run.taro_setup import LEG_R, LEG_L, ARM_R, ARM_L
    # pair_offset の考え方は下の _set_brain_noise_mode と同じ。このファイル自身の
    #   慣例（1306行目付近「muscle if n_act > 90」）どおり、n_act>90 を
    #   筋肉モード（拮抗筋2本展開済み空間）の判定に使う。
    _babble_pair_offset = (n_act // 2) if n_act > 90 else 0
    _babble_cpg = CPG(n_act, leg_r=LEG_R, leg_l=LEG_L, arm_r=ARM_R, arm_l=ARM_L,
                       seed=0, pair_offset=_babble_pair_offset)
    # white モード用。CPG内部の ColoredNoiseGenerator（乱数系列）と混同しないよう
    #   完全に別の乱数インスタンスにする。
    _babble_white_rng = np.random.default_rng(0)

    def _babble_noise():
        """もがき運動の1tick分の探索ノイズを、いまの駆動モード(noise_mode[0])で作る。

        colored のとき、CPG.sample(beta, synergy=False) は内部の
        ColoredNoiseGenerator の出力をそのまま返す（cpg.py 136〜139行目）ので、
        旧実装 ColoredNoiseGenerator(n_act, seed=0).sample(0.7) と
        bit-identical（検証済み。実装の作業記録参照。beta既定0.7のとき）。

        【なぜ、2026-08-11】beta・syn_w は以前 0.7・0.6 に固定されていたが、
        「もがき運動パラメータ調整」スライダー（bab_beta_var・bab_synw_var）で
        毎tick変えられるようにした。ここで`.get()`するので、GUIの現在値が
        必ず反映される（起動時に1回だけ読んで別変数に保存する、という
        過去に3回踏んだ罠は避けている）。
        """
        if noise_mode[0] == "white":
            return _babble_white_rng.standard_normal(n_act)
        return _babble_cpg.sample(
            float(bab_beta_var.get()),
            synergy=(noise_mode[0] == "colored+synergy"),
            syn_w=float(bab_synw_var.get()))

    # ================= 反射+共通駆動（2026-08-12）=========================
    # 【なぜ run.taro_setup.Taro を素直に使わないか】仕様が示す推奨アーキテクチャ
    #   （チェックポイント無しで Taro を直接組み立てる）をそのまま実装すると、
    #   `Taro.__init__` は必ず `env.reset(seed=self.seed)` を呼ぶ（乱数の
    #   再現性のための既存仕様、run/taro_setup.py 314行目）。この env は
    #   Viewer が実際に表示している env そのもの（HybridEnvはこれを包むだけ
    #   ＝内部で同じ物理envをreset）なので、これを呼ぶと**シーン・姿勢編集
    #   スライダーで作った今の姿勢が、脳を読み込んだ瞬間に既定姿勢へ飛ぶ**
    #   （実際に「脳」区画の既存の `_load_brain` にも同じ副作用があるが、
    #   そちらは「モデルを読む」という明示的な操作に対する既知の挙動として
    #   受け入れられている。もがき運動のパラメータをスライダーで調整している
    #   最中に不意に姿勢が飛ぶのは体験として別物と判断し、ここでは避けた）。
    #   そこで、taro_core側の計算式は一切重複させない（検証の落とし穴
    #   チェックリスト項30）という方針は守りつつ、env に触れない最小限の
    #   組み立て＝TaroBrainWithMotor（env非依存）＋
    #   run.taro_setup._setup_reflex_common（env読み取りのみ、reset無し）
    #   を直接使う。moment_1/moment_2・関節グループ・ρ・振動子パラメータの
    #   計算式そのものは taro_core・run.taro_setup 側のものをそのまま呼ぶだけ。
    _reflex_common_holder = [None]     # {"brain": TaroBrainWithMotor, ...} or None
    # 【なぜ、2026-08-12・監査指摘のバグ修正】run/taro_setup._setup_reflex_common
    #   （_build_reflex_common が呼ぶ）は、組み立てるたびに
    #   infant_limbs.disable_limb_tone_spring(model, groups=("arm","leg")) を呼び、
    #   四肢の関節の jnt_stiffness を問答無用で0にする。これを元に戻す経路が
    #   今まで無く、駆動モードを他へ切り替えても四肢のバネが0のまま戻らなかった
    #   （仕様 作業記録（非公開）
    #   2026-08-12_Viewer筋力スライダーと筋緊張バグ修正.md 依頼2）。
    #   既存の limb_tone_apply()/limb_tone_release()（_limb_saved、819〜906行目
    #   付近）が「書き換える前に保存し、離れるときに復元する」という全く同じ型の
    #   問題を正しく解いているので、その型をそのまま踏襲する。
    _rc_stiffness_saved = [None]       # {関節id: 元のjnt_stiffness} or None（未保存）
    _rc_pending_restore = [False]      # 復元が必要か（キーコールバックスレッドから
                                        # 直接 model 配列を書かないための旗。下記参照）

    def _build_reflex_common():
        from run.config import Config
        from run.taro_setup import _setup_reflex_common
        from taro_brain_motor import TaroBrainWithMotor
        from infant_limbs import limb_tone_joints_by_group
        osc_params = {"f0": float(rc_f0_var.get()), "A0": float(rc_a0_var.get()),
                      "tau_f": 2.0, "tau_A": 2.0, "amp": 1.0}
        taro_spec = {
            "actuation": _ACTUATION_MODE, "age_months": _AGE,
            "spinal_drive_mode": "reflex_common",
            "common_drive_rho": float(rc_rho_var.get()),
            "common_drive_grouping": rc_grouping_var.get(),
            "common_drive_osc_params": osc_params,
            "noise": "white",   # spinal_drive_mode=reflex_common はnoise=coloredと
                                 # 同時指定不可（run/config.py のバリデーション）
        }
        cfg = Config(taro_spec, {"seed": 0, "K": 10},
                    scene=(_scene or {}).get("name"), name="viewer_reflex_common")

        class _ReflexCommonHolder:
            pass
        holder = _ReflexCommonHolder()
        # sensory_dim/proprio_dimは反射+共通駆動では一切使わない（motor_cortex・
        #   forward_model_headは呼ばれない）ので、無駄な層を作らないよう最小値にする。
        holder.brain = TaroBrainWithMotor(vocab_size=3, sensory_dim=1,
                                          n_actuators=n_act, proprio_dim=1)
        holder.seed = 0
        # 【なぜここで保存するか】_setup_reflex_common が disable_limb_tone_spring を
        #   呼ぶ**直前**の値を控える。_rc_stiffness_saved[0] が None のとき
        #   （＝このセッションでまだ一度も保存していないとき）だけ保存する。
        #   グルーピング変更（_rc_invalidate）でholderだけ作り直す2回目以降は
        #   None のままではないのでここをスキップし、「既に0になっているものを
        #   保存して0のまま復元する」を避ける（仕様やること2の注意点）。
        if _rc_stiffness_saved[0] is None:
            _pairs = dict(limb_tone_joints_by_group(m, groups=("arm", "leg")))
            _sv = {}
            for j in range(m.njnt):
                nm = (m.joint(j).name or "").split(":")[-1]
                if nm in _pairs:
                    _sv[j] = float(m.jnt_stiffness[j])
            _rc_stiffness_saved[0] = _sv
        ok = _setup_reflex_common(holder, cfg, env, verbose=True)
        if not ok:
            return None
        return holder

    def _rc_restore_stiffness():
        """disable_limb_tone_spring が0にした jnt_stiffness を、保存しておいた
        値へ戻す。反射+共通駆動モードから離れるときに呼ぶ（メインループ側から
        だけ呼ぶこと。model配列への直接書き込みを含むため）。
        """
        sv = _rc_stiffness_saved[0]
        if sv:
            for j, k in sv.items():
                m.jnt_stiffness[j] = k
        _rc_stiffness_saved[0] = None
        # 次に反射+共通駆動へ戻ったとき disable_limb_tone_spring を確実に
        #   再度掛け直させるため、組み立て済みのholderも捨てる。捨てないと
        #   _ensure_reflex_common が古いholderを再利用し、stiffness=0が
        #   復元されたまま（＝バネが効いた状態で反射+共通駆動が動く）になる。
        _reflex_common_holder[0] = None

    def _ensure_reflex_common():
        """必要になった時に作る（もがき運動＋反射+共通駆動を初めて選んだ時、
        またはグルーピングを変えて無効化された直後）。"""
        if _ACTUATION_MODE != "muscle":
            return None
        if _reflex_common_holder[0] is None:
            _reflex_common_holder[0] = _build_reflex_common()
        return _reflex_common_holder[0]

    def _apply_reflex_common_params(holder):
        """rho/f0/A0/k_s のスライダーの現在値を、既存オブジェクトへその場で
        書き込む（作り直さない）。グルーピングだけは _rc_invalidate() 経由で
        作り直す（トポロジーが変わるため属性の書き換えでは対応できない）。
        """
        if holder is None:
            return
        holder.brain.reflex_common.k_s = float(rc_gain_var.get())
        _f0, _a0, _rho = float(rc_f0_var.get()), float(rc_a0_var.get()), float(rc_rho_var.get())
        for _idxs, _group in holder.brain._common_drive_groups.values():
            _group.rho = _rho
            for _osc in [_group.common] + list(_group.indep.values()):
                _osc.f0, _osc.A0 = _f0, _a0

    act = [zero.copy()]

    # ================= 脳（学習したモデル）2026-07-31 =====================
    # 【なぜここか】メインループの直前。太郎一式（脳・小脳・神経調節）は
    #   `run/taro_setup.Taro` が組み立てるので、ここでは**呼ぶだけ**にする。
    # 注意：内受容感覚（空腹・眠気・不快・覚醒）が観測に無いと脳の入力次元が合わない。
    #   ⇒ HybridEnv で包む。e_viewer は `env.step()` の戻り値を使っていないので、
    #     包んでも既存の編集機能は壊れない（2026-07-31 に確認）。
    brains = {}          # {"A": (taro, hidden, prev_a), "B": ...}
    hybrid_env = [None]
    brain_da2 = []       # 行動の変化量（学習ログの da2 と同じ量）
    gbuf = []            # 目標指向の探索が使う「過去に経験した感覚」
    gb_stat = [0, 0]     # [目標指向にした回数, 探索のままにした回数]
    last_obs = [None]    # いちばん新しい観測（脳を切り替えたときに渡し直す）
    babble_da2 = []      # もがき運動の変化量（学習済みと見比べるため）

    def _ensure_hybrid():
        """内臓つきの環境（HybridEnv）を必ず用意する。

        【なぜ、2026-07-31】以前はこれを `_load_brain` の中でだけ作っていた。
        ところが `_load_brain` は「10物理ステップに1回」の判断のタイミングでしか
        呼ばれないのに、`hybrid_env[0].step()` は**毎フレーム**呼ばれる。
        チェックを入れた瞬間が10の倍数でないと
        `AttributeError: 'NoneType' object has no attribute 'step'` で Viewer が落ちた。
        ⇒ 使う側から必ず通るようにした。
        """
        if hybrid_env[0] is None:
            from hybrid_env import HybridEnv
            hybrid_env[0] = HybridEnv(env)
            hybrid_env[0].reset(seed=0)
        return hybrid_env[0]

    def _load_brain(tag, path):
        """モデルを読んで太郎一式を作る。失敗しても Viewer は落とさない。"""
        if not path:
            return None
        p = path if os.path.isabs(path) else os.path.join(_ROOT, path)
        if not os.path.exists(p):
            brain_label.config(text=f"注意見つかりません: {path}", fg="#a33")
            return None
        try:
            if _ROOT not in sys.path:
                sys.path.insert(0, _ROOT)
            from run.config import Config, touch_setting_of
            from run.taro_setup import Taro
            _ensure_hybrid()
            # 【なぜ、2026-08-10】以前は n_act（今の環境の行動次元）から
            #   "muscle 90超/joint" を逆算していたが、この環境は常に筋肉モードで
            #   作られていたため n_act は常に180超＝常に"muscle"と誤判定していた
            #   （監査：作業記録（非公開）
            #   2026-08-10_run系システムとViewerの型バグ横断監査.md「中2」）。
            #   実際に環境を組み立てたときの駆動モード（_ACTUATION_MODE、上で
            #   E_ACTUATION から決定済み）をそのまま使う。これで少なくとも
            #   「今の環境」と「今から作る脳の設計」は必ず一致する。
            # 【なぜ、2026-08-11】以前は taro_spec をここで空の辞書から個別に
            #   組み立てていたため、run/config.py の TARO_DEFAULTS へ新しい
            #   設定キーが追加されるたびに、ここへの転送が個別に漏れていた
            #   （2026-08-07のtaro.noise、2026-08-11の伸張反射＋揺らぐ振動子の
            #   共通駆動＝spinal_drive_mode等4キーで再発。仕様
            #   作業記録（非公開）
            #   担当B節）。run/main.py の edit分岐が実験ファイルの taro欄を
            #   丸ごとJSON化して E_TARO_SPEC_JSON に渡すようにしたので、
            #   まずこれを「土台」として展開する。そのうえで、実際に組み立てた
            #   環境（_ACTUATION_MODE・_AGE・このモデルのパスp）を使う個別計算を
            #   従来どおり上から適用する＝個別計算が必ず勝つ（下の
            #   touch_setting_of・noise/beta/synergy/syn_w の上書きも含め、
            #   優先順位は変えていない）。
            #   注意：E_TARO_SPEC_JSON が無い場合（run/main.py を通さず環境変数
            #   だけで e_viewer.py を直接起動した従来の使い方）は空辞書のままで、
            #   挙動は変更前と同じになる。
            taro_spec = {}
            _spec_json = os.environ.get("E_TARO_SPEC_JSON")
            if _spec_json:
                try:
                    _loaded_spec = json.loads(_spec_json)
                    if isinstance(_loaded_spec, dict):
                        taro_spec.update(_loaded_spec)
                    else:
                        print(f"注意[脳] E_TARO_SPEC_JSON が辞書ではありません"
                              f"（{type(_loaded_spec).__name__}）。無視します", flush=True)
                except (json.JSONDecodeError, TypeError) as _e:      # noqa: BLE001
                    print(f"注意[脳] E_TARO_SPEC_JSON を読めません: "
                          f"{type(_e).__name__}: {_e}。無視して続けます", flush=True)
            taro_spec.update({"actuation": _ACTUATION_MODE,
                              "age_months": _AGE, "model": p})
            # 【なぜ、2026-08-10・さらに確認】上記だけでは「今の環境と脳の設計」は
            #   一致しても、「このチェックポイント自身が実際にどちらの駆動モードで
            #   学習されたか」までは保証できない。学習側（run/taro_setup.py
            #   Taro.save()）は Config の全項目（actuation を含む）を
            #   blob["config"] に保存しているので、touch_setting_of() と同じ
            #   「モデル自身に聞く」考え方でここも確認する。食い違っていたら
            #   黙って読まず、明示的なエラーを出して中止する（Taro._load は
            #   strict=False なので、形の合わない層は例外を出さずに白紙のまま
            #   読み込まれてしまうため）。
            import torch as _torch
            try:
                _blob = _torch.load(p, map_location="cpu", weights_only=False)
                _ckpt_cfg = _blob.get("config") or {}
                _ckpt_act = _ckpt_cfg.get("actuation")
            except Exception as _e:      # noqa: BLE001
                _ckpt_act = None
                print(f"注意[脳] 保存された駆動モードを読み取れません: "
                      f"{type(_e).__name__}: {_e}。この確認はスキップします",
                      flush=True)
            if _ckpt_act is None:
                # 古い形式（config が無い、または actuation キーが無い）保存ファイル。
                #   食い違いの検出はできないが、読み込みそのものは止めない
                #   （touch_setting_of も同様に「読めなければ既定扱い」の方針）。
                print("注意[脳] このモデルの保存データに駆動モードの記録がありません"
                      "（古い形式の可能性）。食い違いの確認をスキップして読み込みます",
                      flush=True)
                # 【なぜ、2026-08-10】上のprintだけだとGUI上には何も出ず、
                #   コンソールを見ないユーザーには気づかれない（2026-07-31付の
                #   別コメントと同じ轍。GUIツールでコンソール出力は見落とされる）。
                #   この分岐は読み込みを継続して最終的に成功するため、直後に
                #   _update_brain_label() が brain_label を上書きしてしまい、
                #   そちらに書いても一瞬で消える。msg は成功後も上書きされない
                #   共通のメッセージ欄なので、こちらに警告を残す。
                msg.config(text="注意このモデルの保存データに駆動モードの記録が"
                                 "ありません（古い形式の可能性）。食い違いの確認を"
                                 "スキップして読み込みました",
                           fg="#c60")
            elif str(_ckpt_act).lower() != _ACTUATION_MODE:
                _msg = (f"駆動モードが食い違います（今の体={_ACTUATION_MODE} / "
                        f"このモデルの学習時={_ckpt_act}）。層の形が合わないため"
                        "読み込みを中止します")
                brain_label.config(text=f"注意{_msg}", fg="#a33")
                print(f"注意[脳] {_msg}", flush=True)
                return None
            taro_spec.update(touch_setting_of(p))
            # 【なぜ、2026-08-07】いま選ばれている駆動モード(noise_mode[0])を
            #   毎回 taro_spec に反映する。white のときは TARO_DEFAULTS の
            #   既定値と完全に同じ値を明示するだけなので、Config の挙動は
            #   変更前と bit-identical（run/config.py 272〜273行目、
            #   キーが無ければ既定値を使う仕組みと同じ結果になる）。
            #   beta/syn_w はUIを持たないので TARO_DEFAULTS の既定値を常に使う
            #   （0.8/0.6、[Tier3]・変更していない）。
            _cfg_noise = ("colored" if noise_mode[0] in ("colored", "colored+synergy")
                          else "white")
            taro_spec.update({
                "noise": _cfg_noise,
                "beta": 0.8,
                "synergy": (noise_mode[0] == "colored+synergy"),
                "syn_w": 0.6,
            })
            cfg = Config(taro_spec, {"seed": 0, "K": 10},
                         scene=(_scene or {}).get("name"), name="viewer")
            t = Taro(cfg, hybrid_env[0], seed=0, verbose=True)
            st = t.init_state(t.first_obs)
            return {"taro": t, "state": st}
        except Exception as e:      # noqa: BLE001
            brain_label.config(text=f"注意読めません: {type(e).__name__}: {e}", fg="#a33")
            print(f"注意[脳] 読めません: {type(e).__name__}: {e}", flush=True)
            return None

    def _ensure_brains():
        """チェックを入れた時に読む（起動を遅くしないため後回しにする）。"""
        want = {"A": brain_a_var.get().strip(), "B": brain_b_var.get().strip()}
        for tag, path in want.items():
            if path and brains.get(tag, {}).get("path") != path:
                got = _load_brain(tag, path)
                brains[tag] = ({"path": path, **got} if got else {"path": path})
            if not path:
                brains.pop(tag, None)
        _update_brain_label()
        return [k for k in ("A", "B") if brains.get(k, {}).get("taro")]

    def _update_brain_label():
        """いまどちらの脳を使っているかを、押した瞬間に画面へ出す。"""
        ok = [k for k in ("A", "B") if brains.get(k, {}).get("taro")]
        now = "B" if brain_which.get() else "A"
        if not ok:
            brain_label.config(text="（まだ読み込んでいません）", fg="#666")
        elif now in ok:
            brain_label.config(
                text=f"いま {now} で動かしています（読み込み済み: {', '.join(ok)}）",
                fg="#0a7")
        else:
            brain_label.config(
                text=f"注意{now} は読み込めていません（読み込み済み: {', '.join(ok)}）",
                fg="#a33")

    def _on_brain_switch():
        """A ⇔ B を切り替えたとき。

        注意：切り替えた側の脳に「いまの観測」を渡し直す。
          【なぜ、2026-07-31】観測は**選んでいる方だけ**更新していたので、
          切り替えると相手は「前に選ばれていたときの観測」から再開してしまう。
          体は動いているのに脳だけ過去を見ている状態になる。
        """
        brain_da2.clear()          # 前の脳の値が混ざらないように捨てる
        gb_stat[0] = gb_stat[1] = 0
        tag = "B" if brain_which.get() else "A"
        if not (brains.get(tag) or {}).get("taro"):
            _ensure_brains()       # まだ読んでいなければここで読む
        b = brains.get(tag) or {}
        if b.get("state") is not None and last_obs[0] is not None:
            b["state"]["obs"] = last_obs[0]
        _update_brain_label()

    _switch_hook[0] = _on_brain_switch

    def _brain_action():
        """いま選んでいる脳に、次の行動を決めてもらう。"""
        tag = "B" if brain_which.get() else "A"
        b = brains.get(tag) or {}
        t, stt = b.get("taro"), b.get("state")
        if not t or stt is None:
            return None
        import torch
        # 注意：`torch.no_grad()` で囲んではいけない。
        #   太郎の潜在推論（予測符号化）は**中で torch.autograd.grad を使う**ので、
        #   勾配を切ると「element 0 of tensors does not require grad」で落ちる。
        #   （2026-07-31 に単体テストで踏んだ。run/viewer.py も no_grad を使っていない）
        #   ⇒ 代わりに各段で detach して、計算グラフが伸び続けないようにする。
        sv = t.fusion.encode(stt["obs"])
        cf = t.target_fusion.encode(stt["obs"]).detach()
        z, _kl, _rc, hn = t.infer_latent(sv, stt["prev_a"], cf, stt["hidden"])
        z = z.detach()
        mean = t.act_mean(z)
        # 目標指向の探索。注意経験のバッファは**常に**溜める。
        #   （ONのときだけ溜める実装にしたら、切り替えるたび64件たまる前にOFFになり
        #     一度も発動しなかった。2026-07-30 のユーザー報告「g押してもなんも変わってない」）
        clp = t.encode_target(stt["obs"]).detach()
        gbuf.append(clp)
        if len(gbuf) > 2000:
            gbuf.pop(0)
        if st_gb.get() and len(gbuf) >= 64:
            if torch.rand(1).item() < float(gb_rate_var.get()):
                g = gbuf[torch.randint(len(gbuf), (1,)).item()]
                mean = t.infer_goal_action(z, clp, mean, g)
                gb_stat[0] += 1
            else:
                gb_stat[1] += 1
        std = float(brain_std_var.get())
        if std > 0:
            a, _lp = t.brain.explore(mean, torch.full_like(mean, std))
            a = a.detach()
        else:
            a = torch.clamp(mean, -1.0, 1.0).detach()
        brain_da2.append(float(((a - stt["prev_a"]) ** 2).mean()))
        if len(brain_da2) > 200:
            brain_da2.pop(0)
        stt["hidden"], stt["prev_a"] = hn.detach(), a
        a_env = t.brain.to_env_action(a)      # 拮抗筋モードなら筋活性化へ写す
        return a_env

    def _set_brain_noise_mode(t, mode):
        """1つの脳(t)に対して、駆動モードをその場で切り替える。

        仕様の設計方針そのまま（run/taro_setup.py 174〜181行目と同じ式）。
        taro_core・run/taro_setup.py 自体は変更せず、公開の状態
        （t.brain.spinal_cpg・enable_spinal_babble・_babble_synergy）を使うだけ。
        """
        if mode == "white":
            # TaroBrainWithMotor.__init__ の初期状態と同じ代入。
            #   次のtickから explore() は白色ガウス分岐に戻る。
            t.brain.spinal_cpg = None
            return
        if t.brain.spinal_cpg is None:
            # white → colored（またはcolored+synergy）。まだCPGが無いので新規に作る。
            from run.taro_setup import LEG_R, LEG_L, ARM_R, ARM_L
            pair_offset = ((t.n_act // 2)
                           if (t.cfg.is_muscle and not t.cfg.antagonist) else 0)
            t.brain.enable_spinal_babble(
                t.n_act, leg_r=LEG_R, leg_l=LEG_L, arm_r=ARM_R, arm_l=ARM_L,
                beta=t.cfg.beta, synergy=(mode == "colored+synergy"),
                syn_w=t.cfg.syn_w, seed=t.seed,
                antagonist=(t.cfg.is_muscle and t.cfg.antagonist),
                co_activation=t.cfg.coactivation, pair_offset=pair_offset)
        else:
            # 既にCPGがある（colored ⇔ colored+synergy の切替）。
            #   同じインスタンスのバッファ・読み出し位置を保つため、
            #   synergyフラグだけ書き換える（作り直さない）。
            t.brain._babble_synergy = (mode == "colored+synergy")

    def _apply_noise_mode_all():
        """現在ロード済みの全ての脳（A・B）に、いまの駆動モードを適用する。

        【なぜ全部か】片方だけ変えると「AとBの違いを見ているつもりが、実は
        ノイズモードの違いを見ていた」という交絡が起きるため（仕様2節）。
        """
        for _tag in ("A", "B"):
            _t = (brains.get(_tag) or {}).get("taro")
            if _t is not None:
                _set_brain_noise_mode(_t, noise_mode[0])

    def _set_noise_mode(new_mode):
        """駆動モードを new_mode に切り替える（Nキー・ラジオボタン共通の実体）。

        【なぜ、2026-08-07】もがき運動（_babble_noise）は毎tick noise_mode[0] を
        直接参照するので、ここで明示的に何かする必要はない。脳（A/B）だけ
        _apply_noise_mode_all() で明示的に反映する。
        注意：この関数は key_callback（tkinterのメインスレッドとは別スレッドの
        可能性がある、1477行目付近の注意参照）とラジオボタンの command
        （tkinterのメインスレッド）の両方から呼ばれる。単純なPython属性・
        リストの書き換えだけにし、tk.StringVar.set() 等の tkinter API を
        直接呼ばない（画面・ラジオボタンの同期はメインループ側の
        noise_label_dirty 処理でまとめて行う）。
        """
        if new_mode not in NOISE_MODES:
            return
        # 【やること2・actuation=muscle必須のガードの防御的な第二段】
        #   ラジオボタン側は既にdisabledにしているが（第一段）、Nキー循環など
        #   別経路から呼ばれても黙って別モードで動く、を避けるためここでも弾く。
        #   ここは key_callback スレッドから呼ばれる可能性があるため、msg.config
        #   のような tkinter API は呼ばず、print だけにする（このファイルの
        #   既存の方針どおり）。
        if new_mode == "reflex_common" and _ACTUATION_MODE != "muscle":
            print(f"注意[脳] 反射+共通駆動は actuation=muscle のときだけ選べます"
                  f"（今の体={_ACTUATION_MODE}）。切り替えません", flush=True)
            return
        # 【なぜ、2026-08-12・監査指摘のバグ修正】反射+共通駆動から離れるとき、
        #   disable_limb_tone_spring が0にした jnt_stiffness を戻す必要がある。
        #   この関数は key_callback スレッドから呼ばれる可能性があるため、
        #   ここでは model 配列（m.jnt_stiffness）を直接書き換えず、旗を立てる
        #   だけにする（noise_label_dirty と同じ「旗を立ててメインループ側で
        #   処理する」パターン）。実際の復元は _rc_restore_stiffness() が
        #   メインループ側で行う。
        if noise_mode[0] == "reflex_common" and new_mode != "reflex_common":
            _rc_pending_restore[0] = True
        noise_mode[0] = new_mode
        _apply_noise_mode_all()
        noise_label_dirty[0] = True
        print(f"[脳] 駆動モードを切り替えました: {noise_mode[0]}", flush=True)

    _noise_ui_hook[0] = _set_noise_mode      # ラジオボタン側の空フックを実体に差し替える

    def _cycle_noise_mode():
        """N キーで white → colored → colored+synergy → (反射+共通駆動) → white
        … と循環させる。actuation=jointのときは反射+共通駆動を飛ばす。"""
        _avail = [m for m in NOISE_MODES
                  if m != "reflex_common" or _ACTUATION_MODE == "muscle"]
        _i = _avail.index(noise_mode[0]) if noise_mode[0] in _avail else -1
        _set_noise_mode(_avail[(_i + 1) % len(_avail)])

    def _key_cb(keycode):
        # 注意：【2026-08-07】この関数は mujoco.viewer.launch_passive の
        #   key_callback として渡され、MuJoCo側のイベント処理スレッド（メイン
        #   ループ＝tkinterを更新しているスレッドとは別）から呼ばれる可能性がある。
        #   tkinterのウィジェットをここで直接触るのは安全でないと判断し、
        #   noise_label_dirty フラグだけ立てて、実際の tk.Label.config はメイン
        #   ループ側（win.update() を呼んでいる箇所と同じスレッド）で行う。
        #   noise_mode[0]・brains[tag]["taro"].brain の書き換え自体は、
        #   taro_core/tools/motor_viewer.py の speed_ref 書き換えと同じパターン
        #   （単純なPython属性の代入）なので、ここで直接行う。
        try:
            ch = chr(keycode)
        except ValueError:
            return
        if ch.upper() == "N":
            _cycle_noise_mode()

    head_w, devs, seens, neck_hist = [], [], [], []
    prev_sacc = [0]
    fire_until = [-1.0]

    with mujoco.viewer.launch_passive(m, d, key_callback=_key_cb) as viewer:
        # 2026-07-27：初期カメラを太郎の顔に寄せる。
        #   これが無いと MuJoCo の既定カメラ（シーン全体を引きで映す）になり、
        #   柵の外から見下ろす画になる。顔の前にある直径2cmのおもちゃは
        #   小さすぎて柵に隠れ、**「おもちゃが出てこない」ように見えた**
        #   （ユーザーの目視 2026-07-27。実際には正しい位置にあった）。
        try:
            # 真上寄りから見下ろす。仰向けの太郎と、顔の前のおもちゃが
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
                # 【2026-07-29】シーン方式。選んだシーンが今と違うなら、
                #   体そのものを作り直す必要がある（geomの寸法・質量・リクライニング角は
                #   モデル構築時に決まるので実行中には変えられない）。
                #   ⇒ E_SCENE を設定して自分を起動し直す。
                #   旧版は条件を1つずつ環境変数に詰め直していたため、詰め忘れると
                #     「別の条件で立ち上がる」ことが起きていた。シーン名1つで済む。
                _pick = scene_var.get()
                _need_rebuild = bool(_pick) and (_scene is None
                                                 or _pick != _scene.get("name"))
                if _need_rebuild:
                    msg.config(text=f"シーン「{_pick}」で体を作り直します。"
                                    "新しい窓が開いたら、この窓は閉じます")
                    win.update_idletasks()
                    win.after(0, lambda: None)
                    _envv = dict(os.environ)
                    _envv["E_SCENE"] = _pick
                    # 注意：シーンが全部を決めるので、古い個別指定は消しておく
                    #   （残っていると「シーンの値と環境変数のどちらが効くのか」が
                    #     曖昧になり、散らばりが復活する）。
                    for _k in ("E_AGE", "E_HEAD_HOLD", "E_FENCE", "E_TOY_RADIUS",
                               "E_RECLINE", "E_EYE_REST_V", "E_HOLD_TILT",
                               "E_TOY_POS", "E_SEAT_FRICTION", "E_TOY_MODE",
                               "E_TOY_SHAPE", "E_TOY_DIST"):
                        _envv.pop(_k, None)
                    # 【2026-07-28 修正】`os.execve` をやめた。
                    #   Windows には本当の exec が無く、Python は「新プロセスを作って
                    #   自分は終了する」動作になる。その結果、新プロセスが親のコンソールを
                    #   失って管理から外れ、**見た目にはアプリが消えた**ように見えた
                    #   （ユーザーの報告「最初からやり直すを押すとアプリが消える」）。
                    #   → 独立したプロセスとして起動し、自分は普通に閉じる。
                    import subprocess
                    _flags = 0
                    if os.name == "nt":
                        # 新しいコンソールを開く＝完全に独立したプロセスになる。
                        # 注意：DETACHED_PROCESS だとログが見えず、失敗しても気づけない
                        _flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
                    try:
                        subprocess.Popen(
                            [sys.executable, os.path.abspath(__file__)],
                            env=_envv, cwd=os.getcwd(), creationflags=_flags)
                        msg.config(text="新しい条件で立ち上げています。"
                                        "別の窓が開くまで少し待ってください")
                        win.update_idletasks()
                    except Exception as _e:
                        msg.config(text=f"注意起動に失敗しました: {_e}")
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

                # 【2026-07-29 修正・重大】シーンから起動したときは
                #   `env.reset()` **だけ**では駄目。リセットは既定の姿勢に戻すので、
                #   シーンの姿勢もおもちゃの位置も消える。
                #   `_restart` の初期値は True なので**起動直後に必ずここを通る**＝
                #   シーンで立ち上げても最初のフレームで既定状態に戻っていた
                #   （ユーザーの報告「おもちゃの位置がまた元に戻ってる」2026-07-29）。
                if _scene is not None:
                    e_scene.reset_to_scene(env, _scene,
                                           hands=(hands if st_hold_head.get() else None),
                                           seed=0)
                    if not st_hold_head.get():
                        hands.release()
                    # 【なぜ、2026-08-07・重大バグ修正】reset_to_scene() は内部で
                    #   e_scene._apply_limb_tone() を無条件に呼び直す（「姿勢を戻した
                    #   ので目標角も取り直す」という設計自体は正しい）。ただし
                    #   チェックボックスの状態を見ないため、OFFにしていても
                    #   毎回ONに戻ってしまっていた。on_limb_tone() を呼び直し、
                    #   現在のチェックボックスの状態で明示的に上書きする
                    #   （ONなら新しい姿勢を目標に取り直す＝従来の意図どおり、
                    #   OFFなら reset_to_scene が入れたバネを明示的に切る）。
                    on_limb_tone()
                    # スライダーもシーンの位置に合わせる（食い違いを残さない）
                    toy_pos0 = np.array(d.qpos[toy_qadr:toy_qadr + 3], dtype=float)
                else:
                    env.reset(seed=0)
                    # やり直しのたびに支え直す（目標角は「支え始めた時点の角度」なので、
                    #   リセット後の姿勢で取り直す必要がある）
                    if st_hold_head.get():
                        hands.hold(target=_hold_tgt)
                    else:
                        hands.release()
                apply_fence(st_fence.get())
                reflex.reset()
                # 【やること・アーキテクチャ6節】反射+共通駆動が既に組み立て済みなら、
                #   基準長L0をリセット後の姿勢に合わせ直す。呼び直さないと、
                #   リセット前の姿勢を基準に反射が働き続け、不自然な動きになる。
                if _reflex_common_holder[0] is not None:
                    _am_reset = env.unwrapped.actuation_model
                    _reflex_common_holder[0].brain.set_reflex_common_baseline(
                        _am_reset.muscle_lengths)
                t_sim, wall0, tick = 0.0, time.time(), 0
                toy_base[0] = None
                head_w.clear(); devs.clear(); seens.clear(); neck_hist.clear()
                parent_log.clear(); _parent[0] = None
                # おもちゃも初期状態に戻す。戻さないと「顔の前へ持っていく」で
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
            # 物理を止めていると反射の指令が実行されない（step の中にあるため）
            if freeze and st_orient.get() and tick % 60 == 0:
                msg.config(text="注意物理を止めています。反射は目を動かせません"
                                "（『物理を止める』を外してください）")
            OR.SACCADE_LATENCY = float(lat_var.get())
            OR.SACCADE_MIN_STRENGTH = float(thr_var.get())
            u._orienting = reflex if st_orient.get() else None
            u._vor = vor if st_vor.get() else None

            # 【2026-07-28 修正】頭を抑えているあいだは、このスライダーを**適用しない**。
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

            # 反射+共通駆動から離れたときの jnt_stiffness 復元（2026-08-12 新設）。
            #   _set_noise_mode() が立てた旗をここ（メインループ、物理を進める
            #   スレッドと同じ）で消費する。key_callback スレッドから直接
            #   model 配列を書かないための仕組み（上のコメント参照）。
            if _rc_pending_restore[0]:
                _rc_restore_stiffness()
                _rc_pending_restore[0] = False

            # 四肢の筋力スライダー（2026-08-12 新設）。毎tick .get() で読み、
            #   「基準値×スライダーの現在値」を毎回計算し直して代入する
            #   （既存の scale_actuator_strength のような累積型は使わない。
            #   累積すると動かすたびに値が積み上がる。仕様の必須条件）。
            #   これにより、スライダーを1.0へ戻せば _fmax_base とビット同一に戻る
            #   （10.0**0.0 == 1.0 は浮動小数で厳密に一致するため）。
            if _fs_available:
                _fs_x = _fs_current_x()
                _fnew = _fmax_base.copy()
                for _aid in _LIMB_AIDS:
                    _fnew[_aid] *= _fs_x
                    _fnew[_aid + _fs_n_act] *= _fs_x
                am.set_fmax(_fnew)
                if tick % 15 == 0:
                    fs_value_label.config(
                        text=f"倍率 {_fs_x:.2f}x  "
                             f"実効fmax(四肢の平均) {float(np.mean(_fnew[_LIMB_AIDS])):.3f}"
                             f"（基準 {float(np.mean(_fmax_base[_LIMB_AIDS])):.3f}）")

            # 環境のスイッチ（柵・実験者の手）を実行中でも反映する
            if bool(st_fence.get()) != fence_on[0]:
                apply_fence(st_fence.get())
            if bool(st_hold_head.get()) != bool(hands.holding):
                if st_hold_head.get():
                    hands.hold(target=_hold_tgt)
                else:
                    hands.release()

            # おもちゃ。揺らしは「固定するか」と独立に効かせる。
            #   固定ON  … スライダーの位置＋揺れ
            #   固定OFF … 環境が置いた位置（視線の正面）＋揺れ
            wob = np.array([0.0, SHAKE_AMP * np.sin(2 * np.pi * SHAKE_HZ * t_sim), 0.0]) \
                if shake_var.get() else np.zeros(3)

            # 親が顔の前へ運んでいる最中（ボタンで発動）。終わるとスライダーに引き継ぐ
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
            # 注意：【2026-07-28 修正】`freeze`（物理を止める）でスライダーの角度を
            #   書き戻していたため、**止めた瞬間に姿勢が作りかけの値へ飛んでいた**
            #   （ユーザーの要望「現状維持のまま物理を止めてほしい」）。
            #   スライダーで姿勢を作りたいときは「姿勢を固定」（st_hold）を使う。
            #   止めるだけなら今の姿勢をそのまま保つ。
            if st_hold.get():
                _neck_tgt = {}
                for jd in joints:
                    ang = np.radians(jd["var"].get())
                    for jid, qadr in jd["pair"]:
                        d.qpos[qadr] = ang
                        d.qvel[int(m.jnt_dofadr[jid])] = 0.0
                    if jd.get("is_neck"):
                        _neck_tgt[jd["key"]] = float(jd["var"].get())
                # 首は実験者の手（バネ）が支えているので、**支える目標角も**
                #   一緒に動かす。これをしないと、スライダーで首を曲げた瞬間に
                #   バネが元の角度へ引き戻し、姿勢が作れない（強さ200N·m/rad）。
                if _neck_tgt and st_hold_head.get() and hands.holding:
                    hands.hold(target=_neck_tgt)
                    _hold_tgt = dict(_neck_tgt)

            # 物理を止めているあいだは step() が呼ばれず、おもちゃを運ぶ処理も
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
                # 注意：体の根元（位置・向き）も戻さない。止めた時点の姿勢を保つ。
                #   `root_qpos0` はリセット直後の値なので、書き戻すと
                #   リクライニングで座っていた体が仰向けの初期位置へ飛ぶ。
                #   「仰向けに戻す」ボタンを使いたいときだけ明示的に戻す。
                d.qvel[:] = 0.0
                d.qacc[:] = 0.0
                mujoco.mj_forward(m, d)
            else:
                # 学習した脳が優先（チェックが入っていれば自発運動より上）
                if st_brain.get():
                    if tick % 10 == 0:          # 1判断＝10物理ステップ（K=10）
                        # 注意：「読んだ結果が空」でも brains には印が残るので、
                        #   `not brains` だけだと**パスを直しても読み直さない**。
                        #   いま選んでいる側の脳ができているかで判断する。
                        _tag = "B" if brain_which.get() else "A"
                        if not (brains.get(_tag) or {}).get("taro"):
                            _ensure_brains()
                        a_env = _brain_action()
                        # 注意：読めていないのに黙って動かないのを止める。
                        #   （2026-07-31：チェックを入れても何も起きず、
                        #     画面にも理由が出ないので原因が分からなかった）
                        if a_env is None and tick % 500 == 0:
                            tag = "B" if brain_which.get() else "A"
                            msg.config(
                                text=f"注意脳{tag} が読めていないので動きません。"
                                     f"パスを確かめてください")
                        if a_env is not None:
                            from run.taro_setup import rescale_action
                            act[0] = rescale_action(
                                a_env, _ensure_hybrid().action_space).astype(np.float32)
                    # 注意：HybridEnv 側で進める（内臓の時間も進める）。
                    #   生の env を進めると内受容感覚が止まったままになる。
                    obs, _r, _te, _tr, _in = _ensure_hybrid().step(act[0])
                    last_obs[0] = obs
                    tag = "B" if brain_which.get() else "A"
                    if brains.get(tag, {}).get("state") is not None:
                        brains[tag]["state"]["obs"] = obs
                elif st_babble.get() and noise_mode[0] == "reflex_common":
                    # 【なぜ、2026-08-12】反射+共通駆動は「①〜⑤」の白色/色付き
                    #   ノイズ経路とは別物（moment_1/moment_2経由で基準長を動かし
                    #   伸張反射を計算する）。run/trainer.py の
                    #   reflex_common_active分岐（K tick中は毎tick計算し直す）と
                    #   同じ考え方で、保持の長さK（bab_k_var）では**間引かない**
                    #   （trainer.py 195行目「K tick中は毎tick」を踏襲）。
                    #   「②共収縮の下駄」（bab_cocon_var）はベース値として流用する
                    #   （仕様のアーキテクチャ節3・新しいスライダーを追加しない）。
                    _rcobj = _ensure_reflex_common()
                    if _rcobj is not None:
                        _apply_reflex_common_params(_rcobj)
                        _am = env.unwrapped.actuation_model
                        _r = _rcobj.brain.step_reflex_common(
                            dt, _am.muscle_lengths, _am.muscle_velocities)
                        _prev_act = act[0].copy()
                        act[0] = np.clip(
                            float(bab_cocon_var.get()) + _r, 0.0, 1.0).astype(np.float32)
                        babble_da2.append(float(((act[0] - _prev_act) ** 2).mean()))
                        if len(babble_da2) > 200:
                            babble_da2.pop(0)
                        env.step(act[0])
                    else:
                        env.step(zero)
                elif st_babble.get():
                    # 【なぜ、2026-08-11】④保持の長さK・②共収縮の下駄・①ゆらぎの大きさを
                    #   もがき運動パラメータ調整スライダーから毎tick読む（過去に3回
                    #   踏んだ「GUIの表示と物理適用が別経路」バグを避けるため、
                    #   ここでキャッシュせず`.get()`のたびに実体へ反映する）。
                    _bab_k = max(1, int(bab_k_var.get()))
                    if tick % _bab_k == 0:
                        _prev_act = act[0].copy()
                        act[0] = np.clip(
                            float(bab_cocon_var.get())
                            + float(bab_std_var.get()) * _babble_noise(),
                            0.0, 1.0).astype(np.float32)
                        # もがき運動の変化量も同じ物差しで測る。
                        #   【なぜ、2026-07-31】ユーザーの目視「学習済みの方が
                        #   ちょっと激しく動いている気がする」を数字で確かめるため。
                        #   注意：ただし脳の dAction2 は**方策の出力**（-1〜1）で測るのに対し、
                        #     こちらは**環境へ送る値**（筋活性化0〜1）。物差しが違うので
                        #     大小をそのまま比べられない。傾向を見るだけに使う。
                        babble_da2.append(float(((act[0] - _prev_act) ** 2).mean()))
                        if len(babble_da2) > 200:
                            babble_da2.pop(0)
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

                j = f"①角度   {rep['angle']:5.1f}°  {'視野内' if rep['in_fov'] else '視野外'}\n"
                j += ("②光線   遮蔽なし\n" if rep["ray_ok"]
                      else f"②光線   {rep['ray_hit']}に遮られている\n")
                if rep.get("pix_seen"):
                    j += (f"③画像   {rep['n_pixels']:4d}画素  "
                          f"中心からのずれ {dev:.2f}（平均 {np.mean(devs):.2f}）")
                else:
                    j += "③画像   映っていない"
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
                # 親としての介入の記録（何回・どれくらいの間隔で持っていったか）
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
                        text="めり込み " + " ".join(f"{n}{v:.1f}mm" for n, v in pen[:3]),
                        fg="#a00")
                else:
                    pen_label.config(text="めり込みなし", fg="#070")

                hw = np.array(head_w) if head_w else np.zeros(1)
                warn = ""
                if not freeze and st_hold.get():
                    warn = "\n「この角度で固定する」がONのまま物理を回しています＝暴れます"
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
                    kind = ("振動する（減衰不足）" if ratio < 0.7
                            else "ちょうど良い" if ratio < 1.6 else "戻りが遅い（減衰過多）")
                    recent = np.array(neck_hist[-100:]) if len(neck_hist) > 20 else np.zeros(1)
                    swing = float(recent.max() - recent.min())
                    txt = (f"首の角度 {ang:7.2f}度（目標 {nk_t.get():5.1f}）  "
                           f"速さ {vel:6.2f}度/s\n"
                           f"直近2秒の振れ幅 {swing:5.2f}度  "
                           f"{'落ち着いている' if swing < 1.0 else 'まだ動いている'}\n"
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
                        txt += f"\n持ち上げテスト：{st}  行き過ぎ {-_drop[0]['peak']:.1f}度"
                    neck_label.config(
                        text=txt, fg="#a00" if ratio < 0.7 and st_neck.get() else "#333")

                seen_pct = float(np.mean(seens)) * 100 if seens else 0.0
                fired = "● 撃った" if t_sim < fire_until[0] else ""
                # おもちゃの状態。最初の1.5秒は「遠くで待機中」＝視界に無いのが正常
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

                # 距離と輻輳（寄り目）の必要角を出す（2026-07-28）。
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
                             f"（瞳孔間 {IPD*100:.1f}cm）太郎は寄り目ができない")
                except Exception:
                    pass

                # 環境の状態＝「今どの条件で見ているか」を常に出す（2026-07-28）。
                #   Viewer と測定で条件が食い違っていた事故を防ぐため。
                _off = hands.offsets() if hands.holding else {}
                _offmax = max((abs(v) for v in _off.values()), default=0.0)
                # 実行速度（2026-07-29 追加）。ユーザーの報告「めっちゃ重い」に対し、
                #   何倍速で動いているかを常に見えるようにする。
                #     実速度 = シミュレーション時間 ÷ 実際に経過した時間
                #     1.0 なら等倍速。0.3 なら現実の3分の1の速さでしか進んでいない
                _wall = max(time.time() - wall0, 1e-6)
                _rate = t_sim / _wall
                _tgt = float(speed_var.get())
                _mark = "" if _rate >= _tgt * 0.9 else "  遅い"
                env_label.config(
                    text=f"体年齢 {_AGE:g}ヶ月（視力も同じ）  "
                         f"柵 {'あり' if fence_on[0] else 'なし'}  "
                         f"リクライニング {_RECLINE:.0f}度\n"
                         f"顎 {(_HOLD_TILT + '度で支える') if _HOLD_TILT else '指定なし'}"
                         f"　眼球の基準 {_EYE_REST_V:+.0f}度\n"
                         f"実験者の手 {'抑えている' if hands.holding else 'なし'}"
                         + (f"（頭のずれ {_offmax:.2f}度）" if hands.holding else "")
                         + f"\n実行速度 {_rate:.2f}倍速（設定 {_tgt:.1f}）"
                           f"　{t_sim:.1f}秒ぶん進んだ / 実時間 {_wall:.0f}秒{_mark}")

                # 左右の目をそれぞれ描く（2026-07-28）
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

                # コピー用のまとめ（「現在の設定をコピー」ボタンが使う）
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
                    f"VOR:{'ON' if st_vor.get() else 'OFF'}\n"
                    f"四肢の筋緊張  {'ON' if st_tone_limb.get() else 'OFF'}"
                    f"（{'新生児の既定姿勢' if lt_mode.get()=='newborn' else '今の姿勢を保つ'}・"
                    f"許すずれ{lt_hold.get():g}度）\n"
                    f"首のバネ  {'ON' if st_neck.get() else 'OFF'}  "
                    f"剛性{nk_k.get():.2f}  減衰{nk_c.get():.4f}  "
                    f"目標{nk_t.get():+.0f}度  "
                    f"{'3軸' if nk_all.get() else '前後だけ'}\n"
                    f"再生      速度{speed_var.get():.1f}倍  "
                    f"自発運動:{'ON' if st_babble.get() else 'OFF'}"
                    + (f"（変化量{np.mean(babble_da2[-50:]):.3f}）"
                       if babble_da2 else "") + "\n"
                    f"          駆動モード:{noise_mode[0]}\n"
                    # 注意：脳の状態は**常に**出す。
                    #   【なぜ、2026-07-31】「動いているとき」だけ出す作りにしたら、
                    #   チェックを入れ忘れて動かないときに理由が分からなかった。
                    #   何もしていないなら「何もしていない」と書く。
                    + "脳        "
                    + ("OFF（チェックを入れると学習した脳が動かします）"
                       if not st_brain.get() else
                       (f"{'B' if brain_which.get() else 'A'} で動かしている"
                        f"  揺らぎ{brain_std_var.get():.3f}"
                        + (f"  dAction2={np.mean(brain_da2[-50:]):.3f}"
                           if brain_da2 else "  注意まだ動いていません")
                        + (f"  目標指向{gb_stat[0]}回/探索{gb_stat[1]}回"
                           if st_gb.get() else "")))
                    + (f"\n          A={os.path.basename(brain_a_var.get())}"
                       if brain_a_var.get() else "")
                    + (f"  B={os.path.basename(brain_b_var.get())}"
                       if brain_b_var.get() else "")
                    + ("\n          注意自発運動と脳が両方ONです。脳が優先されます"
                       if st_brain.get() and st_babble.get() else "")
                    # 【やること・アーキテクチャ4節】反射+共通駆動は自発運動
                    #   （もがき運動）をONにしたときだけ効く（駆動ループの
                    #   elif st_babble.get() 分岐の中でしか呼ばれない）。脳が
                    #   ONの間は無関係なので、選んだのに何も起きない、を
                    #   黙って通さずここで明示する（既存の「両方ON」警告と
                    #   同じ流儀）。
                    + ("\n          注意反射+共通駆動は自発運動（もがき運動）を"
                       "ONにしたときだけ効きます。脳で動かしている間は無関係です"
                       if st_brain.get() and noise_mode[0] == "reflex_common" else "")
                    # 【なぜ、2026-08-12・仕様やること2の注意点】「四肢の筋緊張」
                    #   チェックボックス（limb_tone_apply/release）と反射+共通駆動
                    #   （disable_limb_tone_spring）は同じ jnt_stiffness を奪い合う。
                    #   どちらが勝つかは操作の順序で決まり実装上は未定義（片方だけ
                    #   無効化する等の作り込みまではせず、ここで食い違いを明示する
                    #   ことで「表示と実体が食い違ったまま気づかれない」状態だけを
                    #   避ける、という仕様の最低限を満たす）。
                    + ("\n          注意四肢の筋緊張と反射+共通駆動が同時にONです。"
                       "剛性(jnt_stiffness)を両方が奪い合うため、後から操作した方が"
                       "勝ちます（表示と実体が食い違うおそれ）"
                       if st_tone_limb.get() and noise_mode[0] == "reflex_common" else "")
                    + "\n"
                    "\n【今の状態】\n"
                    f"経過 {t_sim:.1f}秒  サッケード {reflex.n_saccades}発\n"
                    f"反応の強さ {reflex.strength:.3f}（閾値 {thr_var.get():.2f}）\n"
                    f"①角度 {rep['angle']:.1f}度 "
                    f"{'視野内' if rep['in_fov'] else '視野外'}  "
                    f"②光線 {'遮蔽なし' if rep['ray_ok'] else '' + str(rep['ray_hit']) + 'に遮られている'}  "
                    f"③画像 "
                    + (f"{rep['n_pixels']}画素 中心からのずれ {dev:.2f}"
                       if rep.get("pix_seen") else "映っていない") + "\n"
                    f"見えていた割合 {seen_pct:.1f}%  "
                    f"ずれ平均 {(np.mean(devs) if devs else float('nan')):.2f}"
                    f"（反射OFFで0.48）\n"
                    f"首の角度 {_nkang:.1f}度  頭の角速度 平均{hw.mean():.3f}\n"
                    f"目→おもちゃ {de:.1f}cm  肩→ {ds:.1f}cm"
                    f"（腕の{ds/(ARM_REACH*100)*100:.0f}%）\n"
                    f"親の介入 {len(_pl2)}回"
                    + (f"（平均 {_navg:.1f}秒おき）" if _gaps else "")
                )

                # 【なぜ、2026-08-07】key_callback（別スレッドの可能性がある）から
                #   直接 tkinter を触らず、ここ（win.update() と同じメインループの
                #   スレッド）でラベルを更新する。フラグが立っているときだけ書く
                #   （毎tick書くのは無駄なので、変わったときだけでよい）。
                if noise_label_dirty[0]:
                    noise_mode_label.config(
                        text=f"駆動モード: {noise_mode[0]}（Nキー、またはボタンで切替）")
                    # Nキーで切り替えたときも、ラジオボタンの選択が追従するように。
                    #   ここは win.update() と同じメインループのスレッドなので
                    #   tkinterを直接触ってよい（_set_noise_mode 側は触らない設計）。
                    if noise_mode_var.get() != noise_mode[0]:
                        noise_mode_var.set(noise_mode[0])
                    noise_label_dirty[0] = False

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
