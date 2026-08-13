"""太郎を見るための新しいビューア（PySide6版）— 第2段階：体・環境タブの残り＋状態表示。

【これは何か】`run/viewer_tools/e_viewer.py`（tkinter製・約3100行）のUIを、
PySide6へ段階的に移行する計画の**第2段階**の実装。第1段階
（骨格＋シーン区画のみ）は完了済み。今回はそれに次の4つを追加する。

【今回追加したもの】
    ・「体・環境」タブの残り4区画（設計段階2）
        おもちゃ・姿勢（43関節スライダー＋四肢の筋緊張）・反射・首のバネ
    ・浮動小数点スライダーのラッパー（`e_viewer_qt_widgets.FloatSlider`）と
      開閉できる区画（`e_viewer_qt_widgets.CollapsibleSection`）
    ・limb_toneの初回構築時ガード（旧版256〜269行目と同じ回避策の移植。
      2026-08-07に修正された「チェックボックスOFFでも常にONになる」混線
      バグの再発防止）
    ・状態表示欄（`e_viewer_qt_status.compute_status`、タブの上、常時表示）
    ・Windows 11風テーマの統合（`e_viewer_qt_theme.apply_theme`、他の実装担当の
      成果物。存在すれば適用する）

【今回もまだ無いもの（正直に明記する）】
    「駆動」「感覚と報酬」「実行」「測定器」の各タブは中身が無い空の
    プレースホルダのまま（次段階以降）。旧版の「測定器」区画（頭部・眼球の
    測定、触覚順応の監視ラベル等）・保存/読込（viewer_saved.json）機構・
    「仰向けに戻す」ボタン・親としておもちゃを運ぶ以外の測定表示は、今回の
    仕様（体・環境タブの「おもちゃ／姿勢／反射／首のバネ」区画に限定）の
    範囲外のため移植していない。
    実験・観察に使うなら、引き続き旧版（`e_viewer.py`、`Viewerを開く.bat`）を使うこと。
    旧版は一切変更していない。

【設計・仕様】
    作業記録（非公開）
    作業記録（非公開）
    作業記録（非公開）

【使い方】
    .venv/Scripts/python.exe run/viewer_tools/e_viewer_qt.py
    （E_SCENE環境変数でシーンを指定できる。未指定なら既存シーンの先頭を使う。
    ダブルクリックで試すなら「Viewer(新版UI試作)を開く.bat」も使える）
"""
import copy
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
# 【なぜ、2026-08-13】run/viewer_tools/e_viewer.py（33〜60行目）と同じパス設定を
#   そのまま使う。e_scene.py が内部で e_toy_env・infant_body 等
#   （E/scripts、taro_core/src/body）をimportするため、この一覧が要る。
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
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

import numpy as np
import mujoco
import mujoco.viewer
from PySide6 import QtCore, QtWidgets

import e_scene
import e_toy_env as TE
from e_head_hold import CaregiverHands
from infant_limbs import apply_limb_tone, limb_tone_joints
from run.config import TARO_DEFAULTS

from e_viewer_qt_widgets import FloatSlider, CollapsibleSection
from e_viewer_qt_status import compute_status, format_status_text

# 【2026-08-13】シーンが全部を決めるので、体を作り直すときは古い個別指定を
#   消しておく。run/viewer_tools/e_viewer.py 2348〜2352行目と同じ一覧
#   （そのまま踏襲。ここだけ別の一覧にすると「シーンと環境変数のどちらが
#   効くのか」がまた曖昧になるため＝落とし穴チェックリスト項30と同じ考え方）。
_STALE_ENV_KEYS = ("E_AGE", "E_HEAD_HOLD", "E_FENCE", "E_TOY_RADIUS",
                    "E_RECLINE", "E_EYE_REST_V", "E_HOLD_TILT",
                    "E_TOY_POS", "E_SEAT_FRICTION", "E_TOY_MODE",
                    "E_TOY_SHAPE", "E_TOY_DIST")

SHAKE_HZ = 2.5             # 「小さく激しく」＝2.5Hz（run/viewer_tools/e_viewer.py 82行目と同じ値）
SHAKE_AMP = 0.015          # 1.5cm（同上）

# 【なぜここに複製したか、2026-08-13】姿勢区画の43関節を部位ごとにまとめる表
#   （run/viewer_tools/e_viewer.py 99〜133行目 POSE_GROUPS）。旧版ファイル自体は
#   tkinterのGUI構築コードをモジュール読み込み時に実行するため import できない
#   （読み取り専用の参考にとどめる仕様の制約）。データ自体は静的な対応表
#   （関節名の末尾と日本語名）で、旧版・新版どちらでも変わらないはずのものなので、
#   ロジックの二重実装（禁止事項）ではなく、データ表の複製として許容する。
#   変えるときは両方のファイルを同時に直すこと。
POSE_GROUPS = [
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


def _collect_pose_joints(m, d):
    """姿勢を作る関節を集める（移植元：run/viewer_tools/e_viewer.py 483〜515行目）。

    読み取り専用の参考としてロジックだけを写した（旧版ファイル自体は
    importしない）。戻り値は関節ごとの辞書のリスト。
    """
    def _find_joint(name):
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
                if abs(hi - lo) < 1e-9:
                    continue
                joints.append(dict(
                    base=jname, key=jname, group=gname, side=side,
                    jp=(f"{side}{jp}" if side else jp),
                    jid=jid, qadr=qadr,
                    lo=float(lo), hi=float(hi),
                    init=float(np.degrees(d.qpos[qadr])),
                    is_neck=(gname == "首"),
                    twin=(f"left_{base}" if side == "右" else
                          (f"right_{base}" if side == "左" else None))))
    return joints


class MainWindow(QtWidgets.QMainWindow):
    """太郎ビューア（新版・第2段階）。

    「体・環境」タブに、シーン区画（第1段階）＋おもちゃ／姿勢／反射／
    首のバネ（第2段階）が入る。他の4タブは次段階以降で埋める空の
    プレースホルダのまま。タブの上に、タブを切り替えても消えない状態表示欄を置く。
    """

    def __init__(self, scene_names, current_scene_name, joints):
        super().__init__()
        self.setWindowTitle("太郎ビューア（新版UI試作・第2段階）")
        self.resize(900, 760)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        # ---- 状態表示欄（仕様2-3節）----------------------------------------
        # 【なぜタブの外に置くか】「いま何が有効になっているか」をタブを
        #   切り替えても常に見えるようにするため（設計5節の配置案どおり）。
        status_box = QtWidgets.QGroupBox("状態（実体から読んだ値。タブを切り替えても消えません）")
        status_layout = QtWidgets.QVBoxLayout(status_box)
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet("font-family: Consolas;")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)
        root.addWidget(status_box)

        tabs = QtWidgets.QTabWidget()
        root.addWidget(tabs, 1)

        self.tab_body_env = QtWidgets.QWidget()
        tabs.addTab(self.tab_body_env, "体・環境")
        for title in ("駆動", "感覚と報酬", "実行", "測定器"):
            placeholder = QtWidgets.QWidget()
            lay = QtWidgets.QVBoxLayout(placeholder)
            lay.addWidget(QtWidgets.QLabel(
                f"「{title}」タブは第2段階では未実装です（次段階以降で移植予定）"))
            lay.addStretch(1)
            tabs.addTab(placeholder, title)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_body = QtWidgets.QWidget()
        scroll.setWidget(scroll_body)
        outer = QtWidgets.QVBoxLayout(self.tab_body_env)
        outer.addWidget(scroll)
        form_outer = QtWidgets.QVBoxLayout(scroll_body)

        # ================================================================
        # 区画：シーン（環境）— 第1段階からそのまま（要素は変更していない）
        # 移植元：run/viewer_tools/e_viewer.py 605〜746行目付近
        # ================================================================
        box = QtWidgets.QGroupBox("シーン（環境）")
        form_outer.addWidget(box)
        form = QtWidgets.QVBoxLayout(box)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("シーン"))
        self.scene_combo = QtWidgets.QComboBox()
        self.scene_combo.addItems(scene_names or ["（シーンがまだ無い）"])
        if current_scene_name in scene_names:
            self.scene_combo.setCurrentText(current_scene_name)
        row.addWidget(self.scene_combo, 1)
        form.addLayout(row)

        self.scene_note = QtWidgets.QLabel("")
        self.scene_note.setWordWrap(True)
        self.scene_note.setStyleSheet("color: #666;")
        form.addWidget(self.scene_note)

        note_label = QtWidgets.QLabel(
            "「最初からやり直す」を押すと、選んだシーンで始まります\n"
            "別のシーンに切り替えるときは体を作り直すので数十秒かかります")
        note_label.setStyleSheet("color: #a30;")
        form.addWidget(note_label)

        self.chk_fence = QtWidgets.QCheckBox("柵（ベビーサークル）を有効にする")
        form.addWidget(self.chk_fence)
        self.chk_head_hold = QtWidgets.QCheckBox(
            "実験者が頭を抑える（人間の乳児実験と同じ）")
        form.addWidget(self.chk_head_hold)

        self.env_label = QtWidgets.QLabel("")
        self.env_label.setStyleSheet("font-family: Consolas; color: #444;")
        form.addWidget(self.env_label)

        self.msg_label = QtWidgets.QLabel("")
        self.msg_label.setWordWrap(True)
        form.addWidget(self.msg_label)

        self.restart_btn = QtWidgets.QPushButton("最初からやり直す")
        form.addWidget(self.restart_btn)

        # ================================================================
        # 区画1：おもちゃ（移植元：e_viewer.py 764〜886行目）
        # ================================================================
        box_toy = QtWidgets.QGroupBox("おもちゃ")
        form_outer.addWidget(box_toy)
        toy_form = QtWidgets.QVBoxLayout(box_toy)

        self.toy_sliders = []  # [X, Y, Z]
        for label, lo, hi in [("X（頭↔足）", -0.1, 0.5),
                               ("Y（左↔右）", -0.3, 0.3),
                               ("Z（低↔高）", 0.0, 0.5)]:
            s = FloatSlider(label, lo, hi, 0.005, init=0.0, decimals=3)
            toy_form.addWidget(s)
            self.toy_sliders.append(s)

        self.toy_size_slider = FloatSlider("大きさ(半辺)", 0.005, 0.05, 0.0025,
                                            init=0.02, decimals=4)
        toy_form.addWidget(self.toy_size_slider)

        self.chk_toy_shake = QtWidgets.QCheckBox(
            f"小さく激しく揺らす（{SHAKE_HZ}Hz・{SHAKE_AMP*100:.1f}cm）")
        self.chk_toy_shake.setChecked(True)
        toy_form.addWidget(self.chk_toy_shake)

        self.chk_toy_follow = QtWidgets.QCheckBox(
            "スライダーの位置に固定する（外すと環境が視線の正面に置く）")
        toy_form.addWidget(self.chk_toy_follow)

        self.toy_info_label = QtWidgets.QLabel("")
        self.toy_info_label.setStyleSheet("color: #666;")
        toy_form.addWidget(self.toy_info_label)

        parent_label_title = QtWidgets.QLabel("あなたが「親」をやる")
        parent_label_title.setStyleSheet("font-weight: bold;")
        toy_form.addWidget(parent_label_title)
        parent_desc = QtWidgets.QLabel(
            "人間の親は赤ちゃんの顔の向きに合わせておもちゃを見せる。\n"
            "太郎にはそれが無いので、首が動くと視界から外れたままになる。\n"
            "注意：手動は「何が必要か」を掴むためのもの。数値の比較には使えない。")
        parent_desc.setStyleSheet("color: #555;")
        toy_form.addWidget(parent_desc)

        self.toy_carry_sec = FloatSlider("運ぶ秒数", 0.2, 3.0, 0.1, init=1.0,
                                          note="瞬間移動させない（随伴性を壊すため）")
        toy_form.addWidget(self.toy_carry_sec)

        self.toy_dist_slider = FloatSlider("目からの距離[m]", 0.05, 0.40, 0.005,
                                            init=0.086, decimals=3,
                                            note="「顔の前へ持っていく」を押すとこの距離に置く")
        toy_form.addWidget(self.toy_dist_slider)

        self.vergence_label = QtWidgets.QLabel("")
        self.vergence_label.setStyleSheet("color: #06a; font-family: Consolas;")
        toy_form.addWidget(self.vergence_label)

        self.bring_to_face_btn = QtWidgets.QPushButton("顔の前へ持っていく")
        toy_form.addWidget(self.bring_to_face_btn)
        self.parent_label = QtWidgets.QLabel("")
        toy_form.addWidget(self.parent_label)

        # ================================================================
        # 区画2：姿勢（移植元：e_viewer.py 888〜1068行目）
        # ================================================================
        box_pose = QtWidgets.QGroupBox("姿勢")
        form_outer.addWidget(box_pose)
        pose_form = QtWidgets.QVBoxLayout(box_pose)

        self.chk_freeze = QtWidgets.QCheckBox(
            "物理演算を止める（編集モード。重力も衝突も無し）")
        self.chk_freeze.setStyleSheet("color: #a30;")
        pose_form.addWidget(self.chk_freeze)
        self.chk_pose_hold = QtWidgets.QCheckBox(
            "この角度で固定する（物理ONのまま使うと暴れます）")
        pose_form.addWidget(self.chk_pose_hold)

        self.chk_symmetric = QtWidgets.QCheckBox(
            "左右を対称に動かす（片方を動かすともう片方も同じ角度）")
        self.chk_symmetric.setStyleSheet("color: #06a;")
        pose_form.addWidget(self.chk_symmetric)

        btn_row = QtWidgets.QHBoxLayout()
        self.pose_capture_btn = QtWidgets.QPushButton("いまの姿勢を取り込む")
        self.pose_mirror_btn = QtWidgets.QPushButton("右→左に写す")
        btn_row.addWidget(self.pose_capture_btn)
        btn_row.addWidget(self.pose_mirror_btn)
        btn_row.addStretch(1)
        pose_form.addLayout(btn_row)

        pose_hint = QtWidgets.QLabel(
            "首を動かすと、実験者の手が支える目標角も一緒に動きます\n"
            "姿勢を作る手順： ①物理を止める ②角度を決める ③固定をON")
        pose_hint.setStyleSheet("color: #666;")
        pose_form.addWidget(pose_hint)

        # ---- 四肢の筋緊張 ---------------------------------------------------
        tone_box = QtWidgets.QGroupBox("四肢の筋緊張（脱力しても四肢が既定の姿勢に保たれる）")
        pose_form.addWidget(tone_box)
        tone_lay = QtWidgets.QVBoxLayout(tone_box)
        self.chk_limb_tone = QtWidgets.QCheckBox("有効にする")
        tone_lay.addWidget(self.chk_limb_tone)
        radio_row = QtWidgets.QHBoxLayout()
        self.radio_tone_newborn = QtWidgets.QRadioButton("新生児の既定姿勢（文献）")
        self.radio_tone_reach = QtWidgets.QRadioButton("今の姿勢を保つ")
        radio_row.addWidget(self.radio_tone_newborn)
        radio_row.addWidget(self.radio_tone_reach)
        radio_row.addStretch(1)
        tone_lay.addLayout(radio_row)
        self.tone_hold_slider = FloatSlider(
            "許すずれ[度]", 2, 40, 1, init=10.0,
            note="小さいほど硬い。注意：乳児の四肢の筋緊張の実測値は文献に存在しない"
                 "（Tier3・感度分析の対象）")
        tone_lay.addWidget(self.tone_hold_slider)
        self.tone_capture_btn = QtWidgets.QPushButton("いまの姿勢を筋緊張の目標にする")
        tone_lay.addWidget(self.tone_capture_btn)

        # ---- 43関節のスライダー（部位ごとに折りたたみ）------------------------
        self.joint_sliders = {}   # key（関節名。左右付き） -> FloatSlider
        self.joint_meta = {jd["key"]: jd for jd in joints}
        for gname, paired, items, opened in POSE_GROUPS:
            mine = [jd for jd in joints if jd["group"] == gname]
            if not mine:
                continue
            sec = CollapsibleSection(f"{gname}（{len(mine)}）", opened=opened)
            pose_form.addWidget(sec)
            for jd in mine:
                s = FloatSlider(jd["jp"], jd["lo"], jd["hi"], 1.0, init=jd["init"])
                sec.body_layout.addWidget(s)
                self.joint_sliders[jd["key"]] = s

        # ================================================================
        # 区画3：反射（移植元：e_viewer.py 1070〜1085行目）
        # ================================================================
        box_ref = QtWidgets.QGroupBox("反射")
        form_outer.addWidget(box_ref)
        ref_form = QtWidgets.QVBoxLayout(box_ref)
        self.chk_orient = QtWidgets.QCheckBox("視線誘導反射（動くものに目・首を向ける）")
        ref_form.addWidget(self.chk_orient)
        self.reflex_latency = FloatSlider(
            "間隔[秒]", 0.1, 1.5, 0.1, init=0.5,
            note="0.2＝旧設定（撃ちすぎて視界から追い出す）／0.5〜0.9＝新生児の実測")
        ref_form.addWidget(self.reflex_latency)
        self.reflex_threshold = FloatSlider(
            "発火の閾値", 0.0, 0.06, 0.0025, init=0.015, decimals=4,
            note="0.02＝旧設定（低すぎて常に発火）／0.015前後で動きを選べる")
        ref_form.addWidget(self.reflex_threshold)
        self.chk_vor = QtWidgets.QCheckBox("前庭動眼反射（頭の動きを眼で打ち消す）")
        self.chk_vor.setChecked(True)
        ref_form.addWidget(self.chk_vor)

        # ================================================================
        # 区画3b：首のバネ（移植元：e_viewer.py 1087〜1155行目）
        # ================================================================
        box_neck = QtWidgets.QGroupBox("首のバネ（調整中）")
        form_outer.addWidget(box_neck)
        neck_form = QtWidgets.QVBoxLayout(box_neck)
        neck_desc = QtWidgets.QLabel(
            "首にはバネが無く（stiffness=0）、重力で60秒かけて60度倒れ続ける。\n"
            "正常な新生児でも頸部に軽い抵抗はある（過緊張は正常児の0.7%だけ\n"
            "／Amiel-Tison 1977）。死後標本の剛性は屈曲0.175・伸展0.40 Nm/rad\n"
            "（筋を含まない下限値／Luck 2008）。生体の実測値は存在しない。")
        neck_desc.setStyleSheet("color: #555;")
        neck_form.addWidget(neck_desc)
        self.neck_core_label = QtWidgets.QLabel("")
        self.neck_core_label.setStyleSheet("color: #070;")
        neck_form.addWidget(self.neck_core_label)
        self.chk_neck = QtWidgets.QCheckBox("首にバネを効かせる（外すと core の設定を消します）")
        self.chk_neck.setStyleSheet("color: #a30;")
        neck_form.addWidget(self.chk_neck)
        self.neck_k = FloatSlider("剛性[Nm/rad]", 0.0, 0.6, 0.01, init=0.4, decimals=3,
                                   note="死後標本の下限 0.175〜0.40／四肢のトーンは 0.2")
        neck_form.addWidget(self.neck_k)
        self.neck_c = FloatSlider("減衰", 0.0, 0.2, 0.005, init=0.0265, decimals=4,
                                   note="臨界減衰（振動しない最小値）は下に表示")
        neck_form.addWidget(self.neck_c)
        self.neck_t = FloatSlider("目標角[度]", -60.0, 30.0, 1.0, init=-45.0)
        neck_form.addWidget(self.neck_t)
        self.chk_neck_all = QtWidgets.QCheckBox(
            "3軸すべてに効かせる（外すと前後の傾きだけ）リクライニングでは必要")
        neck_form.addWidget(self.chk_neck_all)
        self.neck_label = QtWidgets.QLabel("")
        self.neck_label.setStyleSheet("font-family: Consolas;")
        neck_form.addWidget(self.neck_label)
        self.drop_head_btn = QtWidgets.QPushButton("頭を持ち上げて離す")
        neck_form.addWidget(self.drop_head_btn)

        form_outer.addStretch(1)


def _describe_scene(sc):
    """選んだシーンが成立しているかを説明文にする。

    移植元：run/viewer_tools/e_viewer.py の on_scene_pick()（639〜675行目付近）と
    同じロジック（読み取り専用の参考。文言・判定は変えていない）。
    戻り値：(説明文, 警告か)
    """
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
    return "\n".join(t for t in lines if t), warn


def main():
    scene_names = e_scene.list_scenes()
    if not scene_names:
        print("[e_viewer_qt] シーンが1つもありません。"
              "run/scene_tools/e_scene_make.py で作ってください。", flush=True)
        return 1

    scene_name = os.environ.get("E_SCENE") or scene_names[0]
    if scene_name not in scene_names:
        print(f"[e_viewer_qt] 注意指定されたシーン「{scene_name}」が見つかりません。"
              f"「{scene_names[0]}」を使います。", flush=True)
        scene_name = scene_names[0]

    scene = e_scene.load(scene_name)
    print(f"[scene] 「{scene['name']}」から始めます", flush=True)
    if scene.get("note"):
        print(f"        {scene['note']}", flush=True)

    # actuation_model は指定しない＝e_scene.build() の既定（MuscleModel）を使う。
    #   駆動モード（E_ACTUATION）の切替UIは今回のスコープに入っていない。
    _ACTUATION_MODE = "muscle"

    # ====================================================================
    # 【必須・申し送り事項の移植（仕様2-2節）】limb_toneの初回構築時ガード。
    #
    # 【なぜ】e_scene.build() は scene.setup.limb_tone を無条件に物理へ適用する
    #   ため、Viewer のチェックボックス（st_tone_limb、姿勢区画）がOFFでも
    #   常にONになる混線バグがあった（監査報告 2026-08-07）。初回構築時は
    #   limb_tone を意図的に外し、「四肢の筋緊張を物理へ適用する経路」を
    #   limb_tone_apply()/limb_tone_release()（チェックボックスと連動する経路）
    #   だけに一本化する（移植元：run/viewer_tools/e_viewer.py 256〜269行目）。
    # ====================================================================
    _scene_for_build = scene
    if (scene.get("setup") or {}).get("limb_tone"):
        _scene_for_build = copy.deepcopy(scene)
        _scene_for_build["setup"]["limb_tone"] = None

    env, hands = e_scene.build(_scene_for_build, orient=True, vor=True, seed=0,
                               verbose=True)
    u = env.unwrapped
    m, d = u.model, u.data
    if hands is None:
        hands = CaregiverHands(m, d)

    # 注意：【2026-08-13・移植元 e_viewer.py 276〜283行目】verify() は
    #   limb_tone の初回適用（下の on_limb_tone_toggle() 初回呼び出し）の
    #   **あと**に呼ぶ。limb_tone を外した直後にここで照合すると、
    #   「バネの入った関節数」が保存時と食い違う偽陽性の警告が出てしまう
    #   （limb_toneを使う全シーンで毎回出る）。

    dt = float(m.opt.timestep) * int(u.frame_skip)
    n_act = env.action_space.shape[0]
    zero = np.zeros(n_act, dtype=np.float32)

    _AGE = float(scene["body"]["age_months"])
    _RECLINE = float(scene["world"]["recline_deg"])
    _EYE_REST_V = float(scene["body"]["eye_rest_vertical_deg"])
    _hh = scene["setup"].get("head_hold") or {}
    _tgt = _hh.get("target_deg") or None
    _HOLD_TILT = (str(_tgt["head_tilt"]) if _tgt and "head_tilt" in _tgt else None)
    _hold_tgt = _tgt

    # 柵のgeom（実行時にON/OFFする）。
    #   移植元：run/viewer_tools/e_viewer.py 740〜762行目（apply_fence）と同じロジック。
    fence_gids = [g for g in range(m.ngeom) if "fence" in (m.geom(g).name or "")]
    fence_rgba0 = {g: m.geom_rgba[g].copy() for g in fence_gids}
    fence_con0 = {g: (int(m.geom_contype[g]), int(m.geom_conaffinity[g]))
                  for g in fence_gids}
    fence_on = [bool(scene["world"]["fence"])]

    def apply_fence(want):
        u._fence = bool(want)
        for g in fence_gids:
            if want:
                m.geom_rgba[g] = fence_rgba0[g]
                m.geom_contype[g], m.geom_conaffinity[g] = fence_con0[g]
            else:
                c = fence_rgba0[g].copy()
                c[3] = 0.0
                m.geom_rgba[g] = c
                m.geom_contype[g] = 0
                m.geom_conaffinity[g] = 0
        fence_on[0] = bool(want)

    # ---- 反射（e_orienting_v2）------------------------------------------
    import e_orienting_v2 as OR
    reflex = u._orienting
    vor = u._vor

    # ---- 首の3軸 ----------------------------------------------------------
    neck_ids = {}
    for j in range(m.njnt):
        nm = m.joint(j).name.split(":")[-1]
        if nm in ("head_tilt", "head_swivel", "head_tilt_side"):
            neck_ids[nm] = j
    neck_damp0 = {k: float(m.dof_damping[int(m.jnt_dofadr[v])])
                  for k, v in neck_ids.items()}
    _hb = int(m.body("head").id)
    _neck_arm = float(np.linalg.norm(
        np.array(d.xpos[_hb]) - np.array(d.xpos[int(m.body_parentid[_hb])])))
    NECK_I = float(m.body_inertia[_hb][0]) + float(m.body_mass[_hb]) * _neck_arm ** 2

    _jt0 = neck_ids.get("head_tilt")
    _neck_k0 = float(m.jnt_stiffness[_jt0]) if _jt0 is not None else 0.0
    _neck_t0 = (float(np.degrees(m.qpos_spring[int(m.jnt_qposadr[_jt0])]))
                if _jt0 is not None else 0.0)
    _neck_c0 = (float(m.dof_damping[int(m.jnt_dofadr[_jt0])]) if _jt0 is not None else 0.0265)
    _neck_lo, _neck_hi = (-60.0, 30.0) if _RECLINE <= 0 else (-30.0, 70.0)
    _neck_def = -45.0 if _RECLINE <= 0 else 30.0

    # ---- おもちゃ -----------------------------------------------------------
    toy_bid = int(m.body("test_object1").id)
    toy_jid = int(m.body_jntadr[toy_bid])
    toy_qadr = int(m.jnt_qposadr[toy_jid])
    toy_dof = int(m.jnt_dofadr[toy_jid])
    toy_gadr = int(m.body("test_object1").geomadr[0])

    toy_pos0 = d.qpos[toy_qadr:toy_qadr + 3].copy()
    # 注意：リセット直後のおもちゃは退避位置（登場を遅らせる仕組み）。
    #   そのままスライダーの初期値にすると範囲外になり、少し動かした瞬間に
    #   突然どこかへ飛んで見える（移植元：e_viewer.py 517〜528行目）。
    if float(np.max(np.abs(toy_pos0[:2]))) > 1.0:
        try:
            u._set_anchor()
            toy_pos0 = np.array(u._rest_pos, dtype=float)
        except Exception:
            toy_pos0 = np.array([0.28, 0.0, 0.18], dtype=float)
    toy_size0 = float(m.geom_size[toy_gadr][0])

    try:
        _el = np.array(d.cam_xpos[int(m.camera("eye_left").id)], dtype=float)
        _er = np.array(d.cam_xpos[int(m.camera("eye_right").id)], dtype=float)
        IPD = float(np.linalg.norm(_el - _er))
    except Exception:
        IPD = 0.045

    # ---- 姿勢の関節一覧 -----------------------------------------------------
    joints = _collect_pose_joints(m, d)
    print(f"[pose] 姿勢を作れる関節 {len(joints)} 個", flush=True)

    # ---- 四肢の筋力倍率の基準値（状態表示欄で使う。仕様2-3節2）--------------
    am = u.actuation_model
    _fmax_base = None
    if hasattr(am, "fmax"):
        _f0 = np.asarray(am.fmax, dtype=float)
        if _f0.ndim > 0:
            _fmax_base = _f0.copy()

    # ==================================================================
    # Qtアプリの構築
    # ==================================================================
    app = QtWidgets.QApplication(sys.argv)
    try:
        from e_viewer_qt_theme import apply_theme
        apply_theme(app)
    except Exception as e:
        # 【なぜ】テーマ適用（QPalette＋最小限QSS）に失敗しても、
        #   テーマ無しで起動を続ける（テーマはあくまで見た目の装飾で、
        #   機能には影響しないため）。
        print(f"[theme] 注意テーマの適用に失敗しました（続行します）: {e}", flush=True)

    win = MainWindow(scene_names, scene["name"], joints)

    def on_scene_pick(name):
        try:
            sc = e_scene.load(name)
        except Exception:
            win.scene_note.setText("")
            return
        text, warn = _describe_scene(sc)
        win.scene_note.setText(text)
        win.scene_note.setStyleSheet(f"color: {'#a30' if warn else '#666'};")
        win.chk_fence.setChecked(bool(sc["world"]["fence"]))
        win.chk_head_hold.setChecked(bool(sc["setup"].get("head_hold")))

    win.scene_combo.currentTextChanged.connect(on_scene_pick)
    on_scene_pick(scene["name"])

    # ---- おもちゃ区画の初期値・配線 ---------------------------------------
    for i in range(3):
        win.toy_sliders[i].setValue(float(toy_pos0[i]), block_signal=True)
    win.toy_size_slider.setValue(toy_size0, block_signal=True)
    win.toy_dist_slider.setValue(float(getattr(u, "_toy_dist", 0.086)), block_signal=True)
    win.toy_info_label.setText(
        f"持たせ方 {TE.TOY_MODE} ／ 登場 {TE.TOY_APPEAR_DELAY:.1f}秒後に"
        f"{TE.TOY_APPROACH_SEC:.1f}秒かけて{TE.TOY_APPROACH_FROM}から")
    win.parent_label.setText("まだ介入していません（おもちゃが見えなくなったら押す）")

    # スライダーを動かしたら自動で「固定する」に切り替える
    #   （移植元：e_viewer.py 772〜774行目 on_toy_slider）。
    _syncing = [False]

    def on_toy_slider_changed(_v):
        if not _syncing[0]:
            win.chk_toy_follow.setChecked(True)

    for s in win.toy_sliders:
        s.valueChanged.connect(on_toy_slider_changed)

    toy_base = [None]
    _parent = [None]
    parent_log = []

    def bring_to_face():
        """今の視線の正面へ、指定秒かけておもちゃを運ぶ（移植元：e_viewer.py 819〜850行目）。"""
        cid = int(m.camera("eye_left").id)
        eyes = []
        for nm2 in ("eye_left", "eye_right"):
            try:
                eyes.append(np.array(d.cam_xpos[int(m.camera(nm2).id)], dtype=float))
            except Exception:
                pass
        origin = np.mean(eyes, axis=0) if eyes else np.array(d.cam_xpos[cid], dtype=float)
        fwd = -np.array(d.cam_xmat[cid], dtype=float).reshape(3, 3)[:, 2]
        dist = float(win.toy_dist_slider.value())
        u._toy_dist = dist
        goal = origin + fwd * dist
        if fence_on[0]:
            mgn = 0.03
            goal[0] = float(np.clip(goal[0], -TE.FENCE_HALF_X + mgn, TE.FENCE_HALF_X - mgn))
            goal[1] = float(np.clip(goal[1], -TE.FENCE_HALF_Y + mgn, TE.FENCE_HALF_Y - mgn))
        goal[2] = float(max(goal[2], 0.04))
        start = np.array(d.xpos[toy_bid], dtype=float)
        _parent[0] = {"t": 0.0, "from": start, "to": goal,
                      "sec": max(0.05, float(win.toy_carry_sec.value()))}
        win.chk_toy_follow.setChecked(True)
        parent_log.append(None)
        win.msg_label.setText(f"親が顔の前へ運んでいます（{len(parent_log)}回目）")

    win.bring_to_face_btn.clicked.connect(bring_to_face)

    # ---- 姿勢区画の配線 ------------------------------------------------------
    _sym_busy = [False]
    by_key = {jd["key"]: win.joint_sliders[jd["key"]] for jd in joints}

    def on_joint_changed(jd):
        def _hook(_v):
            if not win.chk_symmetric.isChecked() or _sym_busy[0] or not jd.get("twin"):
                return
            tw = by_key.get(jd["twin"])
            if tw is None:
                return
            _sym_busy[0] = True
            try:
                tw.setValue(win.joint_sliders[jd["key"]].value())
            finally:
                _sym_busy[0] = False
        return _hook

    for jd in joints:
        win.joint_sliders[jd["key"]].valueChanged.connect(on_joint_changed(jd))

    def pose_from_body():
        _sym_busy[0] = True
        try:
            for jd in joints:
                win.joint_sliders[jd["key"]].setValue(
                    round(float(np.degrees(d.qpos[jd["qadr"]])), 1))
        finally:
            _sym_busy[0] = False
        win.msg_label.setText("いまの姿勢をスライダーに取り込みました")

    def pose_mirror_rl():
        _sym_busy[0] = True
        try:
            for jd in joints:
                if jd.get("side") == "右" and jd.get("twin") in by_key:
                    by_key[jd["twin"]].setValue(win.joint_sliders[jd["key"]].value())
        finally:
            _sym_busy[0] = False
        win.msg_label.setText("右の角度を左へ写しました")

    win.pose_capture_btn.clicked.connect(pose_from_body)
    win.pose_mirror_btn.clicked.connect(pose_mirror_rl)

    # ---- 四肢の筋緊張（移植元：e_viewer.py 925〜1012行目）---------------------
    _lt0 = ((scene or {}).get("setup") or {}).get("limb_tone") or {}
    win.chk_limb_tone.setChecked(bool(_lt0))
    if _lt0.get("target_deg"):
        win.radio_tone_newborn.setChecked(True)
    else:
        win.radio_tone_reach.setChecked(True)
    win.tone_hold_slider.setValue(float(_lt0.get("hold_deg", 10.0)), block_signal=True)

    _limb_saved = [None]

    def limb_tone_apply():
        profile = "newborn_flexor" if win.radio_tone_newborn.isChecked() else "reach_limb"
        if _limb_saved[0] is None:
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
                            hold_deg=float(win.tone_hold_slider.value()), verbose=True)
        _mode_jp = "新生児の既定姿勢（文献）" if profile == "newborn_flexor" else "今の姿勢"
        win.msg_label.setText(f"四肢の筋緊張の目標を「{_mode_jp}」にしました"
                              f"（{r['n']}関節・許すずれ{win.tone_hold_slider.value():g}度）")

    def limb_tone_release():
        sv = _limb_saved[0]
        if not sv:
            return
        for nm, (k, sp, dp, j, adr, dof) in sv.items():
            m.jnt_stiffness[j] = k
            m.qpos_spring[adr] = sp
            m.dof_damping[dof] = dp
        win.msg_label.setText("四肢の筋緊張を切りました（腕は重力で落ちます）")

    def on_limb_tone_toggle(*_a):
        (limb_tone_apply if win.chk_limb_tone.isChecked() else limb_tone_release)()

    win.chk_limb_tone.toggled.connect(on_limb_tone_toggle)
    win.radio_tone_newborn.toggled.connect(
        lambda checked: on_limb_tone_toggle() if checked and win.chk_limb_tone.isChecked() else None)

    def limb_tone_capture_current():
        win.radio_tone_reach.setChecked(True)
        win.chk_limb_tone.setChecked(True)
        on_limb_tone_toggle()

    win.tone_capture_btn.clicked.connect(limb_tone_capture_current)

    # 【仕様2-2節】初回構築時のガードで外したlimb_toneを、チェックボックスと
    #   連動する経路（ここ）だけから適用する。旧版と同じ「起動直後に1回、
    #   保存済みの状態を反映する」呼び出し。
    if win.chk_limb_tone.isChecked():
        limb_tone_apply()

    # 【2026-08-13・移植元 e_viewer.py 276〜283行目】limb_toneの初回適用が
    #   終わった後にverify()する（偽陽性の食い違い警告を避けるため）。
    try:
        e_scene.verify(scene, env, strict=False, verbose=True)
    except Exception as e:
        print(f"[scene] 注意verify()に失敗しました（続行します）: {e}", flush=True)

    # ---- 反射区画の初期値 -----------------------------------------------------
    win.reflex_latency.setValue(float(OR.SACCADE_LATENCY), block_signal=True)
    win.reflex_threshold.setValue(float(OR.SACCADE_MIN_STRENGTH), block_signal=True)

    # ---- 首のバネ区画の初期値（移植元：e_viewer.py 1095〜1121行目）------------
    win.neck_core_label.setText(
        f"起動時に core が設定した値：剛性{_neck_k0:.2f} 目標{_neck_t0:+.0f}度 減衰{_neck_c0:.4f}")
    win.chk_neck.setChecked(_neck_k0 > 0.0)
    win.neck_k.setValue(_neck_k0 if _neck_k0 > 0 else 0.4, block_signal=True)
    win.neck_c.setValue(_neck_c0, block_signal=True)
    win.neck_t.setValue(_neck_t0 if _neck_k0 > 0 else _neck_def, block_signal=True)
    win.neck_t._lo, win.neck_t._hi = _neck_lo, _neck_hi   # 姿勢に合わせた範囲へ更新
    win.neck_t._steps = max(1, int(round((_neck_hi - _neck_lo) / win.neck_t._res)))
    win.neck_t.slider.setRange(0, win.neck_t._steps)
    win.neck_t.setValue(_neck_t0 if _neck_k0 > 0 else _neck_def, block_signal=True)
    win.chk_neck_all.setChecked(_RECLINE > 0)

    neck_hist = []
    _drop = [None]

    def drop_head():
        for j in neck_ids.values():
            d.qpos[int(m.jnt_qposadr[j])] = np.radians(40.0)
            d.qvel[int(m.jnt_dofadr[j])] = 0.0
            break
        mujoco.mj_forward(m, d)
        _drop[0] = {"t0": None, "peak": -1e9, "settled": None}
        win.msg_label.setText("頭を40度に持ち上げて離しました。戻り方を見てください")

    win.drop_head_btn.clicked.connect(drop_head)

    restart = [True]
    win.restart_btn.clicked.connect(lambda: restart.__setitem__(0, True))
    win.show()

    # 【なぜ、2026-08-13】Mica（半透明の背景）を試す。仕様3節・4節。
    #   pywinstylesが入っていない環境・Windows以外でも例外を投げず、
    #   単に適用されないだけで起動を継続する（try_apply_mica内部でtry/except
    #   済み。ここでは戻り値をログに出すだけ）。win.show()の**あとに**
    #   呼ぶ必要がある（winId()がOS側のウィンドウハンドルを持つのは表示後）。
    try:
        from e_viewer_qt_theme import try_apply_mica
        if try_apply_mica(win):
            print("[theme] Micaを適用しました（実機での見え方は環境依存）", flush=True)
    except Exception as e:
        print(f"[theme] 注意Mica適用の呼び出し自体に失敗しました（続行します）: {e}",
              flush=True)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        t_sim, wall0, tick = 0.0, time.time(), 0
        while viewer.is_running() and win.isVisible():
            if restart[0]:
                pick = win.scene_combo.currentText()
                need_rebuild = bool(pick) and pick != scene.get("name")
                if need_rebuild:
                    win.msg_label.setText(
                        f"シーン「{pick}」で体を作り直します。"
                        "新しい窓が開いたら、この窓は閉じます")
                    app.processEvents()
                    envv = dict(os.environ)
                    envv["E_SCENE"] = pick
                    for k in _STALE_ENV_KEYS:
                        envv.pop(k, None)
                    import subprocess
                    flags = 0
                    if os.name == "nt":
                        flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
                    try:
                        subprocess.Popen(
                            [sys.executable, os.path.abspath(__file__)],
                            env=envv, cwd=os.getcwd(), creationflags=flags)
                    except Exception as e:
                        win.msg_label.setText(f"注意起動に失敗しました: {e}")
                        restart[0] = False
                        continue
                    try:
                        env.close()
                    except Exception:
                        pass
                    win.close()
                    return 0

                e_scene.reset_to_scene(
                    env, scene,
                    hands=(hands if win.chk_head_hold.isChecked() else None),
                    seed=0)
                if not win.chk_head_hold.isChecked():
                    hands.release()
                apply_fence(win.chk_fence.isChecked())
                toy_base[0] = None
                _syncing[0] = True
                for i in range(3):
                    win.toy_sliders[i].setValue(float(toy_pos0[i]), block_signal=True)
                _syncing[0] = False
                win.chk_toy_follow.setChecked(False)
                t_sim, wall0, tick = 0.0, time.time(), 0
                win.msg_label.setText(f"シーン「{scene['name']}」で始めました")
                restart[0] = False

            freeze = win.chk_freeze.isChecked()

            # ---- 反射のパラメータ（移植元：e_viewer.py 2463〜2466行目）--------
            OR.SACCADE_LATENCY = float(win.reflex_latency.value())
            OR.SACCADE_MIN_STRENGTH = float(win.reflex_threshold.value())
            u._orienting = reflex if win.chk_orient.isChecked() else None
            u._vor = vor if win.chk_vor.isChecked() else None
            if freeze and win.chk_orient.isChecked() and tick % 60 == 0:
                win.msg_label.setText("注意物理を止めています。反射は目を動かせません"
                                      "（『物理を止める』を外してください）")

            # ---- 首のバネ（移植元：e_viewer.py 2468〜2484行目）-----------------
            #   頭を抑えているあいだは適用しない。実験者の手も首のバネも
            #   同じ jnt_stiffness を使うため（落とし穴チェックリスト項62と同型）。
            if not win.chk_head_hold.isChecked():
                _on = win.chk_neck.isChecked()
                _neck_tgt_rad = np.radians(float(win.neck_t.value()))
                for nm, j in neck_ids.items():
                    if nm != "head_tilt" and not win.chk_neck_all.isChecked():
                        m.jnt_stiffness[j] = 0.0
                        m.dof_damping[int(m.jnt_dofadr[j])] = neck_damp0[nm]
                        continue
                    m.jnt_stiffness[j] = float(win.neck_k.value()) if _on else 0.0
                    m.qpos_spring[int(m.jnt_qposadr[j])] = _neck_tgt_rad if _on else 0.0
                    m.dof_damping[int(m.jnt_dofadr[j])] = (
                        float(win.neck_c.value()) if _on else neck_damp0[nm])

            # 環境のスイッチ（柵・実験者の手）を実行中でも反映する
            if bool(win.chk_fence.isChecked()) != fence_on[0]:
                apply_fence(win.chk_fence.isChecked())
            if bool(win.chk_head_hold.isChecked()) != bool(hands.holding):
                if win.chk_head_hold.isChecked():
                    hands.hold(target=_hold_tgt)
                else:
                    hands.release()

            # ---- おもちゃ（移植元：e_viewer.py 2522〜2569行目）------------------
            wob = np.array([0.0, SHAKE_AMP * np.sin(2 * np.pi * SHAKE_HZ * t_sim), 0.0]) \
                if win.chk_toy_shake.isChecked() else np.zeros(3)

            if _parent[0] is not None:
                if parent_log and parent_log[-1] is None:
                    parent_log[-1] = round(t_sim, 2)
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
                        win.toy_sliders[i2].setValue(float(_parent[0]["to"][i2]))
                    _syncing[0] = False
                    _parent[0] = None
            elif win.chk_toy_follow.isChecked():
                pos = np.array([s.value() for s in win.toy_sliders], dtype=float) + wob
                d.qpos[toy_qadr:toy_qadr + 3] = pos
                d.qvel[toy_dof:toy_dof + 6] = 0.0
                u._rest_pos = pos.copy()
            else:
                if toy_base[0] is None and not getattr(u, "_toy_pending", True) \
                        and getattr(u, "_rest_pos", None) is not None:
                    toy_base[0] = np.array(u._rest_pos, dtype=float)
                if toy_base[0] is not None:
                    pos = toy_base[0] + wob
                    u._rest_pos = pos.copy()
                    d.qpos[toy_qadr:toy_qadr + 3] = pos
                    d.qvel[toy_dof:toy_dof + 6] = 0.0
                    if tick % 20 == 0:
                        _syncing[0] = True
                        for i in range(3):
                            if abs(win.toy_sliders[i].value() - toy_base[0][i]) > 1e-4:
                                win.toy_sliders[i].setValue(float(toy_base[0][i]))
                        _syncing[0] = False
            s_val = float(win.toy_size_slider.value())
            if abs(float(m.geom_size[toy_gadr][0]) - s_val) > 1e-9:
                m.geom_size[toy_gadr] = [s_val, s_val, s_val]

            # ---- 姿勢の固定（移植元：e_viewer.py 2571〜2591行目）----------------
            if win.chk_pose_hold.isChecked():
                _neck_tgt = {}
                for jd in joints:
                    ang = np.radians(win.joint_sliders[jd["key"]].value())
                    d.qpos[jd["qadr"]] = ang
                    d.qvel[int(m.jnt_dofadr[jd["jid"]])] = 0.0
                    if jd.get("is_neck"):
                        _neck_tgt[jd["key"]] = float(win.joint_sliders[jd["key"]].value())
                if _neck_tgt and win.chk_head_hold.isChecked() and hands.holding:
                    hands.hold(target=_neck_tgt)
                    _hold_tgt = dict(_neck_tgt)

            if freeze and getattr(u, "_toy_pending", False):
                try:
                    u.place_toy_now()
                    toy_base[0] = np.array(u._rest_pos, dtype=float)
                    _syncing[0] = True
                    for i in range(3):
                        win.toy_sliders[i].setValue(float(u._rest_pos[i]))
                    _syncing[0] = False
                except Exception:
                    pass

            if freeze:
                d.qvel[:] = 0.0
                d.qacc[:] = 0.0
                mujoco.mj_forward(m, d)
            else:
                env.step(zero)
            t_sim += dt
            tick += 1

            if tick % 5 == 0:
                off = hands.offsets() if hands.holding else {}
                offmax = max((abs(v) for v in off.values()), default=0.0)
                win.env_label.setText(
                    f"体年齢 {_AGE:g}ヶ月（視力も同じ）  "
                    f"柵 {'あり' if fence_on[0] else 'なし'}  "
                    f"リクライニング {_RECLINE:.0f}度\n"
                    f"顎 {(_HOLD_TILT + '度で支える') if _HOLD_TILT else '指定なし'}"
                    f"　眼球の基準 {_EYE_REST_V:+.0f}度\n"
                    f"実験者の手 {'抑えている' if hands.holding else 'なし'}"
                    + (f"（頭のずれ {offmax:.2f}度）" if hands.holding else ""))

                # ---- 状態表示欄（仕様2-3節）--------------------------------
                #   compute_status() が毎回、実体（env・actuation_model・
                #   scene・TARO_DEFAULTS）から読み直す。ここでは結果を
                #   ラベルに反映するだけで、値を保持しない。
                st_lines = compute_status(
                    env=env, actuation_model=am, actuation_mode=_ACTUATION_MODE,
                    scene=scene, taro_defaults=TARO_DEFAULTS,
                    fmax_base=_fmax_base, age=_AGE, hands=hands)
                win.status_label.setText(format_status_text(st_lines))

                # ---- 輻輳（寄り目）の表示（移植元：e_viewer.py 2912〜2924行目）--
                try:
                    _eyes = [np.array(d.cam_xpos[int(m.camera(_n).id)], dtype=float)
                             for _n in ("eye_left", "eye_right")]
                    _org = np.mean(_eyes, axis=0)
                    _real = float(np.linalg.norm(
                        np.array(d.xpos[toy_bid], dtype=float) - _org))
                    _verg = 2.0 * np.degrees(np.arctan2(IPD / 2.0, max(_real, 1e-4)))
                    win.vergence_label.setText(
                        f"実際の距離 {_real*100:5.1f}cm（設定 {win.toy_dist_slider.value()*100:.1f}cm）\n"
                        f"両目で見るのに要る寄り目 {_verg:5.1f}度"
                        f"（瞳孔間 {IPD*100:.1f}cm）太郎は寄り目ができない")
                except Exception:
                    pass

                _pl = [x for x in parent_log if x is not None]
                if _pl:
                    gaps = [_pl[i] - _pl[i - 1] for i in range(1, len(_pl))]
                    avg = (sum(gaps) / len(gaps)) if gaps else float("nan")
                    win.parent_label.setText(
                        f"親の介入 {len(_pl)}回"
                        f"（平均 {avg:.1f}秒おき）"
                        f"{'　運搬中' if _parent[0] is not None else ''}")

                # ---- 首のバネの様子（移植元：e_viewer.py 2836〜2868行目）--------
                jt = neck_ids.get("head_tilt")
                if jt is not None:
                    ang = float(np.degrees(d.qpos[int(m.jnt_qposadr[jt])]))
                    vel = float(np.degrees(d.qvel[int(m.jnt_dofadr[jt])]))
                    neck_hist.append(ang)
                    if len(neck_hist) > 400:
                        neck_hist.pop(0)
                    k = max(float(win.neck_k.value()), 1e-9)
                    c_crit = 2.0 * np.sqrt(k * NECK_I)
                    c_now = float(win.neck_c.value())
                    ratio = c_now / c_crit if c_crit > 1e-12 else float("inf")
                    kind = ("振動する（減衰不足）" if ratio < 0.7
                            else "ちょうど良い" if ratio < 1.6 else "戻りが遅い（減衰過多）")
                    recent = np.array(neck_hist[-100:]) if len(neck_hist) > 20 else np.zeros(1)
                    swing = float(recent.max() - recent.min())
                    txt = (f"首の角度 {ang:7.2f}度（目標 {win.neck_t.value():5.1f}）  "
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
                    win.neck_label.setText(txt)

            app.processEvents()
            viewer.sync()
            lag = wall0 + t_sim - time.time()
            if lag > 0:
                time.sleep(min(lag, 0.05))

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
