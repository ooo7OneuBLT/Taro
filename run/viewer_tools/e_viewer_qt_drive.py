"""「駆動」タブ（PySide6版ビューア）— 第5段階：もがき運動・反射+共通駆動の移植。
                                    ＋ 2026-08-16：脳（学習済みモデル）読み込みの移植。

【これは何か】`e_viewer_qt.py`から呼ばれる「駆動」タブの本体。第4段階までは
プレースホルダのみだった。この段階で、旧版`run/viewer_tools/e_viewer.py`の
「再生」区画にあった駆動モード関連のUI・ロジック（もがき運動・反射+共通駆動・
四肢の筋力調整）を移植する。

【移植元（読み取り専用の参考。importしない）】
    run/viewer_tools/e_viewer.py
        1283〜1520行目   UI（自発運動チェック・駆動モード4択・
                        四肢の筋力調整・もがき運動パラメータ調整・
                        反射+共通駆動パラメータ調整）
        1789〜1938行目   CPG構築・_babble_noise()・反射+共通駆動の構築/保持/
                        パラメータ反映
        2227〜2295行目   駆動モードの切替（_set_noise_mode・_cycle_noise_mode・
                        Nキー）
        2614〜2690行目   メインループでの実際の分岐

【新版での違い（意図的な設計上の変更点）】
    1. 旧版はこの区画の状態（CPGインスタンス・反射+共通駆動のholder等）を
       `e_viewer.py`のmain()内のローカル変数（クロージャ）として持っていたが、
       新版はbuild_drive_tab（UI組み立てのみ・env/mに触れない）と
       drive_tick_action（毎tick呼ばれる）が別関数のため、状態は
       `win._drive_state`（辞書）としてwinインスタンスに持たせる
       （仕様3節の指定どおり）。
    2. 旧版はNキーをmujoco.viewer.launch_passiveのkey_callback
       （メインループとは別スレッドの可能性がある）で受けており、
       そのためtkinterウィジェットを直接触れず「旗を立ててメインループ側で
       処理する」という遠回りな設計になっていた（noise_label_dirty・
       _rc_pending_restore）。新版はQShortcut（Qtのメインスレッドで発火）を
       使うため、この種の間接化が丸ごと不要になった（ラジオボタンを直接
       .setChecked()するだけでtoggledシグナルが発火し、ラベル更新も
       自動的に行われる）。
    3. 旧版は「actuation=jointのときreflex_commonのラジオボタン自体を
       選べなくする」という2段防御の第一段（ラジオボタンの無効化）を
       build時点（=UI構築時点）で行っていたが、仕様3節により
       build_drive_tabはenv/mに触れられない（構築時点ではまだ体が
       どちらのactuationか分からない）。そのため今回は防御の第二段
       （drive_tick_action内での実行時ガード。旧版の_set_noise_modeの
       ガードと同じ）だけを実装する。仕様2節の前提どおり、今のViewerは
       常にactuation=muscleで起動されるため、実務上この差は表面化しない
       （作業記録に明記する）。

【設計・仕様】
    作業記録（非公開）

【2026-08-16 追記：脳（学習済みモデル）読み込みの移植】
    依頼の範囲は「①学習済みモデルを1つ読み込んで、その脳で太郎を動かす」だけ
    （脳A/B見比べ・GRU隠れ状態の使い分け・目標指向探索・dAction2表示は範囲外。
    依頼書「移植する／しない」の通り）。

    【移植元（読み取り専用）】
        run/viewer_tools/e_viewer.py
            1620〜1686行目   UI（「脳」区画。今回はA/B・目標指向のUIは作らない）
            2048〜2199行目   _load_brain（①1脳分だけ移植。E_TARO_SPEC_JSON・
                             食い違い時の駆動モード確認はそのまま踏襲）
            2228〜2291行目   _brain_action（目標指向探索を除いた最小限）
            2732〜2760行目   メインループでの分岐（「学習した脳が優先」）
        run/viewer.py（変更禁止・参考のみ）
            135〜169行目     policy_fn（rescale_action・explore・to_env_actionの
                             呼び出し順の参考）

    脳がONで読み込めているときは、生のenvではなくHybridEnv
    （内受容感覚4次元を足すラッパー、taro_core/src/wrapper/hybrid_env.py）で
    毎tick進める（旧版`_ensure_hybrid()`と同じ理由：太郎の観測に内受容感覚が
    無いと入力次元が合わない）。ONでない、または読み込めていないときは
    今までどおり駆動タブ（もがき運動等）の結果で生のenvを進める＝白紙の脳。

    注意（旧版と同じ既知の挙動、そのまま踏襲）：モデルを読み込んだ瞬間、
    `Taro.__init__`が`env.reset(seed=0)`を呼ぶため、スライダーで作った今の姿勢が
    既定姿勢へ飛ぶ（旧版e_viewer.py 1935〜1939行目のコメントに明記されている
    既知の副作用で、モデル読み込みという明示操作に対して受け入れられている）。
"""
from __future__ import annotations

import os

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from e_viewer_qt_widgets import FloatSlider, CollapsibleSection

# このファイルからのTaroリポジトリルート（run/viewer_tools/の2つ上）。
#   モデルパスが相対指定のときの解決に使う（旧版e_viewer.py _ROOTと同じ考え方）。
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      os.pardir, os.pardir))

# 駆動モードの一覧（移植元：e_viewer.py 1306行目 NOISE_MODES と同じ並び）。
NOISE_MODES = ["white", "colored", "colored+synergy", "reflex_common"]
_NOISE_MODE_LABELS = {
    "white": "白色", "colored": "色付き",
    "colored+synergy": "色付き+シナジー", "reflex_common": "反射+共通駆動"}

# 【なぜ200まで対数で振れるか】旧版1343〜1347行目とまったく同じ根拠
#   （0ヶ月児の四肢の筋力が弱すぎて腕が上がらない問題への対応）。
_FS_LOG_MAX = float(np.log10(200.0))

# もがき運動パラメータの既定値（旧版1392行目 _BAB_DEFAULTS と完全一致。
#   「何も操作しなければ挙動は1ビットも変わらない」ことをこの一致で保証する）。
_BAB_DEFAULTS = dict(std=0.174, cocon=0.5, syn_w=0.6, k=10, beta=0.7)

# 反射+共通駆動パラメータの既定値（k_sだけは spinal_cord.stretch_reflex.DEFAULT_K_S
#   を実行時にimportして使う。旧版1442〜1451行目と同じ理由＝値を手で書き写さない）。
_RC_DEFAULTS = dict(rho=0.3, f0=0.5, a0=1.0, grouping="per_limb")


def build_drive_tab(win, tab_widget):
    """タブへUIを組み立てる。envにもmにも触れない（この時点ではまだ使えない）。

    実際の駆動計算に必要な内部状態は drive_tick_action の中で遅延構築し、
    win._drive_state として保持する（3節の指定どおり）。
    """
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll_body = QtWidgets.QWidget()
    scroll.setWidget(scroll_body)
    outer = QtWidgets.QVBoxLayout(tab_widget)
    outer.addWidget(scroll)
    form_outer = QtWidgets.QVBoxLayout(scroll_body)

    # ================================================================
    # 区画：再生速度（移植元：e_viewer.py 1384〜1386行目・3200〜3204行目）
    # ================================================================
    # 【なぜ、2026-08-16】旧版削除にあたっての差分洗い出しで判明した欠落。
    #   新版はこれまで常に等倍速固定だった（メインループのsleep式に速度係数が
    #   無かった）。適用箇所はe_viewer_qt.pyのメインループ（sleep式を
    #   `wall0 + t_sim / speed - time.time()`に変更）。実装時の判断で追加
    #   （仕様に無い機能の復元。境界：このスライダー自体は見た目のみで
    #   ここでは配線しない）。
    box_speed = QtWidgets.QGroupBox("再生速度")
    form_outer.addWidget(box_speed)
    speed_form = QtWidgets.QVBoxLayout(box_speed)
    win.speed_slider = FloatSlider(
        "速度", 0.1, 2.0, 0.1, init=1.0,
        note="1.0＝等倍速。0.25＝4分の1のスロー。"
             "「物理を止める」中は効きません（旧版と同じ）")
    speed_form.addWidget(win.speed_slider)

    # ================================================================
    # 区画：脳（学習したモデル）（2026-08-16新設・移植元：e_viewer.py
    #   1620〜1686行目・2048〜2291行目・2732〜2760行目の①部分だけ）
    # ================================================================
    box_brain = QtWidgets.QGroupBox("脳（学習したモデル）")
    form_outer.addWidget(box_brain)
    brain_form = QtWidgets.QVBoxLayout(box_brain)
    brain_form.addWidget(QtWidgets.QLabel(
        "学習済みモデル(.pt)を読み込んで、その脳で太郎を動かします。"
        "ONのときは駆動モード（下）より優先されます。"))

    brain_path_row = QtWidgets.QHBoxLayout()
    win.brain_path_edit = QtWidgets.QLineEdit(os.environ.get("E_VIEW_MODEL", ""))
    win.brain_path_edit.setPlaceholderText("モデルのパス（例: E/logs/.../model_seed0.pt）")
    brain_path_row.addWidget(win.brain_path_edit)
    win.brain_browse_btn = QtWidgets.QPushButton("参照…")
    brain_path_row.addWidget(win.brain_browse_btn)
    brain_form.addLayout(brain_path_row)

    def _browse_brain_path():
        # 【なぜ環境変数に頼らないか】ユーザー方針（2026-08-07）「run系操作は
        #   編集ウィンドウで完結させる」に沿い、ファイル選択ダイアログで完結させる。
        start_dir = os.path.dirname(win.brain_path_edit.text().strip()) or _ROOT
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            box_brain, "学習済みモデルを選ぶ", start_dir,
            "モデル (*.pt);;すべてのファイル (*)")
        if path:
            win.brain_path_edit.setText(path)

    win.brain_browse_btn.clicked.connect(_browse_brain_path)

    win.brain_std_slider = FloatSlider(
        "探索の揺らぎ", 0.0, 0.5, 0.001,
        init=float(os.environ.get("E_VIEW_STD", "0.174")),
        note="0＝決定的（act_meanをそのまま使う）。既定0.174は学習初期の"
             "実効値相当（旧版と同じ既定値）")
    brain_form.addWidget(win.brain_std_slider)

    win.chk_brain = QtWidgets.QCheckBox("この脳で動かす")
    brain_form.addWidget(win.chk_brain)

    win.brain_label = QtWidgets.QLabel("（まだ読み込んでいません）")
    win.brain_label.setStyleSheet("color: #666;")
    win.brain_label.setWordWrap(True)
    brain_form.addWidget(win.brain_label)

    # ================================================================
    # 区画：駆動モード（移植元：e_viewer.py 1288〜1333行目）
    # ================================================================
    box = QtWidgets.QGroupBox("駆動")
    form_outer.addWidget(box)
    form = QtWidgets.QVBoxLayout(box)

    win.chk_babble = QtWidgets.QCheckBox("自発運動を流す（もがき運動）")
    form.addWidget(win.chk_babble)

    mode_row = QtWidgets.QHBoxLayout()
    mode_row.addWidget(QtWidgets.QLabel("駆動モード:"))
    win.drive_mode_group = QtWidgets.QButtonGroup(tab_widget)
    win.drive_mode_radios = {}
    for val in NOISE_MODES:
        rb = QtWidgets.QRadioButton(_NOISE_MODE_LABELS[val])
        mode_row.addWidget(rb)
        win.drive_mode_group.addButton(rb)
        win.drive_mode_radios[val] = rb
    win.drive_mode_radios["white"].setChecked(True)
    mode_row.addStretch(1)
    form.addLayout(mode_row)

    win.drive_mode_label = QtWidgets.QLabel(
        "駆動モード: white（Nキー、またはラジオボタンで切替）")
    win.drive_mode_label.setStyleSheet("font-family: Consolas; color: #666;")
    form.addWidget(win.drive_mode_label)

    def _on_mode_toggled(val, checked):
        if checked:
            win.drive_mode_label.setText(
                f"駆動モード: {val}（Nキー、またはラジオボタンで切替）")

    for val, rb in win.drive_mode_radios.items():
        rb.toggled.connect(lambda checked, v=val: _on_mode_toggled(v, checked))
        # 反射+共通駆動が選ばれ、または離れた場合に備え、グルーピング区画同様
        # win._drive_state の再構築判定へ反映させる（drive_tick_action側で
        # state["mode_prev"]との比較により検知するので、ここでは何もしない）。

    # ================================================================
    # 区画：四肢の筋力調整（移植元：e_viewer.py 1335〜1373行目）
    # ================================================================
    # 【実装担当の判断（仕様2節）】旧版のコメントどおり「もがき運動・脳・
    #   反射+共通駆動のどの駆動方式でも共通して効く身体側のパラメータ」なので
    #   この「駆動」タブに含めた。ユーザーの元の依頼一覧には明記されていない
    #   判断であり、作業記録に明記する（ユーザーが「実行タブの方がよい」と
    #   判断したら、あとで移すのは容易）。
    sec_fs = CollapsibleSection("四肢の筋力調整", opened=False)
    form.addWidget(sec_fs)
    win.fs_unavailable_label = QtWidgets.QLabel(
        "注意fmaxが定数（キャリブレーションファイル無し）、または\n"
        "actuation=joint のため、このスライダーは効きません")
    win.fs_unavailable_label.setStyleSheet("color: #a33;")
    win.fs_unavailable_label.setWordWrap(True)
    win.fs_unavailable_label.setVisible(False)
    sec_fs.body_layout.addWidget(win.fs_unavailable_label)
    win.fs_log_slider = FloatSlider(
        "筋力倍率(対数)", 0.0, _FS_LOG_MAX, 0.01, init=0.0,
        note="1.0〜200倍を対数目盛で調整（四肢のみ。首・体幹は変わりません）。"
             "既定1.0倍＝何もしなければ挙動は変わりません")
    sec_fs.body_layout.addWidget(win.fs_log_slider)
    win.fs_value_label = QtWidgets.QLabel("倍率 1.00x")
    win.fs_value_label.setStyleSheet("font-family: Consolas; color: #666;")
    sec_fs.body_layout.addWidget(win.fs_value_label)
    win.fs_reset_btn = QtWidgets.QPushButton("既定値に戻す(1.0倍)")
    sec_fs.body_layout.addWidget(win.fs_reset_btn)
    win.fs_reset_btn.clicked.connect(lambda: win.fs_log_slider.setValue(0.0))

    # ================================================================
    # 区画：もがき運動パラメータ調整（移植元：e_viewer.py 1375〜1420行目）
    # ================================================================
    sec_bab = CollapsibleSection("もがき運動パラメータ調整", opened=True)
    form.addWidget(sec_bab)
    win.bab_std_slider = FloatSlider(
        "①ゆらぎの大きさ", 0.0, 0.6, 0.01, init=_BAB_DEFAULTS["std"],
        note="act = 共収縮の下駄 + ①×ノイズ。既定0.174")
    sec_bab.body_layout.addWidget(win.bab_std_slider)
    win.bab_cocon_slider = FloatSlider(
        "②共収縮の下駄", 0.0, 1.0, 0.01, init=_BAB_DEFAULTS["cocon"],
        note="拮抗筋の同時収縮量。既定0.5")
    sec_bab.body_layout.addWidget(win.bab_cocon_slider)
    win.bab_synw_slider = FloatSlider(
        "③シナジーの重み", 0.0, 1.0, 0.01, init=_BAB_DEFAULTS["syn_w"],
        note="駆動モード「色付き+シナジー」のときだけ効く。既定0.6")
    sec_bab.body_layout.addWidget(win.bab_synw_slider)
    win.bab_k_slider = FloatSlider(
        "④保持の長さK", 1, 40, 1, init=_BAB_DEFAULTS["k"],
        note="この物理ステップ数ごとに1回、新しい行動を選び直す。既定10")
    sec_bab.body_layout.addWidget(win.bab_k_slider)
    win.bab_beta_slider = FloatSlider(
        "⑤ノイズの色 beta", 0.0, 2.0, 0.01, init=_BAB_DEFAULTS["beta"],
        note="駆動モード「色付き」「色付き+シナジー」のときだけ効く。既定0.7")
    sec_bab.body_layout.addWidget(win.bab_beta_slider)

    def _bab_reset_defaults():
        win.bab_std_slider.setValue(_BAB_DEFAULTS["std"])
        win.bab_cocon_slider.setValue(_BAB_DEFAULTS["cocon"])
        win.bab_synw_slider.setValue(_BAB_DEFAULTS["syn_w"])
        win.bab_k_slider.setValue(_BAB_DEFAULTS["k"])
        win.bab_beta_slider.setValue(_BAB_DEFAULTS["beta"])

    win.bab_reset_btn = QtWidgets.QPushButton("既定値に戻す")
    sec_bab.body_layout.addWidget(win.bab_reset_btn)
    win.bab_reset_btn.clicked.connect(_bab_reset_defaults)

    # ================================================================
    # 区画：反射+共通駆動パラメータ調整（移植元：e_viewer.py 1422〜1513行目）
    # ================================================================
    sec_rc = CollapsibleSection("反射+共通駆動パラメータ調整", opened=True)
    form.addWidget(sec_rc)
    win.rc_unavailable_label = QtWidgets.QLabel(
        "注意actuation=muscle のときだけ使えます。実行時に判定します")
    win.rc_unavailable_label.setStyleSheet("color: #a33;")
    win.rc_unavailable_label.setWordWrap(True)
    win.rc_unavailable_label.setVisible(False)
    sec_rc.body_layout.addWidget(win.rc_unavailable_label)
    win.rc_rho_slider = FloatSlider(
        "ρ（共通の度合い）", 0.0, 1.0, 0.01, init=_RC_DEFAULTS["rho"],
        note="0=各関節が独立に揺れる／1=グループ内が同じ揺れを共有。既定0.3")
    sec_rc.body_layout.addWidget(win.rc_rho_slider)
    win.rc_f0_slider = FloatSlider(
        "振動子の周波数 f0[Hz]", 0.1, 2.0, 0.01, init=_RC_DEFAULTS["f0"],
        note="揺らぎの中心周波数。既定0.5Hz")
    sec_rc.body_layout.addWidget(win.rc_f0_slider)
    win.rc_a0_slider = FloatSlider(
        "振動子の振幅 A0", 0.0, 2.0, 0.01, init=_RC_DEFAULTS["a0"],
        note="揺らぎの中心振幅（無次元）。既定1.0")
    sec_rc.body_layout.addWidget(win.rc_a0_slider)
    # k_sの既定値表示は実行時（drive_tick_action初回）にDEFAULT_K_Sを読んでから
    # ラベルを更新する（build時点ではspinal_cord側の値をimportしない方針
    # ＝env/mに触れない、という制約とは無関係だが、他のスライダーと import
    # タイミングを揃えるため遅延させる）。ここでは暫定的に5.0（既知の値）を
    # 初期値として置く。
    win.rc_gain_slider = FloatSlider(
        "伸張反射のゲイン k_s", 0.0, 20.0, 0.1, init=5.0,
        note="筋が伸びたときの反射の強さ。既定はspinal_cord/stretch_reflex.py"
             "のDEFAULT_K_S（実行時に確定します）")
    sec_rc.body_layout.addWidget(win.rc_gain_slider)

    grouping_row = QtWidgets.QHBoxLayout()
    grouping_row.addWidget(QtWidgets.QLabel("グルーピング:"))
    win.rc_grouping_group = QtWidgets.QButtonGroup(tab_widget)
    win.rc_grouping_radios = {}
    for val, label in (("none", "なし"), ("per_limb", "四肢ごと"),
                        ("whole_body", "全身")):
        rb = QtWidgets.QRadioButton(label)
        grouping_row.addWidget(rb)
        win.rc_grouping_group.addButton(rb)
        win.rc_grouping_radios[val] = rb
    win.rc_grouping_radios[_RC_DEFAULTS["grouping"]].setChecked(True)
    grouping_row.addStretch(1)
    sec_rc.body_layout.addLayout(grouping_row)
    grouping_note = QtWidgets.QLabel(
        "none にすると各関節が1要素グループになり、ρは実質効かなくなります")
    grouping_note.setStyleSheet("color: #666;")
    sec_rc.body_layout.addWidget(grouping_note)

    def _on_grouping_changed(checked):
        if not checked:
            return
        # 【なぜ、旧版1475〜1486行目と同じ】グルーピングは関節indexのグループ
        # 分けそのもの（トポロジー）を変えるため、rho/f0/A0/k_sのような
        # その場書き換えでは対応できない。次回の drive_tick_action で
        # 作り直させるため、いま持っているholderを捨てるだけにする。
        st = getattr(win, "_drive_state", None)
        if st is not None:
            st["reflex_holder"] = None

    for rb in win.rc_grouping_radios.values():
        rb.toggled.connect(_on_grouping_changed)

    win.rc_joint_label = QtWidgets.QLabel("対象関節数: まだ計算していません（駆動を開始すると実測します）")
    win.rc_joint_label.setStyleSheet("font-family: Consolas; color: #666;")
    sec_rc.body_layout.addWidget(win.rc_joint_label)

    def _rc_reset_defaults():
        win.rc_rho_slider.setValue(_RC_DEFAULTS["rho"])
        win.rc_f0_slider.setValue(_RC_DEFAULTS["f0"])
        win.rc_a0_slider.setValue(_RC_DEFAULTS["a0"])
        st = getattr(win, "_drive_state", None)
        default_k_s = st.get("default_k_s") if st is not None else None
        win.rc_gain_slider.setValue(float(default_k_s) if default_k_s is not None else 5.0)
        win.rc_grouping_radios[_RC_DEFAULTS["grouping"]].setChecked(True)

    win.rc_reset_btn = QtWidgets.QPushButton("既定値に戻す")
    sec_rc.body_layout.addWidget(win.rc_reset_btn)
    win.rc_reset_btn.clicked.connect(_rc_reset_defaults)

    form_outer.addStretch(1)

    # ================================================================
    # Nキーでの駆動モード循環（移植元：e_viewer.py 2268〜2291行目の役割）
    # ================================================================
    # 【なぜQShortcutか、仕様3節】mujoco.viewer.launch_passiveの
    #   key_callbackには頼らない（e_viewer_qt.pyを変更せずに実現できるため。
    #   かつ上のdocstringに書いたとおりQtのメインスレッドで安全に発火する）。
    def _cycle_mode():
        cur = _current_mode(win)
        i = NOISE_MODES.index(cur) if cur in NOISE_MODES else -1
        nxt = NOISE_MODES[(i + 1) % len(NOISE_MODES)]
        win.drive_mode_radios[nxt].setChecked(True)

    win._drive_shortcut = QtGui.QShortcut(QtGui.QKeySequence("N"), tab_widget)
    win._drive_shortcut.setContext(QtCore.Qt.ApplicationShortcut)
    win._drive_shortcut.activated.connect(_cycle_mode)


def _current_mode(win):
    for val, rb in win.drive_mode_radios.items():
        if rb.isChecked():
            return val
    return "white"


def _ensure_state(win, env, m, u, zero):
    """内部状態（CPG・反射+共通駆動のholder等）を遅延構築する。

    【なぜここで、drive_tick_actionの中でだけ】build_drive_tabの時点では
    envもmも使えない（仕様3節）。実際に使えるようになる最初のtickで
    1回だけ構築し、以後は win._drive_state に保持して使い回す。
    """
    if getattr(win, "_drive_state", None) is not None:
        return win._drive_state

    am = u.actuation_model
    fmax_base = None
    if hasattr(am, "fmax"):
        f0 = np.asarray(am.fmax, dtype=float)
        if f0.ndim > 0:
            fmax_base = f0.copy()
    fs_available = fmax_base is not None
    # 対象＝四肢のみ（移植元：e_viewer.py 394〜403行目と同じ判定式。
    #   taro_core/src/body/infant_limbs.py の apply_limb_inversion_fix が
    #   使う判定をそのまま踏襲する。検証の落とし穴チェックリスト項30
    #   「同じ式は複数箇所に置かない」に沿い、除外条件だけをそのまま複製）。
    limb_aids = [i for i in range(m.nu)
                 if not (m.actuator(i).name or "").startswith(("act:head", "act:chest"))]
    fs_n_act = int(getattr(am, "n_actuators", (len(fmax_base) // 2) if fmax_base is not None else 0))

    # actuation=muscle かどうかの実行時判定（旧版394〜403行目のコメントと同じ
    # 事実に基づく：actuation=jointのSpringDamperModelにはmuscle_lengths・
    # moment_1が無い）。
    muscle_ok = hasattr(am, "muscle_lengths") and hasattr(am, "moment_1")

    n_joints = None
    if muscle_ok:
        try:
            from run.taro_setup import _reflex_common_joint_indices
            idx = _reflex_common_joint_indices(env)
            n_joints = sum(len(v) for v in idx.values())
            win.rc_joint_label.setText(f"対象関節数: {n_joints}（両腕・両脚の合計・実測）")
        except Exception as e:
            win.rc_joint_label.setText(f"対象関節数: 計算できません（{e}）")
    else:
        win.rc_joint_label.setText("対象関節数: actuation=muscleでないため計算できません")

    default_k_s = 5.0
    try:
        from spinal_cord.stretch_reflex import DEFAULT_K_S
        default_k_s = float(DEFAULT_K_S)
        win.rc_gain_slider.setValue(default_k_s)
    except Exception:
        pass

    if not fs_available:
        win.fs_unavailable_label.setVisible(True)
        win.fs_log_slider.setEnabled(False)
        win.fs_reset_btn.setEnabled(False)
    if not muscle_ok:
        win.rc_unavailable_label.setVisible(True)
        win.rc_unavailable_label.setText(
            "注意actuation=muscle のときだけ使えます（今の体は対象外）。"
            "ラジオボタンで「反射+共通駆動」を選んでも動きません")

    win._drive_state = dict(
        act=np.asarray(zero, dtype=np.float32).copy(),
        mode_prev="white",
        babble_cpg=None,
        white_rng=np.random.default_rng(0),
        reflex_holder=None,
        rc_stiffness_saved=None,
        fs_available=fs_available,
        fmax_base=fmax_base,
        limb_aids=limb_aids,
        fs_n_act=fs_n_act,
        muscle_ok=muscle_ok,
        n_joints=n_joints,
        default_k_s=default_k_s,
        warned_reflex=False,
        active=False,
        mode="white",
    )
    return win._drive_state


def _apply_limb_strength(win, state, u, tick):
    """四肢の筋力倍率スライダーの現在値を、実体（actuation_model.fmax）へ反映する。

    毎tick「基準値×スライダーの現在値」を計算し直して代入する（累積しない。
    移植元：e_viewer.py 2494〜2511行目と同じ方式。スライダーを1.0へ戻せば
    fmax_base とビット同一に戻る＝10.0**0.0 == 1.0 が浮動小数で厳密に一致するため）。
    """
    if not state["fs_available"]:
        return
    am = u.actuation_model
    x = 10.0 ** float(win.fs_log_slider.value())
    fnew = state["fmax_base"].copy()
    n_act = state["fs_n_act"]
    for aid in state["limb_aids"]:
        fnew[aid] *= x
        fnew[aid + n_act] *= x
    am.set_fmax(fnew)
    if tick % 15 == 0:
        base_mean = float(np.mean(state["fmax_base"][state["limb_aids"]]))
        cur_mean = float(np.mean(fnew[state["limb_aids"]]))
        win.fs_value_label.setText(
            f"倍率 {x:.2f}x  実効fmax(四肢の平均) {cur_mean:.3f}（基準 {base_mean:.3f}）")


def _rc_restore_stiffness(state, m):
    """反射+共通駆動から離れるとき、disable_limb_tone_springが0にした
    jnt_stiffnessを元へ戻す（移植元：e_viewer.py 1901〜1915行目と同じ）。
    """
    sv = state["rc_stiffness_saved"]
    if sv:
        for j, k in sv.items():
            m.jnt_stiffness[j] = k
    state["rc_stiffness_saved"] = None
    # 次に反射+共通駆動へ戻ったとき disable_limb_tone_spring を確実に
    #   再度掛け直させるため、組み立て済みのholderも捨てる（旧版と同じ理由）。
    state["reflex_holder"] = None


def _build_reflex_common(win, state, env, m, u, n_act):
    """反射+共通駆動一式（TaroBrainWithMotor＋反射+共通駆動の配線）を組み立てる。

    移植元：e_viewer.py 1855〜1899行目（_build_reflex_common）。
    env.reset()を一切呼ばない経路（run.taro_setup._setup_reflex_common・
    TaroBrainWithMotor）だけを使う点も同じ（Viewerの今の姿勢を壊さないため）。
    """
    from run.config import Config
    from run.taro_setup import _setup_reflex_common
    from taro_brain_motor import TaroBrainWithMotor
    from infant_limbs import limb_tone_joints_by_group

    osc_params = {"f0": float(win.rc_f0_slider.value()), "A0": float(win.rc_a0_slider.value()),
                  "tau_f": 2.0, "tau_A": 2.0, "amp": 1.0}
    grouping = "per_limb"
    for val, rb in win.rc_grouping_radios.items():
        if rb.isChecked():
            grouping = val
            break
    taro_spec = {
        "actuation": "muscle",
        "spinal_drive_mode": "reflex_common",
        "common_drive_rho": float(win.rc_rho_slider.value()),
        "common_drive_grouping": grouping,
        "common_drive_osc_params": osc_params,
        "noise": "white",   # spinal_drive_mode=reflex_common は noise=colored と
                             # 同時指定不可（run/config.py のバリデーション）
    }
    cfg = Config(taro_spec, {"seed": 0, "K": 10}, scene=None, name="viewer_qt_reflex_common")

    class _ReflexCommonHolder:
        pass

    holder = _ReflexCommonHolder()
    holder.brain = TaroBrainWithMotor(vocab_size=3, sensory_dim=1,
                                       n_actuators=n_act, proprio_dim=1)
    holder.seed = 0

    if state["rc_stiffness_saved"] is None:
        pairs = dict(limb_tone_joints_by_group(m, groups=("arm", "leg")))
        sv = {}
        for j in range(m.njnt):
            nm = (m.joint(j).name or "").split(":")[-1]
            if nm in pairs:
                sv[j] = float(m.jnt_stiffness[j])
        state["rc_stiffness_saved"] = sv

    ok = _setup_reflex_common(holder, cfg, env, verbose=True)
    if not ok:
        return None
    return holder


def _ensure_reflex_common(win, state, env, m, u, n_act):
    if not state["muscle_ok"]:
        return None
    if state["reflex_holder"] is None:
        state["reflex_holder"] = _build_reflex_common(win, state, env, m, u, n_act)
    return state["reflex_holder"]


def _apply_reflex_common_params(win, holder):
    """rho/f0/A0/k_sのスライダーの現在値を、既存オブジェクトへその場で書き込む
    （作り直さない。移植元：e_viewer.py 1926〜1938行目と同じ）。
    """
    if holder is None:
        return
    holder.brain.reflex_common.k_s = float(win.rc_gain_slider.value())
    f0 = float(win.rc_f0_slider.value())
    a0 = float(win.rc_a0_slider.value())
    rho = float(win.rc_rho_slider.value())
    for idxs, group in holder.brain._common_drive_groups.values():
        group.rho = rho
        for osc in [group.common] + list(group.indep.values()):
            osc.f0, osc.A0 = f0, a0


def _babble_noise(win, state, mode, n_act):
    """もがき運動の1tick分の探索ノイズ（移植元：e_viewer.py 1801〜1820行目）。"""
    if mode == "white":
        return state["white_rng"].standard_normal(n_act)
    if state["babble_cpg"] is None:
        from spinal_cord.cpg import CPG
        from run.taro_setup import LEG_R, LEG_L, ARM_R, ARM_L
        pair_offset = (n_act // 2) if n_act > 90 else 0
        state["babble_cpg"] = CPG(n_act, leg_r=LEG_R, leg_l=LEG_L, arm_r=ARM_R,
                                   arm_l=ARM_L, seed=0, pair_offset=pair_offset)
    return state["babble_cpg"].sample(
        float(win.bab_beta_slider.value()),
        synergy=(mode == "colored+synergy"),
        syn_w=float(win.bab_synw_slider.value()))


def drive_tick_action(win, *, env, m, d, u, zero, tick, dt, t_sim, freeze):
    """毎tick、駆動する行動を決める。

    戻り値：今tickに送る行動（np.ndarray、shape=(n_act,)）、または
    None（＝呼び出し側は zero を使う。自発運動オフのときの挙動）。

    freeze=True のときは e_viewer_qt.py 側が既にこの関数を呼ばない
    （main()の`if freeze: ... else: drive_tick_action(...)`分岐）。
    念のためここでも早期リターンする（この関数だけを直接使う将来のコードに
    対する保険。実害は無い）。
    """
    if freeze:
        return None

    n_act = int(zero.shape[0])
    state = _ensure_state(win, env, m, u, zero)

    # 四肢の筋力倍率は駆動モード・自発運動のON/OFFに関係なく毎tick効く
    # （旧版と同じ。「もがき運動・脳・反射+共通駆動のどの駆動方式でも共通して
    #   効く身体側のパラメータ」という位置づけのため）。
    _apply_limb_strength(win, state, u, tick)

    mode = _current_mode(win)
    if mode != state["mode_prev"]:
        if state["mode_prev"] == "reflex_common" and mode != "reflex_common":
            _rc_restore_stiffness(state, m)
        state["mode_prev"] = mode
    state["mode"] = mode
    state["active"] = bool(win.chk_babble.isChecked())

    if not state["active"]:
        return None

    if mode == "reflex_common":
        if not state["muscle_ok"]:
            if not state["warned_reflex"]:
                print("注意[駆動] 反射+共通駆動は actuation=muscle のときだけ選べます"
                      "（今の体は対象外）。ゼロ入力のままにします", flush=True)
                state["warned_reflex"] = True
            return None
        holder = _ensure_reflex_common(win, state, env, m, u, n_act)
        if holder is None:
            return None
        _apply_reflex_common_params(win, holder)
        am = u.actuation_model
        # 【なぜ間引かないか、旧版2644〜2652行目と同じ】反射+共通駆動は
        #   moment_1/moment_2経由で基準長を動かし伸張反射を計算する経路で、
        #   run/trainer.pyのreflex_common_active分岐と同じく毎tick計算し直す。
        #   保持の長さK（bab_k_slider）はこの経路には適用しない。
        r = holder.brain.step_reflex_common(dt, am.muscle_lengths, am.muscle_velocities)
        cocon = float(win.bab_cocon_slider.value())
        act = np.clip(cocon + r, 0.0, 1.0).astype(np.float32)
        state["act"] = act
        return act

    # ---- white / colored / colored+synergy（旧版2668〜2689行目）--------------
    k = max(1, int(round(win.bab_k_slider.value())))
    if tick % k == 0:
        noise = _babble_noise(win, state, mode, n_act)
        cocon = float(win.bab_cocon_slider.value())
        std = float(win.bab_std_slider.value())
        act = np.clip(cocon + std * noise, 0.0, 1.0).astype(np.float32)
        state["act"] = act
    return state["act"]


# ============================================================================
# 脳（学習したモデル）2026-08-16新設。移植元：e_viewer.py 2048〜2291行目・
# 2732〜2760行目のうち①（1脳のみ・見比べ無し・目標指向探索無し）だけ。
# ============================================================================

def _ensure_brain_state(win):
    """脳まわりの内部状態（HybridEnv・Taro一式）を遅延構築する。

    駆動タブの`win._drive_state`と同じ考え方（build_drive_tab はenv/mに
    触れられないため、実際に使えるようになるtick時点で1回だけ作る）。
    """
    if getattr(win, "_brain_state", None) is None:
        win._brain_state = dict(
            hybrid_env=None,   # HybridEnv（内受容感覚つきの環境）。脳を使うときだけ作る
            taro=None,         # run.taro_setup.Taro 一式。読み込み成功時のみ非None
            state=None,        # Taro.init_state() の戻り値（obs/hidden/prev_a）
            path=None,         # 直近に読み込みを試みたパス（変わったら読み直す）
            last_act=None,     # 直近に決めた行動（10tickごとに更新、間は保持）
            da2=[],            # 行動の変化量（学習ログのda2と同じ量。表示用）
        )
    return win._brain_state


def _load_brain_model(win, st, env, m, u, path, age_months, scene_name):
    """モデルを読んで太郎一式を作る。失敗してもViewerは落とさない。

    移植元：e_viewer.py 2078〜2199行目（_load_brain）のうち、脳Bの分岐・
    E_TARO_SPEC_JSONからの土台展開を除いた最小限（今回はViewerの実験ファイル
    経由の起動を前提にしないため。旧版で E_TARO_SPEC_JSON が無い場合の
    経路＝空辞書のまま、と同じ挙動になる）。
    """
    p = path if os.path.isabs(path) else os.path.join(_ROOT, path)
    if not os.path.exists(p):
        win.brain_label.setText(f"注意見つかりません: {path}")
        win.brain_label.setStyleSheet("color: #a33;")
        st["taro"] = None
        st["path"] = path
        return
    try:
        import torch as _torch
        from run.config import Config, touch_setting_of
        from run.taro_setup import Taro

        if st["hybrid_env"] is None:
            from hybrid_env import HybridEnv
            st["hybrid_env"] = HybridEnv(env)

        # 駆動モードの食い違い確認（移植元：e_viewer.py 2140〜2174行目と同じ
        # 考え方。今のViewerは常にactuation=muscleで起動される、という
        # build_drive_tab冒頭の前提に沿い、比較対象は"muscle"固定）。
        _actuation = "muscle"
        try:
            _blob = _torch.load(p, map_location="cpu", weights_only=False)
            _ckpt_cfg = _blob.get("config") or {}
            _ckpt_act = _ckpt_cfg.get("actuation")
        except Exception as _e:      # noqa: BLE001
            _ckpt_act = None
            print(f"注意[脳] 保存された駆動モードを読み取れません: "
                  f"{type(_e).__name__}: {_e}。この確認はスキップします", flush=True)
        if _ckpt_act is not None and str(_ckpt_act).lower() != _actuation:
            _msg = (f"駆動モードが食い違います（今の体={_actuation} / "
                    f"このモデルの学習時={_ckpt_act}）。層の形が合わないため"
                    "読み込みを中止します")
            win.brain_label.setText(f"注意{_msg}")
            win.brain_label.setStyleSheet("color: #a33;")
            print(f"注意[脳] {_msg}", flush=True)
            st["taro"] = None
            st["path"] = path
            return

        taro_spec = dict(touch_setting_of(p))
        taro_spec.update({"actuation": _actuation, "age_months": age_months, "model": p})
        cfg = Config(taro_spec, {"seed": 0, "K": 10}, scene=scene_name, name="viewer_qt_brain")
        # 【注意、旧版1935〜1939行目と同じ既知の挙動】Taro.__init__は必ず
        #   env.reset(seed=self.seed)を呼ぶ（run/taro_setup.py既存仕様）ため、
        #   スライダーで作った今の姿勢が既定姿勢へ飛ぶ。これはモデル読み込みという
        #   明示的な操作の結果として旧版でも受け入れられている副作用であり、
        #   ここでは避けない（避けるにはtaro_core側の設計変更が要る＝範囲外）。
        t = Taro(cfg, st["hybrid_env"], seed=0, verbose=True)
        st["state"] = t.init_state(t.first_obs)
        st["taro"] = t
        st["path"] = path
        st["last_act"] = np.zeros(t.n_act, dtype=np.float32)
        st["da2"].clear()
        win.brain_label.setText(
            f"読み込みました: {os.path.basename(path)}"
            "（注意：読み込んだ瞬間に体が既定姿勢へ戻ります）")
        win.brain_label.setStyleSheet("color: #0a7;")
        print(f"[脳] 読み込みました: {p}", flush=True)
    except Exception as e:      # noqa: BLE001
        win.brain_label.setText(f"注意読めません: {type(e).__name__}: {e}")
        win.brain_label.setStyleSheet("color: #a33;")
        print(f"注意[脳] 読めません: {type(e).__name__}: {e}", flush=True)
        st["taro"] = None
        st["path"] = path


def _brain_step(win, st):
    """いま読み込んでいる脳に、次の行動を決めてもらう（環境のaction_space単位）。

    移植元：e_viewer.py 2248〜2291行目（_brain_action）のうち、目標指向探索
    （gbuf・infer_goal_action）を除いた最小限（依頼の範囲③・④は移植しない）。
    """
    import torch
    from run.taro_setup import rescale_action

    t, stt = st["taro"], st["state"]
    # 注意：`torch.no_grad()`で囲まない（太郎の潜在推論は内部でautograd.gradを
    #   使うため。旧版2256〜2259行目、run/viewer.pyのpolicy_fnと同じ理由）。
    sv = t.fusion.encode(stt["obs"])
    cf = t.target_fusion.encode(stt["obs"]).detach()
    z, _kl, _rc, hn = t.infer_latent(sv, stt["prev_a"], cf, stt["hidden"])
    z = z.detach()
    mean = t.act_mean(z)
    std = float(win.brain_std_slider.value())
    if std > 0:
        a, _lp = t.brain.explore(mean, torch.full_like(mean, std))
        a = a.detach()
    else:
        a = torch.clamp(mean, -1.0, 1.0).detach()
    st["da2"].append(float(((a - stt["prev_a"]) ** 2).mean()))
    if len(st["da2"]) > 200:
        st["da2"].pop(0)
    stt["hidden"], stt["prev_a"] = hn.detach(), a
    a_env = t.brain.to_env_action(a)      # 拮抗筋モードなら筋活性化へ写す
    return rescale_action(a_env, st["hybrid_env"].action_space).astype(np.float32)


def brain_or_env_step(win, *, env, m, u, tick, freeze, age_months, scene_name, fallback_act):
    """1tick分、環境を進める。

    脳（学習したモデル）がONで読み込めていれば、その行動でHybridEnv
    （内受容感覚つきの環境）を進める。ONでない、または読み込めていなければ、
    今までどおり生のenvをfallback_act（駆動タブ＝もがき運動等の結果）で
    進める＝白紙の脳（依頼書「読み込んでいないときは今までどおりの動作」）。

    移植元：e_viewer.py 2732〜2760行目（「学習した脳が優先」の分岐）。
    """
    if freeze:
        return
    chk = getattr(win, "chk_brain", None)
    if chk is not None and chk.isChecked():
        st = _ensure_brain_state(win)
        path = win.brain_path_edit.text().strip()
        if path and st.get("path") != path:
            _load_brain_model(win, st, env, m, u, path, age_months, scene_name)
        if st.get("taro") is not None:
            if tick % 10 == 0:          # 1判断＝10物理ステップ（K=10、旧版と同じ）
                st["last_act"] = _brain_step(win, st)
            obs, _r, _te, _tr, _in = st["hybrid_env"].step(st["last_act"])
            if st.get("state") is not None:
                st["state"]["obs"] = obs
            if tick % 50 == 0:
                recent = st["da2"][-50:]
                avg = sum(recent) / len(recent) if recent else 0.0
                win.brain_label.setText(
                    f"いま動かしています: {os.path.basename(path)}   "
                    f"dAction2 avg={avg:.3f} (n={len(recent)})")
                win.brain_label.setStyleSheet("color: #0a7;")
            return
        # 注意：読めていないのに黙って動かない、を避ける（旧版2742〜2749行目と
        #   同じ理由。チェックを入れても何も起きず理由も出ないと原因が分からない）。
        if tick % 500 == 0:
            _reason = "パスが空です" if not path else "読み込めていません"
            win.brain_label.setText(f"注意脳が{_reason}ので、白紙の脳のまま動かします")
            win.brain_label.setStyleSheet("color: #a33;")
    env.step(fallback_act)
