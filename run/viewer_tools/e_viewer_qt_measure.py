"""「測定器」タブ（PySide6版ビューア）— 第5段階：本移植。

【これは何か】`e_viewer_qt.py`から呼ばれる「測定器」タブの実装。
移植元は`run/viewer_tools/e_viewer.py`の区画4「測定器」（1156〜1185行目、
UI）と、毎tickの計算ロジック（2762〜2834行目・2951〜2968行目）。

【入っているもの】
    ・左目・右目の画像表示（各176x176、検出画素を緑で重ねるON/OFFの
      チェックボックス付き）
    ・判定ラベル（①角度②光線の遮蔽③画素数のいずれか）
    ・距離ラベル（目→おもちゃ・肩→おもちゃ・手→おもちゃ、腕の長さに対する%）
    ・めり込みラベル（おもちゃと他の部位の接触の深さ、上位3件）
    ・頭の角速度ラベル（平均・最大、物理を止めたまま「この角度で固定する」が
      ONのままだと暴れる、という警告つき）

【触っていない範囲】旧版の「見えていた割合」「サッケード発生回数」等
（run_label相当）・輻輳（vergence）表示は、仕様2節の移植範囲に含まれて
いないため、このファイルには含めていない（他タブ・別段階の担当）。

【時間が止まるとキャッシュが切れない罠（検証の落とし穴チェックリスト項53）】
`u.get_vision_obs()`は内部で時刻をキーにした簡易キャッシュを持つ。Viewerで
物理を止めた（freeze）まま使うと、時刻が進まないためキャッシュが永久に
切れず、最初の1枚のまま固まる。旧版と同じく、呼ぶ直前に必ず
`u._vision_t = None; u._vision_cache = None`を実行してから呼ぶ。

【body idのキャッシュについて（仕様3節）】
`m`（MuJoCoモデル）は、`e_viewer_qt.py`の`main()`内で1回だけ構築され、
以後同じプロセス内で作り直されることが無い（シーンを切り替える操作は
`subprocess.Popen`で別プロセスを起動して自分は終了する設計。ノウハウ.md
2026-08-13の記載、および`e_viewer_qt.py`のE_SCENE経由の再起動処理で確認済み）。
`build_measure_tab`が呼ばれるタイミング＝`MainWindow.__init__`も1プロセスに
つき1回だけなので、`update_measure_tab`の初回呼び出し時にbody idを
`win._measure_ids`へキャッシュしても、古いモデルを指したままになる心配はない。

【設計・仕様】
    作業記録（非公開）
"""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

import e_visibility as VIS

# 肩から手先までの長さ[m]（移植元：e_viewer.py 80行目のARM_REACHをそのまま複製。
#   複製している理由：e_viewer.py側は触らない指示のため、共通化はできない）
ARM_REACH = 0.186

_MONO = "font-family: Consolas;"


def build_measure_tab(win, tab_widget):
    """タブへUIを組み立てる（移植元：e_viewer.py 1156〜1185行目）。"""
    lay = QtWidgets.QVBoxLayout(tab_widget)

    title = QtWidgets.QLabel("太郎の目に映っているもの（検出画素は緑）")
    title.setStyleSheet("font-weight: bold;")
    lay.addWidget(title)

    eye_row = QtWidgets.QHBoxLayout()
    win._measure_eye_labels = {}
    for side, jp in (("eye_left", "左目"), ("eye_right", "右目")):
        col = QtWidgets.QVBoxLayout()
        side_title = QtWidgets.QLabel(jp)
        side_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        col.addWidget(side_title)
        img_label = QtWidgets.QLabel()
        img_label.setFixedSize(176, 176)
        # 【なぜ背景色を付けるか】画像が一度も来ていない状態（起動直後）でも
        #   「何も無い」ことが分かるように、黒でも白でもない色にしておく。
        img_label.setStyleSheet("background-color: #333; border: 1px solid #555;")
        col.addWidget(img_label)
        eye_row.addLayout(col)
        win._measure_eye_labels[side] = img_label
    lay.addLayout(eye_row)

    win.chk_measure_mask = QtWidgets.QCheckBox("検出した画素を緑で重ねる")
    win.chk_measure_mask.setChecked(True)
    lay.addWidget(win.chk_measure_mask)

    win.measure_judge_label = QtWidgets.QLabel("")
    win.measure_judge_label.setStyleSheet(_MONO)
    win.measure_judge_label.setWordWrap(True)
    lay.addWidget(win.measure_judge_label)

    win.measure_dist_label = QtWidgets.QLabel("")
    win.measure_dist_label.setStyleSheet(_MONO)
    win.measure_dist_label.setWordWrap(True)
    lay.addWidget(win.measure_dist_label)

    win.measure_pen_label = QtWidgets.QLabel("")
    win.measure_pen_label.setStyleSheet(_MONO)
    win.measure_pen_label.setWordWrap(True)
    lay.addWidget(win.measure_pen_label)

    win.measure_head_label = QtWidgets.QLabel("")
    win.measure_head_label.setStyleSheet(_MONO + "color: #a30;")
    win.measure_head_label.setWordWrap(True)
    lay.addWidget(win.measure_head_label)

    lay.addStretch(1)

    # ---- このタブが持つ状態（毎tick更新される。build_measure_tabは1プロセスに
    #   つき1回しか呼ばれない前提。詳細はモジュールdocstring参照）----------------
    win._measure_ids = None       # body id（初回のupdate_measure_tabで埋める）
    win._measure_head_w = []      # 頭の角速度の直近300サンプル
    win._measure_devs = []        # 画像上の中心からのずれ（見えているtickのみ）


def _body_id_or_none(m, name):
    try:
        return int(m.body(name).id)
    except Exception:
        return None


def update_measure_tab(win, *, m, d, u, env, toy_bid, tick, t_sim):
    """測定器タブの表示を更新する（移植元：e_viewer.py 2762〜2834・2951〜2968行目）。

    呼び出し側（e_viewer_qt.pyのメインループ）が tick % 5 == 0 のときだけ
    呼ぶので、ここでは頻度の判定はしない。
    """
    if not hasattr(win, "measure_judge_label"):
        # build_measure_tab がまだ呼ばれていない（想定外の呼び出し順）場合の安全策
        return None

    # 【検証の落とし穴チェックリスト項53】時間が止まるとキャッシュが切れない。
    #   毎回、呼ぶ直前に必ず捨てる。
    u._vision_t = None
    u._vision_cache = None
    imgs = {}
    try:
        imgs = u.get_vision_obs() or {}
    except Exception:
        imgs = {}
    img = imgs.get("eye_left") if isinstance(imgs, dict) else None

    # ---- ①②③の判定 ------------------------------------------------------
    rep = VIS.report(m, d, toy_bid, img)
    if rep.get("pix_seen"):
        dev = float(np.hypot(rep["cx"], rep["cy"]))
        win._measure_devs.append(dev)
        if len(win._measure_devs) > 300:
            win._measure_devs.pop(0)

    j = f"①角度   {rep['angle']:5.1f}°  {'視野内' if rep['in_fov'] else '視野外'}\n"
    j += ("②光線   遮蔽なし\n" if rep["ray_ok"]
          else f"②光線   {rep['ray_hit']}に遮られている\n")
    if rep.get("pix_seen"):
        _mean_dev = float(np.mean(win._measure_devs)) if win._measure_devs else float("nan")
        j += (f"③画像   {rep['n_pixels']:4d}画素  "
              f"中心からのずれ {dev:.2f}（平均 {_mean_dev:.2f}）")
    else:
        j += "③画像   映っていない"
    win.measure_judge_label.setText(j)
    win.measure_judge_label.setStyleSheet(
        _MONO + ("color: #070;" if rep.get("pix_seen") else "color: #a00;"))

    # ---- body idのキャッシュ（初回のみ。理由はモジュールdocstring参照）---------
    if win._measure_ids is None:
        win._measure_ids = dict(
            head_bid=_body_id_or_none(m, "head"),
            eye_bid=_body_id_or_none(m, "left_eye"),
            sh_bid=_body_id_or_none(m, "right_upper_arm"),
            hand_bid=_body_id_or_none(m, "right_hand"))
    ids = win._measure_ids

    # ---- 距離ラベル -------------------------------------------------------
    if None not in (ids.get("eye_bid"), ids.get("sh_bid"), ids.get("hand_bid")):
        toy = d.xpos[toy_bid]
        de = float(np.linalg.norm(toy - d.xpos[ids["eye_bid"]])) * 100
        ds = float(np.linalg.norm(toy - d.xpos[ids["sh_bid"]])) * 100
        dh = float(np.linalg.norm(toy - d.xpos[ids["hand_bid"]])) * 100
        win.measure_dist_label.setText(
            f"目→おもちゃ {de:5.1f}cm  肩→ {ds:5.1f}cm"
            f"（腕の{ds / (ARM_REACH * 100) * 100:3.0f}%）  手→ {dh:5.1f}cm")
    else:
        win.measure_dist_label.setText("距離：body idが見つかりません")

    # ---- めり込みラベル ----------------------------------------------------
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
    if pen:
        pen.sort(key=lambda x: -x[1])
        win.measure_pen_label.setText(
            "めり込み " + " ".join(f"{n}{v:.1f}mm" for n, v in pen[:3]))
        win.measure_pen_label.setStyleSheet(_MONO + "color: #a00;")
    else:
        win.measure_pen_label.setText("めり込みなし")
        win.measure_pen_label.setStyleSheet(_MONO + "color: #070;")

    # ---- 頭の角速度ラベル ----------------------------------------------------
    if ids.get("head_bid") is not None:
        win._measure_head_w.append(float(np.linalg.norm(d.cvel[ids["head_bid"]][:3])))
        if len(win._measure_head_w) > 300:
            win._measure_head_w.pop(0)
        hw = np.array(win._measure_head_w) if win._measure_head_w else np.zeros(1)
        warn = ""
        # 「物理演算を止める」（chk_freeze）はOFF＝物理が回っているのに、
        #   「この角度で固定する」（chk_pose_hold、旧版のst_hold相当）がONのままだと
        #   関節が固定目標へ強制されながら物理も動くので暴れる（移植元：
        #   e_viewer.py 2830〜2832行目の warn ロジック）。
        freeze = bool(win.chk_freeze.isChecked()) if hasattr(win, "chk_freeze") else False
        hold_on = (bool(win.chk_pose_hold.isChecked())
                   if hasattr(win, "chk_pose_hold") else False)
        if not freeze and hold_on:
            warn = "\n「この角度で固定する」がONのまま物理を回しています＝暴れます"
        win.measure_head_label.setText(
            f"頭の角速度 平均 {hw.mean():.3f} 最大 {hw.max():.3f} rad/s{warn}")

    # ---- 左右の目を描く（移植元：e_viewer.py 2951〜2968行目）--------------------
    #   PySide6ではPIL/ImageTkの代わりにQImage/QPixmapへ直接変換する
    #   （実機での動作確認済み。仕様2節）。
    for side, img_label in win._measure_eye_labels.items():
        im = imgs.get(side) if isinstance(imgs, dict) else None
        if im is None:
            continue
        try:
            arr = np.asarray(im)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.ndim != 3 or arr.shape[2] < 3:
                continue
            if arr.shape[2] > 3:
                arr = arr[:, :, :3]
            arr = np.ascontiguousarray(arr)
            if win.chk_measure_mask.isChecked():
                arr = arr.copy()
                arr[VIS.red_mask(arr)] = [0, 255, 0]
            h, w, _ch = arr.shape
            qimg = QtGui.QImage(arr.tobytes(), w, h, w * 3,
                                 QtGui.QImage.Format.Format_RGB888)
            # QPixmap.fromImage() の時点でデータがコピーされるので、
            #   直後に arr / qimg が破棄されても表示は壊れない（実機で検証済み）。
            pix = QtGui.QPixmap.fromImage(qimg).scaled(
                176, 176,
                QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                QtCore.Qt.TransformationMode.FastTransformation)
            img_label.setPixmap(pix)
        except Exception:
            continue

    return None
