"""太郎を見るための新しいビューア（PySide6版）— 第1段階：骨格＋シーン区画のみ。

【これは何か】`run/viewer_tools/e_viewer.py`（tkinter製・約3100行）のUIを、
PySide6へ段階的に移行する計画の**第1段階**の実装。

【今回作ったもの（できること）】
    ・QMainWindow + QTabWidget の骨格
    ・「体・環境」タブの中の「シーン」区画だけ
      （シーン選択・シーンの説明・柵チェックボックス・頭を支えるチェックボックス・
       状態メッセージ・「最初からやり直す」ボタン）
    ・MuJoCoの物理ループ（旧版の `win.update()` の代わりに `app.processEvents()`
      を使う版。設計が示す置き換え方をそのまま採用）
    ・「最初からやり直す」で、選んだシーンが今と違えば自分を新しいE_SCENE付きで
      再起動して閉じる（旧版と同じパターン）。同じシーンならその場でリセットする。

【今回まだ無いもの（正直に明記する）】
    他の4タブ（駆動／感覚と報酬／実行／測定器）は中身が無い空のプレースホルダ。
    旧版が持つ他の区画（おもちゃ・姿勢・反射・首のバネ・四肢の筋力調整・
    もがき運動パラメータ調整・反射+共通駆動パラメータ調整・口元の報酬・
    触覚の順応・再生（速度調整）・脳・測定器）はまだこちらには無い。
    env_label（状態メッセージ）のうち「実行速度」の行も今回は省略している
    （「再生」区画のスライダー値に依存するため。次段階で「再生」区画を
    移植するときに追加する）。
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
from e_head_hold import CaregiverHands

# 【2026-08-13】シーンが全部を決めるので、体を作り直すときは古い個別指定を
#   消しておく。run/viewer_tools/e_viewer.py 2348〜2352行目と同じ一覧
#   （そのまま踏襲。ここだけ別の一覧にすると「シーンと環境変数のどちらが
#   効くのか」がまた曖昧になるため＝落とし穴チェックリスト項30と同じ考え方）。
_STALE_ENV_KEYS = ("E_AGE", "E_HEAD_HOLD", "E_FENCE", "E_TOY_RADIUS",
                    "E_RECLINE", "E_EYE_REST_V", "E_HOLD_TILT",
                    "E_TOY_POS", "E_SEAT_FRICTION", "E_TOY_MODE",
                    "E_TOY_SHAPE", "E_TOY_DIST")


class MainWindow(QtWidgets.QMainWindow):
    """太郎ビューア（新版・骨格）。

    タブは5つ用意するが、中身があるのは「体・環境」タブの「シーン」区画だけ
    （第1段階のスコープ）。他の4タブは次段階以降で埋める空のプレースホルダ。
    """

    def __init__(self, scene_names, current_scene_name):
        super().__init__()
        self.setWindowTitle("太郎ビューア（新版UI試作・骨格・第1段階）")
        self.resize(760, 560)

        tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(tabs)

        self.tab_body_env = QtWidgets.QWidget()
        tabs.addTab(self.tab_body_env, "体・環境")
        for title in ("駆動", "感覚と報酬", "実行", "測定器"):
            placeholder = QtWidgets.QWidget()
            lay = QtWidgets.QVBoxLayout(placeholder)
            lay.addWidget(QtWidgets.QLabel(
                f"「{title}」タブは第1段階では未実装です（次段階以降で移植予定）"))
            lay.addStretch(1)
            tabs.addTab(placeholder, title)

        outer = QtWidgets.QVBoxLayout(self.tab_body_env)

        # ---- シーン（環境）区画 -------------------------------------------
        # 移植元：run/viewer_tools/e_viewer.py 605〜746行目付近
        #   （読み取り専用の参考。旧版のこのファイルは一切変更していない）。
        box = QtWidgets.QGroupBox("シーン（環境）")
        outer.addWidget(box)
        outer.addStretch(1)
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

        # 【なぜ、2026-08-13・仕様4-3節】旧版の env_label は
        #   体年齢・柵・リクライニング・頭を支える状態・実行速度を出していたが、
        #   「実行速度」は「再生」区画（次段階以降で移植予定）のスライダー値に
        #   依存するため、今回はまだ存在しない。実行速度の行は省略する
        #   （実装担当が判断した段階分けの結果。仕様変更ではない）。
        self.env_label = QtWidgets.QLabel("")
        self.env_label.setStyleSheet("font-family: Consolas; color: #444;")
        form.addWidget(self.env_label)

        self.msg_label = QtWidgets.QLabel("")
        self.msg_label.setWordWrap(True)
        form.addWidget(self.msg_label)

        self.restart_btn = QtWidgets.QPushButton("最初からやり直す")
        form.addWidget(self.restart_btn)


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
    #   第1段階は「シーン」区画だけの移植であり、駆動モード（E_ACTUATION）の
    #   切替UIは今回のスコープに入っていない。
    env, hands = e_scene.build(scene, orient=True, vor=True, seed=0, verbose=True)
    u = env.unwrapped
    m, d = u.model, u.data
    if hands is None:
        hands = CaregiverHands(m, d)
    try:
        e_scene.verify(scene, env, strict=False, verbose=True)
    except Exception as e:
        print(f"[scene] 注意verify()に失敗しました（続行します）: {e}", flush=True)

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

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(scene_names, scene["name"])

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

    restart = [True]
    win.restart_btn.clicked.connect(lambda: restart.__setitem__(0, True))
    win.show()

    with mujoco.viewer.launch_passive(m, d) as viewer:
        t_sim, wall0, tick = 0.0, time.time(), 0
        while viewer.is_running() and win.isVisible():
            if restart[0]:
                # ================================================================
                # 【2026-08-13】旧版と同じパターン：月齢・リクライニング角は
                #   モデル構築時に決まるので実行中には変えられない。選んだシーンが
                #   今と違うなら、E_SCENEを設定して自分を新しいプロセスとして
                #   起動し、自分は閉じる（run/viewer_tools/e_viewer.py
                #   2328〜2384行目と同じロジック。読み取り専用の参考）。
                # ================================================================
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

                # 同じシーンのままの「やり直す」は、その場でリセットする
                #   （run/viewer_tools/e_viewer.py 2386〜2456行目と同じロジック。
                #   このシーンから始めているので _scene is not None 側だけ移植した）。
                e_scene.reset_to_scene(
                    env, scene,
                    hands=(hands if win.chk_head_hold.isChecked() else None),
                    seed=0)
                if not win.chk_head_hold.isChecked():
                    hands.release()
                apply_fence(win.chk_fence.isChecked())
                t_sim, wall0, tick = 0.0, time.time(), 0
                win.msg_label.setText(f"シーン「{scene['name']}」で始めました")
                restart[0] = False

            env.step(zero)
            t_sim += dt
            tick += 1

            # 環境のスイッチ（柵・実験者の手）を実行中でも反映する
            #   （run/viewer_tools/e_viewer.py 2513〜2520行目と同じロジック）。
            if bool(win.chk_fence.isChecked()) != fence_on[0]:
                apply_fence(win.chk_fence.isChecked())
            if bool(win.chk_head_hold.isChecked()) != bool(hands.holding):
                if win.chk_head_hold.isChecked():
                    hands.hold(target=_hold_tgt)
                else:
                    hands.release()

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

            # 【2026-08-13・設計4-2節】旧版の win.update() をここで
            #   app.processEvents() に置き換える。非ブロッキングでQtのイベント
            #   （クリック・コンボボックス操作等）を処理する。
            app.processEvents()
            viewer.sync()
            lag = wall0 + t_sim - time.time()
            if lag > 0:
                time.sleep(min(lag, 0.05))

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
