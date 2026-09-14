"""体ぜんたいスライダー（絶対値化）の往復保証テスト。

【なぜ要るか、2026-08-17】診断の結果、体ぜんたいスライダー6本（前後にたおす／
左右にたおす／ひねる／高さ／前後の位置／左右の位置）が「読み込み時の姿勢からの
差分」だったことが判明した。あるセッションで記録した値を別のシーン・別の
セッションへそのまま適用すると基準がずれて別物の姿勢になる事故が起きていた
（2026-08-16のメモ→6ヶ月シーン機械変換で頭部が床下-21cmに埋まった。詳細：
E/docs/研究日誌_2026-08.md 2026-08-17追記）。これを受けて
run/viewer_tools/e_viewer_qt.py の `_root_quat_from_sliders`/
`_sliders_from_root_quat` を世界座標基準の絶対値へ作り直した。

このテストは、その絶対値化が壊れていないかを恒久的に確認する回帰テスト。
2つを検証する：
    (a) 角度往復：スライダー角度→四元数→スライダー角度、で誤差0.01度未満か
    (b) シーン往復：あるシーンにスライダー相当の絶対値を設定して保存→
        読み込み直して同じ値が読み取れるか（一時フォルダに保存。
        run/scenes/ 配下・E/docs/シーン一覧.md には一切書き込まない）

使い方：
    .venv/Scripts/python.exe run/viewer_tools/test_pose_slider_roundtrip.py
    （QtのGUIは開かない。QT_QPA_PLATFORM=offscreen を既定で設定する）

終了コード：全部合格なら0、1つでも不合格なら1（CIやスクリプトから使える形）。
"""
import os
import shutil
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
# 【なぜ、2026-08-17】e_viewer_qt.py（33〜60行目）と同じパス設定。
#   scene_io.py が内部で e_toy_env・infant_body 等をimportするため必要。
for _p in [os.path.join(_ROOT, "D", "scripts"), os.path.join(_ROOT, "MIMo"),
           os.path.join(_ROOT, "taro_core"),
           os.path.join(_ROOT, "taro_core", "src", "body"),
           os.path.join(_ROOT, "taro_core", "src", "brain"),
           os.path.join(_ROOT, "taro_core", "src", "wrapper"),
           os.path.join(_ROOT, "taro_core", "src", "senses"),
           os.path.join(_ROOT, "run", "scene_tools"),
           os.path.join(_ROOT, "E", "scripts"),
           _ROOT, _HERE]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import mujoco

import scene_io
import e_toy_env as TE
from e_viewer_qt import _root_quat_from_sliders, _sliders_from_root_quat


def _find_root_qadr(m):
    for j in range(m.njnt):
        if (m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                and (m.body(int(m.jnt_bodyid[j])).name or "") == TE.ROOT_BODY):
            return int(m.jnt_qposadr[j])
    return None


def test_angle_roundtrip(tol_deg=0.01):
    """(a) スライダー角度→四元数→スライダー角度の往復誤差を確認する。

    ケースは通常域（±90度程度）に加え、実在シーンの機械走査で見つかった
    極端な値（「座位保持_リクライニング70度」実測roll=152.8, yaw=161.2）と、
    ±180度の境界付近も含める。
    """
    print("=" * 70)
    print("(a) 角度往復テスト（誤差 0.01度未満で合格）")
    print("=" * 70)
    cases = [
        (0.0, 0.0, 0.0),
        (30.0, 0.0, 0.0),
        (0.0, 60.0, 0.0),
        (0.0, 0.0, 90.0),
        (15.0, 25.0, -40.0),
        (0.0, 90.0, 0.0),
        (0.0, -90.0, 0.0),
        (152.8, -22.7, 161.2),      # 実在シーン「座位保持_リクライニング70度」実測値
        (-179.9, 5.0, 179.9),       # ±180度境界
        (77.7, 43.2, -2.9),         # 実在シーンの絶対姿勢の組み合わせ
    ]
    print(f"{'入力(roll,pitch,yaw)':30s} {'復元(roll,pitch,yaw)':30s} {'誤差[度]':>10s}")
    max_err = 0.0
    for roll, pitch, yaw in cases:
        q = _root_quat_from_sliders(roll, pitch, yaw)
        r2, p2, y2 = _sliders_from_root_quat(q)
        err = max(abs(r2 - roll), abs(p2 - pitch), abs(y2 - yaw))
        max_err = max(max_err, err)
        print(f"({roll:7.2f},{pitch:7.2f},{yaw:7.2f})".ljust(30)
              + f"({r2:7.3f},{p2:7.3f},{y2:7.3f})".ljust(30)
              + f"{err:10.5f}")
    ok = max_err < tol_deg
    print(f"\n[{'PASS' if ok else 'FAIL'}] 最大誤差 = {max_err:.5f} 度"
          f"（許容 {tol_deg} 度未満）")
    return ok, max_err


def test_scene_save_load_roundtrip(tol_deg=0.05, tol_cm=0.05):
    """(b) シーンに絶対値を設定して保存→読み込み直し、同じ値が読めるか確認する。

    scene_io.SCENE_DIR を一時フォルダへ差し替えて保存する（run/scenes/ 配下は
    一切変更しない）。save() は末尾で catalog.write_catalog() を自動で呼び、
    これは出力先 E/docs/シーン一覧.md をハードコードしている（catalog.py 34行目・
    OUT_PATH）ため、SCENE_DIRの差し替えだけでは防げない。テスト中だけ
    write_catalog を無効化し、実プロジェクトのファイルには一切触れないようにする。
    """
    print()
    print("=" * 70)
    print("(b) シーン保存→読み込み→スライダー表示値一致テスト")
    print("=" * 70)

    tmp_dir = tempfile.mkdtemp(prefix="pose_slider_roundtrip_")
    orig_scene_dir = scene_io.SCENE_DIR
    import catalog
    orig_write_catalog = catalog.write_catalog
    catalog.write_catalog = lambda *a, **k: None
    try:
        scene_io.SCENE_DIR = tmp_dir

        base = scene_io.load(os.path.join(
            _ROOT, "run", "scenes", "座位_6ヶ月_土台_2026-08-17_fix.json"))
        env, hands = scene_io.build(base, orient=True, vor=True, seed=0, verbose=False)
        u = env.unwrapped
        m, d = u.model, u.data
        root_qadr = _find_root_qadr(m)
        assert root_qadr is not None, "root_qadrが見つからない（テスト前提が崩れている）"

        set_roll, set_pitch, set_yaw = 12.0, -25.0, 47.0
        set_dx, set_dy, set_dz = 3.0, -2.0, 5.0  # cm
        newq = _root_quat_from_sliders(set_roll, set_pitch, set_yaw)
        d.qpos[root_qadr + 3:root_qadr + 7] = newq
        d.qpos[root_qadr:root_qadr + 3] = np.array(
            [set_dx, set_dy, set_dz]) / 100.0
        mujoco.mj_forward(m, d)

        scene_io.save(base, name="_test_pose_slider_roundtrip", env=env, hands=hands,
                     settle_seconds=0, image=False, verbose=False)

        reloaded = scene_io.load(
            os.path.join(tmp_dir, "_test_pose_slider_roundtrip.json"))
        env2, hands2 = scene_io.build(reloaded, orient=True, vor=True, seed=0,
                                     verbose=False)
        u2 = env2.unwrapped
        m2, d2 = u2.model, u2.data
        root_qadr2 = _find_root_qadr(m2)
        assert root_qadr2 is not None

        r2, p2, y2 = _sliders_from_root_quat(d2.qpos[root_qadr2 + 3:root_qadr2 + 7])
        pos2 = d2.qpos[root_qadr2:root_qadr2 + 3]
        dx2, dy2, dz2 = (float(pos2[0]) * 100.0, float(pos2[1]) * 100.0,
                          float(pos2[2]) * 100.0)

        print(f"  設定した値    : roll={set_roll} pitch={set_pitch} yaw={set_yaw}  "
              f"dx={set_dx}cm dy={set_dy}cm dz={set_dz}cm")
        print(f"  保存→再読込値 : roll={r2:.4f} pitch={p2:.4f} yaw={y2:.4f}  "
              f"dx={dx2:.4f}cm dy={dy2:.4f}cm dz={dz2:.4f}cm")
        errs = {
            "roll[度]": abs(r2 - set_roll), "pitch[度]": abs(p2 - set_pitch),
            "yaw[度]": abs(y2 - set_yaw), "dx[cm]": abs(dx2 - set_dx),
            "dy[cm]": abs(dy2 - set_dy), "dz[cm]": abs(dz2 - set_dz),
        }
        print(f"  誤差          : " + ", ".join(f"{k}={v:.5f}" for k, v in errs.items()))
        ok = (errs["roll[度]"] < tol_deg and errs["pitch[度]"] < tol_deg
              and errs["yaw[度]"] < tol_deg and errs["dx[cm]"] < tol_cm
              and errs["dy[cm]"] < tol_cm and errs["dz[cm]"] < tol_cm)
        print(f"\n[{'PASS' if ok else 'FAIL'}] 許容: 角度<{tol_deg}度・位置<{tol_cm}cm")
        return ok, errs
    finally:
        scene_io.SCENE_DIR = orig_scene_dir
        catalog.write_catalog = orig_write_catalog
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    ok1, _ = test_angle_roundtrip()
    ok2, _ = test_scene_save_load_roundtrip()
    print()
    print("=" * 70)
    print(f"総合判定: {'全合格' if (ok1 and ok2) else '不合格あり'}"
          f"（(a)={'PASS' if ok1 else 'FAIL'} / (b)={'PASS' if ok2 else 'FAIL'}）")
    print("=" * 70)
    sys.exit(0 if (ok1 and ok2) else 1)
