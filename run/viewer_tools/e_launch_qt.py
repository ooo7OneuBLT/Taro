"""新版ビューア（PySide6・第1段階）のランチャー — シーンを選んで起動するだけの入口。

【なぜ別ファイルにしたか、2026-08-13】既存の `e_launch.py`（旧版tkinter版の起動経路）は
一切変更しない方針（仕様3節）のため、新版専用にこのファイルを複製した。
中身は `e_launch.py` とほぼ同じ（意図的な複製。旧新を独立させるための制約上、
最小限のコードなので複製のコストは小さいと判断した）。唯一の違いは、
起動する先が `e_viewer.py` ではなく `e_viewer_qt.py` であること。

【これは何か】新版ビューア（第1段階＝骨格＋シーン区画のみ）を試すための入口。
旧版と同じ「シーンを選んで Enter」という使い方ができる。

使い方:
    プロジェクト直下の `Viewer(新版UI試作)を開く.bat` をダブルクリックする
    （コマンドから直接なら .venv/Scripts/python.exe run/viewer_tools/e_launch_qt.py）

    番号を入れて Enter … そのシーンで開く
    そのまま Enter     … 前回と同じシーンで開く
"""
import os
import sys
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_SCENE_TOOLS = os.path.join(_ROOT, "run", "scene_tools")
if _SCENE_TOOLS not in sys.path:
    sys.path.insert(0, _SCENE_TOOLS)

import e_scene  # noqa: E402
# 一覧表示・選択のロジックは旧版のランチャー（e_launch.py）と全く同じものを使う
#   （複製を避けるため、そちらの関数をそのままimportして使い回す）。
from e_launch import (read_last, write_last, select_scene)  # noqa: E402


def main():
    names = e_scene.list_scenes()
    if not names:
        print("シーンが1つもありません。")
        print("  .venv/Scripts/python.exe run/scene_tools/e_scene_make.py  で作れます")
        input("\nEnter で閉じます ")
        return 1

    last = read_last()
    if last not in names:
        last = names[0]

    pick = select_scene(names, last)
    if pick is None:
        return 0
    write_last(pick)

    print(f"\n  「{pick}」で開きます（新版UI試作・骨格・第1段階）。"
          "体を作るのに数十秒かかります …\n")
    envv = dict(os.environ)
    envv["E_SCENE"] = pick
    envv["PYTHONIOENCODING"] = "utf-8"
    for k in ("E_AGE", "E_HEAD_HOLD", "E_FENCE", "E_TOY_RADIUS", "E_RECLINE",
              "E_EYE_REST_V", "E_HOLD_TILT", "E_TOY_POS", "E_SEAT_FRICTION",
              "E_TOY_MODE", "E_TOY_SHAPE", "E_TOY_DIST", "E_ORIENT_V"):
        envv.pop(k, None)
    return subprocess.call([sys.executable,
                            os.path.join(_HERE, "e_viewer_qt.py")],
                           env=envv, cwd=_ROOT)


if __name__ == "__main__":
    sys.exit(main())
