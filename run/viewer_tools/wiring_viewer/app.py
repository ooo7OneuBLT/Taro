"""TaroMap の起動口。

使い方：
    .venv\\Scripts\\python.exe -m run.viewer_tools.wiring_viewer.app

`build_app(argv=None)` は `exec()` を呼ばずに (app, window) を返す。
自動検証（オフスクリーンでの起動・スクロールバー探索など）から使うための入口。
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from PySide6.QtWidgets import QApplication  # noqa: E402

from run.viewer_tools.wiring_viewer.main_window import MainWindow  # noqa: E402


def build_app(argv=None):
    """QApplicationとMainWindowを作って返す（execは呼ばない）。

    【なぜexecを呼ばないか、仕様より】工程Cがオフスクリーンで自動検証する
    （QT_QPA_PLATFORM=offscreen の下で resize や QScrollArea 探索を行う）ために、
    イベントループへ入る前の状態を返す必要がある。
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication(argv if argv is not None else sys.argv)
    # Windows 11 スタイル（角丸・フォーカスリング等）。QApplication全体への
    #   setStyleSheet()は使わない（仕様の禁止事項）。
    try:
        app.setStyle("windows11")
    except Exception:
        pass
    window = MainWindow()
    return app, window


def main():
    app, window = build_app()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
