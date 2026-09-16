"""TaroMap（配線・設定の一覧アプリ）専用のテーマ定義。

【想定外・ファイル名を仕様から変えたことについて、2026-08-15】
仕様は `run/viewer_tools/e_viewer_qt_theme.py`（新規）としてこのファイルを
指定していたが、着手時点で**そのパスには既に別プロジェクトの実ファイルが存在
していた**（`run/viewer_tools/e_viewer_qt.py` という既存の「Viewer新版UI試作」が
`from e_viewer_qt_theme import enable_auto_theme` のように**フラットな相対import**
で読んでいる、コミット済みの現役ファイル）。仕様はこのファイルが新規だという
前提で書かれていたが、実際には既存資産と衝突した。
　⇒ 誤って一度 `git checkout --` で復元できる形で上書きしてしまったが、
　`git checkout -- run/viewer_tools/e_viewer_qt_theme.py` で直後に元へ戻し、
　中身は1バイトも変わっていない（`git diff` で無変更を確認済み）。
　このファイルは衝突を避けるため、`wiring_viewer` パッケージの中
　（自分が単独で担当している領域）に `theme.py` として置く。
　この判断と、この判断が正しいかは作業記録の「上に上げること」に明記する。

【なぜ独立ファイルか】
別プロジェクトに類似のテーマ橋渡しファイルが
既にあるが、それは別リポジトリのビューア専用（別リポジトリ・別venv）。太郎リポジトリ側の
このアプリは独立して動くべきなので、コピーはせず参考程度に見た上で新規に書いた。

【色の選び方】
  bg / card / ink / sub / line / accent … Windows 11 のライト/ダーク配色に寄せた基本色。
  role_innate / role_emergent / role_tool / role_unclassified / role_deviation
    … 機構の分類バッジ用。文字色として使う想定なので、対応する card 色に対して
      WCAG 2.0 のコントラスト比 4.5:1 以上を満たすように選んだ
      （`contrast_check.py` の自己テストで実測済み。全て5.0以上ある）。
  role_provisional … 「これは自動判定・参考表示であり確定値ではない」ことを示す色。
    確定した分類（role_innate 等）と紛れないよう、彩度を落とし、
    UI側では実線でなく破線枠で描く想定（このファイルは色だけを持つ）。

【禁止事項、仕様より】QApplication全体への setStyleSheet() はしない
  （windows11スタイルの角丸・フォーカスリング・ホバーアニメーションが失われることが
  既に実測されている）。QSS はここで定義した色を使い、バッジ等の独自部品にだけ
  局所的に当てる。
"""

THEMES = {
    "windows11": {
        # ---- ライト（Windows 11 既定に近い配色）------------------------
        "bg": "#f3f3f3",
        "card": "#ffffff",
        "ink": "#1b1b1b",
        "sub": "#5f6368",
        "line": "#e0e0e0",
        "accent": "#0067c0",
        "role_innate": "#0f7a3d",         # 生得＝緑
        "role_emergent": "#7a3d9c",       # 創発＝紫
        "role_tool": "#8a5a00",           # 道具＝褐色（金より視認性が高い）
        "role_unclassified": "#5f6368",   # 未分類＝グレー（subと同系）
        "role_deviation": "#c42b1c",      # 逸脱＝赤（Windows11エラー色に近い）
        "role_provisional": "#6b6b6b",    # 自動判定・参考表示＝彩度を落とした灰
    },
    "windows11_dark": {
        # ---- ダーク ------------------------------------------------------
        "bg": "#202020",
        "card": "#2b2b2b",
        "ink": "#f3f3f3",
        "sub": "#c5c5c5",
        "line": "#3d3d3d",
        "accent": "#60cdff",
        "role_innate": "#6ccb5f",
        "role_emergent": "#c58af9",
        "role_tool": "#e3a127",
        "role_unclassified": "#c5c5c5",
        "role_deviation": "#ff99a4",
        "role_provisional": "#9b9b9b",
    },
}


def get_theme(name: str) -> dict:
    """テーマ名から色の辞書を返す。無い名前が来たら windows11（ライト）にする。"""
    return THEMES.get(name, THEMES["windows11"])


def detect_system_theme() -> str:
    """OSのダーク/ライト設定を見て、対応するテーマ名を返す。

    【なぜQGuiApplication.styleHints()か、2026-08-15】仕様の実測情報の通り、
    このPySide6 6.11.1環境には `QStyleHints.colorSchemeChanged` シグナルが存在し、
    ライブでOS設定の切り替えを検知できる。呼び出し側（main_window.py）でこの
    シグナルにフックしてテーマを差し替える。
    """
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QGuiApplication
        hints = QGuiApplication.styleHints()
        scheme = hints.colorScheme()
        if scheme == Qt.ColorScheme.Dark:
            return "windows11_dark"
        return "windows11"
    except Exception:
        # 【なぜ、2026-08-15】QApplication構築前に呼ばれた等、失敗しても
        #   アプリ全体を落とさない（ライト固定にフォールバック）。
        return "windows11"
