"""Viewer(PySide6版)の見た目テーマを定義する。

【なぜ、2026-08-13】ViewerをtkinterからPySide6へ移行する第2段階として、
Windows 11風の見た目を導入する（仕様：
作業記録（非公開））。
将来「macos」等の別テーマを足せるよう、数値の置き場所（THEMES辞書）と
QSS文字列の組み立てロジック（build_stylesheet）を分離してある
（doc/検証の落とし穴チェックリスト.md 項30「同じ定数が複数箇所にあったら
置き場所が間違っている」と同じ考え方。別テーマを足すとき、QSSの組み立て
コードを複製しなくて済む）。

このファイルは、Qtの標準ウィジェットクラス（QPushButton・QCheckBox・
QComboBox・QGroupBox・QLabel等）へ`QApplication.setStyleSheet()`で
スタイルシートを当てるだけの役割に閉じる。独自ウィジェットクラスへの
置き換えは一切要求しない（他の実装担当がe_viewer_qt.py側で標準Qtウィジェットを
使って並行実装しているため）。

使い方（呼び出し側の実装担当向け）：
    from e_viewer_qt_theme import apply_theme
    app = QtWidgets.QApplication(sys.argv)
    apply_theme(app)  # 既定はwindows11

ON/OFF・数値表示・警告のような、標準QSSセレクタだけでは表現しきれない
見た目は、Qtの「動的プロパティ」またはobjectNameで運ぶ想定にしてある
（QSSはCSSと違い、任意のクラス名を割り当てる仕組みが無いため）。
    - ON状態のラベル      ： label.setProperty("state", "on")
    - OFF状態のラベル     ： label.setProperty("state", "off")
    - 矛盾・警告の表示     ： label.setProperty("warn", True)
    - 等幅の数値表示ラベル ： label.setProperty("mono", True)
  プロパティを後から変える場合、Qtの仕様上
  `widget.style().unpolish(widget); widget.style().polish(widget)` を
  呼ばないと再描画に反映されないことがある（Qt公式ドキュメントに明記された
  既知の挙動。呼び出し側の実装で必要になったら対応すること）。
"""

from __future__ import annotations

import sys


# 【なぜ、2026-08-13】このテーマ名を既定として固定し、呼び出し側
# （e_viewer_qt.py）が引数を省略したときの挙動を1箇所で決められるようにする。
DEFAULT_THEME = "windows11"


# 【根拠ラベルについて】
# 数値の由来は3系統ある。
#   [Tier3-土台] 設計の統合設計・案C個別設計がたたき台として示した数値
#     （作業記録（非公開） 6節、
#      案C個別設計 6節）。工学的判断であり一次文献に基づく数値ではない。
#   [Tier2-一般知識] Microsoftが公開しているFluent 2 Design System・WinUI 3の
#     設計トークン（コーナー半径・配色）についての一般的な知識に基づく調整。
#     **注意：このセッションではWebSearchツールが利用できず、仕様が求める
#     「実際にWebSearchで確認する」という手順は実行できていない。** 学習時点の
#     知識（Microsoft Learn・Fluent UI公式資料で広く公開されている値）に基づく
#     判断であり、この場でURLを検証・提示することはできない。この点は
#     作業記録の「想定外」に明記する。
#   [Tier3-新規] たたき台に無かった項目（アクセントカラー・本文の基本文字色・
#     角丸を「グループ枠」と「個別コントロール」で分けたこと）は、今回
#     実装のために新規に判断した値。
THEMES = {
    "windows11": {
        # 全体の背景色。
        # [Tier2-一般知識] たたき台の #f4f5f7 から #f3f3f3 へ調整。
        # WinUI 3のライトテーマの既定背景（SolidBackgroundFillColorBase相当）は
        # #f3f3f3 が広く知られた値であり、Windows 11の設定アプリ等の背景に
        # 近い。#f4f5f7との差はごくわずかで実害は無いが、より実物に近い値に
        # 揃えた。
        "bg": "#f3f3f3",
        # 通常の文字色（黒に寄せすぎない、WinUIのTextFillColorPrimary相当）。
        # [Tier2-一般知識] たたき台に無かった項目。新規に追加。
        "text_color": "#1a1a1a",

        # グループ枠（QGroupBox相当）
        "group_bg": "#ffffff",
        # [Tier2-一般知識] たたき台の #d9dce3 から #e5e5e5 へ調整。
        # WinUI 3のCardStrokeColorDefault（ライト）に近い、より淡いグレーの
        # 枠線が実物に近いという理解に基づく。
        "group_border": "#e5e5e5",
        "group_border_width": 1,
        # [Tier2-一般知識] たたき台の4pxから8pxへ調整。
        # Fluent 2の設計トークンでは、ボタン等の小さいコントロールの角丸
        # （ControlCornerRadius、目安4px前後）と、カード・ダイアログのような
        # 「面」を持つ大きい要素の角丸（OverlayCornerRadius、目安8px前後）が
        # 別の値として使い分けられている。QGroupBoxは区画全体を囲む「面」に
        # あたるため、後者に近い8pxを採用する。
        "group_radius": 8,
        # 個別のコントロール（ボタン・チェックボックス・コンボボックス等）の
        # 角丸。[Tier2-一般知識] 上記の使い分けにより新規追加。
        "control_radius": 4,

        # タブ
        "tab_font_size_pt": 12,
        # 選択中のタブのみ太字にする（1節のたたき台の指定通り）。
        "tab_selected_bold": True,
        # [Tier3-新規] Windows既定のアクセントカラー（Windows Blue）。
        # 選択中タブの下線・チェック済みチェックボックスの塗りに使う。
        "accent": "#0078d4",

        # 区画の見出し文字
        "group_title_font_size_pt": 11,
        "group_title_bold": True,

        # 通常のラベル
        "label_font_size_pt": 10,

        # 数値表示（等幅フォントで桁を揃える。単位は呼び出し側の文言で添える）
        "mono_font_family": "Consolas",
        "mono_font_size_pt": 10,

        # ON/OFF状態の表示色（たたき台をそのまま踏襲）
        "on_color": "#2e7d32",
        "off_color": "#757575",

        # 矛盾・注意の表示（たたき台をそのまま踏襲）
        "warn_color": "#c62828",
        "warn_bg": "#fff8e1",

        # 余白（たたき台をそのまま踏襲。QSSでは制御できないレイアウトの
        # 間隔[layout.setSpacing等]に使う想定の値のため、辞書にだけ残す）
        "spacing_inner_px": 6,
        "spacing_group_px": 12,
    },
}


def theme_names() -> list[str]:
    """登録されているテーマ名の一覧を返す。"""
    return list(THEMES.keys())


def _theme_dict(theme_name: str) -> dict:
    """テーマ名からTHEMES辞書の値を取り出す。未知の名前ならDEFAULT_THEMEへ
    フォールアックし、標準エラー出力に注意文を出す（例外は投げない）。"""
    if theme_name not in THEMES:
        print(
            f"[e_viewer_qt_theme] 未知のテーマ名 '{theme_name}' が指定されました。"
            f"既定のテーマ '{DEFAULT_THEME}' を使用します。"
            f"（登録済みテーマ: {', '.join(theme_names())}）",
            file=sys.stderr,
        )
        theme_name = DEFAULT_THEME
    return THEMES[theme_name]


def build_stylesheet(theme_name: str = DEFAULT_THEME) -> str:
    """指定テーマのQSS文字列を組み立てて返す。

    未知のテーマ名ならDEFAULT_THEMEにフォールバックし、標準エラー出力に
    注意文を出す（例外は投げない）。

    数値はTHEMES[theme_name]の辞書から読むだけで、ここにハードコードしない
    （新しいテーマを足すときにQSSの組み立てコードを複製せずに済むため）。
    """
    resolved_name = theme_name if theme_name in THEMES else DEFAULT_THEME
    t = _theme_dict(theme_name)

    return f"""
/* ==== {resolved_name} テーマ（e_viewer_qt_theme.py で自動生成） ==== */

QMainWindow, QWidget {{
    background-color: {t["bg"]};
    color: {t["text_color"]};
    font-size: {t["label_font_size_pt"]}pt;
}}

/* タブ */
QTabWidget::pane {{
    border: {t["group_border_width"]}px solid {t["group_border"]};
    border-radius: {t["group_radius"]}px;
    background-color: {t["bg"]};
    top: -1px;
}}
QTabBar::tab {{
    font-size: {t["tab_font_size_pt"]}pt;
    padding: 6px 14px;
    margin-right: 2px;
    background-color: {t["bg"]};
    color: {t["text_color"]};
    border: {t["group_border_width"]}px solid {t["group_border"]};
    border-bottom: none;
    border-top-left-radius: {t["control_radius"]}px;
    border-top-right-radius: {t["control_radius"]}px;
}}
QTabBar::tab:selected {{
    font-weight: {"bold" if t["tab_selected_bold"] else "normal"};
    background-color: {t["group_bg"]};
    border-bottom: 2px solid {t["accent"]};
}}
QTabBar::tab:!selected {{
    font-weight: normal;
}}

/* 区画（QGroupBox） */
QGroupBox {{
    background-color: {t["group_bg"]};
    border: {t["group_border_width"]}px solid {t["group_border"]};
    border-radius: {t["group_radius"]}px;
    margin-top: 14px;
    padding: {t["spacing_inner_px"]}px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 8px;
    padding: 0 4px;
    font-size: {t["group_title_font_size_pt"]}pt;
    font-weight: {"bold" if t["group_title_bold"] else "normal"};
    color: {t["text_color"]};
}}

/* 区画どうしの間隔は呼び出し側のレイアウト(layout.setSpacing等)で
   THEMES[...]["spacing_group_px"] を使うことを想定。QSSでは制御できない。 */

/* 通常のラベル */
QLabel {{
    font-size: {t["label_font_size_pt"]}pt;
    color: {t["text_color"]};
}}

/* ボタン・入力コントロール */
QPushButton {{
    background-color: {t["group_bg"]};
    border: {t["group_border_width"]}px solid {t["group_border"]};
    border-radius: {t["control_radius"]}px;
    padding: 4px 12px;
    font-size: {t["label_font_size_pt"]}pt;
    color: {t["text_color"]};
}}
QPushButton:hover {{
    background-color: #f5f5f5;
}}
QPushButton:pressed {{
    background-color: #e5e5e5;
}}
QPushButton:disabled {{
    color: {t["off_color"]};
    background-color: {t["bg"]};
}}

QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {t["group_bg"]};
    border: {t["group_border_width"]}px solid {t["group_border"]};
    border-radius: {t["control_radius"]}px;
    padding: 3px 6px;
    font-size: {t["label_font_size_pt"]}pt;
    color: {t["text_color"]};
}}
QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
    border: {t["group_border_width"]}px solid {t["accent"]};
}}

QCheckBox {{
    font-size: {t["label_font_size_pt"]}pt;
    color: {t["text_color"]};
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 15px;
    height: 15px;
    border: {t["group_border_width"]}px solid {t["group_border"]};
    border-radius: 3px;
    background-color: {t["group_bg"]};
}}
QCheckBox::indicator:checked {{
    background-color: {t["accent"]};
    border: {t["group_border_width"]}px solid {t["accent"]};
}}

QSlider::groove:horizontal {{
    height: 4px;
    background: {t["group_border"]};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {t["accent"]};
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 7px;
}}

/* 数値表示（呼び出し側が setProperty("mono", True) を付けたラベル向け） */
QLabel[mono="true"] {{
    font-family: "{t["mono_font_family"]}";
    font-size: {t["mono_font_size_pt"]}pt;
}}

/* ON/OFF状態（呼び出し側が setProperty("state", "on"/"off") を付けたラベル向け） */
QLabel[state="on"] {{
    color: {t["on_color"]};
    font-weight: bold;
}}
QLabel[state="off"] {{
    color: {t["off_color"]};
}}

/* 矛盾・警告（呼び出し側が setProperty("warn", true) を付けたウィジェット向け） */
QLabel[warn="true"] {{
    color: {t["warn_color"]};
    background-color: {t["warn_bg"]};
    padding: 2px 4px;
    border-radius: {t["control_radius"]}px;
}}
QWidget[warn="true"] {{
    background-color: {t["warn_bg"]};
    border-radius: {t["control_radius"]}px;
}}
""".strip()


def apply_theme(app, theme_name: str = DEFAULT_THEME) -> None:
    """QApplicationインスタンス app にテーマを適用する。

    未知のテーマ名を渡しても例外は投げず、build_stylesheet側の
    フォールバックに任せる。
    """
    app.setStyleSheet(build_stylesheet(theme_name))


if __name__ == "__main__":
    # 単体確認用：QSS文字列が組み立てられることと、文字数をざっと表示する。
    css = build_stylesheet()
    print(f"[ok] build_stylesheet() -> {len(css)} 文字")
    css_unknown = build_stylesheet("no_such_theme")
    print(f"[ok] 未知テーマでフォールバック後 -> {len(css_unknown)} 文字")
    print(f"[ok] theme_names() -> {theme_names()}")
