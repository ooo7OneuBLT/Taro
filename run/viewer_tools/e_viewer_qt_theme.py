"""Viewer(PySide6版)の見た目テーマを定義する。

【なぜ、2026-08-13（第2版・作り直し）】第1版（QApplication.setStyleSheet()で
全ウィジェットへQSSを丸ごとかける方式）を、ユーザーから「全然Windows11っぽく
ない」という評価を受けて作り直した（仕様：
作業記録（非公開））。

【方針転換の理由（実装担当の事前調査、仕様2-1節）】
    この環境の.venvのPySide6(6.11.1)は、Windows上でQt公式ネイティブの
    `windows11`スタイル（QStyleFactoryが持つ、Qt自身が実装したWindows 11風の
    描画ロジック）を既定で使っている（`app.style().objectName()` -> "windows11"、
    実装担当・実装担当ともに実測で確認済み）。

    ところが`QApplication.setStyleSheet()`でQSSを丸ごとかけると、Qtは対象
    ウィジェットの描画を`QStyleSheetStyle`という別のスタイルへ切り替える
    （Qt公式の既知の挙動）。これにより、ネイティブの`windows11`スタイルが
    持っている繊細な角丸・フォーカスリング・ホバー/プレス時のアニメーション
    などが失われ、手書きQSSによる近似（角丸8px/4pxなど）に置き換わって
    しまっていた。これが「全然Windows11っぽくない」の技術的な主因と考えられる。

    そこで本版は、**QSSでの丸ごと上書きをやめ、QPalette（色）中心の調整＋
    どうしても必要な最小限のQSSに切り替える**。QPaletteによる色調整は
    ウィジェットのスタイル（描画ロジック）そのものは切り替えない
    （`QApplication.setPalette()`は色の置き換えだけで、`QStyleSheetStyle`への
    切替を引き起こさない）ため、ネイティブの`windows11`スタイルの描画は
    そのまま活きる。

【テーマを差し替えられる構造は維持する】
    数値の置き場所（THEMES辞書）と、そこから実際のQPalette／QSSを組み立てる
    ロジック（build_palette/build_stylesheet）を分離してある（第1版から
    引き続き。doc/検証の落とし穴チェックリスト.md 項30「同じ定数が複数箇所に
    あったら置き場所が間違っている」と同じ考え方）。将来「windows11」以外の
    テーマ（例："macos"）を足すときは、THEMES辞書に1エントリ追加するだけで
    済む。

使い方（呼び出し側の実装担当向け）：
    from e_viewer_qt_theme import apply_theme
    app = QtWidgets.QApplication(sys.argv)
    apply_theme(app)  # 既定はwindows11

ON/OFF・数値表示・警告のような、パレットだけでは表現しきれない見た目は、
引き続きQtの「動的プロパティ」またはobjectNameで運ぶ（第1版から変更なし）。
    - ON状態のラベル      ： label.setProperty("state", "on")
    - OFF状態のラベル     ： label.setProperty("state", "off")
    - 矛盾・警告の表示     ： label.setProperty("warn", True)
    - 等幅の数値表示ラベル ： label.setProperty("mono", True)
  プロパティを後から変える場合、Qtの仕様上
  `widget.style().unpolish(widget); widget.style().polish(widget)` を
  呼ばないと再描画に反映されないことがある（Qt公式ドキュメントに明記された
  既知の挙動。呼び出し側の実装で必要になったら対応すること）。

【この版で変わったこと（第1版との違い）】
    ・`apply_theme()`が`app.setStyleSheet(全体QSS)`ではなく
      `app.setPalette(...)` + `app.setStyleSheet(最小限QSS)`の2段構えになった。
    ・最小限QSSの対象は、ネイティブの`windows11`スタイルだけでは実現できない
      ものに絞った：
        - QGroupBoxの角丸（ネイティブは直角の枠線を描く。仕様1節が挙げる
          「面の演出」の一種として、角丸だけQSSで補う）
        - state/warn/monoのカスタムプロパティを使うラベルの見た目
          （Qtの標準セレクタに無い概念のため、元々QSSでしか表現できない）
      ボタン・チェックボックス・スライダー・タブ・コンボボックス・
      スピンボックスの外見（角丸・ホバー・プレス時の反応・フォーカスリング）は
      **QSSを当てず、ネイティブの描画に任せる**。
    ・アクセントカラーは`QPalette.Highlight`として設定する。ネイティブの
      `windows11`スタイルは、チェックボックスの塗り・選択状態の表示等に
      この色を実際に参照して描画する（Qt公式のQWindowsVistaStyle/
      QWindows11Style実装がQPaletteのHighlightロールを読む設計のため。
      [Tier2-一般知識]）。
"""

from __future__ import annotations

import sys


# 【なぜ、2026-08-13】このテーマ名を既定として固定し、呼び出し側
# （e_viewer_qt.py）が引数を省略したときの挙動を1箇所で決められるようにする。
DEFAULT_THEME = "windows11"


# 【根拠ラベルについて】
# 数値の由来は3系統ある（第1版から引き継ぎ。今回は色の値自体はほぼ変更していない
# ので、ラベルもそのまま引き継ぐ）。
#   [Tier3-土台] 設計の統合設計・案C個別設計がたたき台として示した数値
#     （作業記録（非公開） 6節、
#      案C個別設計 6節）。工学的判断であり一次文献に基づく数値ではない。
#   [Tier2-一般知識] Microsoftが公開しているFluent 2 Design System・WinUI 3の
#     設計トークン（コーナー半径・配色）についての一般的な知識に基づく調整。
#     **注意：このセッションではWebSearchツールが利用できず、仕様が求める
#     「実際にWebSearchで確認する」という手順は実行できていない。** 学習時点の
#     知識（Microsoft Learn・Fluent UI公式資料で広く公開されている値）に基づく
#     判断であり、この場でURLを検証・提示することはできない。この点は
#     作業記録の「上に上げること」にも明記する。
#   [Tier3-新規] たたき台に無かった項目（アクセントカラー・本文の基本文字色・
#     角丸を「グループ枠」と「個別コントロール」で分けたこと）は、今回
#     実装のために新規に判断した値。
THEMES = {
    "windows11": {
        # 全体の背景色。
        # [Tier2-一般知識] WinUI 3のライトテーマの既定背景
        # （SolidBackgroundFillColorBase相当）は #f3f3f3 が広く知られた値。
        "bg": "#f3f3f3",
        # 通常の文字色（黒に寄せすぎない、WinUIのTextFillColorPrimary相当）。
        "text_color": "#1a1a1a",

        # グループ枠（QGroupBox相当）・入力欄などの「面」の色
        "group_bg": "#ffffff",
        # [Tier2-一般知識] WinUI 3のCardStrokeColorDefault（ライト）に近い、
        # 淡いグレーの枠線。
        "group_border": "#e5e5e5",
        "group_border_width": 1,
        # [Tier2-一般知識] Fluent 2の設計トークンでは、ボタン等の小さい
        # コントロールの角丸（ControlCornerRadius、目安4px前後）と、
        # カード・ダイアログのような「面」を持つ大きい要素の角丸
        # （OverlayCornerRadius、目安8px前後）が別の値として使い分けられている。
        # QGroupBoxは区画全体を囲む「面」にあたるため、後者に近い8pxを採用する。
        # 【この版での使い道が変わった点】第1版はこれを個別コントロール
        # （ボタン等）のQSSにも使っていたが、本版はネイティブ描画に任せるため
        # QGroupBoxの角丸だけに使う。
        "group_radius": 8,
        # 個別のコントロール（ボタン・チェックボックス等）の角丸。
        # 【この版では使わない】ネイティブの`windows11`スタイルが自前で
        # 角丸を描画するため、QSSでの指定は行わない。THEMES辞書には記録として
        # 残す（将来、ネイティブ描画が使えない環境向けの代替テーマを足すときに
        # 再利用できるようにするため）。
        "control_radius": 4,

        # タブ
        "tab_font_size_pt": 12,
        # 選択中のタブのみ太字にする。
        # 【この版での扱い】QTabBarの外見自体（背景・境界線・角丸）はネイティブに
        # 任せるが、太字とフォントサイズはQPaletteで表現できないため、
        # 最小限QSSの対象に残す（フォントの太さ・大きさは「面の演出」ではなく
        # 文字の階層の演出であり、ネイティブスタイルの描画とは競合しない）。
        "tab_selected_bold": True,
        # [Tier3-新規] Windows既定のアクセントカラー（Windows Blue）。
        # QPalette.Highlightとして設定し、ネイティブ描画（チェック済み
        # チェックボックスの塗り・選択中タブの下線等）に使わせる。
        "accent": "#0078d4",

        # 区画の見出し文字
        "group_title_font_size_pt": 11,
        "group_title_bold": True,

        # 通常のラベル
        "label_font_size_pt": 10,

        # 数値表示（等幅フォントで桁を揃える。単位は呼び出し側の文言で添える）
        "mono_font_family": "Consolas",
        "mono_font_size_pt": 10,

        # ON/OFF状態の表示色
        "on_color": "#2e7d32",
        "off_color": "#757575",

        # 矛盾・注意の表示
        "warn_color": "#c62828",
        "warn_bg": "#fff8e1",

        # 余白（QSSでは制御できないレイアウトの間隔[layout.setSpacing等]に
        # 使う想定の値のため、辞書にだけ残す）
        "spacing_inner_px": 6,
        "spacing_group_px": 12,
    },
}


def theme_names() -> list[str]:
    """登録されているテーマ名の一覧を返す。"""
    return list(THEMES.keys())


def _theme_dict(theme_name: str) -> dict:
    """テーマ名からTHEMES辞書の値を取り出す。未知の名前ならDEFAULT_THEMEへ
    フォールバックし、標準エラー出力に注意文を出す（例外は投げない）。"""
    if theme_name not in THEMES:
        print(
            f"[e_viewer_qt_theme] 未知のテーマ名 '{theme_name}' が指定されました。"
            f"既定のテーマ '{DEFAULT_THEME}' を使用します。"
            f"（登録済みテーマ: {', '.join(theme_names())}）",
            file=sys.stderr,
        )
        theme_name = DEFAULT_THEME
    return THEMES[theme_name]


def build_palette(theme_name: str = DEFAULT_THEME):
    """指定テーマのQPaletteを組み立てて返す。

    未知のテーマ名ならDEFAULT_THEMEにフォールバックする（build_stylesheetと
    同じ挙動。_theme_dict()が既にフォールバック+警告出力を行う）。

    ここでの色の割り当ては、Qt公式のQPaletteロールの一般的な意味
    （https://doc.qt.io/qt-6/qpalette.html 相当の定義。[Tier2-一般知識]、
    このセッションではWebSearch不可のため学習時点の知識に基づく）に沿わせる。
    ネイティブの`windows11`スタイルは、ボタン・チェックボックス・スライダー等の
    描画時にこれらのロールを実際に参照するため、QSSを使わずに配色を変えられる。
    """
    from PySide6 import QtGui

    t = _theme_dict(theme_name)
    pal = QtGui.QPalette()

    bg = QtGui.QColor(t["bg"])
    text = QtGui.QColor(t["text_color"])
    base = QtGui.QColor(t["group_bg"])  # 入力欄・カードの下地（白に近い面）
    accent = QtGui.QColor(t["accent"])
    off = QtGui.QColor(t["off_color"])
    border = QtGui.QColor(t["group_border"])

    # 【なぜ、2026-08-13】ここで設定するのは「色」だけで、ウィジェットの
    # 描画ロジック（角丸の描き方・アニメーション）には一切触れない。
    # これがQSS丸ごと上書きとの決定的な違い。
    pal.setColor(QtGui.QPalette.Window, bg)
    pal.setColor(QtGui.QPalette.WindowText, text)
    pal.setColor(QtGui.QPalette.Base, base)
    pal.setColor(QtGui.QPalette.AlternateBase, bg)
    pal.setColor(QtGui.QPalette.Text, text)
    pal.setColor(QtGui.QPalette.Button, bg)
    pal.setColor(QtGui.QPalette.ButtonText, text)
    pal.setColor(QtGui.QPalette.BrightText, QtGui.QColor("#ffffff"))
    pal.setColor(QtGui.QPalette.ToolTipBase, base)
    pal.setColor(QtGui.QPalette.ToolTipText, text)
    pal.setColor(QtGui.QPalette.PlaceholderText, off)
    pal.setColor(QtGui.QPalette.Highlight, accent)
    pal.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#ffffff"))
    pal.setColor(QtGui.QPalette.Link, accent)
    # 【なぜ、2026-08-13】Windowsのネイティブスタイルは無効化状態の文字色に
    # Disabledグループを参照する。ここを設定し忘れると、無効時の文字色だけ
    # 黒背景テーマ等で読めなくなることがある（今回はライトテーマのみだが、
    # 将来ダークテーマを足すときの落とし穴を避けるため、最初から設定しておく）。
    pal.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, off)
    pal.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, off)
    pal.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, off)
    pal.setColor(QtGui.QPalette.Mid, border)

    return pal


def build_stylesheet(theme_name: str = DEFAULT_THEME) -> str:
    """指定テーマの、**最小限の**QSS文字列を組み立てて返す。

    未知のテーマ名ならDEFAULT_THEMEにフォールバックし、標準エラー出力に
    注意文を出す（例外は投げない）。

    【この版での役割変化】第1版はここで全ウィジェットの外見を定義していたが、
    本版は「ネイティブの`windows11`スタイルだけでは表現できないもの」だけに
    絞ってある：
        1. QGroupBoxの角丸（ネイティブは直角の枠線）
        2. QTabBarの文字の太さ・大きさ（フォント階層の演出。QPaletteでは
           表現できない）
        3. state/warn/monoのカスタムプロパティを使うラベル（Qtの標準
           セレクタに元々存在しない概念のため、常にQSSが必要）
    ボタン・チェックボックス・コンボボックス・スピンボックス・スライダーの
    外見（背景・枠線・角丸・ホバー・プレス）は、意図的に**ここに書かない**。
    書いてしまうと、そのウィジェットの描画がQStyleSheetStyleへ切り替わり、
    ネイティブの繊細な描画（フォーカスリングのアニメーション等）が失われる
    （このファイル冒頭のコメント参照）。

    数値はTHEMES[theme_name]の辞書から読むだけで、ここにハードコードしない
    （新しいテーマを足すときにQSSの組み立てコードを複製せずに済むため）。
    """
    resolved_name = theme_name if theme_name in THEMES else DEFAULT_THEME
    t = _theme_dict(theme_name)

    return f"""
/* ==== {resolved_name} テーマ（e_viewer_qt_theme.py で自動生成・最小限QSS） ==== */

/* 区画（QGroupBox）の角丸だけを補う。背景・枠線の色はQPaletteが決めるため
   ここでは触らない（Base/Windowロールをそのまま使わせる）。 */
QGroupBox {{
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
}}

/* タブの文字階層（選択中を太字にする）。背景・枠線・角丸はネイティブに任せる。 */
QTabBar::tab {{
    font-size: {t["tab_font_size_pt"]}pt;
}}
QTabBar::tab:selected {{
    font-weight: {"bold" if t["tab_selected_bold"] else "normal"};
}}
QTabBar::tab:!selected {{
    font-weight: normal;
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
    border-radius: 4px;
}}
QWidget[warn="true"] {{
    background-color: {t["warn_bg"]};
    border-radius: 4px;
}}
""".strip()


def apply_theme(app, theme_name: str = DEFAULT_THEME) -> None:
    """QApplicationインスタンス app にテーマを適用する。

    未知のテーマ名を渡しても例外は投げず、build_palette/build_stylesheet側の
    フォールバックに任せる。

    【この版での挙動】`app.setPalette(...)`で色を、`app.setStyleSheet(...)`で
    最小限のQSS（QGroupBoxの角丸等）を適用する。どちらもネイティブの
    `windows11`スタイルそのものを切り替えるものではない
    （setStyleSheetは対象ウィジェットの描画をQStyleSheetStyleへ切り替えるが、
    本版のQSSはQGroupBox・QTabBar・カスタムプロパティ付きQLabelだけに絞って
    あるため、ボタン・チェックボックス・スライダー等はネイティブ描画のまま）。
    """
    app.setPalette(build_palette(theme_name))
    app.setStyleSheet(build_stylesheet(theme_name))


def try_apply_mica(window, style: str = "mica") -> bool:
    """windowにWindows 11のMica（半透明の背景）を適用してみる。

    【なぜ、2026-08-13】仕様3節・4節。`pywinstyles`（PyPI、CC0ライセンス。
    ライセンスの懸念は無い。作業記録（非公開）
    2026-08-13_ViewerをWindows11風に本気で作り込む.md 2-4節参照）を使う。

    pywinstyles未導入の環境（requirements.txtに追記したが`pip install`前）や、
    Windows以外の環境でも例外を投げず、False を返すだけにする（呼び出し側の
    e_viewer_qt.pyは、テーマ適用と同じ「失敗しても続行する」パターンで
    これを呼ぶ想定）。

    戻り値：適用できたらTrue、できなければFalse（理由は問わない）。

    【実測で分かったこと（作業記録に詳細）】この呼び出し自体はWindows実機で
    例外なく成功する（DWMの属性変更APIが正常応答する）ことを実装担当が確認した。
    ただし、その結果として実際に半透明の背景がデスクトップの壁紙と混ざって
    見えるかどうかは、リモート/仮想デスクトップのような環境ではDWMの
    合成（コンポジット）が行われず黒一色になることがある（実測済み）。
    つまり「呼び出しが成功する」ことと「見た目が本物のMicaになる」ことは
    別であり、後者はユーザーの実機で確認する必要がある。
    """
    try:
        import pywinstyles
    except Exception:
        return False
    try:
        pywinstyles.apply_style(window, style)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    # 単体確認用：QSS/パレットが組み立てられることと、未知テーマの
    # フォールバックが例外を投げずに動くことを確認する。
    css = build_stylesheet()
    print(f"[ok] build_stylesheet() -> {len(css)} 文字")
    css_unknown = build_stylesheet("no_such_theme")
    print(f"[ok] 未知テーマでフォールバック後 -> {len(css_unknown)} 文字")
    print(f"[ok] theme_names() -> {theme_names()}")

    # QPaletteの組み立てはQApplicationが無いとQColor等が使えないことがあるため、
    # ここで実際にQApplicationを作って確認する。
    from PySide6 import QtWidgets
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    pal = build_palette()
    print(f"[ok] build_palette() -> Window={pal.color(pal.ColorRole.Window).name()} "
          f"Highlight={pal.color(pal.ColorRole.Highlight).name()}")
    pal_unknown = build_palette("no_such_theme")
    print(f"[ok] 未知テーマでのbuild_palette() -> "
          f"Window={pal_unknown.color(pal_unknown.ColorRole.Window).name()}")
