"""TaroMap のメインウィンドウ（工程C＝統合済み。6ページとも本物のページを配線）。

【なぜこの形にしたか、2026-08-15】
過去の「別リポジトリのビューア」で、1つのQScrollAreaに7ビューを縦積みし合計高さ2480pxに
なり、1080px画面ではスクロールバーが絶対に消えない失敗が実測されている
（QSplitterが1箇所も無かった）。今回は必ず守る：
  1) 左（ナビ）と右（中身）は QSplitter（横方向）で区切り、ドラッグで幅を変えられる
  2) setMinimumWidth() の固定値だけで幅問題に対処しない
  3) 1画面に主役は1つ。中身のページを7つ縦積みにしない

【工程C・統合時の注意（今後この構造を崩さないこと）】
各page_*.pyは「1ページ＝1つのQScrollArea以下」に留めて作られている
（page_health.pyのみページ自身がQScrollAreaを1つ持つ。他は内部ウィジェット
（QTreeWidget/QTableWidget/QTextEdit等）が自前でスクロールを持つため0個）。
QStackedWidget全体を1つのQScrollAreaでラップしてはいけない（一番背の高い
ページに他のページも道連れでスクロールが出る。実装ノウハウに実測記録あり）。
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QSplitter,
    QStackedWidget,
)

from run.viewer_tools.wiring_viewer.page_classification import PageClassification
from run.viewer_tools.wiring_viewer.page_health import PageHealth
from run.viewer_tools.wiring_viewer.page_logs import PageLogs
from run.viewer_tools.wiring_viewer.page_organs import PageOrgans
from run.viewer_tools.wiring_viewer.page_prediction import PagePrediction
from run.viewer_tools.wiring_viewer.page_research_map import PageResearchMap
from run.viewer_tools.wiring_viewer.page_settings import PageSettings
from run.viewer_tools.wiring_viewer.page_wiring import PageWiring
from run.viewer_tools.wiring_viewer.theme import detect_system_theme, get_theme

# ページの並び（ナビの表示順とそのまま対応する）。
#   【工程C、2026-08-15】ここに本物のページを配線した。並び順はナビの表示順と
#   そのまま対応する（仕様の6ページ・順序どおり）。
#   【2026-08-15 追記】7ページ目「予測のしくみ（実測）」を追加（工程2）。
#   ユーザーが①配線図だけでは「何をもとに何を予測しているか分からない」と
#   評価したため、実測値ベースで予測の2経路（学習側/正解側）を見せる専用ページを足した。
#   【2026-08-15 追記（工程5）】8ページ目「研究マップ」を追加。「1ヶ月前と同じ提案を
#   繰り返される」という追加依頼に対し、実験ファイル×結果CSVの実測一覧を見せる。
PAGE_TITLES = ("配線図", "器官の説明", "設定一覧", "実測ログ", "区分一覧", "健康診断",
               "予測のしくみ（実測）", "研究マップ")


def _build_pages():
    """(タイトル, ページウィジェット) のリストを返す。

    【工程C、2026-08-15】各 page_*.py はいずれも `__init__(self, parent=None)`
    （引数なしで構築可）と `apply_theme(self, theme_name)` を持つ契約で
    作られている（工程Bで4人がそれぞれ個別検証済み）。ここではその契約どおりに
    素朴に構築するだけでよい。
    """
    return [
        ("配線図", PageWiring()),
        ("器官の説明", PageOrgans()),
        ("設定一覧", PageSettings()),
        ("実測ログ", PageLogs()),
        ("区分一覧", PageClassification()),
        ("健康診断", PageHealth()),
        ("予測のしくみ（実測）", PagePrediction()),
        ("研究マップ", PageResearchMap()),
    ]


class MainWindow(QMainWindow):
    """左＝ナビ（QListWidget）、右＝中身（QStackedWidget）。QSplitterで区切る。"""

    NAV_WIDTH_DEFAULT = 200

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("TaroMap ── 太郎の配線・設定を見る")
        self.resize(1400, 860)

        self._theme_name = detect_system_theme()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        self.nav = QListWidget()
        self.nav.setObjectName("navList")
        self.stack = QStackedWidget()
        self.stack.setObjectName("pageStack")

        self._pages = _build_pages()
        for title, widget in self._pages:
            item = QListWidgetItem(title)
            self.nav.addItem(item)
            self.stack.addWidget(widget)

        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(0)

        splitter.addWidget(self.nav)
        splitter.addWidget(self.stack)
        # 【なぜsetSizesか、2026-08-15】setMinimumWidthだけに頼らず、初期の
        #   見た目の比率をここで決める。ユーザーはドラッグで自由に変えられる
        #   （QSplitterなので、固定値で幅問題を作らない）。
        splitter.setSizes([self.NAV_WIDTH_DEFAULT, 1400 - self.NAV_WIDTH_DEFAULT])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)
        self._apply_theme(self._theme_name)
        self._hook_theme_change()

    # ------------------------------------------------------------ テーマ
    def _apply_theme(self, theme_name: str):
        """このアプリ専用の局所QSS（QApplication全体へは絶対に当てない）。"""
        self._theme_name = theme_name
        c = get_theme(theme_name)
        # 【なぜQApplication全体でなくウィジェット単位か、仕様より】
        #   windows11スタイルの角丸・フォーカスリング・ホバーアニメーションが
        #   QApplication.setStyleSheet()で失われることが実測されている。
        #   ⇒ ここではナビとページ台紙にだけ、最小限の色を当てる。
        self.setStyleSheet(f"""
            QMainWindow {{ background: {c['bg']}; }}
            QListWidget#navList {{
                background: {c['card']};
                color: {c['ink']};
                border: none;
                border-right: 1px solid {c['line']};
                font-size: 13px;
            }}
            QListWidget#navList::item {{ padding: 9px 12px; }}
            QListWidget#navList::item:selected {{
                background: {c['accent']};
                color: #ffffff;
            }}
            QStackedWidget#pageStack {{ background: {c['bg']}; }}
            QLabel {{ color: {c['ink']}; }}
        """)
        # 【なぜページ側にも伝播させるか、仕様より】このQSSはナビ/台紙にしか
        #   当たらない。各ページは自前で apply_theme(theme_name) を持っており、
        #   その中で自分の表・グラフ・バッジ色を塗り直す。ここで呼ばないと
        #   テーマ切替時にページの中身だけ古い配色のまま取り残される。
        for _title, widget in getattr(self, "_pages", ()):
            if hasattr(widget, "apply_theme"):
                widget.apply_theme(theme_name)

    def _hook_theme_change(self):
        """OSのダーク/ライト切り替えにライブで追従する。

        【なぜ、仕様より】QStyleHints.colorSchemeChangedシグナルがこの
        PySide6 6.11.1環境に存在することが実測済み。フックできなくても
        アプリを落とさない（初期テーマのまま動く）。
        """
        try:
            from PySide6.QtGui import QGuiApplication
            hints = QGuiApplication.styleHints()
            hints.colorSchemeChanged.connect(
                lambda _scheme: self._apply_theme(detect_system_theme()))
        except Exception:
            pass
