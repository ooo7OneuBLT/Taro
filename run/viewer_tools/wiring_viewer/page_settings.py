"""TaroMap ③設定一覧ページ。

`scan_config.scan_config()` の結果（taro欄・run欄）を表で見せる。
1画面に主役は1つ：上部にtaro/runの切替タブ、その下に検索欄＋表。7つ縦積みにしない。

【幅問題への対処】setMinimumWidth() の固定値だけに頼らない。QTableWidget自体は
QTabWidget→QWidget→QVBoxLayoutの中に入り、列幅は Stretch + interactive の
組み合わせにして、ウィンドウ幅が狭くても横スクロールで内容が読めるようにする
（`main_window.py`のQSplitterで幅そのものはユーザーがドラッグして変えられる）。
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from run.viewer_tools.wiring_viewer.scan_config import scan_config
from run.viewer_tools.wiring_viewer.theme import get_theme

_COLUMNS = ("キー", "既定値", "説明", "環境変数名")


def _value_to_text(v) -> str:
    """既定値をテーブルに出す文字列にする。None・list・dictも読める形にする。"""
    if v is None:
        return "(None)"
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


class _ConfigTable(QTableWidget):
    """taro欄 or run欄の1枚の表。検索フィルタは呼び出し側（PageSettings）が持つ。"""

    def __init__(self, rows: list[dict], parent=None):
        super().__init__(parent)
        self._all_rows = rows
        self.setColumnCount(len(_COLUMNS))
        self.setHorizontalHeaderLabels(_COLUMNS)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setAlternatingRowColors(True)
        self.verticalHeader().setVisible(False)
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.set_filter("")

    def set_filter(self, text: str):
        """キー名 or 説明文に text を含む行だけ表示する（大小文字を無視）。"""
        needle = text.strip().lower()
        if needle:
            rows = [r for r in self._all_rows
                    if needle in r["key"].lower() or needle in str(r["doc"]).lower()]
        else:
            rows = self._all_rows
        self.setRowCount(len(rows))
        for i, r in enumerate(rows):
            self.setItem(i, 0, QTableWidgetItem(r["key"]))
            self.setItem(i, 1, QTableWidgetItem(_value_to_text(r["default"])))
            self.setItem(i, 2, QTableWidgetItem(str(r["doc"])))
            self.setItem(i, 3, QTableWidgetItem(r["envname"] or "(なし)"))


class PageSettings(QWidget):
    """③設定一覧ページ。taro欄／run欄をタブで切替。検索欄でキー・説明を絞り込む。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        result = scan_config()
        self._taro_rows = result["taro"]
        self._run_rows = result["run"]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        title = QLabel("設定一覧（run/config.py の TARO_DEFAULTS / RUN_DEFAULTS）")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        self._search = QLineEdit()
        self._search.setPlaceholderText("キー名・説明文で絞り込み（例: touch）")
        self._search.textChanged.connect(self._on_search_changed)
        layout.addWidget(self._search)

        self._tabs = QTabWidget()
        self._taro_table = _ConfigTable(self._taro_rows)
        self._run_table = _ConfigTable(self._run_rows)
        self._tabs.addTab(self._taro_table, f"taro（{len(self._taro_rows)}）")
        self._tabs.addTab(self._run_table, f"run（{len(self._run_rows)}）")
        layout.addWidget(self._tabs, stretch=1)

        self._count_label = QLabel()
        layout.addWidget(self._count_label)
        self._update_count_label()

    def _on_search_changed(self, text: str):
        self._taro_table.set_filter(text)
        self._run_table.set_filter(text)
        self._update_count_label()

    def _update_count_label(self):
        self._count_label.setText(
            f"taro: {self._taro_table.rowCount()}/{len(self._taro_rows)}件　"
            f"run: {self._run_table.rowCount()}/{len(self._run_rows)}件")

    # ------------------------------------------------------------ テーマ
    def apply_theme(self, theme_name: str):
        c = get_theme(theme_name)
        self.setStyleSheet(f"""
            QLabel#pageTitle {{ color: {c['ink']}; font-size: 15px; font-weight: 600; }}
            QLabel {{ color: {c['ink']}; }}
            QLineEdit {{
                background: {c['card']}; color: {c['ink']};
                border: 1px solid {c['line']}; border-radius: 4px; padding: 4px 8px;
            }}
            QTabWidget::pane {{ border: 1px solid {c['line']}; }}
            QTabBar::tab {{
                background: {c['bg']}; color: {c['ink']};
                padding: 6px 12px; border: 1px solid {c['line']};
            }}
            QTabBar::tab:selected {{ background: {c['card']}; }}
            QTableWidget {{
                background: {c['card']}; color: {c['ink']};
                gridline-color: {c['line']}; alternate-background-color: {c['bg']};
            }}
            QHeaderView::section {{
                background: {c['bg']}; color: {c['ink']};
                border: none; border-bottom: 1px solid {c['line']}; padding: 4px;
            }}
        """)
