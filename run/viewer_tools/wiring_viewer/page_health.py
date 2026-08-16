"""TaroMap ⑥健康診断ページ。

`health_check.find_unwired_classes()` と `health_check.find_uncovered_bool_settings()`
の結果を2つの表で見せる。

【ユーザーの確定判断、仕様より】どちらも完全な保証ができないヒューリスティックで
あり、誤検出がありうる。「エラー」「壊れている」という断定表示はせず、
「要確認（自動判定、誤検出がありうる）」という中立的な言葉を画面の目立つ位置
（各表の直上）に明記する。安全側の断定表示をしない。
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHeaderView,
    QLabel,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from run.viewer_tools.wiring_viewer.health_check import (
    find_uncovered_bool_settings,
    find_unwired_classes,
)
from run.viewer_tools.wiring_viewer.theme import get_theme

_NOTICE_TEXT = ("この下の一覧は自動ヒューリスティックによる「要確認」であり、"
                "「エラー」「壊れている」の確定判定ではありません。"
                "誤検出（正当に配線が無いだけのもの）がありえます。")


class PageHealth(QWidget):
    """1ページ＝1つのQScrollArea以下（main_window.pyの注意事項を厳守）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._theme_name = "windows11"

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        lay = QVBoxLayout(inner)

        self._notice_label = QLabel(_NOTICE_TEXT)
        self._notice_label.setWordWrap(True)
        self._notice_label.setObjectName("healthNotice")
        lay.addWidget(self._notice_label)

        # ---- (a) 未配線検出 ----------------------------------------------
        self._unwired_title = QLabel()
        lay.addWidget(self._unwired_title)
        self.unwired_table = QTableWidget()
        self.unwired_table.setColumnCount(2)
        self.unwired_table.setHorizontalHeaderLabels(["クラス名", "ファイル"])
        self.unwired_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        self.unwired_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        lay.addWidget(self.unwired_table)

        # ---- (b) bool設定カバレッジ ----------------------------------------
        self._bool_title = QLabel()
        lay.addWidget(self._bool_title)
        self.bool_table = QTableWidget()
        self.bool_table.setColumnCount(2)
        self.bool_table.setHorizontalHeaderLabels(["設定キー", "説明"])
        self.bool_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        self.bool_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        lay.addWidget(self.bool_table)

        scroll.setWidget(inner)
        outer.addWidget(scroll)

        self._run_checks()
        self.apply_theme(self._theme_name)

    # -------------------------------------------------------------- 実行
    def _run_checks(self):
        unwired_result = find_unwired_classes()
        self._unwired_title.setText(
            "(a) 未配線検出 ── 要確認（自動判定）　"
            f"taro_core全クラス数={unwired_result['total_classes']} / "
            f"構築を確認できた数={unwired_result['constructed_classes']} / "
            f"未配線とみなした数={len(unwired_result['unwired'])}")
        unwired = unwired_result["unwired"]
        self.unwired_table.setRowCount(len(unwired))
        for row, c in enumerate(unwired):
            self.unwired_table.setItem(row, 0, QTableWidgetItem(c["name"]))
            self.unwired_table.setItem(row, 1, QTableWidgetItem(c["file"]))

        bool_result = find_uncovered_bool_settings()
        self._bool_title.setText(
            "(b) config.pyのbool設定・配線カバレッジ ── 要確認（自動判定）　"
            f"bool設定総数={bool_result['total_bool_settings']} / "
            f"要確認とみなした数={len(bool_result['uncovered'])}")
        uncovered = bool_result["uncovered"]
        self.bool_table.setRowCount(len(uncovered))
        for row, u in enumerate(uncovered):
            self.bool_table.setItem(row, 0, QTableWidgetItem(u["key"]))
            self.bool_table.setItem(row, 1, QTableWidgetItem(u["doc"]))

    # -------------------------------------------------------------- テーマ
    def apply_theme(self, theme_name: str):
        self._theme_name = theme_name
        c = get_theme(theme_name)
        self.setStyleSheet(f"""
            QWidget {{ background: {c['bg']}; color: {c['ink']}; }}
            QLabel#healthNotice {{
                background: {c['card']};
                color: {c['role_provisional']};
                border: 1px dashed {c['role_provisional']};
                padding: 8px;
                font-weight: bold;
            }}
            QTableWidget {{ background: {c['card']}; color: {c['ink']};
                            border: 1px solid {c['line']}; gridline-color: {c['line']}; }}
            QHeaderView::section {{ background: {c['card']}; color: {c['sub']};
                                    border: none; border-bottom: 1px solid {c['line']}; }}
        """)
