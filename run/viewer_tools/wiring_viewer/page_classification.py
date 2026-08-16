"""TaroMap ⑤区分一覧ページ。

`run/wiring_map.NODES`（各要素9フィールド：id, jp, eng, col, tier, desc, fn,
primary_axis, deviation_keywords）を表で一覧化する。

【primary_axisがNoneの行の扱い、ユーザーの確定判断】
空欄にしない。空欄だと「宿題があること自体が見えない」状態になるので、必ず
「未分類」という文字とバッジ色（role_unclassified）を出す。値が
"innate"/"emergent"/"tool"になっている行があれば、対応する
role_innate/role_emergent/role_tool色のバッジにする。

【逸脱リストとの突合せ、ユーザーの確定判断】
行を選ぶと、その jp（日本語名）をキーワードに search_deviations() を呼ぶ。
  ヒットあり → 該当箇所（見出し・行番号・本文）を列挙して見せる。
  ヒットなし → 「未記載の可能性（該当なしとは断定しない）」という**中立**な文言。
  「✅許容」のような安全側の表示は絶対にしない（失敗判定に明記された必須項目）。
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from run.viewer_tools.wiring_viewer.search_deviation_list import search_deviations
from run.viewer_tools.wiring_viewer.theme import get_theme
from run.wiring_map import NODES

_COLUMNS = ("id", "日本語名", "列", "根拠(Tier)", "区分(primary_axis)", "逸脱キーワード")

_AXIS_LABEL = {
    None: "未分類",
    "innate": "生得",
    "emergent": "創発",
    "tool": "道具",
}
_AXIS_ROLE_KEY = {
    None: "role_unclassified",
    "innate": "role_innate",
    "emergent": "role_emergent",
    "tool": "role_tool",
}

_ANY = "(すべて)"


class PageClassification(QWidget):
    """⑤区分一覧ページ。上＝一覧表とフィルタ、下（右）＝選択行の逸脱リスト突合せ結果。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        # NODESの各要素は9フィールド固定のタプル。決め打ちインデックスで展開する
        #   （wiring_map.py側の並びはコメントに明記されており、勝手に増減できない）。
        self._rows = []
        for node in NODES:
            (node_id, jp, eng, col, tier, desc, fn, primary_axis,
             deviation_keywords) = node
            self._rows.append({
                "id": node_id, "jp": jp, "eng": eng, "col": col, "tier": tier,
                "desc": desc, "primary_axis": primary_axis,
                "deviation_keywords": tuple(deviation_keywords or ()),
            })

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        title = QLabel(f"区分一覧（run/wiring_map.NODES、全{len(self._rows)}件）")
        title.setObjectName("pageTitle")
        outer.addWidget(title)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Tier:"))
        self._tier_combo = QComboBox()
        self._tier_combo.addItem(_ANY)
        for tier in sorted({str(r["tier"]) for r in self._rows}):
            self._tier_combo.addItem(tier)
        self._tier_combo.currentTextChanged.connect(self._refresh_table)
        filter_row.addWidget(self._tier_combo)

        filter_row.addWidget(QLabel("区分:"))
        self._axis_combo = QComboBox()
        self._axis_combo.addItem(_ANY)
        for axis in (None, "innate", "emergent", "tool"):
            self._axis_combo.addItem(_AXIS_LABEL[axis])
        self._axis_combo.currentTextChanged.connect(self._refresh_table)
        filter_row.addWidget(self._axis_combo)
        filter_row.addStretch(1)
        outer.addLayout(filter_row)

        splitter = QSplitter(Qt.Orientation.Vertical)

        self._table = QTableWidget()
        self._table.setColumnCount(len(_COLUMNS))
        self._table.setHorizontalHeaderLabels(_COLUMNS)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.setSortingEnabled(True)
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        splitter.addWidget(self._table)

        detail_panel = QWidget()
        detail_layout = QVBoxLayout(detail_panel)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        self._detail_label = QLabel("行を選ぶと、逸脱リストとの突合せ結果をここに表示します。")
        self._detail_label.setObjectName("detailLabel")
        self._detail_label.setWordWrap(True)
        detail_layout.addWidget(self._detail_label)
        self._detail_text = QPlainTextEdit()
        self._detail_text.setReadOnly(True)
        detail_layout.addWidget(self._detail_text)
        splitter.addWidget(detail_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        outer.addWidget(splitter, stretch=1)

        self._refresh_table()

    # ------------------------------------------------------------ 一覧
    def _filtered_rows(self):
        tier_sel = self._tier_combo.currentText()
        axis_sel = self._axis_combo.currentText()
        rows = self._rows
        if tier_sel != _ANY:
            rows = [r for r in rows if str(r["tier"]) == tier_sel]
        if axis_sel != _ANY:
            rows = [r for r in rows if _AXIS_LABEL[r["primary_axis"]] == axis_sel]
        return rows

    def _refresh_table(self, *_args):
        rows = self._filtered_rows()
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            self._table.setItem(i, 0, QTableWidgetItem(r["id"]))
            self._table.setItem(i, 1, QTableWidgetItem(r["jp"]))
            self._table.setItem(i, 2, QTableWidgetItem(r["col"]))
            self._table.setItem(i, 3, QTableWidgetItem(str(r["tier"])))

            axis_item = QTableWidgetItem(_AXIS_LABEL[r["primary_axis"]])
            axis_item.setData(Qt.ItemDataRole.UserRole, r["primary_axis"])
            self._table.setItem(i, 4, axis_item)

            kw_text = "、".join(r["deviation_keywords"]) if r["deviation_keywords"] else "(なし)"
            self._table.setItem(i, 5, QTableWidgetItem(kw_text))
        self._table.setSortingEnabled(True)
        self._colorize_axis_column()
        self._detail_text.clear()
        self._detail_label.setText("行を選ぶと、逸脱リストとの突合せ結果をここに表示します。")

    def _colorize_axis_column(self):
        """区分バッジの色付け。primary_axis=Noneの行は必ず role_unclassified 色になる。"""
        c = get_theme(self._theme_name if hasattr(self, "_theme_name") else "windows11")
        for i in range(self._table.rowCount()):
            item = self._table.item(i, 4)
            if item is None:
                continue
            axis = item.data(Qt.ItemDataRole.UserRole)
            role_key = _AXIS_ROLE_KEY[axis]
            from PySide6.QtGui import QBrush, QColor
            item.setForeground(QBrush(QColor(c[role_key])))

    # ------------------------------------------------------------ 詳細
    def _on_selection_changed(self):
        items = self._table.selectedItems()
        if not items:
            return
        row_idx = items[0].row()
        jp_item = self._table.item(row_idx, 1)
        if jp_item is None:
            return
        jp = jp_item.text()
        hits = search_deviations(jp)
        self._detail_label.setText(f"「{jp}」の逸脱リストとの突合せ結果")
        if hits:
            lines = [f"{len(hits)}件ヒット（逸脱リスト.md）", ""]
            for h in hits:
                lines.append(f"[行{h['line_no']}] 見出し: {h['heading']}")
                lines.append(f"    {h['matched_line']}")
                lines.append("")
            self._detail_text.setPlainText("\n".join(lines))
        else:
            # 【ユーザーの確定判断】ヒット0件を「問題なし」「許容」とは絶対に表示しない。
            #   中立的に「未記載の可能性がある」とだけ伝える。
            self._detail_text.setPlainText(
                "逸脱リストに「" + jp + "」を含む行は見つからなかった。\n"
                "未記載の可能性がある（該当なしとは断定しない）。\n"
                "人間模倣からの逸脱リスト.md を直接確認すること。")

    # ------------------------------------------------------------ テーマ
    def apply_theme(self, theme_name: str):
        self._theme_name = theme_name
        c = get_theme(theme_name)
        self.setStyleSheet(f"""
            QLabel#pageTitle {{ color: {c['ink']}; font-size: 15px; font-weight: 600; }}
            QLabel#detailLabel {{ color: {c['ink']}; font-weight: 600; }}
            QLabel {{ color: {c['ink']}; }}
            QComboBox {{
                background: {c['card']}; color: {c['ink']};
                border: 1px solid {c['line']}; border-radius: 4px; padding: 2px 6px;
            }}
            QTableWidget {{
                background: {c['card']}; color: {c['ink']};
                gridline-color: {c['line']}; alternate-background-color: {c['bg']};
            }}
            QHeaderView::section {{
                background: {c['bg']}; color: {c['ink']};
                border: none; border-bottom: 1px solid {c['line']}; padding: 4px;
            }}
            QPlainTextEdit {{
                background: {c['card']}; color: {c['ink']};
                border: 1px solid {c['line']};
            }}
        """)
        self._colorize_axis_column()
