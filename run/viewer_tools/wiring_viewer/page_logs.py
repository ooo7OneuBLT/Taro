"""TaroMap ④実測ログページ。

左：E/logs配下のラン一覧（起動時は列名すら読まず、ファイル名の列挙だけで高速に表示）。
右：選んだCSV1本だけを pyqtgraph でグラフ化する（ズーム・パンはpyqtgraph標準搭載）。

【なぜ複数CSVを重ねて描かないか、プロジェクトの流儀（feedback-per-sim-graph）】
1つのCSV＝1回の学習ランの記録。複数ランを混ぜて1つのグラフにすると、
「条件の違い」なのか「シードのばらつき」なのか区別できなくなる。
このページは常に「1回選んだら1本のグラフ」のみを描く。

【なぜ起動時に790本を全部読まないか】
`scan_logs.list_log_runs()` はファイル名の一覧だけを返す（中身は読まない）。
実際にCSVの中身（`read_csv_header` → `read_csv_data`）を読むのは、ユーザーが
ツリーでCSVを選び、さらに列を選んだ、その瞬間だけ。
"""
from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

import pyqtgraph as pg

from run.viewer_tools.wiring_viewer.scan_logs import (
    list_log_runs,
    read_csv_data,
    read_csv_header,
)
from run.viewer_tools.wiring_viewer.theme import get_theme

# ステップ列は横軸に固定（どのCSVにも存在する前提の共通列）。無ければ先頭列を使う。
_X_CANDIDATES = ("step", "life_min", "age_months")


class PageLogs(QWidget):
    """左＝ランのツリー（起動時は高速）、右＝選んだ1本のグラフ（遅延読み込み）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._theme_name = "windows11"
        self._current_csv_path = None
        self._current_header = []

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        # ---- 左：ラン一覧 ------------------------------------------------
        left = QWidget()
        left_lay = QVBoxLayout(left)
        self._left_title = QLabel("実測ログ（E/logs）")
        left_lay.addWidget(self._left_title)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        left_lay.addWidget(self.tree)
        self._runs_count_label = QLabel("")
        left_lay.addWidget(self._runs_count_label)

        # ---- 右：グラフ ---------------------------------------------------
        right = QWidget()
        right_lay = QVBoxLayout(right)
        top_row = QHBoxLayout()
        self._selected_label = QLabel("（左でCSVを選んでください）")
        top_row.addWidget(self._selected_label)
        top_row.addStretch(1)
        top_row.addWidget(QLabel("列:"))
        self.column_combo = QComboBox()
        self.column_combo.setMinimumWidth(220)
        top_row.addWidget(self.column_combo)
        right_lay.addLayout(top_row)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        right_lay.addWidget(self.plot_widget)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([260, 900])
        root_layout.addWidget(splitter)

        self._populate_tree()

        self.tree.currentItemChanged.connect(self._on_tree_selection)
        self.column_combo.currentTextChanged.connect(self._on_column_changed)

        self.apply_theme(self._theme_name)

    # -------------------------------------------------------------- 一覧
    def _populate_tree(self):
        """list_log_runs()の結果だけでツリーを組む（CSVの中身には触れない＝高速）。"""
        self.tree.clear()
        runs = list_log_runs()
        total_csv = 0
        for run in runs:
            run_item = QTreeWidgetItem([f"{run['run_name']}  ({len(run['csv_files'])}本)"])
            run_item.setData(0, Qt.ItemDataRole.UserRole, None)
            for csv_name in run["csv_files"]:
                csv_item = QTreeWidgetItem([csv_name])
                csv_item.setData(0, Qt.ItemDataRole.UserRole,
                                  os.path.join(run["dir"], csv_name))
                run_item.addChild(csv_item)
            self.tree.addTopLevelItem(run_item)
            total_csv += len(run["csv_files"])
        self._runs_count_label.setText(f"ラン数: {len(runs)}  /  CSV総数: {total_csv}")

    # -------------------------------------------------------------- 選択
    def _on_tree_selection(self, current, _previous):
        if current is None:
            return
        path = current.data(0, Qt.ItemDataRole.UserRole)
        if not path:
            return  # ランの見出し行（CSVでない）を選んだ場合は何もしない
        self._load_csv(path)

    def _load_csv(self, csv_path: str):
        """選ばれた1本だけをここで初めて読む（列名だけ先に読み、ドロップダウンを作る）。"""
        self._current_csv_path = csv_path
        header = read_csv_header(csv_path)
        self._current_header = header
        self._selected_label.setText(f"選択中: {os.path.basename(csv_path)}")

        self.column_combo.blockSignals(True)
        self.column_combo.clear()
        # x軸候補以外を選択肢にする（x軸自体をy軸として選んでも困らないが、実用上は除く）
        y_candidates = [c for c in header if c not in _X_CANDIDATES] or header
        self.column_combo.addItems(y_candidates)
        self.column_combo.blockSignals(False)

        if y_candidates:
            self._draw(csv_path, y_candidates[0])

    def _on_column_changed(self, column_name: str):
        if not column_name or not self._current_csv_path:
            return
        self._draw(self._current_csv_path, column_name)

    def _draw(self, csv_path: str, y_col: str):
        """列を選んだ瞬間だけCSVの中身を読み、1本のグラフを描く。"""
        x_col = next((c for c in _X_CANDIDATES if c in self._current_header), None)
        columns = [c for c in (x_col, y_col) if c]
        data = read_csv_data(csv_path, columns=columns)

        self.plot_widget.clear()
        if y_col not in data:
            return
        y = data[y_col]
        x = data[x_col] if x_col and x_col in data else None
        theme = get_theme(self._theme_name)
        pen = pg.mkPen(color=theme["accent"], width=2)
        if x is not None:
            self.plot_widget.plot(x, y, pen=pen)
            self.plot_widget.setLabel("bottom", x_col)
        else:
            self.plot_widget.plot(y, pen=pen)
        self.plot_widget.setLabel("left", y_col)
        self.plot_widget.setTitle(f"{os.path.basename(csv_path)} ── {y_col}"
                                   f"（1ラン単独。他ランと混ぜていません）")

    # -------------------------------------------------------------- テーマ
    def apply_theme(self, theme_name: str):
        self._theme_name = theme_name
        c = get_theme(theme_name)
        self.setStyleSheet(f"""
            QWidget {{ background: {c['bg']}; color: {c['ink']}; }}
            QTreeWidget {{ background: {c['card']}; color: {c['ink']};
                           border: 1px solid {c['line']}; }}
            QComboBox {{ background: {c['card']}; color: {c['ink']};
                         border: 1px solid {c['line']}; padding: 2px 6px; }}
        """)
        self.plot_widget.setBackground(c["card"])
        self.plot_widget.getAxis("bottom").setPen(pg.mkPen(c["sub"]))
        self.plot_widget.getAxis("left").setPen(pg.mkPen(c["sub"]))
        self.plot_widget.getAxis("bottom").setTextPen(pg.mkPen(c["ink"]))
        self.plot_widget.getAxis("left").setTextPen(pg.mkPen(c["ink"]))
        # 既に何か描いていれば色（accent）を合わせて再描画する
        if self._current_csv_path and self.column_combo.currentText():
            self._draw(self._current_csv_path, self.column_combo.currentText())
