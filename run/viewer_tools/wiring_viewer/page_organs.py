"""TaroMap ②「器官の説明」ページ。

taro_core/src 配下（body/brain/senses/wrapper）のフォルダ階層をそのままツリー表示し、
選んだファイルのモジュールdocstring全文を右に出す。ファイル一覧は
`scan_docstrings.scan_docstrings()` を毎回呼び直して作る（キャッシュしない。ページを
開き直すたびに taro_core/src の最新状態を読み直す）。

【画面構成、仕様より】
左右2区画のみ（QSplitter）。左＝体の部位フィルタ(小さなQComboBox) + フォルダツリー、
右＝docstring詳細。7つ縦積み等は作らない。

【体の部位フィルタについて】
`body_regions.py` の `_BODY_GROUPS`（somatosensory_cortex.pyの触覚グループ定義）を
そのまま使う、ユーザー確定のフィルタ。独立ページにはしない。ファイル名からの
簡易的な推測でしかなく、厳密な配線対応ではないことを画面上にも明記している。

【捏造しない原則】
docstringが薄い（`scan_docstrings.THIN_DOCSTRING_THRESHOLD` 字未満）ファイルでも、
このページ側で文章を書き足したり内容を推測して埋めたりはしない。「要加筆」の
バッジを付けて、薄いなら薄いまま表示するだけ。

【QScrollAreaについて、main_window.pyの注意を踏まえて】
このページ自身は QScrollArea を作らない。左のツリー(QTreeWidget)・右の詳細表示
(QTextEdit)はどちらもウィジェット自身が内部スクロールを持つため、外側に
QScrollAreaを足す必要が無い（=「1ページに付き0〜1個のQScrollArea」の0側で満たす）。
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSplitter,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from run.viewer_tools.wiring_viewer.body_regions import (
    files_related_to_category,
    region_categories,
)
from run.viewer_tools.wiring_viewer.scan_docstrings import (
    THIN_DOCSTRING_THRESHOLD,
    scan_docstrings,
)
from run.viewer_tools.wiring_viewer.theme import detect_system_theme, get_theme

_ALL_LABEL = "すべて"


class PageOrgans(QWidget):
    """左＝フォルダツリー(+体部位フィルタ)、右＝選択ファイルのdocstring詳細。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("pageOrgans")
        self._theme = get_theme(detect_system_theme())
        self._all_entries = []

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_right())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 760])

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(splitter)

        self.reload()
        self.apply_theme(detect_system_theme())

    # ---------------------------------------------------------------- 構築
    def _build_left(self) -> QWidget:
        left = QWidget()
        left.setObjectName("organsLeft")
        lay = QVBoxLayout(left)
        lay.setContentsMargins(8, 8, 8, 8)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("体の部位で絞り込み:"))
        self.filter_combo = QComboBox()
        self.filter_combo.setObjectName("organsFilterCombo")
        self.filter_combo.addItem(_ALL_LABEL)
        for category in region_categories():
            self.filter_combo.addItem(category)
        self.filter_combo.currentTextChanged.connect(self._apply_filter)
        filter_row.addWidget(self.filter_combo, 1)
        lay.addLayout(filter_row)

        filter_note = QLabel(
            "注意：taro_core/src/senses/somatosensory_cortex.py の _BODY_GROUPS を"
            "使った簡易フィルタです。ファイル名からの推測であり、厳密な配線対応では"
            "ありません。")
        filter_note.setObjectName("organsFilterNote")
        filter_note.setWordWrap(True)
        lay.addWidget(filter_note)

        self.tree = QTreeWidget()
        self.tree.setObjectName("organsTree")
        self.tree.setHeaderHidden(True)
        self.tree.itemClicked.connect(self._on_item_clicked)
        lay.addWidget(self.tree, 1)

        return left

    def _build_right(self) -> QWidget:
        right = QWidget()
        right.setObjectName("organsRight")
        lay = QVBoxLayout(right)
        lay.setContentsMargins(12, 12, 12, 12)

        self.path_label = QLabel("左のツリーからファイルを選んでください。")
        self.path_label.setObjectName("organsPathLabel")
        self.path_label.setWordWrap(True)
        lay.addWidget(self.path_label)

        self.badge_label = QLabel("")
        self.badge_label.setObjectName("organsBadge")
        self.badge_label.setWordWrap(True)
        lay.addWidget(self.badge_label)

        self.docstring_view = QTextEdit()
        self.docstring_view.setObjectName("organsDocstring")
        self.docstring_view.setReadOnly(True)
        lay.addWidget(self.docstring_view, 1)

        threshold_note = QLabel(
            f"「要加筆」バッジは docstring の文字数が {THIN_DOCSTRING_THRESHOLD} 字未満"
            "のファイルに付けています。この数値に強い根拠はなく暫定値です（Tier3）。"
            "薄いdocstringの中身をこのアプリ側で書き足すことはしません。")
        threshold_note.setObjectName("organsThresholdNote")
        threshold_note.setWordWrap(True)
        lay.addWidget(threshold_note)

        return right

    # ---------------------------------------------------------------- データ
    def reload(self):
        """taro_core/src を読み直してツリーを再構築する（キャッシュしない）。

        工程C等、外から明示的に最新状態へ更新させたい呼び出し元向けに公開している。
        """
        self._all_entries = scan_docstrings()
        current = self.filter_combo.currentText() if hasattr(self, "filter_combo") else _ALL_LABEL
        self._apply_filter(current)

    def _apply_filter(self, category_text: str):
        if not category_text or category_text == _ALL_LABEL:
            entries = self._all_entries
        else:
            entries = files_related_to_category(category_text, self._all_entries)
        self._build_tree(entries)

    def _build_tree(self, entries):
        self.tree.clear()
        dir_items = {}
        for entry in sorted(entries, key=lambda e: (e["rel_dir"], e["filename"])):
            rel_dir = entry["rel_dir"]
            parent_item = None
            if rel_dir:
                accum = ""
                for part in rel_dir.split("/"):
                    accum = f"{accum}/{part}" if accum else part
                    if accum not in dir_items:
                        item = QTreeWidgetItem([part])
                        if parent_item is None:
                            self.tree.addTopLevelItem(item)
                        else:
                            parent_item.addChild(item)
                        dir_items[accum] = item
                    parent_item = dir_items[accum]

            label = entry["filename"]
            if entry["thin"]:
                label += "  [要加筆]"
            file_item = QTreeWidgetItem([label])
            file_item.setData(0, Qt.ItemDataRole.UserRole, entry)
            if parent_item is None:
                self.tree.addTopLevelItem(file_item)
            else:
                parent_item.addChild(file_item)

        self.tree.expandAll()

    def _on_item_clicked(self, item: QTreeWidgetItem, _column: int):
        entry = item.data(0, Qt.ItemDataRole.UserRole)
        if entry is None:
            # フォルダ行（ファイルのentryを持たない）がクリックされた。何もしない。
            return
        self._show_entry(entry)

    def _show_entry(self, entry: dict):
        self.path_label.setText(entry["path"])
        c = self._theme
        if entry["thin"]:
            self.badge_label.setText(
                f"要加筆（{entry['char_count']}字 < {THIN_DOCSTRING_THRESHOLD}字・暫定閾値）")
            self.badge_label.setStyleSheet(f"color: {c['role_deviation']}; font-weight: 600;")
        else:
            self.badge_label.setText(f"{entry['char_count']}字")
            self.badge_label.setStyleSheet(f"color: {c['sub']};")
        self.docstring_view.setPlainText(entry["docstring"] or "（docstringがありません）")

    # ---------------------------------------------------------------- テーマ
    def apply_theme(self, theme_name: str):
        """このウィジェット配下にだけ局所QSSを当てる（QApplication全体へは当てない）。"""
        c = get_theme(theme_name)
        self._theme = c
        self.setStyleSheet(f"""
            QWidget#organsLeft, QWidget#organsRight {{ background: {c['bg']}; }}
            QLabel {{ color: {c['ink']}; }}
            QLabel#organsFilterNote, QLabel#organsThresholdNote {{
                color: {c['sub']}; font-size: 11px;
            }}
            QTreeWidget#organsTree {{
                background: {c['card']};
                color: {c['ink']};
                border: 1px solid {c['line']};
            }}
            QTextEdit#organsDocstring {{
                background: {c['card']};
                color: {c['ink']};
                border: 1px solid {c['line']};
            }}
            QComboBox#organsFilterCombo {{
                background: {c['card']};
                color: {c['ink']};
                border: 1px solid {c['line']};
                padding: 2px 6px;
            }}
        """)
        # 選択中のバッジ色もテーマに合わせて更新する。
        if self.badge_label.text():
            selected = self.tree.currentItem()
            entry = selected.data(0, Qt.ItemDataRole.UserRole) if selected else None
            if entry is not None:
                self._show_entry(entry)
