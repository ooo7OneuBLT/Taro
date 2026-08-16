"""TaroMap ⑧「研究マップ」ページ（工程5）。

【なぜ要るか、仕様より】
研究者（高校生1人）が「1ヶ月前と同じ提案を繰り返される」と困っている。原因は
実験記録が全て時系列（研究日誌・実験ファイル）で、「今どうなっているか」
「もう試したか」を引ける形になっていないこと。①は`scan_research_map.
list_experiments()`（工程4）が実測した実験ファイル×結果CSVの一覧をそのまま
検索・並べ替え可能な表で見せる。②は`E/docs/研究マップ_道筋.json`
（研究者本人が手で書く想定のファイル）を読んで表示する補助セクション。

【このページが触ってよいもの・触らないもの】
  ・scan_research_map.py はimportして使うだけ（変更しない・自分で再パースしない）。
  ・page_settings.py（③設定一覧）と同じ「検索欄＋QTableWidget」の考え方を踏襲する
    （コピーではなく、このページの列構成に合わせて作り直す）。
  ・page_wiring.py・page_prediction.py は変更しない。

【①・②の中身を実装担当が創作しないこと、仕様の禁止事項より】
研究マップ_道筋.json・研究マップ_解釈.json の実際の中身（夢・目標・分岐点・解釈）は
研究者本人が書くもの。このページはひな形（空）のときに案内文を出すだけで、
実際の値を作文しない。
"""
from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractItemView,
    QFrame,
    QHeaderView,
    QLabel,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from run.viewer_tools.wiring_viewer import scan_research_map  # noqa: E402
from run.viewer_tools.wiring_viewer.theme import get_theme  # noqa: E402

_MICHISUJI_PATH = os.path.join(_ROOT, "E", "docs", "研究マップ_道筋.json")

_COLUMNS = ("日付", "名前", "note", "結果の数値", "解釈")
_NOTE_PREVIEW_LEN = 40


def _format_result_numbers(entry: dict) -> str:
    """結果の数値をセルに出す1行の文字列にする。

    csv_existsがFalseなら「CSVなし」。Trueで結果の数値がNoneまたは空なら
    「測れず」（NaNをそのまま数字として出さない、仕様の注意）。
    """
    if not entry.get("csv_exists"):
        return "CSVなし"
    値 = entry.get("結果の数値")
    if not 値:
        return "測れず"
    parts = []
    for k, v in 値.items():
        if isinstance(v, float) and v != v:  # NaN判定（v != v はNaNのときだけTrue）
            parts.append(f"{k}=測れず")
        else:
            parts.append(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}")
    return "　".join(parts)


def _note_preview(note: str) -> str:
    note = note or ""
    if len(note) <= _NOTE_PREVIEW_LEN:
        return note
    return note[:_NOTE_PREVIEW_LEN] + "…"


def _sort_key_日付(entry: dict):
    """日付の新しい順に並べる既定キー。日付が無い行は末尾に送る。"""
    d = entry.get("日付")
    if not d:
        return ("9999-99-99",)
    return (d,)


class _ExperimentTable(QTableWidget):
    """①「試したこと一覧」の表本体。検索フィルタはPageResearchMapが呼ぶ。"""

    def __init__(self, rows: list[dict], parent=None):
        super().__init__(parent)
        # 既定は日付の新しい順（無いものは末尾）。
        self._all_rows = sorted(rows, key=_sort_key_日付, reverse=True)
        self.setColumnCount(len(_COLUMNS))
        self.setHorizontalHeaderLabels(_COLUMNS)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(True)
        self.verticalHeader().setVisible(False)
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Interactive)
        self.set_filter("")

    def set_filter(self, text: str):
        """name・note・解釈のいずれかにtextを含む行だけ表示する（大小文字無視）。"""
        # 【なぜsetSortingEnabledを一時的に切るか】setRowCount中に並べ替えが
        # 走ると行の対応がずれる（Qtの既知の落とし穴。page_settings.pyには
        # setSortingEnabledが無いためここで新たに気をつける必要がある）。
        self.setSortingEnabled(False)
        needle = text.strip().lower()
        if needle:
            rows = [r for r in self._all_rows if
                    needle in str(r.get("name", "")).lower()
                    or needle in str(r.get("note", "")).lower()
                    or needle in str(r.get("解釈", "")).lower()]
        else:
            rows = self._all_rows
        self.setRowCount(len(rows))
        for i, r in enumerate(rows):
            日付 = r.get("日付") or "(不明)"
            date_item = QTableWidgetItem(日付)
            if r.get("日付の出どころ") == "ファイル更新日時（推定）":
                date_item.setToolTip("noteに日付が書かれていないため、ファイルの更新日時から推定した値です。")
            self.setItem(i, 0, date_item)

            name_item = QTableWidgetItem(r.get("name") or "(名前なし)")
            name_item.setToolTip(r.get("path", ""))
            self.setItem(i, 1, name_item)

            note_full = r.get("note") or ""
            note_item = QTableWidgetItem(_note_preview(note_full))
            note_item.setToolTip(note_full if note_full else "(noteなし)")
            self.setItem(i, 2, note_item)

            result_item = QTableWidgetItem(_format_result_numbers(r))
            result_item.setToolTip(result_item.text())
            self.setItem(i, 3, result_item)

            解釈 = r.get("解釈") or "未記入"
            interp_item = QTableWidgetItem(解釈)
            if 解釈 == "未記入":
                interp_item.setForeground(Qt.GlobalColor.gray)
                f = interp_item.font()
                f.setItalic(True)
                interp_item.setFont(f)
            self.setItem(i, 4, interp_item)
        self.setSortingEnabled(True)


class PageResearchMap(QWidget):
    """⑧研究マップページ。①試したこと一覧（主役）＋②研究の道筋（補助）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("pageResearchMap")

        self._result = scan_research_map.list_experiments(use_cache=True)
        self._entries = self._result["実験"]

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        title = QLabel("研究マップ ── 何をもう試したか、どこまで来ているか")
        title.setObjectName("pageTitle")
        outer.addWidget(title)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_tab1(), f"① 試したこと一覧（{len(self._entries)}）")
        self._tabs.addTab(self._build_tab2(), "② 研究の道筋")
        outer.addWidget(self._tabs, stretch=1)

    # ---------------------------------------------------- ① 試したこと一覧
    def _build_tab1(self) -> QWidget:
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        self._search = QLineEdit()
        self._search.setPlaceholderText("名前・note・解釈で絞り込み（例: リクライニング）")
        self._search.textChanged.connect(self._on_search_changed)
        lay.addWidget(self._search)

        self._table = _ExperimentTable(self._entries)
        lay.addWidget(self._table, stretch=1)

        self._count_label = QLabel()
        lay.addWidget(self._count_label)
        self._update_count_label()

        gen = self._result.get("生成時刻", "不明")
        from_cache = self._result.get("キャッシュから", False)
        cache_txt = "キャッシュから読みました" if from_cache else "いま実測しました"
        foot = QLabel(f"実測時刻：{gen}　（{cache_txt}）　"
                       "注意：「解釈」は E/docs/研究マップ_解釈.json に手で書いた内容です。"
                       "未記入は本人が未着手か、まだ書いていないだけの可能性があります。")
        foot.setObjectName("rmFootNote")
        foot.setWordWrap(True)
        lay.addWidget(foot)
        return tab

    def _on_search_changed(self, text: str):
        self._table.set_filter(text)
        self._update_count_label()

    def _update_count_label(self):
        self._count_label.setText(
            f"表示中: {self._table.rowCount()}/{len(self._entries)}件")

    # ------------------------------------------------------- ② 研究の道筋
    def _build_tab2(self) -> QWidget:
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        data = self._load_michisuji()
        夢 = (data.get("夢") or "").strip()
        階層 = data.get("階層") or []
        分岐点 = data.get("分岐点") or []

        if not 夢 and not 階層 and not 分岐点:
            note = QLabel(
                "まだ書かれていません"
                f"（{os.path.relpath(_MICHISUJI_PATH, _ROOT).replace(os.sep, '/')} "
                "に手で書くと表示されます）。")
            note.setObjectName("rmEmptyNote")
            note.setWordWrap(True)
            lay.addWidget(note)
            lay.addStretch(1)
            return tab

        if 夢:
            dream_box = QFrame()
            dream_box.setObjectName("rmSectionBox")
            dl = QVBoxLayout(dream_box)
            dl.addWidget(_section_title("夢"))
            dream_lbl = QLabel(夢)
            dream_lbl.setWordWrap(True)
            dl.addWidget(dream_lbl)
            lay.addWidget(dream_box)

        if 階層:
            level_box = QFrame()
            level_box.setObjectName("rmSectionBox")
            ll = QVBoxLayout(level_box)
            ll.addWidget(_section_title("階層（長期〜今ここ）"))
            for item in 階層:
                if not isinstance(item, dict):
                    continue
                text = "　".join(
                    f"{k}: {v}" for k, v in item.items() if str(v).strip())
                if not text:
                    continue
                lbl = QLabel(text)
                lbl.setWordWrap(True)
                ll.addWidget(lbl)
            lay.addWidget(level_box)
        else:
            lay.addWidget(QLabel("階層：まだ書かれていません。"))

        if 分岐点:
            branch_box = QFrame()
            branch_box.setObjectName("rmSectionBox")
            bl = QVBoxLayout(branch_box)
            bl.addWidget(_section_title("分岐点（選んだもの／捨てたもの）"))
            for item in 分岐点:
                if not isinstance(item, dict):
                    continue
                text = "　".join(
                    f"{k}: {v}" for k, v in item.items() if str(v).strip())
                if not text:
                    continue
                lbl = QLabel(text)
                lbl.setWordWrap(True)
                bl.addWidget(lbl)
            lay.addWidget(branch_box)
        else:
            lay.addWidget(QLabel("分岐点：まだ書かれていません。"))

        lay.addStretch(1)
        return tab

    def _load_michisuji(self) -> dict:
        """研究マップ_道筋.json を読む。無い・壊れている・空でも例外を出さない
        （scan_research_map.py の_load_interpretations()と同じ考え方）。"""
        try:
            with open(_MICHISUJI_PATH, encoding="utf-8") as fp:
                data = json.load(fp)
        except (OSError, ValueError):
            return {}
        if not isinstance(data, dict):
            return {}
        return data

    # ------------------------------------------------------------ テーマ
    def apply_theme(self, theme_name: str):
        c = get_theme(theme_name)
        self.setStyleSheet(f"""
            QWidget#pageResearchMap {{ background: {c['bg']}; }}
            QLabel#pageTitle {{ color: {c['ink']}; font-size: 15px; font-weight: 600; }}
            QLabel {{ color: {c['ink']}; }}
            QLabel#rmFootNote {{ color: {c['sub']}; font-size: 11px; }}
            QLabel#rmEmptyNote {{ color: {c['sub']}; font-size: 13px; padding: 10px; }}
            QLabel#rmSectionTitle {{
                font-size: 14px; font-weight: 700; color: {c['ink']};
            }}
            QFrame#rmSectionBox {{
                background: {c['card']}; border: 1px solid {c['line']};
                border-radius: 8px; padding: 6px;
            }}
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


def _section_title(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("rmSectionTitle")
    return lbl


if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    w = PageResearchMap()
    w.apply_theme("windows11")
    w.resize(1200, 800)
    w.show()
    sys.exit(app.exec())
