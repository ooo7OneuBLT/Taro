"""TaroMap ⑧「研究マップ」ページ（page_research_map.py）の検証（工程5）。

PySide6が要るためQT_QPA_PLATFORM=offscreenで実行する（GUIを実際に開かない）。
既存の check_wiring_viewer_prediction_page.py と同じ流儀に合わせる。

使い方：
    set QT_QPA_PLATFORM=offscreen
    .venv\\Scripts\\python.exe run\\tools\\check_wiring_viewer_research_map_page.py
"""
import json
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from PySide6.QtWidgets import QApplication, QScrollArea  # noqa: E402

_APP = QApplication.instance() or QApplication(sys.argv)

from run.viewer_tools.wiring_viewer import scan_research_map  # noqa: E402
from run.viewer_tools.wiring_viewer.page_research_map import PageResearchMap  # noqa: E402
from run.viewer_tools.wiring_viewer.main_window import MainWindow, PAGE_TITLES  # noqa: E402

_MICHISUJI_PATH = os.path.join(_ROOT, "E", "docs", "研究マップ_道筋.json")


def _find_page_scroll_areas(page):
    """ページ自身（及び子孫）が持つQScrollAreaを探す。QTableWidget等が内部に持つ
    スクロールはQAbstractScrollAreaのサブクラスであってQScrollAreaそのものでは
    ないため、ここでは混同しない（実装ノウハウ「QAbstractScrollArea」の注意）。
    findChildrenは複数形を使う（同ノウハウ「findChild(QScrollArea)だけだと」の注意）。
    """
    return list(page.findChildren(QScrollArea))


def test_table_row_count_matches_list_experiments_count():
    """件数（実測194件）と、画面のテーブルの行数が一致すること。"""
    result = scan_research_map.list_experiments(use_cache=True)
    page = PageResearchMap()
    assert page._table.rowCount() == result["件数"], (
        f"テーブル行数={page._table.rowCount()} != list_experimentsの件数{result['件数']}")
    assert result["件数"] > 0, "件数が0（実測できていない可能性）"
    page.close()


def test_csv_exists_rows_show_nonempty_result_numbers():
    """csv_existsがTrueの行で、結果の数値セルが空でないこと（実測158件のはず）。"""
    from run.viewer_tools.wiring_viewer.page_research_map import _format_result_numbers

    result = scan_research_map.list_experiments(use_cache=True)
    csv_exists_entries = [e for e in result["実験"] if e["csv_exists"]]
    assert len(csv_exists_entries) > 0, "csv_existsがTrueの行が1件も無い"
    for entry in csv_exists_entries:
        text = _format_result_numbers(entry)
        assert text, f"{entry['path']} の結果の数値セルが空文字"
        assert text != "CSVなし", f"{entry['path']} はcsv_exists=Trueなのに'CSVなし'と表示"


def test_nan_is_shown_as_readable_text_not_raw_number():
    """NaNが混ざる行で、'nan'という生の数字表記ではなく「測れず」等に置き換わること。"""
    from run.viewer_tools.wiring_viewer.page_research_map import _format_result_numbers

    result = scan_research_map.list_experiments(use_cache=True)
    nan_entries = [
        e for e in result["実験"]
        if e["結果の数値"] and any(
            isinstance(v, float) and v != v for v in e["結果の数値"].values())
    ]
    assert len(nan_entries) > 0, "NaNを含む実験が1件も見つからなかった（検証データが変わった可能性）"
    for entry in nan_entries:
        text = _format_result_numbers(entry)
        assert "nan" not in text.lower(), f"{entry['path']} の表示にNaNが生の数字として出ている: {text}"
        assert "測れず" in text, f"{entry['path']} でNaN箇所が「測れず」に置き換わっていない: {text}"


def test_search_filter_narrows_rows_to_matching_only():
    """検索欄に実在するキーワードを入れると、該当行だけに絞り込まれること。"""
    page = PageResearchMap()
    total = page._table.rowCount()

    # 実在するキーワードを実データから拾う（決め打ちしない）。
    needle = None
    for e in page._entries:
        if "リクライニング" in (e.get("name", "") + e.get("note", "")):
            needle = "リクライニング"
            break
    assert needle is not None, "検証用キーワード'リクライニング'が実データに見つからない"

    expected = sum(
        1 for e in page._entries
        if needle.lower() in str(e.get("name", "")).lower()
        or needle.lower() in str(e.get("note", "")).lower()
        or needle.lower() in str(e.get("解釈", "")).lower())
    assert 0 < expected < total, f"キーワード'{needle}'の該当件数({expected})が全件({total})と同じか0"

    page._search.setText(needle)
    _APP.processEvents()
    assert page._table.rowCount() == expected, (
        f"絞り込み後の行数={page._table.rowCount()} != 期待値{expected}")

    page._search.setText("")
    _APP.processEvents()
    assert page._table.rowCount() == total, "検索欄を空にしても全件に戻らない"
    page.close()


def test_no_overflow_at_1920x1080_and_1280x720():
    for w, h in ((1920, 1080), (1280, 720)):
        page = PageResearchMap()
        page.resize(w, h)
        page.show()
        _APP.processEvents()
        scrolls = _find_page_scroll_areas(page)
        for sa in scrolls:
            vmax = sa.verticalScrollBar().maximum()
            assert vmax == 0, f"{w}x{h}でスクロールが残っている(maximum={vmax})"
        page.close()


def test_main_window_has_research_map_page_and_switches_to_it():
    win = MainWindow()
    assert "研究マップ" in PAGE_TITLES
    idx = PAGE_TITLES.index("研究マップ")
    assert win.nav.count() == len(PAGE_TITLES), (
        f"ナビの項目数={win.nav.count()} != PAGE_TITLES件数{len(PAGE_TITLES)}")
    assert win.nav.item(idx).text() == "研究マップ"

    win.show()
    _APP.processEvents()
    win.nav.setCurrentRow(idx)
    _APP.processEvents()
    assert win.stack.currentIndex() == idx, "ナビ選択でスタックが切り替わっていない"
    current_widget = win.stack.currentWidget()
    assert current_widget.objectName() == "pageResearchMap", (
        f"切り替え後の中身が研究マップページでない: {current_widget.objectName()}")
    win.close()


def test_empty_michisuji_shows_guidance_without_error():
    """研究マップ_道筋.json が空のひな形のままでも例外を出さず案内文が出ること。"""
    with open(_MICHISUJI_PATH, encoding="utf-8") as fp:
        data = json.load(fp)
    is_blank = (not (data.get("夢") or "").strip()
                and not data.get("階層") and not data.get("分岐点"))
    assert is_blank, (
        "研究マップ_道筋.json が既にひな形でなくなっている（このテストの前提が崩れている。"
        "中身が書かれた場合はこのテストの意味が変わるので確認要）")

    page = PageResearchMap()  # 例外を出さず構築できることが最初の条件
    tab2 = page._tabs.widget(1)
    from PySide6.QtWidgets import QLabel
    labels_text = " ".join(lbl.text() for lbl in tab2.findChildren(QLabel))
    assert "まだ書かれていません" in labels_text, "空の道筋ファイルなのに案内文が出ていない"
    assert "研究マップ_道筋.json" in labels_text, "案内文にファイル名が入っていない"
    page.close()


def _run_all() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = []
    for t in tests:
        try:
            t()
            print(f"[ok] {t.__name__}")
        except AssertionError as exc:
            failed.append(t.__name__)
            print(f"[ng] {t.__name__}: {exc}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            failed.append(t.__name__)
            print(f"[error] {t.__name__}: {exc!r}", file=sys.stderr)
    if failed:
        print(f"\n{len(failed)}件失敗: {failed}", file=sys.stderr)
        return 1
    print(f"\nすべて合格（{len(tests)}件）")
    return 0


if __name__ == "__main__":
    sys.exit(_run_all())
