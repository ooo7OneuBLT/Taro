"""TaroMap ⑦「予測のしくみ（実測）」ページ（page_prediction.py）の検証。

PySide6が要るためQT_QPA_PLATFORM=offscreenで実行する（GUIを実際に開かない）。
既存の check_wiring_viewer_*.py と違い、Qtを実際に構築するため、
このファイル自身の先頭で環境変数を設定してから import する必要がある。

使い方：
    set QT_QPA_PLATFORM=offscreen
    .venv\\Scripts\\python.exe run\\tools\\check_wiring_viewer_prediction_page.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from PySide6.QtWidgets import QApplication, QScrollArea  # noqa: E402

_APP = QApplication.instance() or QApplication(sys.argv)

from run.viewer_tools.wiring_viewer.page_prediction import PagePrediction  # noqa: E402
from run.viewer_tools.wiring_viewer.main_window import MainWindow, PAGE_TITLES  # noqa: E402


def _find_page_scroll_areas(page):
    """ページ自身が直接持つQScrollAreaだけを探す（findChildrenは孫も拾うので、
    QTableWidget等が内部に持つスクロールと混同しないよう名前で絞る）。
    実装ノウハウ「QAbstractScrollArea」の注意どおり、findChildrenは複数形を使う。
    """
    return [w for w in page.findChildren(QScrollArea)]


def test_page_has_at_most_one_scroll_area_and_no_overflow_at_1920x1080():
    page = PagePrediction()
    page.resize(1920, 1080)
    page.show()
    _APP.processEvents()

    scrolls = _find_page_scroll_areas(page)
    assert len(scrolls) <= 1, f"QScrollAreaが複数ある: {len(scrolls)}個"
    assert len(scrolls) == 1, "このページは内容が長いのでQScrollAreaが1個あるはず"
    sa = scrolls[0]
    vmax = sa.verticalScrollBar().maximum()
    assert vmax == 0, f"1920x1080でスクロールが残っている(maximum={vmax})"
    page.close()


def test_page_has_no_overflow_at_1280x720():
    page = PagePrediction()
    page.resize(1280, 720)
    page.show()
    _APP.processEvents()

    scrolls = _find_page_scroll_areas(page)
    assert len(scrolls) == 1
    sa = scrolls[0]
    vmax = sa.verticalScrollBar().maximum()
    assert vmax == 0, f"1280x720でスクロールが残っている(maximum={vmax})"
    page.close()


def test_touch_off_dims_the_touch_row_and_touch_on_restores_it():
    """実験ファイルではなく、run.config.Config を直接組み立てて touch を切り替える
    （check_wiring_viewer_dimensions.py と同じ流儀＝実在の実験ファイルに依存しない）。
    """
    from run.config import Config

    page = PagePrediction()

    # 【なぜisVisible()でなくisHidden()か、実装ノウハウより】
    #   ここではページをshow()していない（親ウィンドウ非表示の下では
    #   isVisible()が常にFalseを返し、setVisible(True/False)の切替を検出できない）。
    #   setVisible()で明示的に隠したかどうかはisHidden()で判定する。
    cfg_off = Config(taro={"touch": False})
    page._cfg = cfg_off
    page._measured = page._measured  # 構造自体は不変。ON/OFFだけ差し替える。
    page._refresh_on_off()
    row = page._sense_rows["touch"]
    opacity_off = row["effect"].opacity()
    off_label_shown = not row["off_label"].isHidden()

    cfg_on = Config(taro={"touch": True})
    page._cfg = cfg_on
    page._refresh_on_off()
    opacity_on = row["effect"].opacity()
    off_label_hidden_when_on = row["off_label"].isHidden()

    assert opacity_off < 1.0, f"touch=falseなのに不透明度が下がっていない: {opacity_off}"
    assert off_label_shown, "touch=falseなのにOFFラベルが出ていない"
    assert opacity_on == 1.0, f"touch=trueなのに不透明度が戻っていない: {opacity_on}"
    assert off_label_hidden_when_on, "touch=trueなのにOFFラベルが出たままになっている"
    assert opacity_off != opacity_on, "touch切替の前後で見た目(opacity)が変わっていない"
    page.close()


def test_generation_time_is_shown():
    page = PagePrediction()
    text = page._gen_time_label.text()
    assert page._measured["生成時刻"] in text, "画面に実測モジュールの生成時刻が出ていない"
    page.close()


def test_efference_copy_off_dims_the_section():
    from run.config import Config

    page = PagePrediction()
    cfg_off = Config(taro={"efference_copy": False})
    page._cfg = cfg_off
    page._measured = __import__(
        "run.viewer_tools.wiring_viewer.scan_dimensions", fromlist=["measure_dimensions"]
    ).measure_dimensions(cfg=cfg_off, use_cache=True)
    page._refresh_on_off()
    assert page._efference_effect.opacity() < 1.0, "efference_copy=falseなのに薄くなっていない"
    assert not page._efference_off_label.isHidden(), "efference_copy=falseなのにOFF文言が出ていない"

    cfg_on = Config(taro={"efference_copy": True})
    page._cfg = cfg_on
    page._measured = __import__(
        "run.viewer_tools.wiring_viewer.scan_dimensions", fromlist=["measure_dimensions"]
    ).measure_dimensions(cfg=cfg_on, use_cache=True)
    page._refresh_on_off()
    assert page._efference_effect.opacity() == 1.0
    assert page._efference_off_label.isHidden()
    page.close()


def test_main_window_has_prediction_page_and_switches_to_it():
    win = MainWindow()
    assert "予測のしくみ（実測）" in PAGE_TITLES
    idx = PAGE_TITLES.index("予測のしくみ（実測）")
    assert win.nav.count() == len(PAGE_TITLES), (
        f"ナビの項目数={win.nav.count()} != PAGE_TITLES件数{len(PAGE_TITLES)}")
    assert win.nav.item(idx).text() == "予測のしくみ（実測）"

    win.show()
    _APP.processEvents()
    win.nav.setCurrentRow(idx)
    _APP.processEvents()
    assert win.stack.currentIndex() == idx, "ナビ選択でスタックが切り替わっていない"
    current_widget = win.stack.currentWidget()
    assert current_widget.objectName() == "pagePrediction", (
        f"切り替え後の中身が予測ページでない: {current_widget.objectName()}")
    win.close()


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
