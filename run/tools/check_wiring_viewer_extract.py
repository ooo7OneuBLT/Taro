"""TaroMap（配線可視化アプリ）の各scan_*.pyの抽出が正確かを、Qt起動なしで検証する。

各scan_*.pyはUIから独立した純粋関数（Qt非依存）として作られているため、
ここではQApplicationを一切構築せず、関数を直接呼んで件数を照合するだけの
軽量な検証にする（工程C仕様のとおり）。

使い方（既存の run/tools/check_*.py 群と同じ流儀）:
    .venv\\Scripts\\python.exe run\\tools\\check_wiring_viewer_extract.py
    -> pytestからも `pytest run/tools/check_wiring_viewer_extract.py` で呼べる。
"""
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import run.config as config  # noqa: E402
import run.wiring_map as wiring_map  # noqa: E402
from run.viewer_tools.wiring_viewer.scan_config import scan_config  # noqa: E402
from run.viewer_tools.wiring_viewer.scan_docstrings import scan_docstrings  # noqa: E402
from run.viewer_tools.wiring_viewer.scan_logs import list_log_runs  # noqa: E402
from run.viewer_tools.wiring_viewer.scan_taro_setup import scan_taro_setup  # noqa: E402

_TARO_SETUP_PATH = os.path.join(_ROOT, "run", "taro_setup.py")

# NODES/EDGESの期待件数。仕様に明記された期待値（工程Bで確認済み）。
# 【なぜ数値を決め打ちするか】wiring_map.py の中身が今後増減した場合に、
#   このテストが「意図した変化か・事故か」を人が判断するきっかけになるように、
#   あえて動的に取得せず固定値と突き合わせる。
_EXPECTED_NODE_COUNT = 38
_EXPECTED_EDGE_COUNT = 43


def test_scan_config_count_matches_defaults():
    """scan_config()の件数がTARO_DEFAULTS+RUN_DEFAULTSの件数と一致するか。"""
    result = scan_config()
    n_got = len(result["taro"]) + len(result["run"])
    n_expected = len(config.TARO_DEFAULTS) + len(config.RUN_DEFAULTS)
    assert n_got == n_expected, (
        f"scan_config()合計={n_got} != TARO_DEFAULTS+RUN_DEFAULTS={n_expected}")
    assert len(result["taro"]) == len(config.TARO_DEFAULTS)
    assert len(result["run"]) == len(config.RUN_DEFAULTS)


def test_scan_docstrings_count():
    """taro_core/src配下の実ファイル数（空__init__.py除く）と一致するか。"""
    entries = scan_docstrings()
    n_got = len(entries)
    # 手検算：rglobで全.pyを数え、空__init__.pyだけ除外する（scan_docstrings()自身と
    #   同じ除外条件を、ここでは独立に再実装して突き合わせる＝自己参照にしない）。
    import ast
    from pathlib import Path
    src_root = Path(_ROOT) / "taro_core" / "src"
    n_expected = 0
    for path in sorted(src_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        docstring = ast.get_docstring(tree) or ""
        if path.name == "__init__.py" and not docstring:
            continue
        n_expected += 1
    assert n_got == n_expected, f"scan_docstrings()件数={n_got} != 実測ファイル数={n_expected}"
    # 仕様の想定値（48件）からの逸脱にも気づけるよう、参考情報として出す。
    assert n_got == 48, f"想定値48件から外れている（実測{n_got}件）。taro_core/srcが変化した可能性。"


def test_scan_taro_setup_self_attrs_count_matches_grep():
    """scan_taro_setup()のself_attrs件数が、grep相当の正規表現一致件数と一致するか。"""
    info = scan_taro_setup()
    n_got = len(info["self_attrs"])

    with open(_TARO_SETUP_PATH, encoding="utf-8") as fp:
        lines = fp.readlines()
    pattern = re.compile(r"self\.\w+\s*=\s*[A-Z]\w*\(")
    n_expected = sum(1 for line in lines if pattern.search(line))

    assert n_got == n_expected, (
        f"scan_taro_setup()self_attrs件数={n_got} != grep一致件数={n_expected}")


def test_scan_logs_csv_total_matches_os_walk():
    """scan_logs.list_log_runs()が返すCSV総数が、os.walkで数えた実際の総数と一致するか。"""
    runs = list_log_runs()
    n_got = sum(len(r["csv_files"]) for r in runs)

    logs_dir = os.path.join(_ROOT, "E", "logs")
    n_expected = 0
    for _dirpath, _dirnames, filenames in os.walk(logs_dir):
        n_expected += sum(1 for f in filenames if f.lower().endswith(".csv"))

    assert n_got == n_expected, f"list_log_runs()CSV総数={n_got} != os.walk実測={n_expected}"
    # 仕様の想定値（790件）からの逸脱にも気づけるよう、参考情報として出す。
    assert n_got == 790, f"想定値790件から外れている（実測{n_got}件）。E/logsが変化した可能性。"


def test_wiring_map_node_edge_counts():
    """wiring_map.NODES/EDGESの件数が想定どおり（38個/43本）か。"""
    assert len(wiring_map.NODES) == _EXPECTED_NODE_COUNT, (
        f"NODES件数={len(wiring_map.NODES)} != 想定{_EXPECTED_NODE_COUNT}")
    assert len(wiring_map.EDGES) == _EXPECTED_EDGE_COUNT, (
        f"EDGES件数={len(wiring_map.EDGES)} != 想定{_EXPECTED_EDGE_COUNT}")


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
    if failed:
        print(f"\n{len(failed)}件失敗: {failed}", file=sys.stderr)
        return 1
    print(f"\nすべて合格（{len(tests)}件）")
    return 0


if __name__ == "__main__":
    sys.exit(_run_all())
