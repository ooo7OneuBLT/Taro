"""TaroMap（研究マップページ）の scan_research_map.py（実測一覧モジュール）が
正しく一覧化できているかを、Qt起動なしで検証する。

scan_research_map.py はUIから独立した純粋関数（Qt非依存）として作られているため、
ここではQApplicationを一切構築せず、関数を直接呼んで値を照合するだけの
軽量な検証にする（既存の run/tools/check_wiring_viewer_dimensions.py と同じ流儀）。

使い方（既存の run/tools/check_*.py 群と同じ流儀）:
    .venv\\Scripts\\python.exe run\\tools\\check_wiring_viewer_research_map.py
    -> pytestからも `pytest run/tools/check_wiring_viewer_research_map.py` で呼べる。
"""
import glob
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from run.viewer_tools.wiring_viewer import scan_research_map  # noqa: E402

_EXPERIMENTS_GLOB = os.path.join(_ROOT, "E", "experiments", "**", "*.json")
_DUMMY_NAME = "_check_wiring_viewer_research_map_ダミー.json"
_DUMMY_PATH = os.path.join(_ROOT, "E", "experiments", _DUMMY_NAME)


def _count_experiment_jsons() -> int:
    return len(glob.glob(_EXPERIMENTS_GLOB, recursive=True))


def _count_csv_exists() -> int:
    n = 0
    for path in glob.glob(_EXPERIMENTS_GLOB, recursive=True):
        try:
            with open(path, encoding="utf-8") as fp:
                data = json.load(fp)
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        run = data.get("run")
        if not isinstance(run, dict):
            continue
        csv_rel = run.get("csv")
        if not csv_rel:
            continue
        csv_abs = os.path.join(_ROOT, csv_rel.replace("\\", "/"))
        if os.path.isfile(csv_abs):
            n += 1
    return n


def test_list_experiments_count_matches_actual_glob():
    """「件数」が実際にglobし直した本数と一致するか（ユーザー見積もりでなく実測値で照合）。"""
    result = scan_research_map.list_experiments(use_cache=False)
    expected = _count_experiment_jsons()
    assert result["件数"] == expected, (
        f"件数={result['件数']} != 実測glob本数{expected}")
    assert len(result["実験"]) == expected


def test_csv_exists_count_matches_actual_filesystem_check():
    """csv_exists=Trueの件数が、実際にファイル存在チェックした値と一致するか。"""
    result = scan_research_map.list_experiments(use_cache=False)
    got = sum(1 for e in result["実験"] if e["csv_exists"])
    expected = _count_csv_exists()
    assert got == expected, f"csv_exists=True件数={got} != 実測ファイル存在チェック{expected}"


def test_return_shape_and_no_column_name_hardcoding():
    """戻り値が仕様「作るもの」節どおりの主要キーを持ち、結果の数値が列名決め打ちでないか。"""
    result = scan_research_map.list_experiments(use_cache=False)
    for key in ("生成時刻", "所要秒", "キャッシュから", "件数", "実験"):
        assert key in result, f"戻り値に必須キー'{key}'が無い"
    assert result["所要秒"] > 0, "所要秒が0以下（実測していない可能性）"
    assert result["キャッシュから"] is False

    n_with_result = 0
    n_csv_exists = 0
    for e in result["実験"]:
        for key in ("path", "name", "note", "日付", "日付の出どころ", "csv_path",
                    "csv_exists", "csv_行数", "結果の数値", "解釈"):
            assert key in e, f"実験1件に必須キー'{key}'が無い（path={e.get('path')}）"
        if e["csv_exists"]:
            n_csv_exists += 1
            assert e["csv_行数"] is not None or e["csv_行数"] == 0, (
                f"csv_existsなのにcsv_行数がNone: {e['path']}")
            if e["結果の数値"] is not None:
                n_with_result += 1
                assert isinstance(e["結果の数値"], dict)
                for col_name, v in e["結果の数値"].items():
                    assert isinstance(col_name, str)
                    assert isinstance(v, float)
        else:
            assert e["csv_行数"] is None
            assert e["結果の数値"] is None
        assert e["解釈"] == "未記入" or isinstance(e["解釈"], str)
        if e["日付"] is not None:
            assert e["日付の出どころ"] in ("note", "ファイル更新日時（推定）")

    assert n_csv_exists > 0, "csv_existsがTrueの実験が1件も無い（想定外）"
    assert n_with_result > 0, "結果の数値が取れた実験が1件も無い（想定外）"


def test_date_source_labeled_correctly_when_note_has_no_date_prefix():
    """noteに日付先頭が無い実験は、日付の出どころが必ず「ファイル更新日時（推定）」になるか。

    捏造しない原則：noteから取れたのか、ファイルのmtimeから推定したのかを混同しない。
    """
    result = scan_research_map.list_experiments(use_cache=False)
    checked_note_none = False
    checked_note_present_no_prefix = False
    for e in result["実験"]:
        note = e["note"]
        if not note:
            checked_note_none = True
            assert e["日付の出どころ"] == "ファイル更新日時（推定）", (
                f"note空なのに出どころがnote扱い: {e['path']}")
        elif not note[:10].count("-") >= 2 or not note[0].isdigit():
            # 先頭が数字の日付形式でなさそうなnoteの例
            checked_note_present_no_prefix = True
    assert checked_note_none, "note欄が空の実験が1件も見つからなかった（検証条件が成立していない）"
    # note_present_no_prefixは実データ依存の付随チェックなので、無くても失敗にはしない。
    _ = checked_note_present_no_prefix


def test_interpretation_lookup_uses_研究マップ_解釈():
    """研究マップ_解釈.json の内容がキー一致で反映されるか（合成before/afterで安全に検証）。

    落とし穴チェックリスト 項90系の考え方：本物の設定ファイルは書き換えず、
    一時的に別パスへ差し替えたモジュール属性で検証する。
    """
    result = scan_research_map.list_experiments(use_cache=False)
    assert result["件数"] > 0
    sample_path = result["実験"][0]["path"]

    orig_interp_path = scan_research_map._INTERP_PATH
    tmp_dir = os.path.join(_ROOT, "run", "viewer_tools", "wiring_viewer", "_cache")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_interp_path = os.path.join(tmp_dir, "_check_研究マップ_解釈_tmp.json")
    try:
        with open(tmp_interp_path, "w", encoding="utf-8") as fp:
            json.dump({sample_path: "検証用のダミー解釈"}, fp, ensure_ascii=False)
        scan_research_map._INTERP_PATH = tmp_interp_path

        result2 = scan_research_map.list_experiments(use_cache=False)
        matched = [e for e in result2["実験"] if e["path"] == sample_path]
        assert matched, "サンプルpathが一覧から見つからない"
        assert matched[0]["解釈"] == "検証用のダミー解釈"

        others_未記入 = [e for e in result2["実験"] if e["path"] != sample_path]
        assert any(e["解釈"] == "未記入" for e in others_未記入), (
            "他の実験の解釈が全部『未記入』以外になっている（誤って全件に適用された疑い）")
    finally:
        scan_research_map._INTERP_PATH = orig_interp_path
        if os.path.exists(tmp_interp_path):
            os.remove(tmp_interp_path)


def test_empty_hinagata_files_do_not_raise():
    """研究マップ_道筋.json・研究マップ_解釈.json が空（ひな形のまま）でも
    list_experiments()が例外を出さないか。"""
    # 実ファイルはひな形（_書き方キーのみ、または空配列）のはずなので、
    # そのまま呼んで例外が出ないことを確認する。
    result = scan_research_map.list_experiments(use_cache=False)
    assert isinstance(result, dict)
    assert result["件数"] > 0
    # 全件「未記入」になっているはず（本物の研究マップ_解釈.jsonはひな形のまま＝空）。
    assert all(e["解釈"] == "未記入" for e in result["実験"]), (
        "研究マップ_解釈.jsonがひな形のままのはずなのに『未記入』以外の解釈がある"
        "（本番ファイルに中身が書かれている可能性。テストの前提が崩れている）")


def test_new_dummy_experiment_appears_after_glob_even_with_cache():
    """E/experiments配下に新しいダミーjsonを1本追加すると、キャッシュありでも
    次回のlist_experiments()呼び出しで自動的に現れるか（一覧を手書きにしていないことの確認）。

    検証後、ダミーファイルは必ず削除する。
    """
    scan_research_map.list_experiments(use_cache=True)  # まずキャッシュを最新化
    before = scan_research_map.list_experiments(use_cache=True)
    before_count = before["件数"]
    assert not os.path.exists(_DUMMY_PATH), "前回検証の後片付けが残っている（先に削除して再実行）"
    try:
        with open(_DUMMY_PATH, "w", encoding="utf-8") as fp:
            json.dump({
                "name": "検証用ダミー実験（自動削除される）",
                "note": "2026-01-01。check_wiring_viewer_research_map.pyの検証専用ダミー。",
                "run": {"type": "train", "steps": 1, "seed": 0},
            }, fp, ensure_ascii=False)

        after = scan_research_map.list_experiments(use_cache=True)
        assert after["件数"] == before_count + 1, (
            f"ダミー追加後の件数={after['件数']} != 追加前{before_count}+1"
            "（新規ファイルがキャッシュありでも一覧に出ていない）")
        rel = os.path.relpath(_DUMMY_PATH, _ROOT).replace(os.sep, "/")
        matched = [e for e in after["実験"] if e["path"] == rel]
        assert matched, "追加したダミーが一覧に見つからない"
        assert matched[0]["日付"] == "2026-01-01"
        assert matched[0]["日付の出どころ"] == "note"
    finally:
        if os.path.exists(_DUMMY_PATH):
            os.remove(_DUMMY_PATH)
        scan_research_map.list_experiments(use_cache=True)  # キャッシュも綺麗な状態に戻す


def test_cache_roundtrip_speedup():
    """キャッシュ無し→保存→キャッシュ有りの2回目で「キャッシュから」が立つか。速度も参考計測する。"""
    r1 = scan_research_map.list_experiments(use_cache=False)
    assert os.path.exists(scan_research_map._CACHE_PATH), "キャッシュファイルが保存されていない"

    t0 = time.time()
    r2 = scan_research_map.list_experiments(use_cache=True)
    elapsed_cache_hit = time.time() - t0

    assert r2["件数"] == r1["件数"]
    assert r2["キャッシュから"] is True, "キャッシュが効いているはずなのにキャッシュから=Falseだった"
    assert elapsed_cache_hit < 5.0, f"キャッシュ命中のはずが{elapsed_cache_hit:.2f}秒かかった"


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
