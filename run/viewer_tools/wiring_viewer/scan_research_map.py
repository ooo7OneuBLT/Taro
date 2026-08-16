"""TaroMap（⑦研究マップページ）用：E/experiments 配下の実験ファイルと、その結果CSVを
実測して一覧化する道具（工程4）。

【背景・なぜこれを作るか、仕様より】
研究者（高校生1人）が「1ヶ月前と同じ提案を繰り返される」と困っている。原因は
実験記録が全て時系列（研究日誌・実験ファイル）で、「今どうなっているか」
「もう試したか」を引ける形になっていないこと。ここでは E/experiments/*.json と
その結果CSV（E/logs配下）を実際に読んで一覧化する（Qt非依存の純粋関数）。
画面（Qt）は工程5の担当者が別に作る。

【実測済みの事実、2026-08-15 実装担当が実測（ユーザーの見積もり388本とは食い違う。
実測を正とする）】
`E/experiments/**/*.json` は実際には194本。note欄がある実験は126本、run.csv欄が
ある実験は177本、CSVが実在する実験は158本。全194本のjsonを読み158本のcsvを
全文読んでも実測で約1.2秒（キャッシュ無し時）。

【なぜCSVパーサを書き直さないか】
scan_logs.py の read_csv_data() が既にNaN処理・数値変換の面倒を見ている
（同じ式を複数ファイルに書かないという実装ノウハウの教訓）。ここではそれを
そのまま import して使う。

【キャッシュの設計、なぜ「1本の指紋」ではなく「ファイルごとの(mtime,size)」か】
scan_dimensions.py は依存ファイルが高々8個の固定集合なので「1本の指紋」で
足りたが、ここでは対象が194+158本と多く、しかも実験ファイルは研究の進行に
伴って随時増える（globし直すたびに本数が変わる）。1本でも増減・変更されると
指紋全体が変わって194+158本を毎回全部読み直すことになり、「新しいファイルを
足しても既存分は再パースしない」という要求と合わない。
⇒ 作業記録（非公開） の `_完了キャッシュ` と同じ考え方
（ファイルごとに(mtime,size)をキーにして、変化したものだけ再パースする）を踏襲した。
ディレクトリ一覧化（glob）自体は毎回やり直す＝新しいファイルは必ず次回に出る。
中身の再パースだけ、変化したファイルに絞る。

【日付の出どころを混同しない、捏造しない原則】
noteの先頭が "YYYY-MM-DD。" または "YYYY-MM-DD." の形なら正規表現で日付を取る
（出どころ="note"）。取れなければファイルのmtimeから日付を作り、出どころに
必ず「ファイル更新日時（推定）」と明記する。これを混同すると、研究者が
「note に書いた日付」だと信じて見誤る（捏造しない原則）。
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir))

# 【なぜsys.path操作をここでするか】このファイルは `-m run.viewer_tools.wiring_viewer.
# scan_research_map` のようにパッケージ経由でも、`python scan_research_map.py` の
# ように単独スクリプトとしても実行できるようにしておきたい（既存の
# check_wiring_viewer_dimensions.py 等が単独実行を前提にしている流儀に合わせる）。
# 相対import（`from . import scan_logs`）は単独実行時にエラーになるため、
# 絶対import（`from run.viewer_tools.wiring_viewer import scan_logs`）に統一し、
# そのためにリポジトリルートをsys.pathへ足す。
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from run.viewer_tools.wiring_viewer import scan_logs  # noqa: E402

_EXPERIMENTS_GLOB = os.path.join(_ROOT, "E", "experiments", "**", "*.json")
_CACHE_DIR = os.path.join(_ROOT, "run", "viewer_tools", "wiring_viewer", "_cache")
_CACHE_PATH = os.path.join(_CACHE_DIR, "研究マップ.json")
_INTERP_PATH = os.path.join(_ROOT, "E", "docs", "研究マップ_解釈.json")

_DATE_PREFIX_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})[。.]")


def _to_rel(abs_path: str) -> str:
    """リポジトリルートからの相対パスに直す（表示・解釈ファイルのキーと揃えるため
    区切り文字を "/" に統一する）。"""
    return os.path.relpath(abs_path, _ROOT).replace(os.sep, "/")


def _load_cache() -> dict:
    try:
        with open(_CACHE_PATH, encoding="utf-8") as fp:
            return json.load(fp)
    except (OSError, ValueError):
        return {}


def _save_cache(cache: dict):
    os.makedirs(_CACHE_DIR, exist_ok=True)
    tmp_path = _CACHE_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fp:
        json.dump(cache, fp, ensure_ascii=False, indent=1)
    os.replace(tmp_path, _CACHE_PATH)


def _load_interpretations() -> dict:
    """研究マップ_解釈.json を読む。無い・壊れている場合は空辞書
    （ひな形のままでも例外を出さない、という契約）。"""
    try:
        with open(_INTERP_PATH, encoding="utf-8") as fp:
            data = json.load(fp)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _stat(path: str):
    try:
        st = os.stat(path)
        return st.st_mtime, st.st_size
    except OSError:
        return None


def _parse_experiment_json(abs_path: str) -> dict:
    """実験ファイル1本を読み、name/note/日付/日付の出どころ/csv_pathを取り出す。

    読めない・壊れているjsonは例外を外に出さず、空扱いのdictを返す
    （落とし穴チェックリスト 項78「nanは『ダメ』でなく『測れなかった』」と同じ考え方で、
    1本の異常が一覧全体を止めないようにする）。
    """
    try:
        with open(abs_path, encoding="utf-8") as fp:
            data = json.load(fp)
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}

    name = data.get("name") or ""
    note = data.get("note") or ""

    m = _DATE_PREFIX_RE.match(note) if note else None
    if m:
        日付 = m.group(1)
        出どころ = "note"
    else:
        mtime = os.path.getmtime(abs_path)
        日付 = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
        出どころ = "ファイル更新日時（推定）"

    run = data.get("run")
    csv_rel = None
    if isinstance(run, dict):
        csv_rel = run.get("csv") or None

    csv_path = None
    if csv_rel:
        # 実験ファイルの run.csv はリポジトリルートからの相対パス表記
        # （例: "E/logs/contact_reward_compare/seed0.csv"）。そのまま相対パスとして扱う。
        csv_path = csv_rel.replace("\\", "/")

    return {
        "name": name,
        "note": note,
        "日付": 日付,
        "日付の出どころ": 出どころ,
        "csv_path": csv_path,
    }


def _read_csv_summary(abs_csv_path: str) -> dict:
    """CSV1本の行数と最終行の全数値列を読む（scan_logs.read_csv_data()に丸投げ。
    パーサを書き直さない）。読めなければ空のまま返す（項78と同じ考え方）。
    """
    try:
        cols = scan_logs.read_csv_data(abs_csv_path)
    except (OSError, ValueError):
        return {"csv_行数": None, "結果の数値": None}

    if not cols:
        return {"csv_行数": None, "結果の数値": None}

    n_rows = max((len(arr) for arr in cols.values()), default=0)
    if n_rows == 0:
        return {"csv_行数": 0, "結果の数値": None}

    最終行 = {}
    for col, arr in cols.items():
        if len(arr) == 0:
            continue
        最終行[col] = float(arr[-1])

    return {"csv_行数": n_rows, "結果の数値": 最終行}


def list_experiments(use_cache: bool = True) -> dict:
    """E/experiments配下の実験ファイルとその結果CSVを実測して一覧を返す。

    Args:
        use_cache: True なら run/viewer_tools/wiring_viewer/_cache/研究マップ.json の
            キャッシュを使う。ファイル一覧（glob）自体は毎回やり直すので、
            新しい実験ファイルは次回呼び出しで必ず出る。キャッシュはファイルごとに
            (mtime, size) が一致する場合のみ、そのファイルの再パースを省略する。

    Returns:
        仕様「作るもの」節で定義された辞書
        （生成時刻・所要秒・キャッシュから・件数・実験）。
    """
    t0 = time.time()

    # 【なぜuse_cache=Falseでも_load_cache()自体はする一方、中身を使わないのか】
    # scan_dimensions.pyのmeasure_dimensions()と同じ流儀に合わせる：use_cache=False は
    # 「既存キャッシュの値を信用しない（＝全部読み直す）」の意味であって、
    # 「キャッシュファイルへの保存を止める」の意味ではない。保存は毎回行う
    # （そうしないと、最初にuse_cache=Falseで呼んだだけではキャッシュファイルが
    # 一度も作られず、次にuse_cache=Trueで呼んでも高速化されない）。
    cache = _load_cache() if use_cache else {}
    new_cache: dict = {}
    interpretations = _load_interpretations()

    json_paths = sorted(glob.glob(_EXPERIMENTS_GLOB, recursive=True))

    実験 = []
    any_from_cache = False
    for abs_json in json_paths:
        stat = _stat(abs_json)
        if stat is None:
            continue  # globに出たがその直後に消えた等（想定外・稀）。スキップする
        mtime, size = stat
        rel_json = _to_rel(abs_json)
        cache_key = f"json:{rel_json}"

        cached_entry = cache.get(cache_key)
        if cached_entry and cached_entry.get("mtime") == mtime and cached_entry.get("size") == size:
            parsed = cached_entry["parsed"]
            any_from_cache = True
        else:
            parsed = _parse_experiment_json(abs_json)

        new_cache[cache_key] = {"mtime": mtime, "size": size, "parsed": parsed}

        csv_rel = parsed.get("csv_path")
        csv_abs = os.path.join(_ROOT, csv_rel) if csv_rel else None
        csv_exists = bool(csv_abs and os.path.isfile(csv_abs))

        csv_行数 = None
        結果の数値 = None
        if csv_exists:
            csv_stat = _stat(csv_abs)
            csv_cache_key = f"csv:{csv_rel}"
            csv_cached = cache.get(csv_cache_key)
            if (csv_stat is not None and csv_cached
                    and csv_cached.get("mtime") == csv_stat[0]
                    and csv_cached.get("size") == csv_stat[1]):
                csv_行数 = csv_cached.get("csv_行数")
                結果の数値 = csv_cached.get("結果の数値")
                any_from_cache = True
            else:
                summary = _read_csv_summary(csv_abs)
                csv_行数 = summary["csv_行数"]
                結果の数値 = summary["結果の数値"]
            if csv_stat is not None:
                new_cache[csv_cache_key] = {
                    "mtime": csv_stat[0], "size": csv_stat[1],
                    "csv_行数": csv_行数, "結果の数値": 結果の数値,
                }

        解釈 = interpretations.get(rel_json) or "未記入"

        実験.append({
            "path": rel_json,
            "name": parsed.get("name", ""),
            "note": parsed.get("note", ""),
            "日付": parsed.get("日付"),
            "日付の出どころ": parsed.get("日付の出どころ"),
            "csv_path": csv_rel,
            "csv_exists": csv_exists,
            "csv_行数": csv_行数,
            "結果の数値": 結果の数値,
            "解釈": 解釈,
        })

    _save_cache(new_cache)  # use_cacheに関わらず常に最新化する（上のコメント参照）

    elapsed = time.time() - t0

    return {
        "生成時刻": datetime.now(timezone.utc).astimezone().isoformat(),
        "所要秒": elapsed,
        "キャッシュから": bool(use_cache and any_from_cache),
        "件数": len(実験),
        "実験": 実験,
    }


if __name__ == "__main__":
    # 自己テスト：実際のE/experimentsに対して一覧を作り、件数・所要時間を表示する。
    r1 = list_experiments(use_cache=False)
    print(f"[キャッシュ無し] 件数={r1['件数']} 所要秒={r1['所要秒']:.3f} "
          f"キャッシュから={r1['キャッシュから']}")
    n_csv_exists = sum(1 for e in r1["実験"] if e["csv_exists"])
    n_note = sum(1 for e in r1["実験"] if e["note"])
    print(f"note欄あり={n_note} csv実在={n_csv_exists}")

    r2 = list_experiments(use_cache=True)
    print(f"[キャッシュ有り] 件数={r2['件数']} 所要秒={r2['所要秒']:.3f} "
          f"キャッシュから={r2['キャッシュから']}")
