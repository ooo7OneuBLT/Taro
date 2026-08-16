"""TaroMap（③設定一覧ページ）用：run/config.py の設定項目を実行時に列挙する。

【なぜ実行時に反復するだけにするか、2026-08-15】
2026-08-13に `touch_adaptation` 等の項目が `run/config.py` の TARO_DEFAULTS に
追加されたのに、旧 `run/wiring_map.py` の一覧（NODES）には反映されていなかった。
これは「一覧をどこかにハードコードして手で同期していた」ことが原因。
⇒ ここでは `TARO_DEFAULTS.items()` / `RUN_DEFAULTS.items()` を**そのまま**反復するだけ
にする。config.py に項目が増減すれば、次に scan_config() を呼んだ瞬間に自動で
反映される（同期のずれが原理的に起きない）。

注意：この関数は run/config.py を**読むだけ**（書き換えない・コピーしない）。
"""
from __future__ import annotations

import importlib


def _rows_from(defaults: dict) -> list[dict]:
    """{key: (default, doc, envname)} の辞書を、表示用の行のリストに変換する。

    ハードコードした項目名の一覧は一切持たない。渡された辞書を反復するだけ。
    """
    rows = []
    for key, spec in defaults.items():
        # 【なぜタプルの長さを確認するか】config.py の項目タプルは
        #   (既定値, 説明, 環境変数名) の3要素が前提。将来フィールドが増減しても
        #   ここで落ちて気づけるように、決め打ちのインデックスでなく展開で受ける。
        default, doc, envname = spec
        rows.append({
            "key": key,
            "default": default,
            "doc": doc,
            "envname": envname,
        })
    return rows


def scan_config(taro_defaults: dict | None = None,
                 run_defaults: dict | None = None) -> dict:
    """TARO_DEFAULTS / RUN_DEFAULTS を実行時に反復して一覧化する。

    引数を省略すると run/config.py から動的にimportして使う（本番経路）。
    引数を渡すと、そちらをそのまま反復する（検証用。テスト専用のコピー辞書を
    渡して「項目が増えても自動で拾えるか」を確かめるために使う）。

    戻り値の例：
        {"taro": [{"key": "touch_adaptation", "default": False,
                   "doc": "触覚の順応（末梢＋脳）を有効にする...",
                   "envname": None}, ...],
         "run": [...]}

    件数は必ず len(TARO_DEFAULTS)+len(RUN_DEFAULTS) と一致する
    （taro欄件数 + run欄件数の合計として。ハードコードした一覧は作らない）。
    """
    if taro_defaults is None or run_defaults is None:
        # 【なぜ importlib.import_module か】遅延import することで、
        #   このモジュール自体はQt非依存・config.py非依存のまま保てる
        #   （検証コードから taro_defaults/run_defaults を直接渡して呼べるように）。
        cfg = importlib.import_module("run.config")
        if taro_defaults is None:
            taro_defaults = cfg.TARO_DEFAULTS
        if run_defaults is None:
            run_defaults = cfg.RUN_DEFAULTS
    return {
        "taro": _rows_from(taro_defaults),
        "run": _rows_from(run_defaults),
    }


if __name__ == "__main__":
    # 自己テスト：本物のconfig.pyを読み、件数が一致するかその場で表示する。
    import run.config as _cfg

    result = scan_config()
    n_taro, n_run = len(result["taro"]), len(result["run"])
    n_expected = len(_cfg.TARO_DEFAULTS) + len(_cfg.RUN_DEFAULTS)
    print(f"taro欄: {n_taro}件, run欄: {n_run}件, 合計: {n_taro + n_run}件")
    print(f"期待値（len(TARO_DEFAULTS)+len(RUN_DEFAULTS)）: {n_expected}件")
    print("一致" if n_taro + n_run == n_expected else "不一致（バグ）")
