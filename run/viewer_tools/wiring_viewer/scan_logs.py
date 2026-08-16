"""TaroMap（④実測ログページ）用：E/logs 配下のCSVログを高速に一覧化する。

【なぜ「一覧」と「読み込み」を分けたか】
E/logs には約790本のCSVがある（1本＝1回の学習ランの記録。プロジェクトの流儀として
複数ランを混ぜて1つのグラフにはしない＝ノウハウ feedback-per-sim-graph）。
起動のたびに790本すべての中身を読むと重く、しかも大半は画面に表示されない。
⇒ `list_log_runs()` は**ファイル名の一覧だけ**を返す（中身は読まない・高速）。
   実際にグラフを描く1本だけを、選ばれた瞬間に `read_csv_data()` で読む
   （遅延読み込み。ノウハウ 項87「作る場所と使う場所で頻度差があると落ちる」を踏まえ、
   「一覧を作る」処理と「1本を読む」処理を最初から別関数に分けてある）。

【E/logs の実際の構造、2026-08-15 実装時に確認】
790本のCSVのうち789本は `E/logs/<ラン名>/*.csv` の形でサブフォルダに入っているが、
1本（`E/logs/_scratch_check_seed2_1500.csv`）はサブフォルダを持たず直下に置かれている
（スクラッチ用の検証ファイルと見られる）。さらにサブフォルダの中にもう一段
サブフォルダを持つラン（例：`E/logs/<ラン名>/<子ラン名>/*.csv`）が実在する
（`E/logs` 直下のディレクトリ数52に対し、depth2のディレクトリ数63）。
⇒ 「.csvファイルが直接入っているディレクトリ」を単位として1つの `run` にする
   （os.walk で全ディレクトリを見て、直下に.csvがあるディレクトリだけを拾う）。
   これなら深さに関わらず、どのフォルダも取りこぼさない。
"""
from __future__ import annotations

import csv
import os

import numpy as np

_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir))
_DEFAULT_LOGS_DIR = os.path.join(_ROOT, "E", "logs")


def list_log_runs(root=None) -> list[dict]:
    """E/logs配下のラン一覧を返す（全CSVの中身は読まない。ファイル名の列挙のみ）。

    戻り値：
        [{"run_name": "contact_reward_compare", "csv_files": ["seed0.csv", "seed1.csv", ...],
          "dir": "<root>/contact_reward_compare"}, ...]

    run_name は root からの相対パス（サブフォルダの中のサブフォルダも1つのrunとして
    区別できるよう、ディレクトリ名だけでなく相対パスを使う）。root 直下に置かれた
    CSV（サブフォルダを持たないもの）は run_name "(logs直下)" にまとめる。
    """
    logs_dir = root or _DEFAULT_LOGS_DIR
    if not os.path.isdir(logs_dir):
        return []

    runs = []
    for dirpath, _dirnames, filenames in os.walk(logs_dir):
        csv_files = sorted(f for f in filenames if f.lower().endswith(".csv"))
        if not csv_files:
            continue
        rel = os.path.relpath(dirpath, logs_dir)
        run_name = "(logs直下)" if rel == "." else rel.replace(os.sep, "/")
        runs.append({
            "run_name": run_name,
            "csv_files": csv_files,
            "dir": dirpath,
        })
    # 見やすさのため名前順にしておく（一覧の表示順が実行のたびにばらつかないように）
    runs.sort(key=lambda r: r["run_name"])
    return runs


def read_csv_header(csv_path) -> list[str]:
    """CSVの1行目（列名）だけを読む。本体（数値の行）は読まない（遅延読み込みの前段）。"""
    with open(csv_path, newline="", encoding="utf-8") as fp:
        reader = csv.reader(fp)
        try:
            return next(reader)
        except StopIteration:
            return []


def read_csv_data(csv_path, columns=None) -> dict:
    """選択されたCSV1本だけを実際に読む（画面で選ばれた時だけ呼ぶこと）。

    columns を指定すればその列だけ、指定しなければ全列を読む。
    戻り値は {列名: numpy配列} の辞書。数値に変換できないセルは NaN にする
    （落とし穴チェックリスト 項78「nanは『ダメ』でなく『測れなかった』」を踏まえ、
    読めない値を例外で落とさず NaN として残す＝グラフ側で穴として見える）。
    """
    with open(csv_path, newline="", encoding="utf-8") as fp:
        reader = csv.reader(fp)
        try:
            header = next(reader)
        except StopIteration:
            return {}
        want = columns if columns is not None else header
        want = [c for c in want if c in header]
        idx = {c: header.index(c) for c in want}
        raw = {c: [] for c in want}
        for row in reader:
            for c, i in idx.items():
                if i < len(row):
                    raw[c].append(row[i])
                else:
                    raw[c].append("")
    out = {}
    for c, values in raw.items():
        arr = np.full(len(values), np.nan, dtype=float)
        for i, v in enumerate(values):
            if v == "":
                continue
            try:
                arr[i] = float(v)
            except ValueError:
                pass  # 数値でない列（文字列等）はNaN列のまま返す。ここで落とさない
        out[c] = arr
    return out


if __name__ == "__main__":
    # 自己テスト：実際のE/logsに対して一覧を作り、件数を表示する。
    runs = list_log_runs()
    n_csv = sum(len(r["csv_files"]) for r in runs)
    print(f"ラン数（.csvが直接入っているフォルダの数）: {len(runs)}")
    print(f"CSV総数: {n_csv}")
    if runs:
        first = runs[0]
        first_csv = os.path.join(first["dir"], first["csv_files"][0])
        header = read_csv_header(first_csv)
        print(f"例: {first['run_name']} / {first['csv_files'][0]} 列数={len(header)}")
