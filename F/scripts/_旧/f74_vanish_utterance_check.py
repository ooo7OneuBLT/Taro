# -*- coding: utf-8 -*-
"""F2-74 M2 消失発話の照合。

【仕様】F/docs/二語文/仕様_M2_消えた瞬間に親が「○○ないね」と言う_2026-09-05.md
「後半：実装向け技術付録」「照合」節。

【読むだけ】F/logs/F2-74_M2_消失発話/ の下の
  発話イベント.csv（run/plugins/common/word_learning.py が書く。cause列に
    "vanish"が乗るのはE/scripts/parent_labeling.py._VANISH実装の直接の証拠）
  物体ファイル.csv（run/plugins/common/object_files.py が書く。n_dets列）
を読み、cause=vanish の発話ごとに：
  ・時刻(sim_sec)・テキスト・旧的(target)
  ・その時刻の直近の物体ファイル行のn_dets（0であるべき＝機上が空である証拠）
  ・隠した時刻（前の発話の直後＝refractory/repeat終了時刻の近似として
    「vanish発話時刻 - vanish_gap_before_sec」を使う。実装のタイマーそのもの）
    → 発話 → 次の物が出た時刻（次のcauseなし行のsim_sec、無ければ実験終了）
    の間隔を表にする。
出力：消失発話.csv・結果.json（このスクリプト自身の新規出力。触ってよいファイル）。
"""
import csv
import json
import os

LOG_DIR = os.path.join("F", "logs", "F2-74_M2_消失発話")
EVENTS_CSV = os.path.join(LOG_DIR, "発話イベント.csv")
OBJFILE_CSV = os.path.join(LOG_DIR, "物体ファイル.csv")
OUT_CSV = os.path.join(LOG_DIR, "消失発話.csv")
OUT_JSON = os.path.join(LOG_DIR, "結果.json")

# 仕様書のvanish_gap_before_sec/afterの既定値（シーンJSONと同じ値。
# 「隠した時刻」の近似計算にだけ使う。実測のsim_sec列自体はCSVから読むので
# ここがずれても本体の判定（n_dets=0か）には影響しない）。
VANISH_GAP_BEFORE_SEC = 1.0
VANISH_GAP_AFTER_SEC = 1.5


def load_rows(path):
    with open(path, encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def nearest_n_dets(objfile_rows, t):
    """|sim_time - t| が最小の行のn_detsを返す（無ければNone）。

    【注意・実測で発覚】object_files はinterval_s=1.0ごとの疎いサンプルしか
    持たない（実測：約1.0〜1.1秒間隔）。「t以前で直近」で探すと、たまたま
    サンプル間隔の谷（隠す直前のサンプル）を拾ってしまい、実際には隠した
    直後に撮られた次のサンプル（tのすぐ後）の方がずっと近いのに無視される
    事故が起きた（F2-74机上確認：t=8.1sで t-0.9s=7.2sの行(n_dets=1)を拾い、
    t+0.1s=8.2sの行(n_dets=0)を見落としていた）。前後どちらのサンプルが
    近いかで選ぶよう修正。
    """
    best = None
    best_t = None
    best_diff = None
    for r in objfile_rows:
        rt = float(r["sim_time"])
        diff = abs(rt - t)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_t = rt
            best = int(r["n_dets"])
    return best, best_t


def main():
    speech_rows = load_rows(EVENTS_CSV)
    objfile_rows = load_rows(OBJFILE_CSV) if os.path.exists(OBJFILE_CSV) else []

    vanish_rows = [r for r in speech_rows if r.get("cause") == "vanish"]
    table = []
    for i, r in enumerate(vanish_rows):
        t = float(r["sim_sec"])
        n_dets, n_dets_t = nearest_n_dets(objfile_rows, t)
        hidden_t = t - VANISH_GAP_BEFORE_SEC
        # 次に出た物＝この消失発話より後で、causeが空（=新しい的の初回命名）の
        # 最初の行のsim_sec。無ければ実験の最終行のsim_secを使う（打ち切り）。
        next_t = None
        for r2 in speech_rows:
            if float(r2["sim_sec"]) > t and (r2.get("cause") or "") == "":
                next_t = float(r2["sim_sec"])
                break
        next_note = ""
        if next_t is None:
            # 実験がここで終わり、次の的の命名が起きなかった場合
            #（この消失発話自身が最後の行のケースを含む）。
            last_sec = float(speech_rows[-1]["sim_sec"]) if speech_rows else None
            if last_sec is not None and last_sec > t:
                next_t = last_sec
            else:
                next_t = None
                next_note = "実験終了まで次の物の命名が起きなかった"
        table.append({
            "sim_sec": t,
            "text": r["text"],
            "旧的": r["target"],
            "n_dets_直近行": n_dets,
            "n_dets_行のsim_time": n_dets_t,
            "隠した時刻_近似": round(hidden_t, 3),
            "発話までの間隔_sec": round(t - hidden_t, 3),
            "次の物が出た時刻": next_t,
            "発話から次の物までの間隔_sec": (
                round(next_t - t, 3) if next_t is not None else None),
            "備考": next_note,
        })

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["sim_sec", "text", "旧的", "n_dets_直近行", "n_dets_行のsim_time",
                   "隠した時刻_近似", "発話までの間隔_sec", "次の物が出た時刻",
                   "発話から次の物までの間隔_sec", "備考"])
        for row in table:
            w.writerow([row["sim_sec"], row["text"], row["旧的"],
                       row["n_dets_直近行"], row["n_dets_行のsim_time"],
                       row["隠した時刻_近似"], row["発話までの間隔_sec"],
                       row["次の物が出た時刻"], row["発話から次の物までの間隔_sec"],
                       row["備考"]])

    n_dets_values = [row["n_dets_直近行"] for row in table if row["n_dets_直近行"] is not None]
    result = {
        "消失発話数": len(table),
        "全てn_dets=0か": all(v == 0 for v in n_dets_values) if n_dets_values else None,
        "n_dets_直近行の一覧": n_dets_values,
        "表": table,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
