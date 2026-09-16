# -*- coding: utf-8 -*-
"""F2-75 M3 注意中の物体ファイルと消失信号の照合。

【仕様】F/docs/二語文/仕様_M3_注意中の物体ファイルと消失信号_2026-09-06.md
「後半：実装担当向け技術付録」「3. 照合」節。

【読むだけ】F/logs/F2-75_M3_消失信号/ の下の
  注意.csv         （run/plugins/common/object_files.py が書く。attend=True時のみ）
  発話イベント.csv （run/plugins/common/word_learning.py が書く。cause列に"vanish"が
                     乗るのが親の「○○ないね」、cause=""が新しい的の初回命名）
を読み、cause=vanish の発話（消失イベント）ごとに：
  T0 = 隠した時刻（親の消失発話時刻 − vanish_gap_before_sec）
  T1 = 注意.csvでvanishedが最初に真になった時刻（T0以降で探す）
  Tp = 親の消失発話時刻（発話イベント.csvのsim_sec）
  T1-T0（≤0.6であるべき）・Tp-T1（>0であるべき）
  T1時点のattended_id・nearest_word・target_word・target_cos
  誤検出：物が見えている区間（同じ的の直前の命名〜隠すまで）にvanishedが
  真になった回数
出力：消失信号.csv・結果.json・図_消失信号.png（このスクリプト自身の新規出力）。
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

LOG_DIR = os.path.join("F", "logs", "F2-75_M3_消失信号")
ATTEND_CSV = os.path.join(LOG_DIR, "注意.csv")
SPEECH_CSV = os.path.join(LOG_DIR, "発話イベント.csv")
OUT_CSV = os.path.join(LOG_DIR, "消失信号.csv")
OUT_JSON = os.path.join(LOG_DIR, "結果.json")
OUT_PNG = os.path.join(LOG_DIR, "図_消失信号.png")

# シーン側の既定値（E/scripts/parent_labeling.py vanish_gap_before_sec の既定・
# F2-74/F2-75と同じシーンで使われている値。仕様書「時計」節：物体ファイルは
# 0.5秒ごとに動くのでvanish_gap_before_sec=1.0秒より速く、隠して最悪0.6秒で
# 「消えた」が立つ計算になる）。
VANISH_GAP_BEFORE_SEC = 1.0
# T1をT0以降で探す上限（見つからない異常時に無限に遡らないための打ち切り）。
SEARCH_WINDOW_SEC = 5.0


def load_rows(path):
    with open(path, encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def _to_bool(s):
    return str(s).strip().lower() in ("true", "1")


def find_t1(attend_rows, t0, tp):
    """T0以降・Tp+SEARCH_WINDOW_SEC以内で、vanishedが最初に真になった行を返す。
    無ければ (None, None)。"""
    best = None
    for r in attend_rows:
        t = float(r["sim_time"])
        if t < t0 or t > tp + SEARCH_WINDOW_SEC:
            continue
        if _to_bool(r["vanished"]):
            if best is None or t < float(best["sim_time"]):
                best = r
    if best is None:
        return None, None
    return float(best["sim_time"]), best


def find_naming_before(speech_rows, target, before_t):
    """targetについて、before_t（隠した消失発話のsim_sec）より前で一番新しい
    命名イベント（cause=""またはcause="repeat"）のsim_secを返す。無ければNone。
    ＝この消失イベントで「物が見えていた区間」の開始点（親がこの物を見せ始めた
    時刻）の近似。"""
    best_t = None
    for r in speech_rows:
        if r.get("target") != target:
            continue
        cause = r.get("cause") or ""
        if cause not in ("", "repeat"):
            continue
        t = float(r["sim_sec"])
        if t < before_t and (best_t is None or t > best_t):
            best_t = t
    return best_t


def count_false_vanish(attend_rows, start_t, end_t):
    """[start_t, end_t) の間にvanishedが真になった行数（物が見えているはずの
    区間での誤検出）。start_tがNoneなら数えない（判定不能のため0件扱い・
    備考に記録）。"""
    if start_t is None:
        return 0
    n = 0
    for r in attend_rows:
        t = float(r["sim_time"])
        if start_t <= t < end_t and _to_bool(r["vanished"]):
            n += 1
    return n


def main():
    if not os.path.exists(ATTEND_CSV):
        raise SystemExit("注意.csvが無い（object_filesのattend=trueで走らせたか確認）: %s" % ATTEND_CSV)
    attend_rows = load_rows(ATTEND_CSV)
    speech_rows = load_rows(SPEECH_CSV)

    vanish_rows = [r for r in speech_rows if r.get("cause") == "vanish"]

    table = []
    total_false = 0
    for r in vanish_rows:
        tp = float(r["sim_sec"])
        target = r.get("target")
        t0 = round(tp - VANISH_GAP_BEFORE_SEC, 3)
        t1, t1_row = find_t1(attend_rows, t0, tp)
        naming_t = find_naming_before(speech_rows, target, tp)
        false_n = count_false_vanish(attend_rows, naming_t, t0)
        total_false += false_n

        row = {
            "Tp_sim_sec": tp,
            "target": target,
            "T0_隠した時刻": t0,
            "T1_消えたが立った時刻": t1,
            "T1-T0": round(t1 - t0, 3) if t1 is not None else None,
            "Tp-T1": round(tp - t1, 3) if t1 is not None else None,
            "attended_id": t1_row["attended_id"] if t1_row else "",
            "nearest_word": t1_row["nearest_word"] if t1_row else "",
            "target_word": t1_row["target_word"] if t1_row else "",
            "target_cos": t1_row["target_cos"] if t1_row else "",
            "命名時刻_可視区間開始": naming_t,
            "可視区間中の誤検出回数": false_n,
            "備考": "" if t1 is not None else "T1が見つからない（vanishedが立たなかった）",
        }
        table.append(row)

    os.makedirs(LOG_DIR, exist_ok=True)
    fieldnames = ["Tp_sim_sec", "target", "T0_隠した時刻", "T1_消えたが立った時刻",
                  "T1-T0", "Tp-T1", "attended_id", "nearest_word", "target_word",
                  "target_cos", "命名時刻_可視区間開始", "可視区間中の誤検出回数", "備考"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(fieldnames)
        for row in table:
            w.writerow([row[k] for k in fieldnames])

    t1_minus_t0 = [row["T1-T0"] for row in table if row["T1-T0"] is not None]
    tp_minus_t1 = [row["Tp-T1"] for row in table if row["Tp-T1"] is not None]
    nearest_matches_target = [
        row for row in table
        if row["target_word"] and row["nearest_word"] == row["target_word"]]

    result = {
        "消失イベント数": len(table),
        "T1見つからず数": sum(1 for row in table if row["T1-T0"] is None),
        "T1-T0_全て0.6以下か": all(v <= 0.6 for v in t1_minus_t0) if t1_minus_t0 else None,
        "T1-T0_一覧": t1_minus_t0,
        "Tp-T1_全て正か": all(v > 0.0 for v in tp_minus_t1) if tp_minus_t1 else None,
        "Tp-T1_一覧": tp_minus_t1,
        "nearest_word_target_word一致数": len(nearest_matches_target),
        "target_word判定可能イベント数": sum(1 for row in table if row["target_word"]),
        "可視区間中の誤検出_合計": total_false,
        "表": table,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # ---- 図 --------------------------------------------------------------
    fp_font = "C:/Windows/Fonts/meiryo.ttc"
    font_manager.fontManager.addfont(fp_font)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp_font).get_name()

    ts = [float(r["sim_time"]) for r in attend_rows]
    vanished_v = [1 if _to_bool(r["vanished"]) else 0 for r in attend_rows]
    misses_v = [float(r["misses"]) if r["misses"] not in ("", None) else 0.0
                for r in attend_rows]

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.step(ts, vanished_v, where="post", color="red", label="vanished(0/1)")
    ax1.set_ylabel("vanished(0/1)", color="red")
    ax1.set_ylim(-0.1, 1.4)
    ax2 = ax1.twinx()
    ax2.plot(ts, misses_v, color="gray", alpha=0.6, label="misses")
    ax2.set_ylabel("misses", color="gray")

    for r in vanish_rows:
        ax1.axvline(float(r["sim_sec"]), color="orange", linestyle="--", linewidth=1)
    for r in speech_rows:
        if (r.get("cause") or "") == "":
            ax1.axvline(float(r["sim_sec"]), color="blue", linestyle=":", linewidth=1)

    ax1.set_xlabel("sim_time (s)")
    ax1.set_title("M3 消失信号：vanished/misses と親の消失発話（橙）・的の切り替わり（青）")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110)
    plt.close(fig)
    print("保存: %s" % os.path.abspath(OUT_PNG))


if __name__ == "__main__":
    main()
