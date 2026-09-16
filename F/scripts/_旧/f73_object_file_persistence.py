# -*- coding: utf-8 -*-
"""F2-73 M1.5 物体ファイルの持続確認の解析。

【仕様】F/docs/二語文/仕様_M1.5_物体ファイルの持続確認_2026-09-05.md 「4. 解析」。
`物体ファイル.csv`（run/plugins/common/object_files.py が書く。列は同ファイルの
docstring参照。今回追加した misses・since_seen・app_cos_created・unmatched を使う）
を読み、file_id ごとの持続表と、n_files/n_dets/misses推移の図を出す。

【触ってよいファイル】このファイル自身と、出力先
F/logs/F2-73_M1.5_物体ファイルの持続/（持続.csv・結果.json・図_持続.png）だけ。
入力（物体ファイル.csv・太郎の発話.csv）は読むだけで変更しない。
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

LOG_DIR = os.path.join("F", "logs", "F2-73_M1.5_物体ファイルの持続")
OBJ_CSV = os.path.join(LOG_DIR, "物体ファイル.csv")
SPEECH_CSV = os.path.join(LOG_DIR, "太郎の発話.csv")
OUT_TABLE_CSV = os.path.join(LOG_DIR, "持続.csv")
OUT_RESULT_JSON = os.path.join(LOG_DIR, "結果.json")
OUT_FIG = os.path.join(LOG_DIR, "図_持続.png")


def _setup_font():
    fp = "C:/Windows/Fonts/meiryo.ttc"
    font_manager.fontManager.addfont(fp)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()


def _f(x):
    """CSVの数値セル（空文字ありうる）をfloatへ。空なら None。"""
    if x is None or x == "":
        return None
    return float(x)


def load_rows(path):
    with open(path, encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def build_table(rows):
    """file_id ごとの持続の要点を組み立てる（仕様書「4. 解析」の表の列）。"""
    by_file = {}
    for r in rows:
        fid = r["file_id"]
        if fid == "":
            continue
        by_file.setdefault(fid, []).append(r)

    table = []
    for fid, frows in sorted(by_file.items(), key=lambda kv: int(kv[0])):
        created_t = None
        last_matched_t = None
        first_unmatched_t = None
        lost_t = None
        misses_max = 0
        last_unmatched_app_cos = None
        for r in frows:
            t = float(r["sim_time"])
            ev = r["event"]
            misses = _f(r["misses"])
            if misses is not None:
                misses_max = max(misses_max, misses)
            if ev == "created":
                created_t = t
            elif ev == "matched":
                last_matched_t = t
            elif ev == "unmatched":
                if first_unmatched_t is None:
                    first_unmatched_t = t
                last_unmatched_app_cos = _f(r["app_cos_created"])
            elif ev == "lost":
                lost_t = t
        table.append({
            "file_id": fid,
            "created_sim_time": created_t,
            "last_matched_sim_time": last_matched_t,
            "first_unmatched_sim_time": first_unmatched_t,
            "lost_sim_time": lost_t,
            "misses_max": misses_max,
            "last_unmatched_app_cos_created": last_unmatched_app_cos,
        })
    return table


def load_toy_switch_times(speech_csv):
    """太郎の発話.csv があれば toy 列（無ければ target 列）の変化点をsim_timeで返す。
    無い・使えないときは空リスト（図側は縦線を単に描かないだけ＝任意の補助情報）。
    """
    if not os.path.exists(speech_csv):
        return []
    rows = load_rows(speech_csv)
    if not rows:
        return []
    key = None
    for cand in ("toy", "target"):
        if cand in rows[0]:
            key = cand
            break
    if key is None:
        return []
    switches = []
    prev = None
    for r in rows:
        cur = r.get(key)
        if cur and cur != prev:
            t = r.get("sim_sec") or r.get("sim_time") or r.get("t")
            if t not in (None, ""):
                switches.append(float(t))
            prev = cur
    return switches


def make_figure(rows, switch_times, out_path):
    _setup_font()
    step_rows = {}
    for r in rows:
        t = float(r["sim_time"])
        step_rows.setdefault(t, {"n_files": int(r["n_files"]), "n_dets": int(r["n_dets"])})
    ts = sorted(step_rows.keys())
    n_files = [step_rows[t]["n_files"] for t in ts]
    n_dets = [step_rows[t]["n_dets"] for t in ts]

    by_file_ts = {}
    for r in rows:
        fid = r["file_id"]
        if fid == "":
            continue
        misses = _f(r["misses"])
        if misses is None:
            continue
        by_file_ts.setdefault(fid, ([], []))
        by_file_ts[fid][0].append(float(r["sim_time"]))
        by_file_ts[fid][1].append(misses)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax1.plot(ts, n_files, marker="o", ms=3, label="n_files（物体ファイル数）")
    ax1.plot(ts, n_dets, marker="x", ms=3, label="n_dets（今コマの検出数）")
    ax1.set_ylabel("個数")
    ax1.set_title("F2-73 M1.5：物体ファイルの持続（n_files/n_dets の推移）")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)

    for fid, (fts, fmiss) in sorted(by_file_ts.items(), key=lambda kv: int(kv[0])):
        ax2.plot(fts, fmiss, marker=".", ms=3, label=f"file_id={fid}")
    ax2.set_xlabel("sim_time [秒]")
    ax2.set_ylabel("misses（連続で対応がつかなかったコマ数）")
    ax2.legend(loc="upper right", fontsize=8, ncol=2)
    ax2.grid(alpha=0.3)

    for t in switch_times:
        ax1.axvline(t, color="gray", ls="--", alpha=0.5)
        ax2.axvline(t, color="gray", ls="--", alpha=0.5)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    rows = load_rows(OBJ_CSV)
    table = build_table(rows)
    switch_times = load_toy_switch_times(SPEECH_CSV)

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(OUT_TABLE_CSV, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["file_id", "created_sim_time", "last_matched_sim_time",
                    "first_unmatched_sim_time", "lost_sim_time", "misses_max",
                    "last_unmatched_app_cos_created"])
        for row in table:
            w.writerow([row["file_id"], row["created_sim_time"],
                        row["last_matched_sim_time"], row["first_unmatched_sim_time"],
                        row["lost_sim_time"], row["misses_max"],
                        row["last_unmatched_app_cos_created"]])

    persist_secs = []
    for row in table:
        if row["first_unmatched_sim_time"] is not None and row["last_matched_sim_time"] is not None:
            persist_secs.append(row["first_unmatched_sim_time"] - row["last_matched_sim_time"])

    result = {
        "n_files_total": len(table),
        "n_switch_marks": len(switch_times),
        "switch_times": switch_times,
        "table": table,
        "persist_after_last_matched_sec_max": (max(persist_secs) if persist_secs else None),
    }
    with open(OUT_RESULT_JSON, "w", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2)

    make_figure(rows, switch_times, OUT_FIG)

    print(f"file数: {len(table)}")
    for row in table:
        print(row)
    print(f"表: {OUT_TABLE_CSV}")
    print(f"結果: {OUT_RESULT_JSON}")
    print(f"図: {OUT_FIG}")


if __name__ == "__main__":
    main()
