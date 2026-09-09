# -*- coding: utf-8 -*-
"""物体ファイル（`物体ファイル.csv`）の枚数・分裂・寿命を集計する。

【仕様】F/docs/二語文/仕様_物体ファイルの人間寄せ_凝集性・連続性・上限_2026-09-09.md
「後半」3節、および仕様_予測して確かめる検出_2026-09-09「後半」3節（mode列の追加分）。

引数：ログディレクトリ（例 F/logs/F2-100pre_物体ファイル人間寄せ_短い走行）。

【読むだけ】<ログディレクトリ>/物体ファイル.csv（`run/plugins/common/object_files.py`
  が書く。列：step, sim_time, n_dets, n_files, file_id, x, y, area, event,
  misses, since_seen, app_cos_created, mode）。mode列が無い旧形式のCSVでも
  読めるようにする（無ければ全行"scan"扱い＝旧ログとの互換）。
  <ログディレクトリ>/run.csv（あれば）。mode別の処理時間
  （object_files_confirm_ms・object_files_scan_ms 列）を平均する。

【出す数値】
  1. 同時カード数（n_files）の平均・最大（検出コマ単位。同じstepの行は同じ
     n_files を持つので、step単位で重複を除いて平均する）
  2. 検出で確認中（misses==0）の平均枚数（検出コマごとに event が
     matched/created/revived の行数を数え、その平均。misses==0はこの3イベント
     に限られるので同値）
  3. 分裂率＝1コマで matched+created+revived の行数が2件以上の検出コマの割合
  4. created/revived/lost の件数（ログ全体でのイベント行数）
  5. カードの寿命の中央値（sim_time単位）＝file_idごとに最初の
     created/revived から、その後の lost までの sim_time差。lostが無い
     （ログ末尾でまだ生きている）file_idは打ち切りなので中央値の計算から除く
  6.【新設・予測して確かめる検出】mode別の検出コマ件数
     （confirm/scan/scan_change/scan_miss/scan_empty）
  7.【新設】見失い率＝確認コマ(mode=="confirm")でのunmatchedの行数の合計
     ÷確認コマでのn_filesの合計（＝確認したカードの延べ枚数）
  8.【新設】mode別の処理時間の平均（run.csvのobject_files_confirm_ms・
     object_files_scan_ms列の非空値平均。run.csvが無い/列が無ければNone）

【出力】<ログディレクトリ>/結果_物体ファイル.json（他のf8x/f9x出力と名前が
  衝突しないよう「物体ファイル」を付ける）
"""
import csv
import json
import os
import sys
from statistics import median


def _to_float(s):
    if s in (None, ""):
        return None
    return float(s)


def _to_int(s):
    if s in (None, ""):
        return None
    return int(float(s))


def main(log_dir):
    csv_path = os.path.join(log_dir, "物体ファイル.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as fp:
        for r in csv.DictReader(fp):
            rows.append(r)

    # ---- 1. 同時カード数（n_files）の平均・最大：検出コマ（step）単位 --------
    n_files_by_step = {}
    for r in rows:
        step = _to_int(r["step"])
        nf = _to_int(r["n_files"])
        if step is None or nf is None:
            continue
        n_files_by_step[step] = nf   # 同じstepの行は同じn_filesのはず（上書きでよい）
    n_files_vals = list(n_files_by_step.values())
    n_files_mean = round(sum(n_files_vals) / len(n_files_vals), 3) if n_files_vals else None
    n_files_max = max(n_files_vals) if n_files_vals else None

    # ---- 2・3. 検出コマごとの matched+created+revived の件数 -----------------
    confirmed_events = {"matched", "created", "revived"}
    confirmed_by_step = {}
    for r in rows:
        step = _to_int(r["step"])
        ev = r["event"]
        if step is None or ev not in confirmed_events:
            continue
        confirmed_by_step[step] = confirmed_by_step.get(step, 0) + 1

    all_steps = sorted(n_files_by_step.keys())
    confirmed_counts = [confirmed_by_step.get(s, 0) for s in all_steps]
    confirmed_mean = (round(sum(confirmed_counts) / len(confirmed_counts), 3)
                       if confirmed_counts else None)
    split_steps = sum(1 for c in confirmed_counts if c >= 2)
    split_rate = round(split_steps / len(all_steps), 4) if all_steps else None

    # ---- 4. created/revived/lost の件数（ログ全体） --------------------------
    n_created = sum(1 for r in rows if r["event"] == "created")
    n_revived = sum(1 for r in rows if r["event"] == "revived")
    n_lost = sum(1 for r in rows if r["event"] == "lost")

    # ---- 5. カードの寿命の中央値（sim_time、created/revivedからlostまで） -----
    birth = {}      # file_id -> 最初のcreated/revivedのsim_time
    death = {}      # file_id -> lostのsim_time（複数回作り直された場合は最後を使う）
    for r in rows:
        fid = r["file_id"]
        if fid == "":
            continue
        ev = r["event"]
        t = _to_float(r["sim_time"])
        if t is None:
            continue
        if ev in ("created", "revived") and fid not in birth:
            birth[fid] = t
        if ev == "lost":
            death[fid] = t
    lifespans = []
    for fid, t_birth in birth.items():
        t_death = death.get(fid)
        if t_death is None or t_death < t_birth:
            continue   # ログ末尾でまだ生きている＝打ち切り、中央値から除く
        lifespans.append(t_death - t_birth)
    lifespan_median = round(median(lifespans), 3) if lifespans else None

    # ---- 6. mode別の検出コマ件数 --------------------------------------------
    # 【互換】mode列が無い旧ログでは全行 "" になる→ "scan" 扱いにする。
    mode_by_step = {}
    for r in rows:
        step = _to_int(r["step"])
        if step is None:
            continue
        mode_by_step[step] = r.get("mode") or "scan"
    mode_counts = {}
    for m in mode_by_step.values():
        mode_counts[m] = mode_counts.get(m, 0) + 1

    # ---- 7. 見失い率：確認コマ(mode=="confirm")でのunmatched行 / n_filesの延べ --
    confirm_steps = [s for s, m in mode_by_step.items() if m == "confirm"]
    confirm_steps_set = set(confirm_steps)
    confirm_unmatched = 0
    for r in rows:
        step = _to_int(r["step"])
        if step in confirm_steps_set and r["event"] == "unmatched":
            confirm_unmatched += 1
    confirm_card_total = sum(n_files_by_step.get(s, 0) for s in confirm_steps)
    miss_rate = (round(confirm_unmatched / confirm_card_total, 4)
                 if confirm_card_total else None)

    # ---- 8. mode別処理時間（run.csvのobject_files_confirm_ms/scan_ms列） ------
    confirm_ms_vals, scan_ms_vals = [], []
    run_csv_path = os.path.join(log_dir, "run.csv")
    if os.path.exists(run_csv_path):
        with open(run_csv_path, "r", encoding="utf-8", newline="") as fp:
            for r in csv.DictReader(fp):
                v = _to_float(r.get("object_files_confirm_ms"))
                if v is not None:
                    confirm_ms_vals.append(v)
                v = _to_float(r.get("object_files_scan_ms"))
                if v is not None:
                    scan_ms_vals.append(v)
    confirm_ms_mean = (round(sum(confirm_ms_vals) / len(confirm_ms_vals), 3)
                        if confirm_ms_vals else None)
    scan_ms_mean = (round(sum(scan_ms_vals) / len(scan_ms_vals), 3)
                     if scan_ms_vals else None)

    result = {
        "n_files_mean": n_files_mean,
        "n_files_max": n_files_max,
        "confirmed_mean": confirmed_mean,
        "split_rate": split_rate,
        "n_created": n_created,
        "n_revived": n_revived,
        "n_lost": n_lost,
        "lifespan_median_s": lifespan_median,
        "n_detect_steps": len(all_steps),
        "n_lifespan_samples": len(lifespans),
        "mode_counts": mode_counts,
        "miss_rate_confirm": miss_rate,
        "n_confirm_steps": len(confirm_steps),
        "confirm_card_total": confirm_card_total,
        "confirm_unmatched_total": confirm_unmatched,
        "object_files_confirm_ms_mean": confirm_ms_mean,
        "object_files_scan_ms_mean": scan_ms_mean,
    }

    out_path = os.path.join(log_dir, "結果_物体ファイル.json")
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("out:", out_path)
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python f95_object_file_stats.py <log dir>")
        sys.exit(1)
    main(sys.argv[1])
