# -*- coding: utf-8 -*-
"""物体ファイル（`物体ファイル.csv`）の枚数・分裂・寿命を集計する。

【仕様】F/docs/二語文/仕様_物体ファイルの人間寄せ_凝集性・連続性・上限_2026-09-09.md
「後半」3節、および仕様_見る側_道を1本にする_2026-09-09.md「後半」6節
（conf_*集計の廃止・mode列のframe/first一本化・n_pointsの平均を追加）。

引数：ログディレクトリ（例 F/logs/F2-100pre_物体ファイル人間寄せ_短い走行）。

【読むだけ】<ログディレクトリ>/物体ファイル.csv（`run/plugins/common/object_files.py`
  が書く。列：step, sim_time, n_dets, n_files, file_id, x, y, area, event,
  misses, since_seen, app_cos_created, mode, n_points, ...）。mode列が無い
  旧形式のCSVでも読めるようにする（無ければ全行"scan"扱い＝旧ログとの互換）。
  <ログディレクトリ>/run.csv（あれば）。mode別の処理時間
  （object_files_confirm_ms・object_files_scan_ms 列）を平均する
  （道を1本にした後の走行ではこの2列自体が出ないためNoneのまま）。

【出す数値】
  1. 同時カード数（n_files）の平均・最大（検出コマ単位。同じstepの行は同じ
     n_files を持つので、step単位で重複を除いて平均する）。合わせてn_points
     （段3で集めた点の数）の平均も出す。
  2. 検出で確認中（misses==0）の平均枚数（検出コマごとに event が
     matched/created/revived の行数を数え、その平均。misses==0はこの3イベント
     に限られるので同値）
  3. 分裂率＝1コマで matched+created+revived の行数が2件以上の検出コマの割合
  4. created/revived/lost の件数（ログ全体でのイベント行数）
  5. カードの寿命の中央値（sim_time単位）＝file_idごとに最初の
     created/revived から、その後の lost までの sim_time差。lostが無い
     （ログ末尾でまだ生きている）file_idは打ち切りなので中央値の計算から除く
  6. mode別の検出コマ件数（道を1本にした後は"frame"／最初のコマだけ"first"。
     旧ログでは confirm/scan/scan_change/scan_miss/scan_empty/onset/explore
     も出る）
  7. 見失い率＝検出コマ(mode=="frame"。旧confirmに相当)でのunmatchedの行数の
     合計÷検出コマでのn_filesの合計（＝確認したカードの延べ枚数）
  8. mode別の処理時間の平均（run.csvのobject_files_confirm_ms・
     object_files_scan_ms列の非空値平均。列が無ければNone）

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
    # 【2026-09-09・仕様_見る側_道を1本にする 後半1節】n_points（段3で集めた
    #   点の数）の平均。n_points列が無い旧ログでは空のまま（互換）。
    n_points_by_step = {}
    for r in rows:
        step = _to_int(r["step"])
        nf = _to_int(r["n_files"])
        if step is not None and nf is not None:
            n_files_by_step[step] = nf   # 同じstepの行は同じn_filesのはず（上書きでよい）
        npv = _to_int(r.get("n_points"))
        if step is not None and npv is not None:
            n_points_by_step[step] = npv
    n_files_vals = list(n_files_by_step.values())
    n_files_mean = round(sum(n_files_vals) / len(n_files_vals), 3) if n_files_vals else None
    n_files_max = max(n_files_vals) if n_files_vals else None
    n_points_vals = list(n_points_by_step.values())
    n_points_mean = round(sum(n_points_vals) / len(n_points_vals), 3) if n_points_vals else None

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

    # ---- 4. created/revived/lost/absorbed の件数（ログ全体） ------------------
    n_created = sum(1 for r in rows if r["event"] == "created")
    n_revived = sum(1 for r in rows if r["event"] == "revived")
    n_lost = sum(1 for r in rows if r["event"] == "lost")
    # 【2026-09-09・重複をなくす「後半」5節】absorbed（既存カードに吸収された検出）
    #   の件数。absorbed列が無い旧ログでは0のまま（互換）。
    n_absorbed = sum(1 for r in rows if r["event"] == "absorbed")
    # 【2026-09-09・仕様_見る側の段構成_実装 7節】individuated（段4：位置は
    #   同じだが見た目で別の物と判定して作った新カード）の件数。event列に
    #   individuatedが無い旧ログでは0のまま（互換）。
    n_individuated = sum(1 for r in rows if r["event"] == "individuated")

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
    # 【2026-09-09・仕様_見る側の段構成_実装 7節】mode列にはpreattentive有効時
    #   "onset"（段1の説明できない変化を切り出した）・"explore"（段6の探索点を
    #   切り出した）も入り得る。集計は元からmode文字列ごとに数える汎用実装
    #   なので、この2値も新しいキーとして自然にmode_countsへ出る（コード変更なし）。
    mode_by_step = {}
    for r in rows:
        step = _to_int(r["step"])
        if step is None:
            continue
        mode_by_step[step] = r.get("mode") or "scan"
    mode_counts = {}
    for m in mode_by_step.values():
        mode_counts[m] = mode_counts.get(m, 0) + 1

    # ---- 7. 見失い率：検出コマ(mode=="frame"。旧mode=="confirm"に相当)で
    #      「直前に見えていた（misses==0）カード」のうち、外れた(unmatched)割合
    # ---------------------------------------------------------------------------
    # 【2026-09-09・仕様_見る側_道を1本にする】検出コマの処理が1本化された
    #   （旧confirm/scan/scan_change/scan_miss/onset/explore →
    #   frame・最初のコマだけfirst）ので、判定はmode=="frame"に一本化する
    #   （毎コマ既存カードとの照合が起きる点は旧confirmと同じ）。
    # 【2026-09-09・追記「直し」2】隠れた物の確認の失敗（misses>0のカードが
    #   もう一度外れる）は正しい結果であり見失いではない。分母を「直前misses==0
    #   だったカード」に絞る。unmatched行のmisses列は増加後の値（従来必ず+1ずつ
    #   増える）なので misses_after-1 が直前値。matched行はmisses=0（更新後）
    #   なので、直前値を知るには履歴（file_idごとの前回misses）をたどる必要が
    #   ある（rowsをstep順にたどりながら追跡する）。
    confirm_steps = [s for s, m in mode_by_step.items() if m == "frame"]
    confirm_steps_set = set(confirm_steps)

    # 【2026-09-09・道を1本にする】「注意中」判定に使う(step -> attended_id)。
    #   注意.csvが無い（attend=False）走行では空のまま。
    attended_by_step = {}
    attend_csv_path = os.path.join(log_dir, "注意.csv")
    # 【2026-09-09・仕様_見る側の段構成_実装 7節】札の混入率＝attended_idごとに
    #   target_word（空文字を除く）の異なり数が2以上の割合。行を1回で読むために
    #   ここでtarget_wordも合わせて集める（attend=False・注意.csvが無い走行では
    #   words_by_attended_idが空のままlabel_mix_rate=None・n_attended_ids=0）。
    words_by_attended_id = {}
    if os.path.exists(attend_csv_path):
        with open(attend_csv_path, "r", encoding="utf-8", newline="") as fp:
            for r in csv.DictReader(fp):
                step = _to_int(r["step"])
                if step is not None:
                    attended_by_step[step] = r.get("attended_id") or ""
                aid = r.get("attended_id") or ""
                if not aid:
                    continue
                word = (r.get("target_word") or "").strip()
                if not word:
                    continue
                words_by_attended_id.setdefault(aid, set()).add(word)

    n_attended_ids = len(set(attended_by_step.get(s, "") for s in attended_by_step
                              if attended_by_step.get(s, "")))
    mixed_ids = sum(1 for aid, words in words_by_attended_id.items() if len(words) >= 2)
    # 【仕様に無かった判断】分母はn_attended_ids（注意された全id）。target_word
    #   が一度も無いidは異なりword数0なので「混入」に数えない（自然に薄まる）。
    label_mix_rate = (round(mixed_ids / n_attended_ids, 4) if n_attended_ids else None)

    _sorted_rows = sorted(rows, key=lambda r: (_to_int(r["step"]) if _to_int(r["step"]) is not None else -1))
    _last_misses = {}
    confirm_denom = 0
    confirm_unmatched = 0
    # 【追記3「合否」節】見失い率＝「注意中で直前まで見えていたカード」限定版。
    attended_confirm_denom = 0
    attended_confirm_unmatched = 0
    for r in _sorted_rows:
        fid = r["file_id"]
        if fid == "":
            continue
        ev = r["event"]
        step = _to_int(r["step"])
        if ev == "lost":
            _last_misses.pop(fid, None)
            continue
        if ev in ("matched", "created", "revived", "absorbed"):
            misses_after = 0
        elif ev == "unmatched":
            mv = _to_int(r["misses"])
            misses_after = mv
        else:
            continue
        prior = None
        if ev == "unmatched":
            prior = (misses_after - 1) if misses_after is not None else None
        elif ev in ("matched", "absorbed"):
            prior = _last_misses.get(fid)
        # created/revived は「直前に見えていた」の定義に当てはまらない
        # （新規/復活の瞬間で、直前状態が無いか別物）ので分母に数えない。
        is_confirm_step = step in confirm_steps_set
        if is_confirm_step and ev in ("matched", "unmatched", "absorbed") and prior == 0:
            confirm_denom += 1
            if ev == "unmatched":
                confirm_unmatched += 1
        # 注意中＝その時点のattended_idがこのfile_idと一致、かつ
        # 「直前まで見えていた」(prior==0)。
        if is_confirm_step and ev in ("matched", "unmatched"):
            is_attended = (attended_by_step.get(step, "") == fid)
            if is_attended and prior == 0:
                attended_confirm_denom += 1
                if ev == "unmatched":
                    attended_confirm_unmatched += 1
        if misses_after is not None:
            _last_misses[fid] = misses_after
    confirm_card_total = confirm_denom
    miss_rate = (round(confirm_unmatched / confirm_card_total, 4)
                 if confirm_card_total else None)
    # 【2026-09-09・追記3「合否」節】合否判定に使う定義：「注意中で直前まで
    #   見えていたカード」限定の見失い率（miss_rate_confirmは注意の有無を
    #   問わない全カード版で従来どおり残す）。
    miss_rate_attended = (round(attended_confirm_unmatched / attended_confirm_denom, 4)
                           if attended_confirm_denom else None)

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

    # ---- 9. 注意の持続・切り替え（重複をなくす「後半」5節） -------------------
    # 【仕様に無かった判断】注意.csvのdist_centerは画面中心からの距離であって
    #   旧attended_idとの位置差ではないため、「切り替え先が20px以内だったか」は
    #   注意.csv単独では計算できない。物体ファイル.csv（同じstep・file_idの
    #   x,y、matched/created/revived/absorbed行）と突き合わせて位置差を求めた。
    attend_persistence_median = None
    attend_switch_count = None
    attend_switch_close_rate = None
    attend_csv_path = os.path.join(log_dir, "注意.csv")
    if os.path.exists(attend_csv_path):
        with open(attend_csv_path, "r", encoding="utf-8", newline="") as fp:
            arows = [r for r in csv.DictReader(fp)]
        arows.sort(key=lambda r: (_to_int(r["step"]) if _to_int(r["step"]) is not None else -1))
        seq = [(r.get("attended_id") or "", _to_int(r["step"])) for r in arows]
        # 持続：連続して同じattended_id（空文字＝注意対象なしは除く）が続くコマ数。
        runs = []
        i = 0
        while i < len(seq):
            j = i
            while j < len(seq) and seq[j][0] == seq[i][0]:
                j += 1
            runs.append((seq[i][0], j - i))
            i = j
        persistence_lengths = [ln for aid, ln in runs if aid != ""]
        attend_persistence_median = (round(median(persistence_lengths), 3)
                                      if persistence_lengths else None)
        # 位置の突き合わせ用：(step, file_id) -> (x, y)（物体ファイル.csvから）
        pos_by_step_fid = {}
        for r in rows:
            if r["event"] in ("matched", "created", "revived", "absorbed"):
                step = _to_int(r["step"])
                fid = r["file_id"]
                x, y = _to_float(r["x"]), _to_float(r["y"])
                if step is not None and fid and x is not None and y is not None:
                    pos_by_step_fid[(step, fid)] = (x, y)
        switches = 0
        switch_close = 0
        prev_aid, prev_step = None, None
        for aid, step in seq:
            if prev_aid is not None and aid != "" and prev_aid != "" and aid != prev_aid:
                switches += 1
                p_old = pos_by_step_fid.get((prev_step, prev_aid))
                p_new = pos_by_step_fid.get((step, aid))
                if p_old is not None and p_new is not None:
                    d = ((p_old[0] - p_new[0]) ** 2 + (p_old[1] - p_new[1]) ** 2) ** 0.5
                    if d <= 20.0:
                        switch_close += 1
            prev_aid, prev_step = aid, step
        attend_switch_count = switches
        attend_switch_close_rate = (round(switch_close / switches, 4) if switches else None)

    result = {
        "n_files_mean": n_files_mean,
        "n_files_max": n_files_max,
        # 【2026-09-09・仕様_見る側_道を1本にする 後半6節】段3で集めた点の数
        #   （n_points列）の平均。n_points列が無い旧ログではNoneのまま。
        "n_points_mean": n_points_mean,
        "confirmed_mean": confirmed_mean,
        "split_rate": split_rate,
        "n_created": n_created,
        "n_revived": n_revived,
        "n_lost": n_lost,
        "n_absorbed": n_absorbed,
        "n_individuated": n_individuated,
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
        "attend_persistence_median_steps": attend_persistence_median,
        "attend_switch_count": attend_switch_count,
        "attend_switch_close_rate": attend_switch_close_rate,
        "miss_rate_confirm_attended": miss_rate_attended,
        "attended_confirm_card_total": attended_confirm_denom,
        "attended_confirm_unmatched_total": attended_confirm_unmatched,
        # 【2026-09-09・仕様_見る側の段構成_実装 7節】札の混入率。
        #   注意.csvが無い/target_wordが一度も無い走行ではNone・0のまま。
        "label_mix_rate": label_mix_rate,
        "n_attended_ids": n_attended_ids,
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
