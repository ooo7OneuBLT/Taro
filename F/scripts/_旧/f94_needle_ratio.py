# -*- coding: utf-8 -*-
"""出来事の針の比（消失・戻った瞬間の跳ね ÷ 注意の切り替えの瞬間の跳ね）を測る。

【仕様】F/docs/二語文/仕様_M7b-1改4_V8設定と驚きの基準の初期化_2026-09-10.md
「後半：実装向け技術付録」2節。

引数：ログディレクトリ（例 F/logs/F2-99_M7b1改4_V8設定_学習）。

【読むだけ】<ログディレクトリ>/世界の予測器.csv（主CSV。attend_port,
  z_vec_att, z_vec_max を含む、run/plugins/common/world_predictor_log.py が書く）
  <ログディレクトリ>/世界の予測器_物ごと.csv（物ごとCSV。file_id, attended,
  visible, vanished, err_state, z_state, z_vec を含む）

【出す数値（仕様「後半」2節）】
  ①消失の瞬間（物ごとCSVでvanished 0→1）と戻った瞬間（1→0かつvisible 1）の
    注意中の物（=そのtickの主CSVのerr_state、＝ctx.last_world_predが持つ
    「注意中の物」のerr_state）の平均
  ②注意の切り替えの瞬間（物ごとCSVでattendedのfile_idが前tickと違う）の平均
  ③それ以外（①②のどちらでもない主CSVの全tick）の平均
  ④針の比＝①÷②
  ⑤物が生まれて3秒（物ごとCSVでfile_idの最初30tick）のz_stateの最小と平均、
    z_vecの同じもの（全物プールした値と、年齢別（0〜29tick目）の平均カーブ）
  ⑥同じ物（file_id）が2回以上消えた場合の、1回目と最後の消失の針
    （消失イベント単独のerr_state、①のような複数イベント平均ではなく
    「その1回」の値）の平均（繰り返しへの慣れ。1回目の平均・最後の平均・
    最後/1回目の比を出す）

【出力】（このスクリプト自身の新規出力、同フォルダ。他のf8x/f9xと名前が衝突
  しないよう「_針」を付ける）
  世界の予測器_針.png（①〜③の棒グラフ、⑤の年齢別z折れ線）
  結果_針.json

【実装判断・仕様に無かった点】
  - ①②③の「注意中の物のerr_state」は、消失イベント自体を検出するのは
    物ごとCSV（どの物が消えたか＝全物ぶん拾える）だが、値そのものは同じstepの
    主CSVのerr_state列（＝ctx.last_world_predが持つ「今注意している物」の
    err_state）から取る、と読んだ。消えた物と注意中の物が一致しない
    tickも含まれ得るが、仕様の文言どおり「注意中の物の」err_stateを使う。
  - ⑥の「針」は①のような複数イベントの平均ではなく、そのイベント単発の
    値（消失tickの主CSV err_state）とした（「1回目・最後」という個別イベント
    指定の言い回しのため）。
"""
import csv
import json
import os
import sys

LOG_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    "F", "logs", "F2-99_M7b1改4_V8設定_学習")
MAIN_CSV = os.path.join(LOG_DIR, "世界の予測器.csv")
BY_FILE_CSV = os.path.join(LOG_DIR, "世界の予測器_物ごと.csv")
OUT_PNG = os.path.join(LOG_DIR, "世界の予測器_針.png")
OUT_JSON = os.path.join(LOG_DIR, "結果_針.json")

AGE_WINDOW_TICKS = 30  # 仕様「後半」2節⑤：「物が生まれて3秒」＝最初30tick（interval_s=0.1想定）


def _f(r, key, default=None):
    v = r.get(key, "")
    if v in (None, ""):
        return default
    try:
        return float(v)
    except ValueError:
        return default


def load_main_rows():
    if not os.path.exists(MAIN_CSV):
        raise FileNotFoundError(f"{MAIN_CSV} が無い（先に走行が必要）")
    rows = []
    with open(MAIN_CSV, encoding="utf-8") as fp:
        for r in csv.DictReader(fp):
            rows.append({
                "step": int(float(r.get("step", 0) or 0)),
                "t_sec": _f(r, "t_sec", 0.0),
                "err_state": _f(r, "err_state"),
            })
    rows.sort(key=lambda r: r["step"])
    return rows


def load_by_file_rows():
    if not os.path.exists(BY_FILE_CSV):
        raise FileNotFoundError(f"{BY_FILE_CSV} が無い（物ごとCSVが出ていない＝ports未使用の走行）")
    rows = []
    with open(BY_FILE_CSV, encoding="utf-8") as fp:
        for r in csv.DictReader(fp):
            fid_raw = r.get("file_id", "")
            try:
                file_id = int(float(fid_raw))
            except ValueError:
                continue
            rows.append({
                "step": int(float(r.get("step", 0) or 0)),
                "t_sec": _f(r, "t_sec", 0.0),
                "file_id": file_id,
                "attended": (r.get("attended", "") or "").strip() in ("True", "1", "1.0"),
                "visible": _f(r, "visible", 0.0),
                "vanished": _f(r, "vanished", 0.0),
                "err_state": _f(r, "err_state"),
                "z_state": _f(r, "z_state"),
                "z_vec": _f(r, "z_vec"),
                # 【2026-09-09・仕様_見る側の段構成_実装 7節】masked列が無い旧ログ
                #   では常にFalse扱い（互換）。
                "masked": (r.get("masked", "") or "").strip() in ("True", "1", "1.0"),
            })
    rows.sort(key=lambda r: (r["step"], r["file_id"]))
    return rows


def find_masked_steps(by_rows):
    """【2026-09-09・仕様_見る側の段構成_実装 7節】masked==Trueの物ごと行が
    1つでもあるstepの集合（そのtickは注意中の物がmaskedだった可能性が高い、
    という近似。物ごと単位のmaskedをtick単位に落とすための仕様に無かった
    判断。厳密には「注意中の物のmasked」を見るべきだが、trainer.py側で
    maskedはtick単位で決まる値（moving/切り替え直後）なので、同じtickの
    物ごと行は全て同じmasked値を持つ＝どのfile_idの行を見ても同じ）。"""
    return set(r["step"] for r in by_rows if r["masked"])


def find_vanish_return_events(by_rows):
    """物ごとに vanished 0→1（消失）・1→0かつvisible1（戻った）の遷移tickを拾う。

    戻り値：[{"step":int, "file_id":int, "kind":"vanish"/"return"}]（stepの昇順ではなく
    file_id別に検出した後まとめてstep順にソートする）。
    """
    events = []
    by_fid = {}
    for r in by_rows:
        by_fid.setdefault(r["file_id"], []).append(r)
    for fid, rs in by_fid.items():
        rs = sorted(rs, key=lambda r: r["step"])
        prev_vanished = None
        for r in rs:
            v = (r["vanished"] or 0.0) >= 0.5
            if prev_vanished is not None:
                if v and not prev_vanished:
                    events.append({"step": r["step"], "file_id": fid, "kind": "vanish"})
                elif (not v) and prev_vanished and (r["visible"] or 0.0) >= 0.5:
                    events.append({"step": r["step"], "file_id": fid, "kind": "return"})
            prev_vanished = v
    events.sort(key=lambda e: e["step"])
    return events


def find_attend_switch_events(by_rows):
    """attendedがTrueの行を追い、file_idが前tickと違うtickを拾う（仕様「後半」2節②）。"""
    attended_by_step = {}
    for r in by_rows:
        if r["attended"]:
            attended_by_step[r["step"]] = r["file_id"]
    steps = sorted(attended_by_step.keys())
    events = []
    prev_id = None
    for i, s in enumerate(steps):
        fid = attended_by_step[s]
        if prev_id is not None and fid != prev_id:
            events.append({"step": s, "file_id": fid})
        prev_id = fid
    return events


def find_first_last_vanish_per_file(by_rows):
    """file_idごとにvanish事象（vanished0→1）を集め、2回以上あるものだけ
    {file_id: (最初のstep, 最後のstep)} を返す（仕様「後半」2節⑥）。
    """
    by_fid_vanish_steps = {}
    by_fid = {}
    for r in by_rows:
        by_fid.setdefault(r["file_id"], []).append(r)
    for fid, rs in by_fid.items():
        rs = sorted(rs, key=lambda r: r["step"])
        prev_vanished = None
        steps = []
        for r in rs:
            v = (r["vanished"] or 0.0) >= 0.5
            if prev_vanished is not None and v and not prev_vanished:
                steps.append(r["step"])
            prev_vanished = v
        if len(steps) >= 2:
            by_fid_vanish_steps[fid] = (steps[0], steps[-1])
    return by_fid_vanish_steps


def compute_needle_ratio(main_rows, vanish_return_events, switch_events, masked_steps=None):
    """masked_steps（【2026-09-09・仕様_見る側の段構成_実装 7節】）：Noneまたは
    空集合なら従来どおり（既定不変・列が無い旧ログではNoneのまま呼ばれる）。
    渡すと①②からmasked_stepsを除いた版の針の比も計算して返す
    （needle_ratio_unmasked用）。"""
    main_by_step = {r["step"]: r for r in main_rows}

    vr_steps = set(e["step"] for e in vanish_return_events)
    sw_steps = set(e["step"] for e in switch_events)

    def vals_at(steps):
        out = []
        for s in steps:
            r = main_by_step.get(s)
            if r is not None and r["err_state"] is not None:
                out.append(r["err_state"])
        return out

    vals_vr = vals_at(vr_steps)
    vals_sw = vals_at(sw_steps - vr_steps)   # ①②が同tickで重なったら①側を優先（実装判断）
    other_steps = [r["step"] for r in main_rows
                   if r["step"] not in vr_steps and r["step"] not in sw_steps
                   and r["err_state"] is not None]
    vals_other = vals_at(other_steps)

    mean1 = sum(vals_vr) / len(vals_vr) if vals_vr else None
    mean2 = sum(vals_sw) / len(vals_sw) if vals_sw else None
    mean3 = sum(vals_other) / len(vals_other) if vals_other else None
    ratio = (mean1 / mean2) if (mean1 is not None and mean2 not in (None, 0)) else None

    result = {
        "1_消失戻り_平均err_state": mean1, "1_件数": len(vals_vr),
        "2_注意切替_平均err_state": mean2, "2_件数": len(vals_sw),
        "3_それ以外_平均err_state": mean3, "3_件数": len(vals_other),
        "4_針の比": ratio,
    }

    if masked_steps:
        vals_vr_u = vals_at(vr_steps - masked_steps)
        vals_sw_u = vals_at((sw_steps - vr_steps) - masked_steps)
        mean1_u = sum(vals_vr_u) / len(vals_vr_u) if vals_vr_u else None
        mean2_u = sum(vals_sw_u) / len(vals_sw_u) if vals_sw_u else None
        ratio_u = (mean1_u / mean2_u) if (mean1_u is not None and mean2_u not in (None, 0)) else None
        result["needle_ratio_unmasked"] = ratio_u
        result["1_消失戻り_平均err_state_unmasked"] = mean1_u
        result["1_件数_unmasked"] = len(vals_vr_u)
        result["2_注意切替_平均err_state_unmasked"] = mean2_u
        result["2_件数_unmasked"] = len(vals_sw_u)

    return result


def compute_age_stats(by_rows):
    """file_idごとの最初AGE_WINDOW_TICKS行のz_state/z_vecをプールし、
    最小・平均と、年齢別（0〜29tick目）の平均カーブを返す。
    """
    by_fid = {}
    for r in by_rows:
        by_fid.setdefault(r["file_id"], []).append(r)

    pooled_z_state, pooled_z_vec = [], []
    age_z_state = [[] for _ in range(AGE_WINDOW_TICKS)]
    age_z_vec = [[] for _ in range(AGE_WINDOW_TICKS)]
    for fid, rs in by_fid.items():
        rs = sorted(rs, key=lambda r: r["step"])[:AGE_WINDOW_TICKS]
        for i, r in enumerate(rs):
            if r["z_state"] is not None:
                pooled_z_state.append(r["z_state"])
                age_z_state[i].append(r["z_state"])
            if r["z_vec"] is not None:
                pooled_z_vec.append(r["z_vec"])
                age_z_vec[i].append(r["z_vec"])

    def mean_or_none(v):
        return sum(v) / len(v) if v else None

    age_curve_state = [mean_or_none(v) for v in age_z_state]
    age_curve_vec = [mean_or_none(v) for v in age_z_vec]

    return {
        "5_生後3秒_z_state_最小": min(pooled_z_state) if pooled_z_state else None,
        "5_生後3秒_z_state_平均": mean_or_none(pooled_z_state),
        "5_生後3秒_z_vec_最小": min(pooled_z_vec) if pooled_z_vec else None,
        "5_生後3秒_z_vec_平均": mean_or_none(pooled_z_vec),
        "5_件数_物": len(by_fid),
    }, age_curve_state, age_curve_vec


def compute_habituation(main_rows, first_last_by_file):
    main_by_step = {r["step"]: r for r in main_rows}
    firsts, lasts = [], []
    for fid, (s_first, s_last) in first_last_by_file.items():
        r_first = main_by_step.get(s_first)
        r_last = main_by_step.get(s_last)
        if r_first is not None and r_first["err_state"] is not None:
            firsts.append(r_first["err_state"])
        if r_last is not None and r_last["err_state"] is not None:
            lasts.append(r_last["err_state"])
    mean_first = sum(firsts) / len(firsts) if firsts else None
    mean_last = sum(lasts) / len(lasts) if lasts else None
    ratio = (mean_last / mean_first) if (mean_first not in (None, 0) and mean_last is not None) else None
    return {
        "6_繰返し2回以上の物数": len(first_last_by_file),
        "6_1回目消失の針_平均": mean_first,
        "6_最後消失の針_平均": mean_last,
        "6_最後_1回目の比": ratio,
    }


def setup_font():
    from matplotlib import font_manager, rcParams
    names = {f.name for f in font_manager.fontManager.ttflist}
    for c in ("BIZ UDGothic", "Yu Gothic", "Meiryo", "MS Gothic"):
        if c in names:
            rcParams["font.family"] = c
            return c
    return None


def plot(ratio_stats, age_curve_state, age_curve_vec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_font()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax0 = axes[0]
    labels = ["①消失/戻り", "②注意切替", "③それ以外"]
    vals = [ratio_stats.get("1_消失戻り_平均err_state"),
            ratio_stats.get("2_注意切替_平均err_state"),
            ratio_stats.get("3_それ以外_平均err_state")]
    vals_plot = [v if v is not None else 0.0 for v in vals]
    ax0.bar(labels, vals_plot, color=["firebrick", "darkorange", "lightgray"])
    ax0.set_ylabel("err_state（注意中の物）")
    ratio = ratio_stats.get("4_針の比")
    ax0.set_title(f"出来事の針（①〜③）\n針の比(①/②) = {ratio if ratio is not None else 'N/A'}")

    ax1 = axes[1]
    xs = list(range(len(age_curve_state)))
    ys_state = [v for v in age_curve_state]
    ys_vec = [v for v in age_curve_vec]
    xs_s = [x for x, y in zip(xs, ys_state) if y is not None]
    ys_s = [y for y in ys_state if y is not None]
    xs_v = [x for x, y in zip(xs, ys_vec) if y is not None]
    ys_v = [y for y in ys_vec if y is not None]
    ax1.plot(xs_s, ys_s, color="steelblue", label="z_state（出来事の驚き）")
    ax1.plot(xs_v, ys_v, color="seagreen", label="z_vec（新奇さ）")
    ax1.axhline(0.0, color="black", linewidth=0.6)
    ax1.set_xlabel("物が生まれてからのtick（0〜29）")
    ax1.set_ylabel("z（全物平均）")
    ax1.set_title("年齢別のz（生後3秒）")
    ax1.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)


def main():
    main_rows = load_main_rows()
    by_rows = load_by_file_rows()

    vanish_return_events = find_vanish_return_events(by_rows)
    switch_events = find_attend_switch_events(by_rows)
    # 【2026-09-09・仕様_見る側の段構成_実装 7節】masked列が無い旧ログでは
    #   全行False扱い（load_by_file_rows）なので masked_steps は空集合になり、
    #   needle_ratio_unmaskedはJSONに出ない＝従来どおり（既定不変）。
    masked_steps = find_masked_steps(by_rows)
    ratio_stats = compute_needle_ratio(main_rows, vanish_return_events, switch_events,
                                        masked_steps=masked_steps)

    age_stats, age_curve_state, age_curve_vec = compute_age_stats(by_rows)

    first_last_by_file = find_first_last_vanish_per_file(by_rows)
    habituation_stats = compute_habituation(main_rows, first_last_by_file)

    try:
        plot(ratio_stats, age_curve_state, age_curve_vec)
    except Exception as e:
        print("[針図の作成に失敗]", repr(e))

    result = {}
    result.update(ratio_stats)
    result.update(age_stats)
    result.update(habituation_stats)
    result["_年齢別z_state_カーブ"] = age_curve_state
    result["_年齢別z_vec_カーブ"] = age_curve_vec
    result["消失戻り事象数"] = len(vanish_return_events)
    result["注意切替事象数"] = len(switch_events)

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
