# -*- coding: utf-8 -*-
"""驚き→青斑核（M7b-1）の配線が効いているかを図にする。

【仕様】F/docs/二語文/仕様_M7b-1_物ごとの予測器と驚きの配線_2026-09-09.md
「後半：実装担当向け技術付録」7節。

引数：ログディレクトリ（例 F/logs/F2-93_M7b1_驚き配線_学習）。

【読むだけ】<ログディレクトリ>/世界の予測器.csv
  （run/plugins/common/world_predictor_log.py が書く。列は
  step,t_sec,present,visible,vanished,parent_spoke,parent_text,
  err_state,err_vec,err_parent,err_slow,err_total,baseline,z,
  n_files,z_max,z_max_id,ne_level）
  present/visible/vanishedは「注意中の物」のもの（trainer.py
  ._world_predictor_stepがattended_pack由来で置く）。z_max/ne_levelは
  M7b-1で追記した列（multi_object無効の実験では空文字＝f90はそのCSVでは
  意味のある図を作れないので、行が読めなければ空の結果で終える）。

出力（このスクリプト自身の新規出力、すべて同フォルダ）：
  驚きと覚醒_消失前後の平均.png
  驚きと覚醒_推移_拡大.png
  結果.json

【実装判断・仕様に無かった点】
  - 「消失」「出現」イベントはvanished/visible列の0→1遷移（f89と同じ流儀）。
  - NEの上がり幅＝「0〜1秒の最大」−「直前2秒の平均」（仕様書7節の文言どおり）。
  - 「戻るまでの秒数（半分に戻る時間）」＝ピーク（0〜1秒の最大の時刻）から、
    ne_levelが「直前2秒の平均 + 上がり幅/2」以下に初めて下がる時刻までの差
    （見つからなければその事象は戻り時間の平均に含めない）。
  - 拡大図（170〜200秒）の範囲にデータが無ければ、そのログの実データ範囲の
    末尾30秒に自動で切り替える（短い走行でも図が空にならないようにするため）。
"""
import csv
import json
import os
import sys

LOG_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    "F", "logs", "F2-93_M7b1_驚き配線_学習")
CSV_PATH = os.path.join(LOG_DIR, "世界の予測器.csv")
OUT_ALIGNED = os.path.join(LOG_DIR, "驚きと覚醒_消失前後の平均.png")
OUT_ZOOM = os.path.join(LOG_DIR, "驚きと覚醒_推移_拡大.png")
OUT_JSON = os.path.join(LOG_DIR, "結果.json")

WINDOW_SEC = 5.0           # 消失/出現 前後の平均window
JUMP_WINDOW_SEC = 1.0      # 上がり幅を見る「直後」の窓
BASELINE_WINDOW_SEC = 2.0  # 上がり幅を見る「直前」の窓
ZOOM_LO, ZOOM_HI = 170.0, 200.0
ZOOM_FALLBACK_SEC = 30.0   # 170-200秒にデータが無いときの代替窓の長さ


def load_rows():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"{CSV_PATH} が無い（先に走行が必要）")
    rows = []
    with open(CSV_PATH, encoding="utf-8") as fp:
        for r in csv.DictReader(fp):
            def f(key, default=None):
                v = r.get(key, "")
                if v in (None, ""):
                    return default
                try:
                    return float(v)
                except ValueError:
                    return default
            rows.append({
                "step": int(float(r.get("step", 0) or 0)),
                "t_sec": f("t_sec", 0.0),
                "visible": f("visible", 0.0),
                "vanished": f("vanished", 0.0),
                "n_files": f("n_files"),
                "z_max": f("z_max"),
                "z_max_id": r.get("z_max_id", "") or "",
                "ne_level": f("ne_level"),
            })
    rows.sort(key=lambda r: r["step"])
    return rows


def find_transitions(rows, key):
    """key列の0→1遷移を[{"t":秒,"idx":行index}]で返す（f89.find_vanish_eventsと同じ流儀）。"""
    events = []
    prev = None
    for i, r in enumerate(rows):
        v = (r[key] or 0.0) >= 0.5
        if prev is not None and v and not prev:
            events.append({"t": r["t_sec"], "idx": i})
        prev = v
    return events


def setup_font():
    from matplotlib import font_manager, rcParams
    names = {f.name for f in font_manager.fontManager.ttflist}
    for c in ("BIZ UDGothic", "Yu Gothic", "Meiryo", "MS Gothic"):
        if c in names:
            rcParams["font.family"] = c
            return c
    return None


def _median_dt(rows):
    dt_candidates = [rows[i + 1]["t_sec"] - rows[i]["t_sec"]
                     for i in range(len(rows) - 1) if rows[i + 1]["t_sec"] > rows[i]["t_sec"]]
    return sorted(dt_candidates)[len(dt_candidates) // 2] if dt_candidates else 0.1


def _aligned_curve(rows, events, key, window_sec, dt):
    """各eventのt=0を揃え、±window_secのkey系列を全事象ぶん集める（f89._aligned_curveと同じ流儀）。"""
    if not events:
        return [], []
    grid = []
    g = -window_sec
    while g <= window_sec + 1e-9:
        grid.append(round(g, 3))
        g += dt

    curves = []
    for ev in events:
        t0 = ev["t"]
        vals = []
        for g in grid:
            target_t = t0 + g
            best = min(rows, key=lambda r: abs(r["t_sec"] - target_t))
            if abs(best["t_sec"] - target_t) > dt * 2:
                vals.append(None)
            else:
                vals.append(best[key])
        curves.append(vals)
    return grid, curves


def _mean_std(grid, curves):
    import statistics
    means, stds = [], []
    for i in range(len(grid)):
        vals = [c[i] for c in curves if c[i] is not None]
        if vals:
            means.append(sum(vals) / len(vals))
            stds.append(statistics.pstdev(vals) if len(vals) > 1 else 0.0)
        else:
            means.append(None)
            stds.append(0.0)
    return means, stds


def plot_aligned(rows, vanish_events, appear_events, dt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_font()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    panels = [
        (axes[0][0], vanish_events, "z_max", "消失（vanished 0→1）前後の z_max"),
        (axes[0][1], vanish_events, "ne_level", "消失（vanished 0→1）前後の ne_level"),
        (axes[1][0], appear_events, "z_max", "出現（visible 0→1）前後の z_max"),
        (axes[1][1], appear_events, "ne_level", "出現（visible 0→1）前後の ne_level"),
    ]
    for ax, events, key, title in panels:
        grid, curves = _aligned_curve(rows, events, key, WINDOW_SEC, dt)
        if not grid:
            ax.set_title(title + "（事象なし）")
            continue
        means, stds = _mean_std(grid, curves)
        xs = [grid[i] for i in range(len(grid)) if means[i] is not None]
        ys = [means[i] for i in range(len(grid)) if means[i] is not None]
        es = [stds[i] for i in range(len(grid)) if means[i] is not None]
        color = "firebrick" if key == "z_max" else "seagreen"
        ax.plot(xs, ys, color=color)
        lo = [y - s for y, s in zip(ys, es)]
        hi = [y + s for y, s in zip(ys, es)]
        ax.fill_between(xs, lo, hi, color=color, alpha=0.2)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_title(f"{title}（n={len(events)}）")
        ax.set_ylabel(key)
    for ax in axes[-1]:
        ax.set_xlabel("t - 事象時刻（秒）")
    fig.suptitle("驚き(z_max)と覚醒(ne_level)：消失/出現の前後（帯＝標準偏差）")
    fig.tight_layout()
    fig.savefig(OUT_ALIGNED, dpi=120)
    plt.close(fig)


def plot_zoom(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_font()

    lo, hi = ZOOM_LO, ZOOM_HI
    in_range = [r for r in rows if lo <= r["t_sec"] <= hi]
    if not in_range:
        # 【実装判断・2026-09-09、仕様に明記が無いため理由を残す】170-200秒に
        #   データが無い短い走行でも図が空にならないよう、実データ範囲の末尾へ切り替える。
        if rows:
            hi = rows[-1]["t_sec"]
            lo = max(rows[0]["t_sec"], hi - ZOOM_FALLBACK_SEC)
            in_range = [r for r in rows if lo <= r["t_sec"] <= hi]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    t = [r["t_sec"] for r in in_range]
    axes[0].plot(t, [r["z_max"] for r in in_range], color="firebrick", label="z_max")
    ax0b = axes[0].twinx()
    ax0b.plot(t, [r["ne_level"] for r in in_range], color="seagreen", alpha=0.8, label="ne_level")
    axes[0].set_ylabel("z_max", color="firebrick")
    ax0b.set_ylabel("ne_level", color="seagreen")
    axes[0].set_title(f"z_max・ne_level の推移（{lo:.0f}〜{hi:.0f}秒）")

    axes[1].step(t, [r["vanished"] for r in in_range], where="post", color="red", label="vanished（注意中の物）")
    axes[1].step(t, [r["visible"] for r in in_range], where="post", color="blue", alpha=0.6, label="visible（注意中の物）")
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].set_xlabel("時間（秒）")
    axes[1].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_ZOOM, dpi=120)
    plt.close(fig)


def compute_ne_rise_and_recovery(rows, events):
    """各事象の「ne上がり幅」と「半分に戻るまでの秒数」を求める（仕様7節）。

    上がり幅 = 0〜1秒のne_levelの最大 - 直前2秒の平均。
    戻り時間 = ピーク時刻から、ne_levelが「直前2秒平均 + 上がり幅/2」以下に
    初めて下がる時刻までの差（見つからなければその事象は除外）。
    """
    rises = []
    recoveries = []
    for ev in events:
        t0 = ev["t"]
        before = [r["ne_level"] for r in rows
                  if t0 - BASELINE_WINDOW_SEC <= r["t_sec"] < t0 and r["ne_level"] is not None]
        after = [(r["t_sec"], r["ne_level"]) for r in rows
                 if t0 <= r["t_sec"] <= t0 + JUMP_WINDOW_SEC and r["ne_level"] is not None]
        if not before or not after:
            continue
        base = sum(before) / len(before)
        peak_t, peak_v = max(after, key=lambda pair: pair[1])
        rise = peak_v - base
        rises.append(rise)
        if rise <= 0:
            continue
        half_level = base + rise / 2.0
        later = [(r["t_sec"], r["ne_level"]) for r in rows
                 if r["t_sec"] > peak_t and r["ne_level"] is not None]
        rec = next((t for t, v in later if v <= half_level), None)
        if rec is not None:
            recoveries.append(rec - peak_t)
    return rises, recoveries


def main():
    rows = load_rows()
    dt = _median_dt(rows)
    vanish_events = find_transitions(rows, "vanished")
    appear_events = find_transitions(rows, "visible")

    try:
        plot_aligned(rows, vanish_events, appear_events, dt)
    except Exception as e:
        print("[消失前後の平均 図の作成に失敗]", repr(e))
    try:
        plot_zoom(rows)
    except Exception as e:
        print("[推移_拡大 図の作成に失敗]", repr(e))

    rises, recoveries = compute_ne_rise_and_recovery(rows, vanish_events)

    result = {
        "行数": len(rows),
        "消失事象数": len(vanish_events),
        "出現事象数": len(appear_events),
        "ne上がり幅_平均": (sum(rises) / len(rises)) if rises else None,
        "ne戻り秒数_平均": (sum(recoveries) / len(recoveries)) if recoveries else None,
        "ne戻り秒数_事象数": len(recoveries),
    }
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
