# -*- coding: utf-8 -*-
"""世界の予測器（M7a）の誤差の推移を図にする。

【仕様】F/docs/二語文/仕様_M7a_世界の予測器_測るだけ_2026-09-08.md
「後半：実装担当向け技術付録」5節。

引数：ログディレクトリ（例 F/logs/F2-89_M7a_世界の予測器_学習）。

【読むだけ】<ログディレクトリ>/世界の予測器.csv
  （run/plugins/common/world_predictor_log.py が書く。列は
  step,t_sec,present,visible,vanished,parent_spoke,parent_text,
  err_state,err_vec,err_parent,err_slow,err_total,baseline,z）
  err_slowは追記2026-09-08（仕様書末尾「追記」節）。遅い層（h_slow）が
  次tickの速い層（h_fast）の状態をどれだけ当てられているかの誤差。
  err_totalには含めない。

出力（このスクリプト自身の新規出力、すべて同フォルダ）：
  世界の予測器_誤差の推移.png
  世界の予測器_消失前後の平均.png
  世界の予測器_慣れ.png
  結果.json

【実装判断・仕様に無かった点】
  - baselineは太郎側ではerr_total 1本しか追わない（err_state/err_vec/err_parentの
    個別baselineは無い）。「誤差の推移」図の3段すべてに同じbaseline(err_total)を
    薄い線で重ねた（3本別々のbaselineを作る指示は無いため）。
  - 「消失」イベントは vanished列の 0→1 遷移（前の行がvanished<0.5かつ今の行が
    vanished>=0.5）とした（注意.csvと同じ流儀、f84_heldout_vanish_score.pyの
    find_vanish_eventsを参考にした）。
  - 「親の発話」イベントは parent_spoke列が>0.5の行そのもの（tick単位、1行=1発話
    イベントの意味。複数tick連続で1.0になることは無い想定＝word_productionの
    発話は基本1tickのイベントのため）。
"""
import csv
import json
import os
import sys

LOG_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    "F", "logs", "F2-89_M7a_世界の予測器_学習")
CSV_PATH = os.path.join(LOG_DIR, "世界の予測器.csv")
OUT_TIMELINE = os.path.join(LOG_DIR, "世界の予測器_誤差の推移.png")
OUT_ALIGNED = os.path.join(LOG_DIR, "世界の予測器_消失前後の平均.png")
OUT_HABIT = os.path.join(LOG_DIR, "世界の予測器_慣れ.png")
OUT_JSON = os.path.join(LOG_DIR, "結果.json")

WINDOW_SEC = 5.0          # 消失前後の平均window
JUMP_WINDOW_SEC = 1.0     # 跳ねの高さを見る「直後」の窓
BASELINE_WINDOW_SEC = 2.0  # 跳ねの高さを見る「直前」の窓


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
                "present": f("present", 0.0),
                "visible": f("visible", 0.0),
                "vanished": f("vanished", 0.0),
                "parent_spoke": f("parent_spoke", 0.0),
                "parent_text": r.get("parent_text", "") or "",
                "err_state": f("err_state"),
                "err_vec": f("err_vec"),
                "err_parent": f("err_parent"),
                "err_slow": f("err_slow"),
                "err_total": f("err_total"),
                "baseline": f("baseline"),
                "z": f("z"),
            })
    rows.sort(key=lambda r: r["step"])
    return rows


def find_vanish_events(rows):
    """vanished列の0→1遷移を[{"t":秒,"idx":行index}]で返す。"""
    events = []
    prev = None
    for i, r in enumerate(rows):
        v = r["vanished"] >= 0.5
        if prev is not None and v and not prev:
            events.append({"t": r["t_sec"], "idx": i})
        prev = v
    return events


def find_parent_events(rows):
    return [{"t": r["t_sec"], "idx": i, "text": r["parent_text"]}
            for i, r in enumerate(rows) if r["parent_spoke"] >= 0.5]


def setup_font():
    from matplotlib import font_manager, rcParams
    names = {f.name for f in font_manager.fontManager.ttflist}
    for c in ("BIZ UDGothic", "Yu Gothic", "Meiryo", "MS Gothic"):
        if c in names:
            rcParams["font.family"] = c
            return c
    return None


def plot_timeline(rows, vanish_events, parent_events):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_font()

    t = [r["t_sec"] for r in rows]
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    series = [("err_state", "err_state（物の状態）"),
              ("err_vec", "err_vec（見た目）"),
              ("err_parent", "err_parent（親の発話）"),
              ("err_slow", "err_slow（遅い層→速い層の予測）")]
    baseline = [r["baseline"] for r in rows]
    for ax, (key, label) in zip(axes, series):
        y = [r[key] for r in rows]
        ax.plot(t, y, color="steelblue", linewidth=1.0, label=label)
        ax.plot(t, baseline, color="gray", linewidth=0.8, alpha=0.5,
                linestyle="--", label="baseline(err_total)")
        for ev in vanish_events:
            ax.axvline(ev["t"], color="red", alpha=0.4, linewidth=1.0)
        for ev in parent_events:
            ax.axvline(ev["t"], color="blue", alpha=0.4, linewidth=1.0)
            if ev["text"]:
                ax.annotate(ev["text"], (ev["t"], ax.get_ylim()[1]),
                            fontsize=6, color="blue", rotation=90,
                            va="top", ha="center")
        ax.set_ylabel(label)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("時間（秒）")
    fig.suptitle("世界の予測器：誤差の推移（赤=消失の立ち上がり、青=親の発話）")
    fig.tight_layout()
    fig.savefig(OUT_TIMELINE, dpi=120)
    plt.close(fig)


def _aligned_curve(rows, events, key, window_sec):
    """各eventのt=0を揃え、±window_secのkey系列を全事象ぶん集める。

    戻り値：(共通の相対時刻グリッド, 各事象の値配列のリスト)
    """
    if not events:
        return [], []
    dt_candidates = [rows[i + 1]["t_sec"] - rows[i]["t_sec"]
                     for i in range(len(rows) - 1) if rows[i + 1]["t_sec"] > rows[i]["t_sec"]]
    dt = sorted(dt_candidates)[len(dt_candidates) // 2] if dt_candidates else 0.1
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
            # 最も近い行を1つ選ぶ（線形補間はしない・簡便法）。
            best = min(rows, key=lambda r: abs(r["t_sec"] - target_t))
            if abs(best["t_sec"] - target_t) > dt * 2:
                vals.append(None)
            else:
                vals.append(best[key])
        curves.append(vals)
    return grid, curves


def plot_aligned(rows, vanish_events, parent_events):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import statistics
    setup_font()

    # 【追記2026-09-08】err_slowを1枚足す（消失（vanished 0→1）前後。
    #   err_state・err_vecと同じ「物の変化」に対する反応を見る軸のため、
    #   err_stateと同じvanish_eventsに揃えた[実装判断・仕様に無かった点]。
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, events, key, title in [
            (axes[0], vanish_events, "err_state", "消失（vanished 0→1）前後のerr_state"),
            (axes[1], parent_events, "err_parent", "親の発話 前後のerr_parent"),
            (axes[2], vanish_events, "err_slow", "消失（vanished 0→1）前後のerr_slow")]:
        grid, curves = _aligned_curve(rows, events, key, WINDOW_SEC)
        if not grid:
            ax.set_title(title + "（事象なし）")
            continue
        means, stds = [], []
        for i in range(len(grid)):
            vals = [c[i] for c in curves if c[i] is not None]
            if vals:
                means.append(sum(vals) / len(vals))
                stds.append(statistics.pstdev(vals) if len(vals) > 1 else 0.0)
            else:
                means.append(None)
                stds.append(0.0)
        xs = [grid[i] for i in range(len(grid)) if means[i] is not None]
        ys = [means[i] for i in range(len(grid)) if means[i] is not None]
        es = [stds[i] for i in range(len(grid)) if means[i] is not None]
        ax.plot(xs, ys, color="darkorange")
        lo = [y - s for y, s in zip(ys, es)]
        hi = [y + s for y, s in zip(ys, es)]
        ax.fill_between(xs, lo, hi, color="darkorange", alpha=0.2)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_title(f"{title}（n={len(events)}）")
        ax.set_xlabel("t - 事象時刻（秒）")
        ax.set_ylabel(key)
    fig.tight_layout()
    fig.savefig(OUT_ALIGNED, dpi=120)
    plt.close(fig)


def compute_jumps(rows, events, key):
    """各事象の跳ねの高さ = 直後(0〜JUMP_WINDOW_SEC秒)の最大 - 直前(BASELINE_WINDOW_SEC秒)の平均。"""
    jumps = []
    for ev in events:
        t0 = ev["t"]
        after = [r[key] for r in rows
                 if t0 <= r["t_sec"] <= t0 + JUMP_WINDOW_SEC and r[key] is not None]
        before = [r[key] for r in rows
                  if t0 - BASELINE_WINDOW_SEC <= r["t_sec"] < t0 and r[key] is not None]
        if not after or not before:
            continue
        jumps.append(max(after) - (sum(before) / len(before)))
    return jumps


def plot_habituation(vanish_jumps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_font()

    fig, ax = plt.subplots(figsize=(8, 4))
    if vanish_jumps:
        ax.plot(range(1, len(vanish_jumps) + 1), vanish_jumps, marker="o", color="firebrick")
    ax.set_xlabel("消失事象の順番（1回目、2回目…）")
    ax.set_ylabel("跳ねの高さ（err_state）")
    ax.set_title("慣れ：消失のたびに跳ねが下がるか")
    fig.tight_layout()
    fig.savefig(OUT_HABIT, dpi=120)
    plt.close(fig)


def main():
    rows = load_rows()
    vanish_events = find_vanish_events(rows)
    parent_events = find_parent_events(rows)

    try:
        plot_timeline(rows, vanish_events, parent_events)
    except Exception as e:
        print("[誤差の推移 図の作成に失敗]", repr(e))
    try:
        plot_aligned(rows, vanish_events, parent_events)
    except Exception as e:
        print("[消失前後の平均 図の作成に失敗]", repr(e))

    vanish_jumps = compute_jumps(rows, vanish_events, "err_state")
    try:
        plot_habituation(vanish_jumps)
    except Exception as e:
        print("[慣れ 図の作成に失敗]", repr(e))

    result = {
        "行数": len(rows),
        "消失事象数": len(vanish_events),
        "親の発話事象数": len(parent_events),
        "跳ねの平均": (sum(vanish_jumps) / len(vanish_jumps)) if vanish_jumps else None,
        "1回目の跳ね": vanish_jumps[0] if vanish_jumps else None,
        "最後の跳ね": vanish_jumps[-1] if vanish_jumps else None,
        "1回目と最後の跳ねの比": (
            (vanish_jumps[-1] / vanish_jumps[0])
            if vanish_jumps and vanish_jumps[0] not in (0, None) else None),
    }
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
