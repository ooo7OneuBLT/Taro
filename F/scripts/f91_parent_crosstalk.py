# -*- coding: utf-8 -*-
"""親の発話が物の予測を乱していないか（混線）を見る図。

【仕様】F/docs/二語文/仕様_M7b-1改_ポート型の世界の予測器_2026-09-09.md
「後半：実装担当向け技術付録」5節。

引数：ログディレクトリ（例 F/logs/F2-96_M7b1改_ポート型_学習）。

【読むだけ】<ログディレクトリ>/世界の予測器.csv
  （run/plugins/common/world_predictor_log.py が書く列。
  parent_spoke, err_state, err_parent, err_slow を使う）。

【出力】（このスクリプト自身の新規出力、同フォルダ）
  世界の予測器_親の発話混線.png
  結果_混線.json（結果.json は上書きしない、別ファイル）

【合否（仕様「合否」節）】親の発話（parent_spoke 0→1）の前後±5秒の事象平均で、
  0秒の跳ね（0〜0.3秒の最大 − 直前2秒の平均）が0.3以下ならF2-94（1.7）から
  改善したとみなせる。同じ跳ねをerr_parent・err_slowでも出す
  （err_parentは「親の発話を当てられたか」自体の指標なので跳ねて当然だが、
  比較のため同じ図式で出す）。
"""
import csv
import json
import os
import statistics
import sys

LOG_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    "F", "logs", "F2-96_M7b1改_ポート型_学習")
CSV_PATH = os.path.join(LOG_DIR, "世界の予測器.csv")
OUT_PNG = os.path.join(LOG_DIR, "世界の予測器_親の発話混線.png")
OUT_JSON = os.path.join(LOG_DIR, "結果_混線.json")

WINDOW_SEC = 5.0
# 【仕様「後半」5節】「0秒の跳ね（0〜0.3秒の最大−直前2秒の平均）」。
JUMP_WINDOW_SEC = 0.3
BASELINE_WINDOW_SEC = 2.0


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
                "parent_spoke": f("parent_spoke", 0.0),
                "parent_text": r.get("parent_text", "") or "",
                "err_state": f("err_state"),
                "err_parent": f("err_parent"),
                "err_slow": f("err_slow"),
            })
    rows.sort(key=lambda r: r["step"])
    return rows


def find_parent_onset(rows):
    """parent_spoke列の0→1遷移（f89のfind_vanish_eventsと同じ流儀）。

    【なぜspoke>=0.5の行そのものではなく0→1遷移にしたか】仕様「後半」5節
    「親の発話（parent_spoke 0→1）を0秒に揃えた」の指示どおり。f89の
    find_parent_events（spoke>=0.5の行そのもの）とは違う定義であることに注意
    （f89のコメントは「複数tick連続で1.0になることは無い想定」としていたが、
    このスクリプトでは前提を仮定せず遷移で拾う方が安全）。
    """
    events = []
    prev = None
    for i, r in enumerate(rows):
        v = r["parent_spoke"] >= 0.5
        if prev is not None and v and not prev:
            events.append({"t": r["t_sec"], "idx": i, "text": r["parent_text"]})
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


def _aligned_curve(rows, events, key, window_sec):
    """f89_world_predictor_plot.py の _aligned_curve と同じ簡便法（最近傍の行を選ぶ）。"""
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
            best = min(rows, key=lambda r: abs(r["t_sec"] - target_t))
            if abs(best["t_sec"] - target_t) > dt * 2:
                vals.append(None)
            else:
                vals.append(best[key])
        curves.append(vals)
    return grid, curves


def compute_jump(rows, events, key):
    """各事象の跳ねの高さ＝直後(0〜JUMP_WINDOW_SEC秒)の最大−直前(BASELINE_WINDOW_SEC秒)の平均。"""
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


def plot(rows, events):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_font()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, key, title in [
            (axes[0], "err_state", "親の発話 前後のerr_state（物の状態、混線を見る本命）"),
            (axes[1], "err_parent", "親の発話 前後のerr_parent（参考。当てられて当然の跳ね）"),
            (axes[2], "err_slow", "親の発話 前後のerr_slow（遅い層→速い層の予測）")]:
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
        ax.set_xlabel("t - 発話時刻（秒）")
        ax.set_ylabel(key)
    fig.suptitle("親の発話がおきたときの混線（0秒=parent_spoke 0→1）")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)


def main():
    rows = load_rows()
    events = find_parent_onset(rows)

    try:
        plot(rows, events)
    except Exception as e:
        print("[混線図の作成に失敗]", repr(e))

    jumps_state = compute_jump(rows, events, "err_state")
    jumps_parent = compute_jump(rows, events, "err_parent")
    jumps_slow = compute_jump(rows, events, "err_slow")

    result = {
        "行数": len(rows),
        "親の発話事象数": len(events),
        "err_state_0秒の跳ね_平均": (sum(jumps_state) / len(jumps_state)) if jumps_state else None,
        "err_state_0秒の跳ね_最大": max(jumps_state) if jumps_state else None,
        "err_parent_0秒の跳ね_平均": (sum(jumps_parent) / len(jumps_parent)) if jumps_parent else None,
        "err_slow_0秒の跳ね_平均": (sum(jumps_slow) / len(jumps_slow)) if jumps_slow else None,
    }
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
