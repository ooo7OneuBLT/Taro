# -*- coding: utf-8 -*-
"""ポートごとの驚き（attend_port）を集計する図。

【仕様】F/docs/二語文/仕様_M7b-1改2_遅い層の学習の道筋とポートごとの驚き_2026-09-09.md
「後半：実装担当向け技術付録」3節。

引数：ログディレクトリ（例 F/logs/F2-97_M7b1改2_遅い層の学習_学習）。

【読むだけ】<ログディレクトリ>/世界の予測器.csv
  （run/plugins/common/world_predictor_log.py が書く列。
  attend_port, z_hearing, z_body, z_max_id, parent_spoke, vanished を使う）。

【出力】（このスクリプト自身の新規出力、同フォルダ。他のf8x/f9xと名前が衝突しないよう
  「_ポート注意」を付ける）
  世界の予測器_ポート注意.png（3枚並び）
  結果_ポート注意.json

【出す数値（仕様「後半」3節）】
  1. attend_port の種類別の割合（vision / hearing / body / 空）
  2. 親の発話（parent_spoke 0→1）±2秒の各tickで、attend_port が "hearing" に
     なっている割合（0秒側に寄っているはず）
  3. 消失（vanished 0→1）±2秒の各tickで、attend_port が「そのtickの
     z_max_id の視覚（"vision:<z_max_id>"）」に一致している割合
"""
import csv
import json
import os
import sys

LOG_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    "F", "logs", "F2-97_M7b1改2_遅い層の学習_学習")
CSV_PATH = os.path.join(LOG_DIR, "世界の予測器.csv")
OUT_PNG = os.path.join(LOG_DIR, "世界の予測器_ポート注意.png")
OUT_JSON = os.path.join(LOG_DIR, "結果_ポート注意.json")

WINDOW_SEC = 2.0  # 仕様「後半」3節：親の発話・消失 それぞれ±2秒


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

            z_max_id_raw = r.get("z_max_id", "")
            z_max_id = None
            if z_max_id_raw not in (None, ""):
                try:
                    z_max_id = int(float(z_max_id_raw))
                except ValueError:
                    z_max_id = None

            rows.append({
                "step": int(float(r.get("step", 0) or 0)),
                "t_sec": f("t_sec", 0.0),
                "parent_spoke": f("parent_spoke", 0.0),
                "vanished": f("vanished", 0.0),
                "z_hearing": f("z_hearing"),
                "z_body": f("z_body"),
                "z_max_id": z_max_id,
                # attend_port は文字列列（"vision:<id>" / "hearing" / "body" / ""）。
                # 数値化しないでそのまま使う。
                "attend_port": (r.get("attend_port", "") or ""),
            })
    rows.sort(key=lambda r: r["step"])
    return rows


def find_onset_events(rows, key):
    """key列の0→1遷移（f89のfind_vanish_events・f91のfind_parent_onsetと同じ流儀）。"""
    events = []
    prev = None
    for i, r in enumerate(rows):
        v = r[key] >= 0.5
        if prev is not None and v and not prev:
            events.append({"t": r["t_sec"], "idx": i})
        prev = v
    return events


def port_kind(attend_port):
    """attend_port文字列 -> "vision"/"hearing"/"body"/"空" の種類。"""
    if not attend_port:
        return "空"
    if attend_port.startswith("vision:"):
        return "vision"
    if attend_port == "hearing":
        return "hearing"
    if attend_port == "body":
        return "body"
    return "その他"


def compute_kind_ratio(rows):
    counts = {"vision": 0, "hearing": 0, "body": 0, "空": 0, "その他": 0}
    for r in rows:
        counts[port_kind(r["attend_port"])] += 1
    total = len(rows) if rows else 1
    return {k: v / total for k, v in counts.items()}, counts


def _window_rows(rows, events, window_sec):
    """各事象について±window_secに入る行を集める（イベントごとのリスト）。"""
    out = []
    for ev in events:
        t0 = ev["t"]
        in_window = [r for r in rows if t0 - window_sec <= r["t_sec"] <= t0 + window_sec]
        out.append(in_window)
    return out


def hearing_ratio_around_parent(rows, events, window_sec=WINDOW_SEC):
    """親の発話±window_secの全tickをプールし、attend_port=="hearing"の割合。"""
    groups = _window_rows(rows, events, window_sec)
    pooled = [r for g in groups for r in g]
    if not pooled:
        return None, 0
    n_hearing = sum(1 for r in pooled if r["attend_port"] == "hearing")
    return n_hearing / len(pooled), len(pooled)


def vision_match_ratio_around_vanish(rows, events, window_sec=WINDOW_SEC):
    """消失±window_secの全tickをプールし、attend_portが
    「そのtickのz_max_idの視覚」と一致している割合。z_max_idがNoneの行は分母から除く
    （仕様「その tick の z_max_id の視覚」＝tickごとに変わる的、と読んだ）。
    """
    groups = _window_rows(rows, events, window_sec)
    pooled = [r for g in groups for r in g]
    denom = [r for r in pooled if r["z_max_id"] is not None]
    if not denom:
        return None, 0
    n_match = sum(1 for r in denom if r["attend_port"] == f"vision:{r['z_max_id']}")
    return n_match / len(denom), len(denom)


def _aligned_indicator_curve(rows, events, indicator_fn, window_sec):
    """f91の_aligned_curveと同じ簡便法（最近傍の行）で、indicator_fn(row)->0/1/Noneの
    平均カーブ（時刻ずれ -window_sec〜+window_secのgrid）を作る。図示専用。"""
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
                vals.append(indicator_fn(best))
            _ = best
        curves.append(vals)
    return grid, curves


def setup_font():
    from matplotlib import font_manager, rcParams
    names = {f.name for f in font_manager.fontManager.ttflist}
    for c in ("BIZ UDGothic", "Yu Gothic", "Meiryo", "MS Gothic"):
        if c in names:
            rcParams["font.family"] = c
            return c
    return None


def plot(rows, kind_ratio, parent_events, vanish_events):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_font()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # 1枚目：attend_port種類別の割合（棒グラフ）。
    ax0 = axes[0]
    kinds = ["vision", "hearing", "body", "空"]
    vals = [kind_ratio.get(k, 0.0) for k in kinds]
    ax0.bar(kinds, vals, color=["steelblue", "darkorange", "seagreen", "lightgray"])
    ax0.set_ylim(0, 1)
    ax0.set_title(f"attend_port 種類別の割合（全{len(rows)}tick）")
    ax0.set_ylabel("割合")

    # 2枚目：親の発話±2秒、attend_port=="hearing"の平均カーブ。
    ax1 = axes[1]
    grid, curves = _aligned_indicator_curve(
        rows, parent_events,
        lambda r: 1.0 if r["attend_port"] == "hearing" else 0.0,
        WINDOW_SEC)
    if grid:
        means = []
        for i in range(len(grid)):
            v = [c[i] for c in curves if c[i] is not None]
            means.append(sum(v) / len(v) if v else None)
        xs = [grid[i] for i in range(len(grid)) if means[i] is not None]
        ys = [means[i] for i in range(len(grid)) if means[i] is not None]
        ax1.plot(xs, ys, color="darkorange")
        ax1.axvline(0.0, color="black", linewidth=0.8)
        ax1.set_ylim(0, 1)
    ax1.set_title(f"親の発話 前後（n={len(parent_events)}）\nattend_port==hearing の割合")
    ax1.set_xlabel("t - 発話時刻（秒）")

    # 3枚目：消失±2秒、attend_portがそのtickのz_max_idの視覚と一致する平均カーブ。
    ax2 = axes[2]
    grid2, curves2 = _aligned_indicator_curve(
        rows, vanish_events,
        lambda r: (1.0 if (r["z_max_id"] is not None
                            and r["attend_port"] == f"vision:{r['z_max_id']}")
                    else (None if r["z_max_id"] is None else 0.0)),
        WINDOW_SEC)
    if grid2:
        means2 = []
        for i in range(len(grid2)):
            v = [c[i] for c in curves2 if c[i] is not None]
            means2.append(sum(v) / len(v) if v else None)
        xs2 = [grid2[i] for i in range(len(grid2)) if means2[i] is not None]
        ys2 = [means2[i] for i in range(len(grid2)) if means2[i] is not None]
        ax2.plot(xs2, ys2, color="firebrick")
        ax2.axvline(0.0, color="black", linewidth=0.8)
        ax2.set_ylim(0, 1)
    ax2.set_title(f"消失 前後（n={len(vanish_events)}）\nattend_port==その物の視覚 の割合")
    ax2.set_xlabel("t - 消失時刻（秒）")

    fig.suptitle("ポートごとの驚き（attend_port）の集計")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)


def main():
    rows = load_rows()
    kind_ratio, kind_counts = compute_kind_ratio(rows)

    parent_events = find_onset_events(rows, "parent_spoke")
    vanish_events = find_onset_events(rows, "vanished")

    hearing_ratio, hearing_n = hearing_ratio_around_parent(rows, parent_events)
    vision_match_ratio, vision_n = vision_match_ratio_around_vanish(rows, vanish_events)

    try:
        plot(rows, kind_ratio, parent_events, vanish_events)
    except Exception as e:
        print("[ポート注意図の作成に失敗]", repr(e))

    result = {
        "行数": len(rows),
        "attend_port_種類別割合": kind_ratio,
        "attend_port_種類別件数": kind_counts,
        "親の発話事象数": len(parent_events),
        "親の発話±2秒_hearing最多割合": hearing_ratio,
        "親の発話±2秒_対象tick数": hearing_n,
        "消失事象数": len(vanish_events),
        "消失±2秒_その物の視覚最多割合": vision_match_ratio,
        "消失±2秒_対象tick数": vision_n,
    }
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
