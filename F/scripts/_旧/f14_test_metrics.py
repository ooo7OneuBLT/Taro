"""F1-4h語彙テストの集計 — WordSchedule（parent_labeling.py）が書き出したCSVを読み、
語イベントごとに①初見先②注視時間を計算して表で出す。

【設計】F/docs/設計_F1-4h_語彙テストの測定装置.md 技術付録「4. 記録と集計」。
測定装置そのもの（毎stepの記録）はE/scripts/parent_labeling.WordScheduleが担う。
このスクリプトは走行後に読むだけの独立した集計ツール（run/main.pyのハーネスとは
無関係。F/scripts/f14_test_metrics.py 単体で完結する）。

入力CSVの列（WordSchedule._ensure_csvが書く）: t, a1, a2, word
    t     エピソード開始からの秒数
    a1    視線とtoy1（test_object1）の中心のなす角度[度]（求まらなければ空欄）
    a2    同・toy2（test_object2）
    word  そのstepで注入された語（無ければ空）

指標（技術付録4節の定義）：
    初見先   語の後、window秒以内に最初にthresh度以内へ入ったおもちゃ
             （入らなければ「なし」）
    注視時間 語後window秒間の、各おもちゃthresh度以内の滞在秒（台形近似ではなく
             「次サンプルまでの時間ぶん、そのおもちゃを見ていたとみなす」区分求積。
             理由：MuJoCoの物理stepは離散なので、サンプル間の姿勢は不明＝
             直前のサンプルの状態が次のサンプルまで続いたとみなすのが最も単純で
             恣意性が少ない）。

使い方:
    python f14_test_metrics.py CSVのパス [--window 6.0] [--thresh 10.0] [--out 出力.csv]
"""
import argparse
import csv
import os


def _read_rows(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8") as fp:
        for r in csv.DictReader(fp):
            rows.append({
                "t": float(r["t"]),
                "a1": (None if r["a1"] == "" else float(r["a1"])),
                "a2": (None if r["a2"] == "" else float(r["a2"])),
                "word": r.get("word") or "",
            })
    rows.sort(key=lambda r: r["t"])
    return rows


def _find_events(rows):
    """語が注入された行（word非空）を時刻順に返す。"""
    return [r for r in rows if r["word"]]


def _first_look(rows, t0, window, thresh):
    """t0以降・window秒以内で、最初にthresh度以内へ入ったおもちゃ名を返す。

    同じ行でtoy1・toy2の両方が条件を満たしたときは、角度が小さいほう
    （より正面に近いほう）を採用する。見つからなければ "なし"。
    """
    t1 = t0 + window
    for r in rows:
        if r["t"] < t0 or r["t"] > t1:
            continue
        c1 = r["a1"] is not None and r["a1"] <= thresh
        c2 = r["a2"] is not None and r["a2"] <= thresh
        if c1 and c2:
            return "toy1" if r["a1"] <= r["a2"] else "toy2"
        if c1:
            return "toy1"
        if c2:
            return "toy2"
    return "なし"


def _looking_time(rows, t0, window, thresh):
    """t0以降・window秒間の、toy1/toy2それぞれの滞在秒（区分求積・冒頭docstring参照）。"""
    t1 = t0 + window
    win_rows = [r for r in rows if t0 <= r["t"] <= t1]
    total1 = 0.0
    total2 = 0.0
    for i, r in enumerate(win_rows):
        nxt_t = win_rows[i + 1]["t"] if i + 1 < len(win_rows) else t1
        dt = max(0.0, min(nxt_t, t1) - r["t"])
        if r["a1"] is not None and r["a1"] <= thresh:
            total1 += dt
        if r["a2"] is not None and r["a2"] <= thresh:
            total2 += dt
    return total1, total2


def compute_metrics(csv_path, window=6.0, thresh=10.0):
    rows = _read_rows(csv_path)
    events = _find_events(rows)
    out = []
    for idx, ev in enumerate(events):
        t0 = ev["t"]
        first = _first_look(rows, t0, window, thresh)
        look1, look2 = _looking_time(rows, t0, window, thresh)
        out.append({
            "index": idx,
            "t": t0,
            "word": ev["word"],
            "first_look": first,
            "look_toy1_sec": round(look1, 3),
            "look_toy2_sec": round(look2, 3),
        })
    return out


def _print_table(metrics):
    header = f"{'#':>3} {'t[s]':>8} {'語':<10} {'初見先':<8} {'toy1[s]':>9} {'toy2[s]':>9}"
    print(header)
    print("-" * len(header))
    for m in metrics:
        print(f"{m['index']:>3} {m['t']:>8.2f} {m['word']:<10} "
              f"{m['first_look']:<8} {m['look_toy1_sec']:>9.3f} {m['look_toy2_sec']:>9.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv_path", help="WordScheduleが書き出したCSV")
    ap.add_argument("--window", type=float, default=6.0, help="語後の測定秒数（既定6.0）")
    ap.add_argument("--thresh", type=float, default=10.0,
                    help="『見ている』とみなす角度[度]（既定10.0）")
    ap.add_argument("--out", default=None, help="集計結果をCSVでも保存する場合のパス")
    args = ap.parse_args()

    metrics = compute_metrics(args.csv_path, window=args.window, thresh=args.thresh)
    if not metrics:
        print("[f14_test_metrics] 語イベントが1件もありません（CSVのword列を確認）。")
        return
    _print_table(metrics)

    if args.out:
        d = os.path.dirname(args.out)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=list(metrics[0].keys()))
            w.writeheader()
            w.writerows(metrics)
        print(f"[f14_test_metrics] 保存しました → {args.out}")


if __name__ == "__main__":
    main()
