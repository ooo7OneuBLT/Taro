# -*- coding: utf-8 -*-
"""K1の読み：探し物を教えたとき、太郎はそこへ届いたか。

【仕様】`F/docs/二語文/仕様_K1_記憶からの山_2026-09-11.md`。

【数え方】的（`goal_id` のカード）の位置から半径16画素に、注意点
（`attn_x`/`attn_y`）が2秒以内に入ったか。

**数える前に外すもの**：始めの時点で注意が的から32画素以内にあった試行。
すでにそこを見ていたのだから、届いても目的の効果ではない
（ユーザー指摘 2026-09-11。F2-126 では70件中18件が該当し、これを外すと
0点が 20% → 15% に下がった）。

【0点】15%（F2-126 の有効52試行、95%で±10pt）。
合格＝これを明確に超える（目安40%以上）かつ100%でない。

【効かなかったときに見るところ】内訳の列。山が小さいのか（`g_at_goal`）、
復帰抑制に消されたのか（`ior_at_goal`）、溜めが目立ちに埋もれたのか
（`acc_at_goal` 対 `acc_at_win`）。**「口が死んでいる」と決めつけない。**

【似た道具】`F/scripts/f102_stay_probe.py`＝注意が留まれない原因の切り分け。
こちらは「目的を渡したとき届いたか」を数える。

使い方:
  python F/scripts/f108_k1_goal_reach.py F/logs/F2-129_K1_記憶からの山
  python F/scripts/f108_k1_goal_reach.py <ログ> --baseline   # 目的なしの走行の0点を測る
"""
import argparse
import collections
import csv
import math
import os
import statistics
import sys

HIT_R = 16.0        # 当たりとみなす半径［画素］
EXCLUDE_R = 32.0    # 始めからこれより近ければ試行に数えない［画素］
WIN_S = 2.0         # 届いたかを見る窓［秒］
HOLD_S = 3.0        # 試行の間隔［秒］（0点の測り方と揃える）
BASELINE = 0.15     # 0点（F2-126 の有効52試行）


def _load(log_dir):
    p = os.path.join(log_dir, "物体ファイル.csv")
    if not os.path.exists(p):
        sys.exit("見つからない: %s" % p)
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _per_frame(rows, use_goal):
    """コマごとに {注意点, 的の位置, 見失い中カード} を集める。"""
    per = collections.defaultdict(lambda: {"attn": None, "goal": None, "miss": []})
    for r in rows:
        try:
            t = float(r["sim_time"])
        except (ValueError, KeyError):
            continue
        if r.get("attn_x"):
            try:
                per[t]["attn"] = (float(r["attn_x"]), float(r["attn_y"]))
            except ValueError:
                pass
        # 目的つきの走行：goal_id のカードの位置をその行から拾う
        if use_goal and r.get("goal_id") and r.get("file_id") == r.get("goal_id"):
            try:
                per[t]["goal"] = (r["goal_id"], float(r["x"]), float(r["y"]))
            except ValueError:
                pass
        if r.get("event") == "unmatched" and r.get("file_id") and r.get("x"):
            try:
                per[t]["miss"].append((r["file_id"], float(r["x"]), float(r["y"])))
            except ValueError:
                pass
    return per


def main(log_dir, baseline_mode):
    rows = _load(log_dir)
    has_goal = any(r.get("goal_id") for r in rows)
    use_goal = has_goal and not baseline_mode
    if baseline_mode:
        print("【0点を測る】目的を使わず、見失い中カードを的とみなして数える\n")
    elif not has_goal:
        sys.exit("このログに goal_id が無い。目的なしの走行なら --baseline を付ける")

    per = _per_frame(rows, use_goal)
    ts = sorted(per)
    if not ts:
        sys.exit("読める行が無い")

    trials, excluded = [], 0
    nxt = ts[0]
    for t in ts:
        if t < nxt:
            continue
        d = per[t]
        if d["attn"] is None:
            continue
        if use_goal:
            tgt = d["goal"]
        else:
            tgt = d["miss"][0] if d["miss"] else None
        if tgt is None:
            continue
        fid, fx, fy = tgt
        ax, ay = d["attn"]
        d0 = math.hypot(ax - fx, ay - fy)
        nxt = t + HOLD_S
        if d0 <= EXCLUDE_R:          # すでにそこを見ていた＝数えない
            excluded += 1
            continue
        hit = False
        for u in ts:
            if not (t <= u <= t + WIN_S):
                continue
            a = per[u]["attn"]
            if a and math.hypot(a[0] - fx, a[1] - fy) <= HIT_R:
                hit = True
                break
        edge = (min(fx, fy) < 32.0 or max(fx, fy) > 192.0)   # 的が画像の端か
        trials.append({"t": t, "d0": d0, "hit": hit, "edge": edge})

    if not trials:
        sys.exit("有効な試行が0件。始めから近すぎる試行ばかりか、的が立っていない")

    n = len(trials)
    hits = sum(1 for x in trials if x["hit"])
    p = hits / n
    se = math.sqrt(p * (1 - p) / n)
    print("有効な試行 %d 件（始めから%.0fpx以内で外したもの %d 件）" % (n, EXCLUDE_R, excluded))
    print("始めの距離：中央値 %.1fpx / 最小 %.1f / 最大 %.1f"
          % (statistics.median(x["d0"] for x in trials),
             min(x["d0"] for x in trials), max(x["d0"] for x in trials)))
    print("\n**%.0f秒以内に的へ届いた：%d / %d ＝ %.0f%%（95%%で ±%.0fpt）**"
          % (WIN_S, hits, n, 100 * p, 196 * se))

    for tag, sel in (("的が画像の端", True), ("的が中の方", False)):
        g = [x for x in trials if x["edge"] is sel]
        if g:
            print("  %s：%d / %d ＝ %.0f%%"
                  % (tag, sum(1 for x in g if x["hit"]), len(g),
                     100 * sum(1 for x in g if x["hit"]) / len(g)))

    if baseline_mode:
        print("\n（この値を0点として使う）")
        return
    print("\n0点は %.0f%%。合格＝これを明確に超え（目安40%%以上）、かつ100%%でないこと。"
          % (100 * BASELINE))
    if n < 40:
        print("判定しない：有効な試行が40件未満（0点は52件なので比較にならない）")
    elif p >= 1.0:
        print("**不合格**：100%＝目的が強すぎて目立ちとの競争になっていない（乗っ取りと同じ）")
    elif p >= 0.40:
        print("**合格**：口は生きている。K2（目立つ方を無視して探し物を選ぶ）へ")
    else:
        print("**未達**。下の内訳で原因を分ける。『口が死んでいる』と決めつけない")

    if not use_goal:
        return
    vals = collections.defaultdict(list)
    for r in rows:
        for k in ("g_at_goal", "sal_at_goal", "ior_at_goal", "acc_at_goal",
                  "g_at_win", "sal_at_win", "ior_at_win", "acc_at_win"):
            v = r.get(k)
            if v not in (None, ""):
                try:
                    vals[k].append(float(v))
                except ValueError:
                    pass
    if vals:
        print("\n【内訳】各項の中央値（的の升 / 勝った升）")
        print("| 項 | 的の升 | 勝った升 |")
        print("|---|---|---|")
        for a, b, name in (("g_at_goal", "g_at_win", "目的の山"),
                           ("sal_at_goal", "sal_at_win", "目立ち"),
                           ("ior_at_goal", "ior_at_win", "復帰抑制"),
                           ("acc_at_goal", "acc_at_win", "溜め")):
            f = lambda k: ("%.3f" % statistics.median(vals[k])) if vals.get(k) else "—"
            print("| %s | %s | %s |" % (name, f(a), f(b)))
        print("\n読み方：目的の山が目立ちよりずっと小さければ g を上げる。")
        print("的の升の復帰抑制が大きければ、直前にそこを見て抑制が残っている。")
        print("溜めが的で小さく勝った升で大きければ、目立ちに押し負けている。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir")
    ap.add_argument("--baseline", action="store_true",
                    help="目的なしの走行から0点を測る")
    a = ap.parse_args()
    main(a.log_dir, a.baseline)
