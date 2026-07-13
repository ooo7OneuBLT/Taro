"""
検証②：「まんま」への意味付き分析

7日間のログから：
- 食事文脈（feed）での模倣報酬の推移
- 非食事文脈（comfort）での模倣報酬の推移
- 太郎の出力に「まんま」の構成音が含まれる率の推移
"""

import json
import os
import sys

def load_log(run_name):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "logs", run_name, "turns.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def analyze(logs, window=10):
    feed = [r for r in logs if r["context"] == "feed"]
    comfort = [r for r in logs if r["context"] == "comfort"]

    print(f"=== まんま学習 検証結果 ===")
    print(f"総発話ターン数: {len(logs)}")
    print(f"  食事文脈（まんま）: {len(feed)}回")
    print(f"  その他（よしよし等）: {len(comfort)}回")
    print()

    # --- r_imit の前半 vs 後半比較 ---
    def avg_imit(records):
        if not records:
            return 0.0
        return sum(r["r_imit"] for r in records) / len(records)

    mid = len(logs) // 2
    feed_early = [r for r in feed if r["turn"] <= mid]
    feed_late  = [r for r in feed if r["turn"] > mid]
    comf_early = [r for r in comfort if r["turn"] <= mid]
    comf_late  = [r for r in comfort if r["turn"] > mid]

    print("--- 模倣報酬（r_imit）の前半 vs 後半 ---")
    print(f"  食事文脈   前半: {avg_imit(feed_early):.3f}  後半: {avg_imit(feed_late):.3f}  差: {avg_imit(feed_late)-avg_imit(feed_early):+.3f}")
    print(f"  その他文脈 前半: {avg_imit(comf_early):.3f}  後半: {avg_imit(comf_late):.3f}  差: {avg_imit(comf_late)-avg_imit(comf_early):+.3f}")
    print()

    # --- 太郎の出力にまんまの音が含まれる率 ---
    # 「まんま」の構成文字チェック（ま / ん / ま）
    MAMAMA_CHARS = set("まんま")

    def mamama_score(text):
        """出力がどれくらい『まんま』の文字で構成されているか（0〜1）"""
        if not text:
            return 0.0
        hits = sum(1 for c in text if c in MAMAMA_CHARS)
        return hits / len(text)

    def exact_mamama(text):
        return "まんま" in text or text == "まんま"

    feed_exact_early = sum(1 for r in feed_early if exact_mamama(r["taro"]))
    feed_exact_late  = sum(1 for r in feed_late  if exact_mamama(r["taro"]))
    comf_exact_early = sum(1 for r in comf_early if exact_mamama(r["taro"]))
    comf_exact_late  = sum(1 for r in comf_late  if exact_mamama(r["taro"]))

    def safe_rate(n, d):
        return f"{n}/{d} = {n/d:.1%}" if d > 0 else "0/0"

    print("--- 太郎の出力に「まんま」が含まれる率 ---")
    print(f"  食事文脈   前半: {safe_rate(feed_exact_early, len(feed_early))}  後半: {safe_rate(feed_exact_late, len(feed_late))}")
    print(f"  その他文脈 前半: {safe_rate(comf_exact_early, len(comf_early))}  後半: {safe_rate(comf_exact_late, len(comf_late))}")
    print()

    # --- 全発話サンプル（最新10件の食事文脈）---
    print("--- 食事文脈の最新10件 ---")
    for r in feed[-10:]:
        day = r["sim_seconds"] // 86400 + 1
        mark = "★" if exact_mamama(r["taro"]) else "  "
        print(f"  {mark} Day{day} turn={r['turn']:3d} | 親「{r['parent']}」→ 太郎「{r['taro']}」| r_imit={r['r_imit']:.2f} hunger={r['hunger']:.2f}")

    print()
    print("--- その他文脈の最新10件 ---")
    for r in comfort[-10:]:
        day = r["sim_seconds"] // 86400 + 1
        mark = "★" if exact_mamama(r["taro"]) else "  "
        print(f"  {mark} Day{day} turn={r['turn']:3d} | 親「{r['parent']}」→ 太郎「{r['taro']}」| r_imit={r['r_imit']:.2f}")

    # --- 日別の模倣報酬推移 ---
    print()
    print("--- 日別 r_imit 推移 ---")
    print(f"{'日':>4} | {'食事r_imit':>10} | {'その他r_imit':>12} | {'まんま出現(食事)':>16} | {'まんま出現(その他)':>18}")
    print("-" * 70)
    for day in range(1, 8):
        day_start = (day - 1) * 86400
        day_end   = day * 86400
        fd = [r for r in feed    if day_start <= r["sim_seconds"] < day_end]
        cd = [r for r in comfort if day_start <= r["sim_seconds"] < day_end]
        fd_r = avg_imit(fd)
        cd_r = avg_imit(cd)
        fd_m = sum(1 for r in fd if exact_mamama(r["taro"]))
        cd_m = sum(1 for r in cd if exact_mamama(r["taro"]))
        fd_rate = f"{fd_m}/{len(fd)}({fd_m/len(fd):.0%})" if fd else "-"
        cd_rate = f"{cd_m}/{len(cd)}({cd_m/len(cd):.0%})" if cd else "-"
        print(f"  Day{day} | {fd_r:>10.3f} | {cd_r:>12.3f} | {fd_rate:>16} | {cd_rate:>18}")


if __name__ == "__main__":
    run_name = sys.argv[1] if len(sys.argv) > 1 else "verification_mamama_7days"
    logs = load_log(run_name)
    analyze(logs)
