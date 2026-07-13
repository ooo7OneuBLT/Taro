# NN発火(fire=活性上位32ユニット)が年齢とともに「まばら→特定ユニットに集中」
# しているかを測る。ユーザーの観察（真ん中の点が発火しなくなった）の裏取り。
import json, sys
from collections import Counter
import math

path = sys.argv[1] if len(sys.argv) > 1 else \
    r"C:\claude\AI\Taro\replayViewer\data\trace_365d_s1\trace.jsonl"
MONTH = 2592000

by_month = {}   # month -> Counter(unit -> 出現回数)
events_m = {}   # month -> 発火イベント数
with open(path, encoding="utf-8") as f:
    for line in f:
        if '"fire"' not in line:
            continue
        o = json.loads(line)
        fire = o.get("fire")
        if not isinstance(fire, list) or not fire:
            continue
        m = int(o["t"] // MONTH)
        by_month.setdefault(m, Counter()).update(fire)
        events_m[m] = events_m.get(m, 0) + 1

print("月 | 発火イベント数 | 出現した種類 | 80%を占めるユニット数 | 正規化エントロピー")
print("   (種類/エントロピー小 = 特定ユニットに集中)")
print("-" * 80)
for m in sorted(by_month):
    c = by_month[m]
    total = sum(c.values())
    distinct = len(c)
    # 80%カバーに必要なユニット数
    cum = 0; need = 0
    for _, v in c.most_common():
        cum += v; need += 1
        if cum >= 0.8 * total:
            break
    # エントロピー（0..1、1=完全に均等＝まばら）
    ent = -sum((v/total) * math.log(v/total) for v in c.values())
    nent = ent / math.log(128)
    print(f"{m:2d} | {events_m[m]:6d} | {distinct:3d}/128 | {need:3d} | {nent:.3f}")

# 生後1か月 vs 11か月で「常連ユニット」top10を比較
def top(m, k=10):
    return [f"{u}({v})" for u, v in by_month[m].most_common(k)]
lo = min(by_month); hi = max(k for k in by_month if events_m[k] > 50)
print(f"\n{lo}か月の常連top10: {top(lo)}")
print(f"{hi}か月の常連top10: {top(hi)}")
