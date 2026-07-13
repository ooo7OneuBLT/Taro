# 要求語(word_request)3281回の中身を分析する。
# 問い：自発発話が年齢とともに「まんま」へ音韻収束しているか／空腹はどの水準か。
import json, sys, difflib
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else \
    r"C:\claude\AI\Taro\replayViewer\data\trace_365d_s1\trace.jsonl"
TARGET = "まんま"
MONTH = 2592000  # 秒/月(30日)

def sim(s):
    return difflib.SequenceMatcher(None, s, TARGET).ratio()

reqs = []
with open(path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if '"word_request"' not in line:
            continue
        o = json.loads(line)
        reqs.append((o["t"], o.get("utter", ""), o.get("hunger", 0.0)))

print(f"総要求語数: {len(reqs)}")
print(f"空腹hunger: 全て>0.5(定義上)  平均={sum(r[2] for r in reqs)/len(reqs):.3f}  "
      f"最大={max(r[2] for r in reqs):.3f}")

# 月ごとに集計
buckets = {}
for t, utter, hunger in reqs:
    m = int(t // MONTH)
    buckets.setdefault(m, []).append((utter, hunger, sim(utter)))

print("\n月 | 回数 | 平均空腹 | 平均まんま類似度 | 「まんま」完全一致% | 頻出発話(上位3)")
print("-" * 95)
for m in sorted(buckets):
    b = buckets[m]
    n = len(b)
    mh = sum(x[1] for x in b) / n
    ms = sum(x[2] for x in b) / n
    exact = sum(1 for x in b if x[0] == TARGET) / n * 100
    top = Counter(x[0] for x in b).most_common(3)
    tops = " ".join(f"{w}×{c}" for w, c in top)
    print(f"{m:2d} | {n:4d} | {mh:.3f} | {ms:.3f} | {exact:5.1f}% | {tops}")

# 全期間の発話ランキング
print("\n=== 全期間の発話トップ15 ===")
for w, c in Counter(r[1] for r in reqs).most_common(15):
    print(f"  {w!r:12} ×{c}  (類似度{sim(w):.2f})")
