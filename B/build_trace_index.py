"""
trace.jsonl（生・全イベント）から、replayViewerが読む「粗い概観ログ」と
「マイルストーン」を作る後処理。生ログには一切触らない（読むだけ）。

拡張ポイント（すべて1行足すだけ）：
  - 粒度を増やす     → BUCKETS に1行
  - マイルストーンを増やす → MILESTONES に (名前, 判定関数) を1行
  - 集約する数値を増やす → 何もしなくてよい。イベントに含まれる数値フィールドを
                          自動検出して全て平均する（core_b.trace_event に数値を
                          1行足せば、ここは無修正で概観に反映される）

使い方: python build_trace_index.py [trace.jsonl のパス]
        既定は logs/trace.jsonl。出力は同じフォルダに
        trace_overview_{hour,day,month,year}.jsonl と milestones.json。
"""
import json
import sys
import os
from collections import Counter, defaultdict

# 粒度（秒）。増やしたければ1行足すだけ。
BUCKETS = {
    "hour":  3600,
    "day":   86400,
    "month": 2592000,   # 30日
    "year":  31536000,  # 365日
}

# マイルストーン：最初に条件を満たしたイベントを記録する。増やすのは1行。
MILESTONES = [
    ("初めての喃語",           lambda e: e.get("kind") == "babble"),
    ("初めてまんまを含む喃語",   lambda e: "まんま" in e.get("utter", "")),
    ("初めての要求語",          lambda e: e.get("kind") == "word_request"),
    ("初めての授乳",           lambda e: e.get("kind") == "feed"),
    # 例）("初めて理解", lambda e: e.get("satiety", 0) > 0.5),  # 数値を足せばこう増やせる
]

NON_METRIC_KEYS = {"t"}  # 数値だが集約対象にしないキー


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def load_events(path):
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if o.get("type") == "event":
                events.append(o)
    return events


def build(raw_path):
    out_dir = os.path.dirname(os.path.abspath(raw_path))
    events = load_events(raw_path)
    if not events:
        print("イベントがありません")
        return

    # 集約する数値キーを自動検出（どのイベントかに数値として現れる全フィールド）
    metric_keys = sorted({k for e in events for k, v in e.items()
                          if _is_num(v) and k not in NON_METRIC_KEYS})

    # 粒度ごとに概観を書き出す
    for gran, size in BUCKETS.items():
        buckets = defaultdict(lambda: {"sum": defaultdict(float), "n": defaultdict(int),
                                       "kinds": Counter(), "count": 0})
        for e in events:
            slot = buckets[e["t"] // size]
            slot["count"] += 1
            slot["kinds"][e.get("kind", "?")] += 1
            for k in metric_keys:
                if k in e and _is_num(e[k]):
                    slot["sum"][k] += e[k]
                    slot["n"][k] += 1
        path = os.path.join(out_dir, f"trace_overview_{gran}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for b in sorted(buckets):
                slot = buckets[b]
                gauges = {k: round(slot["sum"][k] / slot["n"][k], 3)
                          for k in metric_keys if slot["n"][k]}
                f.write(json.dumps({"type": "bucket", "gran": gran, "t": b * size,
                                    "gauges": gauges, "counts": dict(slot["kinds"]),
                                    "count": slot["count"]}, ensure_ascii=False) + "\n")
        print(f"  {gran:6s}: {len(buckets):6d} バケツ → {os.path.basename(path)}")

    # マイルストーン（最初の1回だけ）
    found = {}
    for e in events:
        for name, test in MILESTONES:
            if name in found:
                continue
            try:
                if test(e):
                    found[name] = {"name": name, "t": e["t"],
                                   "kind": e.get("kind"), "utter": e.get("utter", "")}
            except Exception:
                pass
    ms = [found[name] for name, _ in MILESTONES if name in found]
    with open(os.path.join(out_dir, "milestones.json"), "w", encoding="utf-8") as f:
        json.dump(ms, f, ensure_ascii=False, indent=2)

    print(f"  検出した数値: {metric_keys}")
    print(f"  マイルストーン: {len(ms)}件 → milestones.json")


if __name__ == "__main__":
    raw = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "logs", "trace.jsonl")
    print(f"[集約] {raw}")
    build(raw)
