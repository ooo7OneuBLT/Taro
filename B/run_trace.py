"""
replayViewer用のトレースログ(trace.jsonl)を生成する。

短期間だけ走らせて、太郎の中で「いつ・どの部品が発火し・数値がどうだったか」を
1行1イベントで書き出す。出力を replayViewer の「ファイルを選択」で読み込むと
再生できる。フル1年は数十万イベントになるので、まずは短期間で。

使い方: python run_trace.py [日数]   （既定7日）
"""
import sys
import os

days = int(sys.argv[1]) if len(sys.argv) > 1 else 7
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
torch.set_num_threads(2)

out_path = os.path.join(os.path.dirname(__file__), "logs", "trace.jsonl")
os.makedirs(os.path.dirname(out_path), exist_ok=True)

from environment.parent_sim_b import run_simulation_b

r = run_simulation_b(
    max_sim_seconds=days * 86400,
    verbose=False,
    run_name=f"trace_{days}d",
    trace_path=out_path,
)

n = sum(1 for _ in open(out_path, encoding="utf-8"))
print(f"[完了] {days}日分 → {out_path}")
print(f"トレース行数（イベント数）: {n}")
print(f"泣き{r['cry_count']} 食事{r['feed_count']} 喃語{r['babble_count']} 要求語{r['request_count']}")
