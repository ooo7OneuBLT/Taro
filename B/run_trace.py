"""
短期間のシミュを走らせ、replayViewer用ログを出力する。
出力先は run_simulation_b の既定＝ replayViewer/data/<run名>/（trace.jsonl＋概観＋
milestones を自動生成）。長い走行では頻出イベント（喃語など）は自動で間引かれる。

使い方: python run_trace.py [日数]   （既定7日）
再生  : replayViewer を開き、下部「ファイルを選択」で出力フォルダを選ぶ。
"""
import sys
import os
import random

days = int(sys.argv[1]) if len(sys.argv) > 1 else 7
seed = int(sys.argv[2]) if len(sys.argv) > 2 else None
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
torch.set_num_threads(2)
if seed is not None:                 # 特定シードで録画（例：検証で理解が強く出たシード）
    random.seed(seed)
    torch.manual_seed(seed)

run_name = f"trace_{days}d" + (f"_s{seed}" if seed is not None else "")
from environment.parent_sim_b import run_simulation_b

r = run_simulation_b(max_sim_seconds=days * 86400, verbose=True, run_name=run_name)

print(f"[完了] {days}日分  泣き{r['cry_count']} 食事{r['feed_count']} "
      f"喃語{r['babble_count']} 要求語{r['request_count']}")
print(f"replayViewer用ログ: {r['trace_dir']}")
print("→ replayViewer下部の「ファイルを選択」でこのフォルダを選ぶと再生できます。")
