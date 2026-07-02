"""
replayViewer用のトレースログを生成し、粒度別の概観まで作って
replayViewer/application/ に配置する（開くと自動で実データが再生される）。

短期間だけ走らせて、太郎の中で「いつ・どの部品が発火し・何を話し・数値がどうだったか」を
1行1イベントで書き出す。フル1年は数十万イベントになるので、まずは短期間で。

使い方: python run_trace.py [日数]   （既定7日）
出力  : logs/ に trace.jsonl・trace_overview_*.jsonl・milestones.json
        同じものを ../replayViewer/application/ にもコピー（viewerが自動読込）
"""
import sys
import os
import shutil

days = int(sys.argv[1]) if len(sys.argv) > 1 else 7
here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(here, "src"))

import torch
torch.set_num_threads(2)

logs_dir = os.path.join(here, "logs")
out_path = os.path.join(logs_dir, "trace.jsonl")
os.makedirs(logs_dir, exist_ok=True)

from environment.parent_sim_b import run_simulation_b

print(f"[1/3] {days}日分のシミュレーション（トレース記録あり）…")
r = run_simulation_b(
    max_sim_seconds=days * 86400,
    verbose=False,
    run_name=f"trace_{days}d",
    trace_path=out_path,
)
n = sum(1 for _ in open(out_path, encoding="utf-8"))
print(f"      → {out_path}（{n}イベント）")
print(f"      泣き{r['cry_count']} 食事{r['feed_count']} 喃語{r['babble_count']} 要求語{r['request_count']}")

print("[2/3] 粒度別の概観＋マイルストーンを集約…")
import build_trace_index
build_trace_index.build(out_path)

print("[3/3] replayViewer/application/ にコピー（開くと自動で読み込まれる）…")
app_dir = os.path.join(here, "..", "replayViewer", "application")
copied = []
if os.path.isdir(app_dir):
    for name in ["trace.jsonl", "trace_overview_hour.jsonl", "trace_overview_day.jsonl",
                 "trace_overview_month.jsonl", "trace_overview_year.jsonl", "milestones.json"]:
        src = os.path.join(logs_dir, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(app_dir, name)); copied.append(name)
    print(f"      コピー: {', '.join(copied)}")
else:
    print("      （replayViewer/application/ が見つからないのでコピーは省略。folderで選択可）")

print("[完了] replayViewer を開くと実データが再生されます。")
