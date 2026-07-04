"""B5-5(B5-2の実装ズレ修正=言葉らしさで気づかれやすさ)込みの2年シミュ。使い方: python run_b5_5_730.py <seed>"""
import sys, os, random
import perf_setup  # torch importより前に：BLASスレッド制限＋優先度DOWN
import torch
torch.set_num_threads(2)

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
random.seed(seed)
torch.manual_seed(seed)

from environment.parent_sim_b import run_simulation_b

r = run_simulation_b(max_sim_seconds=730 * 86400, verbose=True,
                     run_name=f"trace_730d_b5_5_s{seed}")
print(f"[完了 B5-5] seed{seed}  要求語{r['request_count']}  {r['trace_dir']}")
