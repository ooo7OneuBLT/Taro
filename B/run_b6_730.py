"""B6(聞く側の統計的分節=原-辞書)込みの2年シミュ。使い方: python run_b6_730.py <seed>"""
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
                     run_name=f"trace_730d_b6_s{seed}")
env = r["env"]
vocab = env.vocab
lines = [f"[完了 B6] seed{seed}  要求語{r['request_count']}  {r['trace_dir']}",
         "=== 原-辞書 上位15 ==="]
for chunk, cnt in env.lexicon.top(15):
    lines.append(f"  「{vocab.decode(list(chunk))}」(len{len(chunk)}) : {cnt}回")
lines.append(f"語彙サイズ: {len(env.lexicon.counts)}")
out_path = os.path.join(os.path.dirname(__file__), "logs", f"b6_lexicon_s{seed}.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"[辞書サマリ] {out_path}")
