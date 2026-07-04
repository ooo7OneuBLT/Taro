"""B6-4(語↔状態の連合=mand化)込みの2年シミュ。使い方: python run_b6_4_730.py <seed> [gain]
gain省略=2.0(ON)、gain=0で連合オフ(除去テスト用)。"""
import sys, os, random
import perf_setup  # BLASスレッド制限＋優先度DOWN
import torch
torch.set_num_threads(2)

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
gain = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
random.seed(seed)
torch.manual_seed(seed)

from environment.parent_sim_b import run_simulation_b
from environment.core_b import TaroEnvironmentB

# gainを差し込むため、環境生成後に上書きする必要がある。run_simulation_bは内部でenvを作るので
# クラス既定値を書き換えてから走らせる（このプロセス内だけ・除去テスト用）。
_orig_init = TaroEnvironmentB.__init__
def _patched(self, *a, **k):
    _orig_init(self, *a, **k)
    self._assoc_gain = gain
TaroEnvironmentB.__init__ = _patched

tag = "on" if gain > 0 else "off"
r = run_simulation_b(max_sim_seconds=730 * 86400, verbose=True,
                     run_name=f"trace_730d_b6_4_{tag}_s{seed}")
env = r["env"]; vocab = env.vocab
lines = [f"[完了 B6-4 gain={gain}] seed{seed} 要求語{r['request_count']} {r['trace_dir']}",
         "=== 各語の連合[空腹,眠気,不快]・頻度 ==="]
for chunk, cnt in env.lexicon.top(8):
    a = env.lexicon.assoc(chunk)
    lines.append(f"  {vocab.decode(list(chunk))}: 連合={tuple(round(x,2) for x in a) if a else None} 頻度{cnt}")
out = os.path.join(os.path.dirname(__file__), "logs", f"b6_4_{tag}_s{seed}_summary.txt")
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"[summary] {out}")
