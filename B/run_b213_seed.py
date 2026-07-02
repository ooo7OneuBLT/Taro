"""
B2-13（#4=親の返事を聞く経路 入り）検証 — 1シードだけ実行する単体版。
3本を同時に（並列）走らせて壁時計を短縮するために分割した。

測定（#4の効果・#1安定性・#2音韻・崩壊 を1回で）：
  word_contrib = 満腹時の まんま−あうあ  … 語の寄与（＝食べ物の先取り＝理解）
  phon_contrib = 満腹時の まんま−ままん  … 音韻の特異性（まんまとままんを区別できるか）
  collapse     = 満腹時に3語とも>0.9      … 「常に食べ物予期」への崩壊フラグ

使い方: python run_b213_seed.py <seed> [months]   （既定12ヶ月）
出力  : logs/b213_seed<seed>.json（集計用）と標準出力
"""
import sys
import os
import json
import time
import random

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
months = int(sys.argv[2]) if len(sys.argv) > 2 else 12

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import torch
torch.set_num_threads(2)
random.seed(seed)
torch.manual_seed(seed)
try:
    import psutil
    psutil.Process(os.getpid()).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    pass

from environment.parent_sim_b import run_simulation_b

t0 = time.time()
r = run_simulation_b(max_sim_seconds=months * 2592000, verbose=False,
                     run_name=f"B2-13_s{seed}_{months}mo")
env = r["env"]

words = ["まんま", "ままん", "あうあ"]
res = {}
for hv, hn in [(0.1, "満腹"), (0.9, "空腹")]:
    for w in words:
        res[(hn, w)] = env.comprehension_probe(w, hv, n_samples=200).get("satiety")

fm, fmm, fa = res[("満腹", "まんま")], res[("満腹", "ままん")], res[("満腹", "あうあ")]
out = {
    "seed": seed, "months": months, "minutes": round((time.time() - t0) / 60, 1),
    "feed": r["feed_count"], "meal": r["meal_count"], "meal_low": r["meal_low_hunger"],
    "満腹": {w: (round(res[("満腹", w)], 4) if res[("満腹", w)] is not None else None) for w in words},
    "空腹": {w: (round(res[("空腹", w)], 4) if res[("空腹", w)] is not None else None) for w in words},
    "word_contrib": round(fm - fa, 4),      # 理解（語→食べ物の先取り）
    "phon_contrib": round(fm - fmm, 4),     # 音韻（まんまとままんの区別）
    "collapse": bool(min(fm, fmm, fa) > 0.9),  # 3語とも高い＝常に食べ物
}
logs = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(logs, exist_ok=True)
with open(os.path.join(logs, f"b213_seed{seed}.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(json.dumps(out, ensure_ascii=False))
