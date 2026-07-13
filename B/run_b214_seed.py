"""
B2-14（授乳を需要ベース＋年齢graded に作り替え）検証 — 1シードだけ実行する単体版。
3本を並列に走らせて壁時計を短縮する。測定は run_b213_seed.py と同じ
（word_contrib=理解, phon_contrib=音韻, collapse=崩壊）。

使い方: python run_b214_seed.py <seed> [months]   （既定12ヶ月）
出力  : logs/b214_seed<seed>.json ＋ replayViewer/data/B2-14_s<seed>_<months>mo/（自動）
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
                     run_name=f"B2-14_s{seed}_{months}mo")
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
    "word_contrib": round(fm - fa, 4),
    "phon_contrib": round(fm - fmm, 4),
    "collapse": bool(min(fm, fmm, fa) > 0.9),
    "trace_dir": r.get("trace_dir"),
}
logs = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(logs, exist_ok=True)
with open(os.path.join(logs, f"b214_seed{seed}.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(json.dumps(out, ensure_ascii=False))
