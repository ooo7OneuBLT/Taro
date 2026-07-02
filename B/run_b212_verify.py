"""
B2-12 検証 — 混線崩し（空腹較正＋時間割授乳）の後、再定義した理解の指標が
「正・安定」になったかを複数シードで確かめる。

主指標（参考文献§9）：hunger を固定して“聞いた語だけ”変えたときの食べ物予期の差
  語の寄与 = 満腹時の「まんま予期 − あうあ予期」
これが正なら「まんま→食べ物」の先取り＝初期・連合的理解の証拠。複数シードで符号が
安定して正かを見る（n=1問題への対応）。

使い方: python run_b212_verify.py [月数] [シード数]   （既定 12ヶ月・3シード）
出力: logs/b212_verify_{月数}mo.txt
"""
import sys
import os
import time
import random

months = int(sys.argv[1]) if len(sys.argv) > 1 else 12
n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
torch.set_num_threads(2)
try:
    import psutil
    psutil.Process(os.getpid()).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except ImportError:
    pass

from environment.parent_sim_b import run_simulation_b

log_path = os.path.join(os.path.dirname(__file__), "logs", f"b212_verify_{months}mo.txt")
os.makedirs(os.path.dirname(log_path), exist_ok=True)

words = ["まんま", "ままん", "あうあ"]
hungers = [(0.1, "満腹"), (0.9, "空腹")]


with open(log_path, "w", encoding="utf-8") as logf:
    def log(msg):
        logf.write(msg + "\n")
        logf.flush()
        print(msg, flush=True)

    log(f"[B2-12検証] {months}ヶ月 × {n_seeds}シード  開始 {time.strftime('%H:%M:%S')}")
    log("主指標：満腹時の『まんま予期 − あうあ予期』（＝hunger固定での語の寄与）が"
        "正・安定なら理解の証拠\n")

    full_contribs = []   # 満腹時の まんま−あうあ を全シード分ためる
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        random.seed(seed)
        t0 = time.time()
        r = run_simulation_b(max_sim_seconds=months * 2592000, verbose=False,
                             run_name=f"B2-12_verify_s{seed}_{months}mo")
        env = r["env"]
        log(f"--- seed={seed}  学習{(time.time()-t0)/60:.1f}分  "
            f"食事{r['feed_count']}(時間割{r['meal_count']}/満腹寄り{r['meal_low_hunger']}) "
            f"喃語{r['babble_count']}")

        sat = {}
        for hv, hname in hungers:
            for w in words:
                sat[(hname, w)] = env.comprehension_probe(w, hv, n_samples=200).get("satiety")

        for hv, hname in hungers:
            sm, ss, sa = sat[(hname, "まんま")], sat[(hname, "ままん")], sat[(hname, "あうあ")]
            if sm is not None:
                contrib = sm - sa
                if hname == "満腹":
                    full_contribs.append(contrib)
                mark = "  ★語で食べ物を先取り" if contrib > 0.05 else "  （語の寄与ほぼ無し）"
                log(f"    [{hname}] まんま={sm:.4f} ままん={ss:.4f} あうあ={sa:.4f} "
                    f"→ 語の寄与={contrib:+.4f}{mark if hname=='満腹' else ''}")

    log("\n[まとめ] 満腹時『まんま−あうあ』（語の寄与）の全シード：")
    log("  " + "  ".join(f"s{i}={c:+.3f}" for i, c in enumerate(full_contribs)))
    if full_contribs:
        mean = sum(full_contribs) / len(full_contribs)
        n_pos = sum(1 for c in full_contribs if c > 0.05)
        log(f"  平均={mean:+.4f}  正(>0.05)のシード数={n_pos}/{len(full_contribs)}")
        verdict = ("安定して正＝理解の証拠（混線崩しが効いた）" if n_pos == len(full_contribs)
                   else "ばらつく＝まだ不安定" if n_pos > 0
                   else "ほぼ0＝まだ理解せず")
        log(f"  判定：{verdict}")
    log(f"\n[完了] {time.strftime('%H:%M:%S')}")
