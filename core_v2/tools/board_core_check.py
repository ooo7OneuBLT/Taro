"""本体の脳（core_v2）で盤面の1段目・2段目を回し、下書き（core_v2/board/mind.py）と同じ線で確かめる。

使い方：
  python -m core_v2.tools.board_core_check [--seeds 20] [--days 5] [--days2 15] [--procs 4] [--out 出力フォルダ]
1人1日およそ7秒（1分に脳を8回進めるため）。人ごとに別の手で並べて回す。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from multiprocessing import Pool

from core_v2.board.core_run import grow_core
from core_v2.tools.board_stage1 import alone_asks, escalation, summarize, target_share


def _one(args):
    seed, days, days2 = args
    out = {}
    cry = grow_core("cry", seed, days)
    out["cry"] = cry["episodes"]
    out["reach"] = grow_core("reach", seed, days)["episodes"]
    frozen = grow_core("cry", seed, days, frozen=True)
    out["frozen"] = frozen["episodes"]
    out["mama_good"] = grow_core("cry", seed, days2, stage=2, brain=copy.deepcopy(cry["brain"]))["episodes"]
    out["papa_good"] = grow_core("cry", seed, days2, stage=2, swap=True, brain=copy.deepcopy(cry["brain"]))["episodes"]
    out["frozen2"] = grow_core("cry", seed, days2, stage=2, brain=frozen["brain"])["episodes"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--days2", type=int, default=15)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", default="board_core_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with Pool(args.procs) as pool:
        per_seed = pool.map(_one, [(s, args.days, args.days2) for s in range(args.seeds)])
    runs = {k: [{"episodes": r[k]} for r in per_seed] for k in per_seed[0]}
    res = {}
    for name, rule in (("cry", "cry"), ("reach", "reach"), ("frozen", "cry")):
        s = summarize(runs[name], rule)
        res[name] = {"share": s, "esc": escalation(runs[name]) if rule == "cry" else None}
        print(f"1段目 {name}: 最初3回 {s['first']:.2f} → 最後10回 {s['last']:.2f}  "
              f"(7割以上 {sum(x >= 0.7 for x in s['last_each'])}/{len(s['last_each'])}人)"
              + (f"  無視10分後の強まり {res[name]['esc']['rise']:+.3f}" if res[name]["esc"] else ""))
    for name, who in (("mama_good", "ママ"), ("papa_good", "パパ"), ("frozen2", "ママ")):
        s = target_share(runs[name], who)
        a = alone_asks(runs[name], "パパ")
        res[name] = {"target": s, "alone_papa": a}
        print(f"2段目 {name}: 最初3回 {s['first']:.2f} → 最後10回 {s['last']:.2f}  "
              f"(7割以上 {sum(x >= 0.7 for x in s['last_each'])}/{len(s['last_each'])}人)"
              f"  パパだけの時の合図 {a['per_episode']:.1f}回")
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
