"""本体の脳（core_v2）で盤面の1段目・2段目・3段目を回し、合格の線で確かめる。

3段目（言葉で頼む、2026-10-19）の線：
* G 言葉で頼む：親が話し言葉に応える世界で、最後の10回のおなかすきの合図のうち「まんま」が半分以上（20人の平均）
* H 教えていない：親が言葉に応えない世界では、泣くままでいる（最後の10回の合図の半分以上が泣く）
3段目は2段目（ママがよく応える世界）まで育った同じ脳の続きから、--days3 日。

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
    seed, days, days2, days3 = args
    out = {}
    cry = grow_core("cry", seed, days)
    out["cry"] = cry["episodes"]
    out["reach"] = grow_core("reach", seed, days)["episodes"]
    frozen = grow_core("cry", seed, days, frozen=True)
    out["frozen"] = frozen["episodes"]
    mama = grow_core("cry", seed, days2, stage=2, brain=copy.deepcopy(cry["brain"]))
    out["mama_good"] = mama["episodes"]
    out["words"] = grow_core("cry", seed, days3, stage=3, brain=copy.deepcopy(mama["brain"]))["episodes"]
    out["no_words"] = grow_core("cry", seed, days3, stage=3, brain=copy.deepcopy(mama["brain"]),
                                understand_words=False)["episodes"]
    out["papa_good"] = grow_core("cry", seed, days2, stage=2, swap=True, brain=copy.deepcopy(cry["brain"]))["episodes"]
    out["frozen2"] = grow_core("cry", seed, days2, stage=2, brain=frozen["brain"])["episodes"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--days2", type=int, default=15)
    ap.add_argument("--days3", type=int, default=15)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", default="board_core_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with Pool(args.procs) as pool:
        per_seed = pool.map(_one, [(s, args.days, args.days2, args.days3) for s in range(args.seeds)])
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
    for name in ("words", "no_words"):
        s = {k: summarize(runs[name], k) for k in ("say:まんま", "say:みるく", "cry", "reach")}
        said = [sum(1 for e in ep if e[1].startswith("say:")) / len(ep) for r in runs[name] for ep in r["episodes"][-10:]]
        res[name] = {k: v for k, v in s.items()} | {"say_all": sum(said) / len(said)}
        print(f"3段目 {name}: まんま {s['say:まんま']['first']:.2f} → {s['say:まんま']['last']:.2f}"
              f"（半分以上 {sum(x >= 0.5 for x in s['say:まんま']['last_each'])}/{len(s['say:まんま']['last_each'])}人）"
              f"  みるく {s['say:みるく']['last']:.2f}  泣く {s['cry']['first']:.2f} → {s['cry']['last']:.2f}"
              f"  語ぜんぶ {res[name]['say_all']:.2f}")
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
