"""盤面の世界の1段目「泣いて頼む」と2段目「相手を選ぶ」：20人の太郎を育てて確かめる（数秒）。

合格の線（回す前に決めた）：
1段目（大人はママだけ）
* A 覚える：ママが泣き声に応える世界で、おなかすきの時に「泣く」を選ぶ割合が、最後の10回で7割以上（20人の平均）
* B 教えていない：ママが手に応える世界では、同じ太郎が「手を伸ばす」を7割以上選ぶ
* C 強く頼む：無視されて10分以上たった合図の強さの上がり方が、覚えない太郎より大きい（いら立ちの効果）
* 対照：覚えない太郎（worth を止めた太郎）は5割のまま
2段目（ママとパパ。ママはよく応え、パパはあまり応えない）
* D 相手を選ぶ：2人とも部屋にいる時、合図をママに向ける割合が最後の10回で7割以上
* E 教えていない：応え方を入れ替えた世界では、パパに7割以上向ける
* F 決めつけない：パパしかいない時も、パパに頼む（おなかすき1回あたり合図1回以上）

使い方：
  python -m core_v2.tools.board_stage1 [--seeds 20] [--days 5] [--days2 15] [--out 出力フォルダ] [--plot]
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

from core_v2.board.mind import Mind
from core_v2.board.world import Board, MINUTES_PER_DAY, stage1_adults, stage2_adults


def grow(rule: str, seed: int, days: int, frozen: bool = False, keep_log: bool = False,
         stage: int = 1, swap: bool = False, mind: Mind | None = None) -> dict:
    """1人の太郎を days 日育てる。2段目は、1段目（ママだけ）で days 日育った同じ太郎の続きから始める
    （育ちの段：0から育て直さない）。swap の世界では1段目のママがよく応え、2段目からママがあまり応えなくなる。"""
    if stage == 2 and mind is None:
        mind = grow(rule, seed, days, frozen, stage=1)["mind"]
    adults = stage1_adults() if stage == 1 else stage2_adults(swap)
    board = Board(rule=rule, seed=seed + (500 if stage == 2 else 0), adults=adults)
    mind = mind or Mind(seed=seed, frozen=frozen)
    episodes = []          # おなかすき1回ごと：[(分, 行動, 強さ, 向けた相手, いた人), ...]
    cur = []
    log = []
    for t in range(days * MINUTES_PER_DAY):
        present = board.present()
        action, strength, target = mind.act(board.fullness, present)
        f0 = board.fullness
        obs = board.step(action, strength, target)
        inner = mind.feel(obs)
        if action != "nothing":
            cur.append((t, action, strength, target, tuple(present)))
        if keep_log:
            log.append({"t": t, "fullness": round(f0, 3), "action": action, "strength": round(strength, 3),
                        "target": target, "present": present, "busy": obs["busy"], "fed": obs["fed"],
                        "stress": round(inner["stress"], 3), "joy": round(inner["joy"], 3),
                        "v": {who: {s: round(mind.value(s, who), 3) for s in ("cry", "reach")}
                              for who in [a.name for a in adults]}})
        if obs["fed"]:
            if cur:
                episodes.append(cur)
            cur = []
    return {"episodes": episodes, "log": log, "mind": mind}


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def share(ep, rule):
    return sum(1 for e in ep if e[1] == rule) / len(ep)


def summarize(runs, rule, n_first=3, n_last=10):
    curve = defaultdict(list)
    first, last = [], []
    for r in runs:
        eps = r["episodes"]
        for i, ep in enumerate(eps[:40]):
            curve[i].append(share(ep, rule))
        first.append(mean([share(ep, rule) for ep in eps[:n_first]]))
        last.append(mean([share(ep, rule) for ep in eps[-n_last:]]))
    return {"curve": [mean(curve[i]) for i in sorted(curve)], "first": mean(first), "last": mean(last),
            "last_each": last}


def escalation(runs, n_skip=5):
    """育った後（6回目以降）、無視されて10分以上たった泣き声の強さ − そのおなかすきの最初の泣き声の強さ。"""
    rises = []
    for r in runs:
        for ep in r["episodes"][n_skip:]:
            cries = [(t, s) for t, a, s, *_ in ep if a == "cry"]
            if not cries:
                continue
            late = [s for t, s in cries if t - ep[0][0] >= 10]
            if late:
                rises.append(mean(late) - cries[0][1])
    return {"n": len(rises), "rise": mean(rises)}


def target_share(runs, who, both=("ママ", "パパ"), n_first=3, n_last=10):
    """2人とも部屋にいた時の合図のうち、who に向けた割合。"""
    curve = defaultdict(list)
    first, last = [], []
    for r in runs:
        eps = [[e for e in ep if set(both) <= set(e[4])] for ep in r["episodes"]]
        eps = [ep for ep in eps if ep]
        sh = [sum(1 for e in ep if e[3] == who) / len(ep) for ep in eps]
        for i, x in enumerate(sh[:40]):
            curve[i].append(x)
        first.append(mean(sh[:n_first]))
        last.append(mean(sh[-n_last:]))
    return {"curve": [mean(curve[i]) for i in sorted(curve)], "first": mean(first), "last": mean(last),
            "last_each": last}


def alone_asks(runs, who="パパ"):
    """who しかいない時のおなかすきで、who に向けた合図の回数（1回あたり）。"""
    counts = []
    for r in runs:
        for ep in r["episodes"]:
            only = [e for e in ep if e[4] == (who,)]
            if only:
                counts.append(sum(1 for e in only if e[3] == who))
    return {"n": len(counts), "per_episode": mean(counts)}


def run_all(seeds: int, days: int, days2: int = 15) -> dict:
    res = {}
    for name, rule, frozen in (("cry", "cry", False), ("reach", "reach", False), ("frozen", "cry", True)):
        runs = [grow(rule, s, days, frozen) for s in range(seeds)]
        res[name] = {"share": summarize(runs, rule), "esc": escalation(runs) if rule == "cry" else None}
    for name, swap, who in (("mama_good", False, "ママ"), ("papa_good", True, "パパ")):
        runs = [grow("cry", s, days2, stage=2, swap=swap) for s in range(seeds)]
        res[name] = {"target": target_share(runs, who), "alone_papa": alone_asks(runs, "パパ")}
    frozen2 = [grow("cry", s, days2, frozen=True, stage=2) for s in range(seeds)]
    res["frozen2"] = {"target": target_share(frozen2, "ママ")}
    res["sample_log"] = grow("cry", 0, days, keep_log=True)["log"]
    res["sample_log2"] = grow("cry", 0, days2, keep_log=True, stage=2)["log"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--days2", type=int, default=15, help="2段目の日数（1段目で --days 日育った後の続き）")
    ap.add_argument("--out", default="board_stage1_out")
    ap.add_argument("--plot", action="store_true", help="図も描く（matplotlib が要る）")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    res = run_all(args.seeds, args.days, args.days2)
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False)
    for name in ("cry", "reach", "frozen"):
        s = res[name]["share"]
        print(f"1段目 {name}: 最初3回 {s['first']:.2f} → 最後10回 {s['last']:.2f}  "
              f"(7割以上 {sum(x >= 0.7 for x in s['last_each'])}/{len(s['last_each'])}人)"
              + (f"  無視10分後の強まり {res[name]['esc']['rise']:+.3f}" if res[name]["esc"] else ""))
    for name in ("mama_good", "papa_good", "frozen2"):
        s = res[name]["target"]
        extra = f"  パパだけの時の合図 {res[name]['alone_papa']['per_episode']:.1f}回" if "alone_papa" in res[name] else ""
        print(f"2段目 {name}: 最初3回 {s['first']:.2f} → 最後10回 {s['last']:.2f}  "
              f"(7割以上 {sum(x >= 0.7 for x in s['last_each'])}/{len(s['last_each'])}人){extra}")
    if args.plot:
        plot(res, os.path.join(args.out, "stage1.png"))


def plot(res: dict, path: str, day: int = 14):
    """図：左上＝1段目の合図、右上＝2段目の相手、下＝育った太郎の半日（2段目）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "IPAGothic"
    fig = plt.figure(figsize=(13, 8.5))
    ax1 = fig.add_subplot(2, 2, 1)
    ax3 = fig.add_subplot(2, 2, 2)
    ax2 = fig.add_subplot(2, 1, 2)
    for name, label, color in (("cry", "泣き声に応える世界：泣く", "#2a6fdb"),
                               ("reach", "手に応える世界：手を伸ばす", "#e07b00"),
                               ("frozen", "覚えない太郎：泣く", "#888888")):
        c = res[name]["share"]["curve"]
        ax1.plot(range(1, len(c) + 1), c, label=label, color=color, lw=2)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("おなかすきの回")
    ax1.set_ylabel("通じる合図を選んだ割合（20人の平均）")
    ax1.set_title("1段目：通じる合図を自分で見つける")
    ax1.legend(fontsize=9, loc="lower right")
    for name, label, color in (("mama_good", "ママがよく応える世界：ママへ", "#d6336c"),
                               ("papa_good", "パパがよく応える世界：パパへ", "#2b8a3e"),
                               ("frozen2", "覚えない太郎：ママへ", "#888888")):
        c = res[name]["target"]["curve"]
        ax3.plot(range(1, len(c) + 1), c, label=label, color=color, lw=2)
    ax3.set_ylim(0, 1)
    ax3.set_xlabel("2人ともいたおなかすきの回")
    ax3.set_ylabel("よく応える人に向けた割合（20人の平均）")
    ax3.set_title("2段目：応えてくれる人に頼むようになる")
    ax3.legend(fontsize=9, loc="lower right")

    log = [r for r in res["sample_log2"] if day * MINUTES_PER_DAY + 6 * 60 <= r["t"] < day * MINUTES_PER_DAY + 14 * 60]
    hours = [(r["t"] % MINUTES_PER_DAY) / 60 for r in log]
    ax2.plot(hours, [r["fullness"] for r in log], color="#2a9d8f", lw=2, label="お腹（満ち具合）")
    ax2.plot(hours, [r["stress"] for r in log], color="#c92a2a", lw=1, alpha=0.7, label="いら立ち")
    for h, r in zip(hours, log):
        if "ママ" not in r["present"]:
            ax2.axvspan(h, h + 1 / 60, ymin=0, ymax=0.04, color="#d6336c", lw=0)
    for who, color, mk in (("ママ", "#d6336c", "o"), ("パパ", "#2b8a3e", "s")):
        pts = [(h, r["strength"]) for h, r in zip(hours, log) if r["action"] == "cry" and r["target"] == who]
        ax2.scatter([h for h, _ in pts], [s for _, s in pts], s=12, marker=mk, color=color, label=f"{who}に泣く（高さ＝強さ）")
    for h in [h for h, r in zip(hours, log) if r["fed"]]:
        ax2.axvline(h, color="#2a9d8f", lw=1, ls="--")
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("時刻（2段目の太郎・最後の日）　点線＝ミルク　下の赤い帯＝ママがいない時間")
    ax2.set_title("育った太郎の半日")
    ax2.legend(fontsize=9, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
