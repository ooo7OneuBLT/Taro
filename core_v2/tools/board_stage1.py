"""新1段目「泣いて頼む」：盤面の世界で20人の太郎を育てて確かめる（数秒）。

確かめること（合格の線は回す前に決めた）：
* A 覚える：ママが泣き声に応える世界で、お腹がすいて何かする時に「泣く」を選ぶ割合が、
  生まれてすぐ（最初のおなかすき3回）の約5割から、最後の10回で7割以上になる（20人の平均）
* B 教えていない：ママが手を伸ばす方に応える世界では、同じ太郎が「手を伸ばす」を7割以上選ぶようになる
* C 強く頼む：ママが忙しくて応えない間、泣き声が強くなっていく
  （最初の合図と、無視されて10分以上たった合図の強さを比べる。育った後の方が上がり方が大きい）
* 対照：覚えない太郎（覚える所を止めた太郎）は5割のまま

使い方：
  python -m core_v2.tools.board_stage1 [--seeds 20] [--days 5] [--out 出力フォルダ]
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

from core_v2.board.mind import Mind
from core_v2.board.world import Board, MINUTES_PER_DAY


class FrozenMind(Mind):
    """覚えない太郎（対照）：値を生まれた時のまま止める。"""

    def feel(self, obs, fullness_before):
        tries, relief = dict(self.tries), dict(self.relief)
        out = super().feel(obs, fullness_before)
        self.tries, self.relief = tries, relief
        return out


def grow(rule: str, seed: int, days: int, frozen: bool = False, keep_log: bool = False) -> dict:
    board = Board(rule=rule, seed=seed)
    mind = (FrozenMind if frozen else Mind)(seed=seed)
    episodes = []          # おなかすき1回ごと：[(分, 行動, 強さ, ママが忙しいか), ...]
    cur = []
    log = []
    for t in range(days * MINUTES_PER_DAY):
        f0 = board.fullness
        action, strength = mind.act(f0)
        obs = board.step(action, strength)
        inner = mind.feel(obs, f0)
        if action != "nothing":
            cur.append((t, action, strength, obs["busy"]))
        if keep_log:
            log.append({"t": t, "fullness": round(f0, 3), "action": action, "strength": round(strength, 3),
                        "busy": obs["busy"], "fed": obs["fed"], "checked": obs["checked"],
                        "frustration": round(inner["frustration"], 3), "joy": round(inner["joy"], 3),
                        "v_cry": round(mind.value("cry"), 3), "v_reach": round(mind.value("reach"), 3)})
        if obs["fed"]:
            if cur:
                episodes.append(cur)
            cur = []
    return {"episodes": episodes, "log": log, "values": {a: mind.value(a) for a in ("cry", "reach")}}


def share(ep, rule):
    return sum(1 for _, a, _, _ in ep if a == rule) / len(ep)


def escalation(ep):
    """最初の合図の強さと、最初から10分以上たった合図の強さの平均。10分以上待った回だけ。"""
    t0 = ep[0][0]
    late = [s for t, a, s, _ in ep if a == "cry" and t - t0 >= 10]
    first = [s for t, a, s, _ in ep if a == "cry"][:1]
    if not late or not first:
        return None
    return first[0], sum(late) / len(late)


def summarize(runs, rule, n_first=3, n_last=10):
    curve = defaultdict(list)
    first, last = [], []
    for r in runs:
        eps = r["episodes"]
        for i, ep in enumerate(eps[:40]):
            curve[i].append(share(ep, rule))
        first.append(sum(share(ep, rule) for ep in eps[:n_first]) / n_first)
        last.append(sum(share(ep, rule) for ep in eps[-n_last:]) / n_last)
    mean = lambda xs: sum(xs) / len(xs)
    return {"curve": [mean(curve[i]) for i in sorted(curve)], "first": mean(first), "last": mean(last),
            "last_each": last}


def esc_summary(runs, n_early=5):
    early, late = [], []
    for r in runs:
        eps = r["episodes"]
        for i, ep in enumerate(eps):
            e = escalation(ep)
            if e:
                (early if i < n_early else late).append(e)
    def avg(xs):
        if not xs:
            return None
        return {"n": len(xs), "first": sum(a for a, _ in xs) / len(xs), "after10": sum(b for _, b in xs) / len(xs)}
    return {"early": avg(early), "late": avg(late)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--out", default="board_stage1_out")
    ap.add_argument("--plot", action="store_true", help="図も描く（matplotlib が要る）")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    res = {}
    for name, rule, frozen in (("cry", "cry", False), ("reach", "reach", False), ("frozen", "cry", True)):
        runs = [grow(rule, s, args.days, frozen) for s in range(args.seeds)]
        res[name] = {"share": summarize(runs, rule), "esc": esc_summary(runs) if rule == "cry" else None}
    sample = grow("cry", 0, args.days, keep_log=True)
    res["sample_log"] = sample["log"]
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False)
    for name in ("cry", "reach", "frozen"):
        s = res[name]["share"]
        print(f"{name}: 最初3回 {s['first']:.2f} → 最後10回 {s['last']:.2f}  "
              f"(7割以上 {sum(x >= 0.7 for x in s['last_each'])}/{len(s['last_each'])}人)")
        if res[name]["esc"]:
            print("  強さ:", json.dumps(res[name]["esc"], ensure_ascii=False))
    if args.plot:
        plot(res, os.path.join(args.out, "stage1.png"))


def plot(res: dict, path: str, day: int = 3):
    """図：左＝おなかすきの回ごとの「当たりの合図」を選んだ割合、右＝育った太郎の1日の一部。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "IPAGothic"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 1.5]})
    for name, label, color in (("cry", "ママが泣き声に応える世界：泣く", "#2a6fdb"),
                               ("reach", "ママが手に応える世界：手を伸ばす", "#e07b00"),
                               ("frozen", "覚えない太郎：泣く", "#888888")):
        c = res[name]["share"]["curve"]
        ax1.plot(range(1, len(c) + 1), c, label=label, color=color, lw=2)
    ax1.axhline(0.5, color="#bbb", lw=1, ls=":")
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("おなかすきの回（生まれてから）")
    ax1.set_ylabel("当たりの合図を選んだ割合（20人の平均）")
    ax1.set_title("教えていないのに、ママに通じる合図を選ぶようになる")
    ax1.legend(fontsize=9, loc="lower right")

    log = [r for r in res["sample_log"] if day * MINUTES_PER_DAY + 6 * 60 <= r["t"] < day * MINUTES_PER_DAY + 14 * 60]
    hours = [(r["t"] % MINUTES_PER_DAY) / 60 for r in log]
    ax2.plot(hours, [r["fullness"] for r in log], color="#2a9d8f", lw=2, label="お腹（満ち具合）")
    for h, r in zip(hours, log):
        if r["busy"]:
            ax2.axvspan(h, h + 1 / 60, color="#f4d6d6", lw=0)
    cry = [(h, r["strength"]) for h, r in zip(hours, log) if r["action"] == "cry"]
    reach = [(h, r["strength"]) for h, r in zip(hours, log) if r["action"] == "reach"]
    ax2.scatter([h for h, _ in cry], [s for _, s in cry], s=10, color="#2a6fdb", label="泣く（高さ＝強さ）")
    ax2.scatter([h for h, _ in reach], [s for _, s in reach], s=10, marker="^", color="#e07b00", label="手を伸ばす")
    fed = [h for h, r in zip(hours, log) if r["fed"]]
    for h in fed:
        ax2.axvline(h, color="#2a9d8f", lw=1, ls="--")
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("時刻（4日目）　赤い帯＝ママが忙しい　点線＝ミルク")
    ax2.set_title("育った太郎の半日")
    ax2.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
