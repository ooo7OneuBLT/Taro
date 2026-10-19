"""本体の脳（core_v2）で盤面の1段目・2段目・3段目を回し、合格の線で確かめる。

3段目（言葉で頼む、2026-10-19）の線：
* G 言葉で頼む：親が話し言葉に応える世界で、最後の10回のおなかすきの合図のうち「まんま」か「みるく」（親に通じる
  食べ物の語）が半分以上（20人の平均）
* H 教えていない：親が言葉に応えない世界では、泣くままでいる（最後の10回の合図の半分以上が泣く）
3段目は2段目（ママがよく応える世界）まで育った同じ脳の続きから、--days3 日。

4段目（あれなに？）の線（3段目まで育った脳の続き。目の前に物24種類 --days4 日、そのあと初めて見る物6種類 --days4b 日）：
* G4 わからない物の前で、問う語（なに・なあに・これ…）を言う割合が、知っている物の前の2倍以上（4段目の全部の日、合計）
* H4 問う語に名前で答える世界の太郎は、答えない世界の太郎より、わからない物の前で問う語を2倍以上言う（4段目の全部の日）
* I4 初めて見る物でも同じ（初めて見る物の --days4b 日、答える世界÷答えない世界が2倍以上）
5段目（断る・前のことを話す）の線（4段目まで育った脳の続き、--days5 日。対照＝言葉に応えず、話を特に喜ばない世界）：
* I5 断る語：最後の半分で、すすめた人に向けた断り（泣く・手で押す・いや・いらない）のうち、言葉が半分以上（20人の合計）
* J5 おなかがすいた時には断る語をほとんど言わない（最後の半分のおなかすきの合図のうち5%未満）
* K5 対照の世界では、言葉の断りが I5 の半分未満
* L5 話す：最後の半分で、相手がいなかった間の物を話す割合（話せた人×分あたり）が、相手も見ていた物を話す割合の2倍以上
* M5 対照の世界では L5 の比が、元の世界の比の半分以下

使い方：
  python -m core_v2.tools.board_core_check [--seeds 20] [--days 5] [--days2 15] [--days3 15] [--days4 5] [--days4b 3]
                                          [--days5 15] [--procs 4] [--out 出力フォルダ]
1人1日およそ7秒（1分に脳を8回進めるため）。人ごとに別の手で並べて回す。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from multiprocessing import Pool

import torch

from core_v2.board.core_run import NEW_OBJECTS, grow_core, new_brain
from core_v2.board.world import MINUTES_PER_DAY, QUESTION_WORDS, REFUSE_WORDS
from core_v2.tools.board_stage1 import alone_asks, escalation, summarize, target_share


def _one(args):
    seed, days, days2, days3, days4, days4b, days5, controls, out_dir, brains = args
    path = os.path.join(out_dir, f"seed_{seed}.json")
    if os.path.exists(path):                     # 前に終わった人は読むだけ（止めて回し直せる）
        with open(path) as f:
            return json.load(f)
    out = {}
    saved3 = os.path.join(brains, f"brain3_{seed}.pt") if brains else None
    if saved3 and os.path.exists(saved3):
        # 3段目まで育った太郎を読み、4・5段目だけ回す（育った太郎は段ごとに保存し、0から育て直さない）
        b3 = new_brain(seed)
        b3.load_state_dict(torch.load(saved3, weights_only=False))
        controls = False
    else:
        cry = grow_core("cry", seed, days)
        out["cry"] = cry["episodes"]
        if controls:
            out["reach"] = grow_core("reach", seed, days)["episodes"]
            frozen = grow_core("cry", seed, days, frozen=True)
            out["frozen"] = frozen["episodes"]
        mama = grow_core("cry", seed, days2, stage=2, brain=copy.deepcopy(cry["brain"]))
        out["mama_good"] = mama["episodes"]
        words = grow_core("cry", seed, days3, stage=3, brain=copy.deepcopy(mama["brain"]))
        out["words"] = words["episodes"]
        out["no_words"] = grow_core("cry", seed, days3, stage=3, brain=copy.deepcopy(mama["brain"]),
                                    understand_words=False)["episodes"]
        b3 = words["brain"]
        torch.save(b3.state_dict(), os.path.join(out_dir, f"brain3_{seed}.pt"))
    if days4:
        for arm, answer in (("ask", True), ("no_ask", False)):
            r4 = grow_core("cry", seed, days4, stage=4, brain=copy.deepcopy(b3), answer_questions=answer)
            out[arm] = r4["curious"]
            r4b = grow_core("cry", seed, days4b, stage=4, brain=r4["brain"], answer_questions=answer,
                            objects=NEW_OBJECTS)
            out[arm + "_new"] = r4b["curious"]
            if answer:
                b4 = r4b["brain"]
        for arm, kind in (("five", True), ("five_ctrl", False)):
            r5 = grow_core("cry", seed, days5, stage=5, brain=copy.deepcopy(b4), understand_words=kind, joy_news=kind)
            out[arm] = {k: r5[k] for k in ("offers", "tells", "can_tell", "episodes")}
    if controls:
        out["papa_good"] = grow_core("cry", seed, days2, stage=2, swap=True,
                                     brain=copy.deepcopy(cry["brain"]))["episodes"]
        out["frozen2"] = grow_core("cry", seed, days2, stage=2, brain=frozen["brain"])["episodes"]
    with open(path, "w") as f:                   # 1人ずつ書く
        json.dump(out, f, ensure_ascii=False)
    with open(path) as f:
        return json.load(f)                      # 読み直した形にそろえる（タプル→リスト）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--days2", type=int, default=15)
    ap.add_argument("--days3", type=int, default=15)
    ap.add_argument("--days4", type=int, default=5, help="4段目の日数（0なら4段目・5段目を回さない）")
    ap.add_argument("--days4b", type=int, default=3)
    ap.add_argument("--days5", type=int, default=15)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", default="board_core_out")
    ap.add_argument("--brains3", default=None,
                    help="3段目まで育った太郎（brain3_種.pt）のあるフォルダ。あれば1〜3段目を飛ばして4・5段目だけ回す")
    ap.add_argument("--no-controls", action="store_true",
                    help="1・2段目の対照（手・覚えない・パパがよく応える・覚えない2段目）を回さない")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    per_seed = []
    with Pool(args.procs) as pool:
        for r in pool.imap(_one, [(s, args.days, args.days2, args.days3, args.days4, args.days4b, args.days5,
                                   not args.no_controls, args.out, args.brains3) for s in range(args.seeds)]):
            per_seed.append(r)
            print(f"  {len(per_seed)}/{args.seeds}人 終わり", flush=True)   # 途中経過
    runs = {k: [{"episodes": r[k]} for r in per_seed] for k in per_seed[0]}
    with open(os.path.join(args.out, "per_seed.json"), "w") as f:
        json.dump(per_seed, f, ensure_ascii=False)
    res = {}
    for name, rule in (("cry", "cry"), ("reach", "reach"), ("frozen", "cry")):
        if name not in runs:
            continue
        s = summarize(runs[name], rule)
        res[name] = {"share": s, "esc": escalation(runs[name]) if rule == "cry" else None}
        print(f"1段目 {name}: 最初3回 {s['first']:.2f} → 最後10回 {s['last']:.2f}  "
              f"(7割以上 {sum(x >= 0.7 for x in s['last_each'])}/{len(s['last_each'])}人)"
              + (f"  無視10分後の強まり {res[name]['esc']['rise']:+.3f}" if res[name]["esc"] else ""))
    for name, who in (("mama_good", "ママ"), ("papa_good", "パパ"), ("frozen2", "ママ")):
        if name not in runs:
            continue
        s = target_share(runs[name], who)
        a = alone_asks(runs[name], "パパ")
        res[name] = {"target": s, "alone_papa": a}
        print(f"2段目 {name}: 最初3回 {s['first']:.2f} → 最後10回 {s['last']:.2f}  "
              f"(7割以上 {sum(x >= 0.7 for x in s['last_each'])}/{len(s['last_each'])}人)"
              f"  パパだけの時の合図 {a['per_episode']:.1f}回")
    for name in ("words", "no_words"):
        if name not in runs:
            continue
        s = {k: summarize(runs[name], k) for k in ("say:まんま", "say:みるく", "cry", "reach")}
        said = [sum(1 for e in ep if e[1].startswith("say:")) / len(ep) for r in runs[name] for ep in r["episodes"][-10:]]
        res[name] = {k: v for k, v in s.items()} | {"say_all": sum(said) / len(said)}
        print(f"3段目 {name}: まんま {s['say:まんま']['first']:.2f} → {s['say:まんま']['last']:.2f}"
              f"（半分以上 {sum(x >= 0.5 for x in s['say:まんま']['last_each'])}/{len(s['say:まんま']['last_each'])}人）"
              f"  みるく {s['say:みるく']['last']:.2f}  まんまか みるく {s['say:まんま']['last'] + s['say:みるく']['last']:.2f}"
              f"（半分以上 {sum(a + b >= 0.5 for a, b in zip(s['say:まんま']['last_each'], s['say:みるく']['last_each']))}人）  泣く {s['cry']['first']:.2f} → {s['cry']['last']:.2f}"
              f"  語ぜんぶ {res[name]['say_all']:.2f}")
    if args.days4:
        res |= stage4(per_seed, args.days4)
        res |= stage5(per_seed, args.days5)
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False)


def _asks(logs, t_from=0):
    """curious の記録 → {"unknown": (問う語の回数, 分), "known": (同)}。"""
    out = {"unknown": [0, 0], "known": [0, 0]}
    for log in logs:
        for t, _, known, act in log:
            if t < t_from:
                continue
            k = out["known" if known else "unknown"]
            k[1] += 1
            k[0] += act.startswith("say:") and act[4:] in QUESTION_WORDS
    return out


def _rate(x):
    return x[0] / x[1] if x[1] else float("nan")


def stage4(per_seed, days4):
    res = {}
    # 4段目の全部の日で数える（物24種類は3日目ごろにほぼ全部わかり、後半はわからない物の前の分がほとんど無い）
    for arm in ("ask", "no_ask"):
        res[arm] = _asks([r[arm] for r in per_seed])
        res[arm + "_new"] = _asks([r[arm + "_new"] for r in per_seed])
    a, n = res["ask"], res["no_ask"]
    g4 = _rate(a["unknown"]) / max(_rate(a["known"]), 1e-9)
    h4 = _rate(a["unknown"]) / max(_rate(n["unknown"]), 1e-9)
    i4 = _rate(res["ask_new"]["unknown"]) / max(_rate(res["no_ask_new"]["unknown"]), 1e-9)
    res["lines4"] = {"G4": g4, "H4": h4, "I4": i4}
    print(f"4段目 問う語（1分あたり）：わからない物の前 {_rate(a['unknown']):.4f}（{a['unknown'][1]}分）"
          f"・知っている物の前 {_rate(a['known']):.4f}（{a['known'][1]}分） → G4 {g4:.1f}倍")
    print(f"4段目 答えない世界：わからない物の前 {_rate(n['unknown']):.4f} → H4 {h4:.1f}倍")
    print(f"4段目 初めて見る物：答える世界 {_rate(res['ask_new']['unknown']):.4f}・答えない世界 "
          f"{_rate(res['no_ask_new']['unknown']):.4f} → I4 {i4:.1f}倍")
    return res


def stage5(per_seed, days5):
    res = {}
    half = days5 * MINUTES_PER_DAY // 2
    for arm in ("five", "five_ctrl"):
        words = cries = hungry_refuse = hungry_all = 0
        tell = {"tell_new": 0, "tell_old": 0}
        avail = {"tell_new": 0, "tell_old": 0}
        overfed = offers_n = 0
        for r in per_seed:
            d = r[arm]
            for t, act, to_offerer, _, over in d["offers"]:
                if t < half:
                    continue
                overfed += over
                offers_n += 1
                if to_offerer and act in ("cry", "reach"):
                    cries += 1
                elif to_offerer and act.startswith("say:") and act[4:] in REFUSE_WORDS:
                    words += 1
            for ep in d["episodes"]:
                for e in ep:
                    if e[0] >= half:
                        hungry_all += 1
                        hungry_refuse += e[1].startswith("say:") and e[1][4:] in REFUSE_WORDS
            for t, kind, *_ in d["tells"]:
                if t >= half and kind in tell:
                    tell[kind] += 1
            for t, a in d["can_tell"]:
                if t >= half:
                    for k in avail:
                        avail[k] += a[k]
        share = words / max(words + cries, 1)
        rate = {k: tell[k] / max(avail[k], 1) for k in tell}
        ratio = rate["tell_new"] / max(rate["tell_old"], 1e-9)
        res[arm] = {"refuse_words": words, "refuse_cry_reach": cries, "word_share": share,
                    "hungry_refuse_share": hungry_refuse / max(hungry_all, 1), "overfed_per_offer_min": overfed / max(offers_n, 1),
                    "tell": tell, "avail": avail, "tell_rate": rate, "news_ratio": ratio}
        print(f"5段目 {arm}: 断り 言葉 {words}・泣く/手 {cries}（言葉の割合 {share:.2f}）  おなかすきの合図のうち断る語 "
              f"{res[arm]['hungry_refuse_share']:.3f}  話す：いなかった間の物 {tell['tell_new']}回/{avail['tell_new']}人分"
              f"・見ていた物 {tell['tell_old']}回/{avail['tell_old']}人分 → 比 {ratio:.1f}")
    f, c = res["five"], res["five_ctrl"]
    res["lines5"] = {"I5": f["word_share"], "J5": f["hungry_refuse_share"], "K5": c["word_share"] / max(f["word_share"], 1e-9),
                     "L5": f["news_ratio"], "M5": c["news_ratio"] / max(f["news_ratio"], 1e-9)}
    print("5段目の線:", {k: round(v, 3) for k, v in res["lines5"].items()})
    return res


if __name__ == "__main__":
    main()
