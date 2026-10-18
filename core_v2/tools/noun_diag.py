"""段階9：試験場（noun_bed）の A と C がどこで落ちるかを分けて測る道具（直しではない。測るだけ）。

太郎を1人ずつ noun_bed と同じに育て（既定は種類の層を止めた kind_off）、意味の辞書の表 W を語ごとの分に分ける。
W は「見た目 × 語の荷札」の足し算なので、20語の荷札 Q（64×20）で W = P Qᵀ と書ける（P の列 P_w＝語 w を聞いたときに
見ていた荷札の合計）。P = W Q (QᵀQ)⁻¹ で正確に取り出せる。語 w を聞いて思い出す荷札は
W q_w = P_w ＋ Σ_{w'≠w} (q_w'·q_w) P_w'（後ろが「ほかの語の混ざり」。語の荷札どうしが直交していないために入る）。

A（語→物）を、思い出す荷札だけ取り替えて同じ決まり（gaze の線 MATCH_COS 以上でいちばん似た物）で試す：
* 理想：その語の覚えた個体の見えを全部平均した荷札（目の力の上限）
* 今の W／混ざり無し（P_w だけ）／語ごとに回数で割る（混ざりは残す）／今の W で線なし（いちばん似た物）
C（音→語）を、表だけ取り替えて試す：今の W／音が窓の全部にあった場合（同じ W）／音→語の別の表（回数で足す・語ごとに平均）

使い方：python -m core_v2.tools.noun_diag [--taros 0-2] [--cond kind_off] [--scenes 300] [--bank core_v2/data/段階9_見た目の素.pt]
出力：G/logs/段階9_名詞_診断/まとめ.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter

import numpy as np
import torch

from ..boxes.gaze import MATCH_COS
from ..boxes.soundin import sound_tag
from ..formats import make_tag
from ..tagging import noisy, word_tag
from ..world_eye import instances as ins
from ..world_eye import sounds as snd
from . import noun_bed as nb

WORDS = ins.WORDS
LOOK_TICKS = int(nb.LOOK_S / nb.TICK_S)
WINDOW_TICKS = 30          # 語が届く前に見ていた物を数える 3秒（semantic.LOOK_WINDOW_S）
AFTER_TICKS = 10           # 名残りのうち、ほぼ効く 1秒


def _mk(x) -> torch.Tensor:
    return make_tag(x).double()


def grow_logged(seed: int, bank: nb.Bank, cond: str, n_scenes: int):
    """noun_bed と同じに育て、回ごとに (聞いた語, 音が来たか, 真ん中の物の語番号) を残す。"""
    bed = nb.NounBed(seed, bank, kind_learning="kind_off" not in cond, one_inst=cond.startswith("one_inst"))
    log = []
    orig = bed._tick

    def tick(objs, ego=nb._EGO0, fire=None, word=None, sound=None):
        mid = [o["i"] for o in objs if abs(o["c"] - nb.CENTRE) < 1.0]
        log.append((word, sound is not None, mid[0] if mid else None))
        return orig(objs, ego=ego, fire=fire, word=word, sound=sound)

    bed._tick = tick
    bed.grow(n_scenes)
    bed.boxes["semantic"].learning = False
    bed.boxes["kind"].learning = False
    return bed, log


def naming_stats(log) -> dict:
    """語ごとに：名づけの回数、窓（前3秒＋後1秒）のうち名づけた物を見ていた割合、窓に物の音があった回数。"""
    n, share, snd_n = Counter(), {w: [] for w in WORDS}, Counter()
    for t, (word, _, _) in enumerate(log):
        if word is None:
            continue
        i = WORDS.index(word)
        win = log[max(0, t - WINDOW_TICKS): t + AFTER_TICKS]
        seen = [mid for _, _, mid in win if mid is not None]
        n[word] += 1
        share[word].append(sum(m == i for m in seen) / max(1, len(seen)))
        snd_n[word] += any(s for _, s, _ in win)
    return {"n": n, "share": {w: float(np.mean(v)) if v else 0.0 for w, v in share.items()}, "sound": snd_n}


def a_trial(bank: nb.Bank, want: dict, gate: float, rng) -> dict:
    """noun_bed.look_trial と同じ決まりで、思い出す荷札 want[i] だけ渡して A を試す。"""
    res, tc = {}, []
    for i, w in enumerate(WORDS):
        hit = 0
        for _ in range(nb.N_TRIALS):
            others = rng.sample([j for j in range(len(WORDS)) if j != i], 3)
            cands = [(i, rng.choice(ins.TRAIN_INST))] + [(j, rng.choice(ins.TRAIN_INST)) for j in others]
            cnt = [0] * 4
            for _ in range(LOOK_TICKS):
                cs = [float(noisy(bank.tag(a, k, rng.choice(bank.views(a, k)))).double() @ want[i]) for a, k in cands]
                tc.append(cs[0])
                ok = [j for j, c in enumerate(cs) if c >= gate]
                cnt[max(ok, key=lambda j: cs[j]) if ok else rng.randrange(4)] += 1
            best = max(cnt)
            hit += rng.choice([j for j, c in enumerate(cnt) if c == best]) == 0
        res[w] = hit / nb.N_TRIALS
    return {"mean": round(float(np.mean(list(res.values()))), 3), "words_70": sum(v >= 0.7 for v in res.values()),
            "target_cos": round(float(np.mean(tc)), 3), "over_gate": round(float(np.mean(np.array(tc) >= gate)), 3),
            "by_word": res}


def c_trial(score_fn, tests: dict) -> dict:
    d = {w: sum(WORDS[int(torch.argmax(score_fn(x)))] == w for x in xs) / len(xs) for w, xs in tests.items()}
    return {"C6": round(float(np.mean([d[w] for w in ins.C_WORDS])), 3), "by_word": d}


def diagnose(bed: nb.NounBed, log, seed: int) -> dict:
    bank = bed.bank
    nW = len(WORDS)
    Q = torch.stack([word_tag(w) for w in WORDS]).double().T
    G = Q.T @ Q
    W = bed.boxes["semantic"].W.double()
    P = W @ Q @ torch.linalg.inv(G)
    st = naming_stats(log)
    nvec = torch.tensor([max(1, st["n"][w]) for w in WORDS], dtype=torch.float64)
    proto = {i: _mk(torch.stack([bank.tag(i, k, v) for k in ins.TRAIN_INST for v in bank.views(i, k)]).mean(0))
             for i in range(nW)}
    allv = torch.stack([bank.tag(i, k, v) for i in range(nW) for k in range(ins.N_INST) for v in bank.views(i, k)]).double()
    m = allv.mean(0)
    off = ~torch.eye(nW, dtype=torch.bool)
    out = {"seed": seed,
           "表の共通の向き": {"平均の長さ": round(float(m.norm()), 3),
                         "見えの長さ²のうち共通の向きの割合": round(float(((allv @ (m / m.norm())) ** 2).mean()), 3)},
           "語の荷札どうしの cos": {"絶対値の平均": round(float(G[off].abs().mean()), 3), "最大": round(float(G[off].abs().max()), 3)},
           "名づけ": {w: {"回数": st["n"][w], "名づけた物を見ていた割合": round(st["share"][w], 2),
                       "混ざり/自分": round(float((W @ Q[:, i] - P[:, i]).norm() / P[:, i].norm()), 2)}
                   for i, w in enumerate(WORDS)}}
    torch.manual_seed(1000 + seed)
    variants = {
        "理想（覚えた個体の平均）": ({i: proto[i] for i in range(nW)}, MATCH_COS),
        "今の W": ({i: _mk(W @ Q[:, i]) for i in range(nW)}, MATCH_COS),
        "混ざり無し（P_w だけ）": ({i: _mk(P[:, i]) for i in range(nW)}, MATCH_COS),
        "語ごとに回数で割る（混ざりは残す）": ({i: _mk((P / nvec) @ G[:, i]) for i in range(nW)}, MATCH_COS),
        "今の W・線なし（いちばん似た物）": ({i: _mk(W @ Q[:, i]) for i in range(nW)}, -2.0),
    }
    out["A"] = {}
    for name, (want, gate) in variants.items():
        want = {i: noisy(v.float()).double() for i, v in want.items()}      # 思い出しの雑音（semantic と同じ）
        out["A"][name] = a_trial(bank, want, gate, random.Random(5 + seed))
    sw = list(ins.SOUND_WORDS)
    nb_ = {w: _mk(torch.stack([sound_tag(snd.synth(ins.KIND_OF[w], k, seed=s)) for k in range(ins.N_INST) for s in range(3)])
                  .mean(0)) for w in sw}
    tests = {w: [sound_tag(snd.synth(ins.KIND_OF[w], random.Random(t).randrange(ins.N_INST), seed=10 ** 7 + t)).double()
                 for t in range(nb.N_TRIALS)] for w in sw}
    full, sep, avg = P.clone(), torch.zeros_like(P), torch.zeros_like(P)
    for w in sw:
        j = WORDS.index(w)
        full[:, j] += st["sound"][w] * nb_[w]
        sep[:, j] = st["sound"][w] * nb_[w]
        avg[:, j] = nb_[w]
    M = torch.stack([nb_[w] for w in sw])
    out["音の荷札どうしの cos"] = {"語": sw, "cos": np.round((M @ M.T).numpy(), 2).tolist()}
    out["P_w のうち自分の音の向きの割合"] = {w: round(float(P[:, WORDS.index(w)] @ nb_[w] / P[:, WORDS.index(w)].norm()), 3)
                                  for w in sw}
    out["C"] = {"今の W": c_trial(lambda x: Q.T @ (W.T @ x), tests),
                "音が窓の全部（同じ W）": c_trial(lambda x: Q.T @ ((full @ Q.T).T @ x), tests),
                "音→語の別の表（回数で足す）": c_trial(lambda x: sep.T @ x, tests),
                "音→語の別の表（語ごとに平均）": c_trial(lambda x: avg.T @ x, tests)}
    return out


def main(argv=None):
    torch.set_num_threads(1)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--taros", default="0-2")
    ap.add_argument("--cond", default="kind_off")
    ap.add_argument("--scenes", type=int, default=300)
    ap.add_argument("--bank", default="core_v2/data/段階9_見た目の素.pt")
    ap.add_argument("--log-dir", default="段階9_名詞_診断")
    a = ap.parse_args(argv)
    bank = nb.Bank(nb.load_bank(a.bank))
    log_dir = os.path.join(nb.REPO, "G", "logs", a.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    res = []
    for seed in nb.parse_seeds(a.taros):
        bed, log = grow_logged(seed, bank, a.cond, a.scenes)
        r = diagnose(bed, log, seed)
        res.append(r)
        print(f"[太郎{seed}] A " + "  ".join(f"{k}: {v['mean']}（{v['words_70']}語、線越え {v['over_gate']}）"
                                           for k, v in r["A"].items()), flush=True)
        print(f"[太郎{seed}] C " + "  ".join(f"{k}: {v['C6']}" for k, v in r["C"].items()), flush=True)
    summ = {"cond": a.cond, "taros": [r["seed"] for r in res],
            "A": {k: round(float(np.mean([r["A"][k]["mean"] for r in res])), 3) for k in res[0]["A"]},
            "A_words_70": {k: [r["A"][k]["words_70"] for r in res] for k in res[0]["A"]},
            "C6": {k: round(float(np.mean([r["C"][k]["C6"] for r in res])), 3) for k in res[0]["C"]},
            "太郎ごと": res}
    with open(os.path.join(log_dir, "まとめ.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps({k: summ[k] for k in ("A", "A_words_70", "C6")}, ensure_ascii=False))
    return summ


if __name__ == "__main__":
    main()
