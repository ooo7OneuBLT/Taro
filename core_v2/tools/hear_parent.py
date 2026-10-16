"""親の話しかけを耳だけで聞かせ、聞き分ける係を育てる道具（段階5・仕様書 §3・§5 の1）。

脳も環境も使わない。聞かせたあと、耳だけのテストをして、育った耳を保存する。
* A：聞かせていない文（別の種）で、語の切れ目の当たり（文の中の切れ目の 正しく切れた割合・切った所が正しい割合）
  と、物の名前をそのまま切り出せた割合。「ぼおる」「こっぷ」が知っている語になったか
* B：同じ語を別の文で聞いたときの荷札の似かた（0.9以上）、違う語どうし（0.5以下）
* C（耳だけ）：「でんしゃはどこ？」で荷札がゼロ

使い方：
  python -m core_v2.tools.hear_parent [文の数=1000] [--seed 1] [--out G/models/段階5_耳.pt] [--log-dir 段階5_耳だけ]
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import time

import torch

from .. import run_v2
from ..boxes.listen import Listen, kana_only
from .parent_speech import FRAMES, UNHEARD, WORDS, gold_words, make_corpus

TEST_SEED = 99


def cuts_of(chunks: list[str]) -> set[int]:
    pos, out = 0, set()
    for c in chunks[:-1]:
        pos += len(c)
        out.add(pos)
    return out


def ear_tests(ear: Listen) -> dict:
    known = ear.known()
    hit = n_gold = n_cut = 0
    name_ok = name_n = 0
    examples = {}
    for s in make_corpus(300, seed=TEST_SEED):
        gold = [kana_only(w) for w in gold_words(s)]
        got = ear.segment(s)
        g, c = cuts_of(gold), cuts_of(got)
        hit += len(g & c)
        n_gold += len(g)
        n_cut += len(c)
        names = [w for w in gold if w in WORDS]
        if names:
            name_n += 1
            name_ok += names[0] in got
    for s in ["ぼおるはどこ？", "こっぷだね", "おおきいぼうし", "でんしゃはどこ？", "とけいをみて"]:
        examples[s] = ear.segment(s)
    recall = hit / max(n_gold, 1)
    precision = hit / max(n_cut, 1)

    def tag(sentence):
        return ear.step({"earin": sentence}).get("semantic", torch.zeros(64))

    same, diff = [], []
    tags = {w: [tag(f.format(w=w)) for f in FRAMES] for w in WORDS}
    for w in WORDS:
        for a, b in itertools.combinations(tags[w], 2):
            same.append(float(a @ b))
    for w1, w2 in itertools.combinations(WORDS, 2):
        diff.append(float(tags[w1][0] @ tags[w2][0]))
    unknown = {s: float(torch.linalg.vector_norm(tag(s))) for s in
               ["でんしゃはどこ？", "とけいだね"] + [f.format(w=u) for u in UNHEARD for f in FRAMES[:3]]}
    res = {
        "A": {"boundary_recall": round(recall, 3), "boundary_precision": round(precision, 3),
              "name_cut_out": f"{name_ok}/{name_n}", "known_has_ball_cup": {"ぼおる", "こっぷ"} <= known,
              "known_words": sorted(known, key=lambda k: -ear.alone[k]), "examples": examples},
        "B": {"same_min": round(min(same), 3), "diff_max": round(max(diff), 3),
              "diff_mean": round(sum(diff) / len(diff), 3)},
        "C_ear": {"all_zero": all(v == 0.0 for v in unknown.values()), "norms": unknown},
    }
    res["A"]["pass"] = recall >= 0.8 and precision >= 0.8 and res["A"]["known_has_ball_cup"]
    res["B"]["pass"] = res["B"]["same_min"] >= 0.9 and res["B"]["diff_max"] <= 0.5
    return res


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    opts = {"--seed": "1", "--out": os.path.join("G", "models", "段階5_耳.pt"), "--log-dir": "段階5_耳だけ"}
    pos, i = [], 0
    while i < len(argv):
        if argv[i] in opts:
            opts[argv[i]] = argv[i + 1]
            i += 2
            continue
        pos.append(argv[i])
        i += 1
    n = int(pos[0]) if pos else 1000
    t0 = time.time()
    ear = Listen(learning=True)
    for s in make_corpus(n, seed=int(opts["--seed"])):
        ear.step({"earin": s})
    ear.learning = False
    res = ear_tests(ear)
    res["n_sentences"] = n
    res["end_mean"] = round(float(ear.end_mean), 4)
    res["seconds"] = round(time.time() - t0, 1)
    try:
        from ..save import code_version
        res["code"] = code_version()
    except Exception:  # noqa: BLE001
        pass
    out = os.path.join(run_v2.REPO, opts["--out"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ear.save_ear(out, {"n_sentences": n, "seed": int(opts["--seed"])})
    log_dir = os.path.join(run_v2.REPO, "G", "logs", opts["--log-dir"])
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, f"耳だけ_{n}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: res[k] for k in ("A", "B", "C_ear")}, ensure_ascii=False, indent=1))
    print(f"保存: {out}")
    return res


if __name__ == "__main__":
    main()
