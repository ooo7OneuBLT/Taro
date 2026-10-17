"""意味の辞書の試験場（2026-10-17新設）：本番の「体験の録画」を、意味の辞書の覚え方だけに流し込んで比べる。

本番（core_v2/tools/smith_yu.py --ticks）の0.1秒ごとの記録 <mode>_seed<種>_回ごと.csv から、
* 毎回、太郎がどの物を見ていたか（look 列：視線から10度以内で一番近い物の番号。空＝どれも見ていない）
* 学ぶ画面で、意味の辞書に語が届いた回と、その語（semantic.event が learn / learn_skip の行の listen.word）
を読み、覚え方を取りかえて W（語×見た目の表）を計算し直す。体・絵・目の NN は回さない（1回数秒）。

近似（本番と違う所）：
* 物の見た目の荷札は、物ごとの決まった乱数（長さ1、物どうしはほぼ直角）で代える。本番は名簿の行の見た目の
  平均で、似た物どうしは少し似る。なので recall の強さ（‖W q‖）は本番と同じにはならない
* 「見ていた」は名簿の「今見ている行」ではなく、視線から10度以内の物（look 列）
* 覚え方を変えても、目の動きは録画のまま（本当は思い出した物を見に行くので変わる）→ 最後は本物で確かめる
覚え方の変種：
* before_s：語が届く前の何秒を振り返るか（今の太郎は 3.0、objfile.LOOK_WINDOW_S）
* echo_s：語が届いたあと、何秒まで耳に残して、その間に見た物も数えるか（今の太郎は 0）
強さ＝その窓で物を見ていた時間のうち、その物を見ていた割合（今の覚え方B と同じ。見ていない回は数えない）。
窓の中で何も見ていなければ、その語は捨てる（learn_skip）。

使い方:
  python G/scripts/semantic_bench.py G/logs/段階6_1回確かめ2/B_seed0_回ごと.csv [--json G/logs/段階6_1回確かめ2/B_seed0.json]
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "G", "scripts"))

TICK_S = 0.1
RECALL_MIN = 2.5          # core_v2/boxes/semantic.py と同じ


def load(csv_path):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    looks = [int(r["look"]) if r.get("look", "") != "" else None for r in rows]
    # 名簿が「今見ている行」を持っていた回（objfile.rows に * がある）。見ている行が物でない（look 空）なら -1＝物以外
    held = [(int(r["look"]) if r.get("look", "") != "" else -1) if "*" in r.get("objfile.rows", "") else None
            for r in rows]
    # 語は1回前の行の listen.word にある（耳の出力は次の回に意味の辞書へ届く。二重バッファ）
    events = []
    for i, r in enumerate(rows):
        if r.get("semantic.event") in ("learn", "learn_skip") and r.get("phase") == "train":
            w = next((rows[j]["listen.word"] for j in range(i, max(-1, i - 4), -1) if rows[j].get("listen.word")), "")
            if w:
                events.append((i, w))
    actual = {"learn": sum(1 for r in rows if r.get("semantic.event") == "learn" and r.get("phase") == "train"),
              "skip": sum(1 for r in rows if r.get("semantic.event") == "learn_skip" and r.get("phase") == "train")}
    return looks, held, events, actual


def run(looks, events, word_of_obj, before_s=3.0, echo_s=0.0, n_obj=6, tag_seed=0, tau_s=None):
    """tau_s を与えると、語が届いたあとは打ち切りではなく「語の残り」exp(−経過/tau) を掛けて数える
    （届く前は今の覚え方と同じ重み1。届いたあとは 5×tau まで見る）。"""
    from core_v2.tagging import word_tag
    rng = np.random.default_rng(tag_seed)
    A = rng.standard_normal((n_obj + 1, 64))          # 最後の1つ＝物以外（床のかけらなど）
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    W = np.zeros((64, 64))
    nb = int(round(before_s / TICK_S))
    ne = int(round((5 * tau_s if tau_s else echo_s) / TICK_S))
    n_learn = n_skip = 0
    for i, w in events:
        seg = looks[max(0, i - nb + 1): i + ne + 1]
        t_rel = [(j - min(i, nb - 1)) * TICK_S for j in range(len(seg))]     # 語が届いた回＝0
        wts = {}
        for k, tr in zip(seg, t_rel):
            if k is None:
                continue
            wt = 1.0 if (tau_s is None or tr <= 0) else float(np.exp(-tr / tau_s))
            wts[k] = wts.get(k, 0.0) + wt
        tot = sum(wts.values())
        if tot <= 0:
            n_skip += 1
            continue
        q = word_tag(w).numpy().astype(float)
        for k, wt in wts.items():
            W += (wt / tot) * np.outer(A[k], q)      # k=-1 は A[-1]＝物以外
        n_learn += 1
    res = {"before_s": before_s, "echo_s": echo_s, "tau_s": tau_s, "learn": n_learn, "skip": n_skip, "words": {}}
    correct = recalled_correct = 0
    for k, w in word_of_obj.items():
        r = W @ word_tag(w).numpy().astype(float)
        norm = float(np.linalg.norm(r))
        sims = A[:n_obj] @ (r / (norm + 1e-12))
        best = int(np.argmax(sims))
        margin = float(sims[k] - max(s for j, s in enumerate(sims) if j != k))
        ok = best == k
        correct += ok
        recalled_correct += ok and norm >= RECALL_MIN
        res["words"][w] = {"norm": round(norm, 2), "best_is_correct": ok, "margin": round(margin, 3)}
    res["correct"] = correct
    res["recalled_and_correct"] = recalled_correct
    return res


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    csv_path = argv[0]
    jpath = argv[argv.index("--json") + 1] if "--json" in argv else csv_path.replace("_回ごと.csv", ".json")
    info = json.load(open(jpath, encoding="utf-8"))
    from smith_yu_schedule import OBJECTS
    names = [o["name"] for o in OBJECTS]
    word_of_obj = {names.index(n): w for w, n in info["word_of"].items()}
    looks, held, events, actual = load(csv_path)
    print(f"録画：学ぶ画面で語が届いた回 {len(events)}（本番の実際：覚えた {actual['learn']}・捨てた {actual['skip']}）")
    print(f"本番の recall の強さ：{info.get('recall_norm')}")
    out = []
    for src_name, src in (("視線10度以内（look）", looks), ("名簿の「今見ている行」（本番と同じ）", held)):
      print(f"--- 「見ていた」の決め方：{src_name}")
      for before in (3.0,):
        for echo, tau in ((0.0, None), (0.5, None), (1.0, None), (1.5, None), (2.0, None), (3.0, None),
                          (0.0, 0.5), (0.0, 1.0), (0.0, 2.0)):
            r = run(src, events, word_of_obj, before, echo, tau_s=tau)
            r["source"] = src_name
            out.append(r)
            ws = " ".join(f"{w}:{v['norm']}{'○' if v['best_is_correct'] else '×'}" for w, v in r["words"].items())
            lab = f"後は薄れる τ={tau}秒" if tau else f"後{echo}秒"
            print(f"前{before}秒・{lab}：覚えた {r['learn']}・捨てた {r['skip']}・正しい物が一番 {r['correct']}/6"
                  f"（線を越えて正しい {r['recalled_and_correct']}/6）  {ws}")
    with open(os.path.join(os.path.dirname(csv_path), "試験場_" + os.path.basename(csv_path).replace(".csv", ".json")),
              "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out


if __name__ == "__main__":
    main()
