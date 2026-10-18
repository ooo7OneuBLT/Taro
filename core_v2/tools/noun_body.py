"""段階9の合格テストを目だけの場面で（体ありの A・B4・B2・C）。段3の写しの太郎を読んで回す。

体なしの試験場（tools/noun_bed.py の probe）と同じ試し・同じ合格の線（仕様書 §4）を、目だけの場面へ移した：
* 1回の試し：その語の部屋で、候補の物を並べる（4択は 2×2、2択は横に2つ。向きは乱数）。太郎は PREVIEW_S（2秒）黙って自由に見る。
  そのあと親が「〇〇はどこ？」と言い、聞いた回から LOOK_S（2秒＝20回）のあいだに、いちばん長く見た物（視線の角が
  noun_scene.looking_at の決まりで近い物）を選んだ物とする。どれも見なければ外れ。物を片付けて GAP_S（0.5秒）あける
* A：覚えた個体の4択（ほかの3つは別の語の覚えた個体）　B4：見せていない個体（個体3）の4択　B2：見せていない個体と、
  覚えた個体の色（色A）と同じ色の別の種類の個体の2択（形を選ぶ割合）　C：物を出さずに音だけ鳴らし（env.object_sound）、
  音の入口から意味の辞書へ届いた荷札で、20語のうちいちばん強い語が音の主か（noun_bed と同じ数え方）
* 試しのあいだは覚えない（意味の辞書・耳・種類の層の learning を切る）。太郎は試しごとに脳の「その場の状態」だけ消す（学んだ中身は残る）
* 合格の線（1人の太郎）：A は 16語以上が 70%以上、B4 は平均 60%以上、B2 は 70%以上、C は音の6語で平均 60%以上

使い方：
  python -m core_v2.tools.noun_body [--from auto|写しの名前] [--seed 0] [--eye dinov2|table] [--objects made|real]
                                    [--trials 10] [--words 全部|語,語] [--image-trials 4] [--log-dir 段階9_体あり]
  --from auto：育ち帳の段3（world=eye）のいちばん新しい写し（種 --seed）
出力：G/logs/<log-dir>_<物>_<目>_種<seed>/ の まとめ.json・試し.csv、最初の --image-trials 回の試しの前後の画像（部屋ごとのフォルダ）
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch

from ..world_eye import instances as ins
from ..world_eye import noun_scene as ns
from . import growth
from .grow_names import EyeTaro

PREVIEW_S = 2.0
LOOK_S = 2.0
GAP_S = 0.5
N_TRIALS = 10


class Tester(EyeTaro):
    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, all_insts=True, **kw)
        b = self.brain.boxes
        b["semantic"].learning = False
        b["listen"].learning = False
        b["kind"].learning = False
        self.rows: list[dict] = []
        self.n_trial = 0

    def look_trial(self, word: str, cands: list[tuple[int, int]], test: str, rng: random.Random) -> int:
        """候補 [(語の番号, 個体)] を並べ、語を言い、いちばん長く見た候補の番号（どれも見なければ −1）。"""
        env = self.use_room(ins.ROOM_OF[word])
        placed = ns.place(env, [(ins.WORDS[i], k, rng.randrange(ns.N_VIEWS)) for i, k in cands], cause=f"test_{test}")
        self.brain.reset()
        for _ in range(int(round(PREVIEW_S / ns.TICK_S))):
            self.tick()
        target = next((p.name for p in placed if p.word == word), "")
        env.say(f"{word}はどこ？", target=target, cause=f"test_{test}")
        counts = [0] * len(cands)
        for _ in range(int(round(LOOK_S / ns.TICK_S))):
            self.tick()
            p = ns.looking_at(env, placed)
            if p is not None:
                counts[placed.index(p)] += 1
        ns.clear(env, placed, cause=f"test_{test}_end")
        for _ in range(int(round(GAP_S / ns.TICK_S))):
            self.tick()
        best = max(counts)
        pick = rng.choice([j for j, c in enumerate(counts) if c == best]) if best > 0 else -1
        self.rows.append({"trial": self.n_trial, "test": test, "word": word,
                          "cands": " ".join(f"{ins.WORDS[i]}{k}" for i, k in cands), "looks": counts, "pick": pick})
        self.n_trial += 1
        return pick

    def sound_trial(self, word: str, k: int, seed: int) -> bool:
        from ..tagging import word_tag
        from ..world_eye import sounds as snd
        env = self.use_room(ins.ROOM_OF[word])
        self.brain.reset()
        for _ in range(3):
            self.tick()
        env.object_sound(f"音_{word}", snd.synth(ins.KIND_OF[word], k, seed=seed), cause="test_C")
        n = None
        for _ in range(3):              # 環境→音の入口→意味の辞書：届いた回の荷札を使う
            self.tick()
            v = self.brain.buffer[("soundin", "semantic")]
            if float(torch.linalg.vector_norm(v)) > 0.0:
                n = v
                break
        if n is None:
            hit = False
        else:
            wt = torch.stack([word_tag(w) for w in ins.WORDS])
            score = self.brain.boxes["semantic"].word_scores(wt, n=n)
            hit = ins.WORDS[int(torch.argmax(score))] == word
        self.rows.append({"trial": self.n_trial, "test": "C", "word": word, "cands": "", "looks": "", "pick": int(hit)})
        self.n_trial += 1
        return hit


def probe(t: Tester, seed: int, words: list[str], n_trials: int, image_trials: int = 4) -> dict:
    """noun_bed.probe と同じ並べ方（乱数の種も同じ決め方）。"""
    rng = random.Random(1000 + seed)
    W, nW = ins.WORDS, len(ins.WORDS)
    out = {"A": {}, "B4": {}, "B2": {}, "C": {}}
    for w in words:
        i = W.index(w)
        a = b4 = b2 = 0
        ca = ins.kind_colors(w)[0]
        foils = [(j, m) for j in range(nW) if j != i for m in ins.TRAIN_INST if ins.instance(W[j], m)["color"] == ca]
        for _ in range(n_trials):
            if t.n_trial >= image_trials:
                t.save_images(False)
            others = rng.sample([j for j in range(nW) if j != i], 3)
            tk = rng.choice(ins.TRAIN_INST)
            cands = [(i, tk)] + [(j, rng.choice(ins.TRAIN_INST)) for j in others]
            order = list(range(4))
            rng.shuffle(order)
            a += t.look_trial(w, [cands[o] for o in order], "A", rng) == order.index(0)
            cands = [(i, ins.TEST_INST)] + [(j, rng.randrange(ins.N_INST)) for j in others]
            rng.shuffle(order)
            b4 += t.look_trial(w, [cands[o] for o in order], "B4", rng) == order.index(0)
            foil = rng.choice(foils) if foils else (others[0], 0)
            pair = [(i, ins.TEST_INST), foil]
            sw = rng.random() < 0.5
            b2 += t.look_trial(w, pair[::-1] if sw else pair, "B2", rng) == (1 if sw else 0)
        out["A"][w], out["B4"][w], out["B2"][w] = a / n_trials, b4 / n_trials, b2 / n_trials
    for w in [w for w in ins.SOUND_WORDS if w in words]:
        hit = sum(t.sound_trial(w, rng.randrange(ins.N_INST), 10 ** 7 + k) for k in range(n_trials))
        out["C"][w] = hit / n_trials
    return out


def judge(p: dict) -> dict:
    A, B4, B2, C = p["A"], p["B4"], p["B2"], p["C"]
    m = lambda d: round(float(np.mean(list(d.values()))), 3) if d else None  # noqa: E731
    c6 = [C[w] for w in ins.C_WORDS if w in C]
    s = {"A_words_70": sum(v >= 0.7 for v in A.values()), "A_words": len(A), "A_mean": m(A), "B4_mean": m(B4),
         "B2_mean": m(B2), "C6_mean": round(float(np.mean(c6)), 3) if c6 else None, "C_spoon": C.get("すぷうん")}
    full = len(A) == len(ins.WORDS)
    s["A_pass"] = (s["A_words_70"] >= 16) if full else None
    s["B4_pass"] = s["B4_mean"] is not None and s["B4_mean"] >= 0.6
    s["B2_pass"] = s["B2_mean"] is not None and s["B2_mean"] >= 0.7
    s["C_pass"] = s["C6_mean"] is not None and s["C6_mean"] >= 0.6
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from", dest="start", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eye", default="dinov2", choices=["dinov2", "table"])
    ap.add_argument("--table", default=None)
    ap.add_argument("--objects", default=None, choices=[None, "made", "real"], help="既定：写しの物")
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--words", default="", help="語,語（既定は20語全部）")
    ap.add_argument("--image-trials", type=int, default=4)
    ap.add_argument("--log-dir", default="段階9_体あり")
    a = ap.parse_args(argv)
    row = growth.latest(3, seed=a.seed, world="eye") if a.start == "auto" else growth.find(a.start)
    objects = a.objects or row["conditions"].get("objects", "made")
    tag = f"{objects}_{a.eye}_種{a.seed}"
    log_root = os.path.join(growth.REPO, "G", "logs", f"{a.log_dir}_{tag}")
    os.makedirs(log_root, exist_ok=True)
    words = [w for w in a.words.split(",") if w] or list(ins.WORDS)
    t = Tester(a.seed, objects, a.eye, a.table, log_root=log_root, start=row["name"])
    print(f"[段階9 体あり] 写し：{row['name']}　目：{t.eye_info}　物：{objects}　語 {len(words)}　試し {a.trials}", flush=True)
    t0 = time.time()
    try:
        p = probe(t, a.seed, words, a.trials, a.image_trials)
    finally:
        t.close()
    s = judge(p)
    res = {"snapshot": row["name"], "snapshot_conditions": row["conditions"], "eye": t.eye_info, "objects": objects,
           "seed": a.seed, "trials": a.trials, "summary": s, "probe": p, "wall_s": round(time.time() - t0, 1),
           "s_per_trial": round((time.time() - t0) / max(1, t.n_trial), 2), "log_root": log_root}
    with open(os.path.join(log_root, "まとめ.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    with open(os.path.join(log_root, "試し.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["trial", "test", "word", "cands", "looks", "pick"])
        wr.writeheader()
        wr.writerows(t.rows)
    print(json.dumps(s, ensure_ascii=False), f"（{res['wall_s']}秒、1回の試し {res['s_per_trial']}秒）", flush=True)
    return res


if __name__ == "__main__":
    main()
