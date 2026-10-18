"""段階9：体なしの「名詞を厚く」の試験場（仕様書 `2026-10-18_段階9_仕様書_名詞を厚く.md` §4・付録）。

本物の係：種類の層（kind）・物の名簿（objfile）・意味の辞書（semantic、覚え方B）・物の音の入口（soundin）を、脳と同じ二重バッファ
（voice_bed.SubBrain）で回す。目・耳・目を向ける反射は無い：
* 目の入口の代わりに、物ごとの (荷札, 場所の山) と目の動きの地図を手で kind へ渡す。荷札は見た目の素の表の 384 を
  tagging.appearance_tag で縮め、目の入口と同じ雑音（tagging.noisy）を足したもの
* 耳の代わりに、親の語の荷札（tagging.word_tag）をその回だけ意味の辞書と種類の層へ渡す（耳は段1で20語を覚えている前提）
* 目を向ける先は、太郎が自分で決める代わりに、物を見る長さ（2〜4秒）を乱数で決めて次の物へ移る（自由に見る）。
  移るときは、目を向ける反射の指令の写しを名簿へ、目の動きの地図を kind→名簿へ渡す（nai_bed と同じ）

**見た目の素の表**（--bank）：
* 既定は「代わりの表」（make_standin。**DINOv2 ではない、試験場を作るための作り物**）：384 の見た目＝
  共通の向き M（どの物にもある。DINOv2 の特徴は物どうしでも似ている）＋種類の向き S（形）＋色の向き C（パレットの8色で共通）
  ＋個体の向き 0.5·I（大きさ・模様・部品の長さ）＋向きの変化 0.5·V（左右の角度で滑らかに回る）。M・S・C の重みはどれも1：
  仕様書の心配「単色の単純な形では DINOv2 の特徴に色が強く出る」の、色と形が同じ強さの場合。0.5 は切りのよい値。
  どれも当て推量で、PC の本物の表（tools/make_appearance_bank.py、`core_v2/data/段階9_見た目の素.pt`）が来たら差し替える
* `--bank core_v2/data/段階9_見た目の素.pt` で本物の表（拾えなかった向きは使わない）

**育て方**（親の場面。ほうびは無い）：場面1つ＝部屋を1つ選び、その部屋の語から3つ、覚える個体（既定：個体0〜2、--one-inst なら個体0だけ）を
並べる（左・真ん中・右、8マスずつ）。12秒。太郎は1つの物を 2〜4秒見て、別の物へ移る。見ている間、親は物を回す（1秒ごとに
向きが隣へ）。見始めて 1.5秒で親が P_NAME（半分）の割合でその物の名前を言う。音のある物は、見始めて 0.5秒に P_SOUND（半分）の
割合で親が動かして鳴る。場面と場面のあいだは 3秒、何も無い。

**試し**（覚えるのを止めた写しで。語を聞いて2秒＝20回のあいだにいちばん長く見た物で点を付ける）：
* 目を向ける先は、目を向ける先を選ぶ係（gaze）の決まりそのまま：思い出した荷札と物の荷札の cos が MATCH_COS（0.7）以上の物の中で
  いちばん似た物を見る。どれも届かなければ（または思い出せなければ）物を1つ乱数で見る（自由に見る＝偶然）。毎回、目の入口の雑音と向きは新しく引く
* A：覚えた個体の4択（ほかの3つは別の語の覚えた個体）。語ごとに N_TRIALS 回
* B4：見せていない個体（個体3）の4択。B2：見せていない個体と「覚えた個体の色（色A）と同じ色の、別の種類の個体」の2択（形を選ぶ割合）
* C：物を隠して音だけ（見せていない雑音の種、個体は4つから乱数）→ Wᵀ(v·N) のいちばん強い語が音の主か（20語から、偶然5%）
* まず測る（先行研究 5節）：荷札の似かた（同じ種類の別の個体・別の種類・別の種類で同じ色）を、種類の層の前と後で

使い方：
  python -m core_v2.tools.noun_bed [--taros 0-4] [--conds full,kind_off,one_inst,one_inst_kind_off] [--scenes 300]
                                   [--bank 代わり|<表の .pt>] [--log-dir 段階9_名詞]
                                   [--semantic table|sparse] [--sound mean|time]   （まばら化の試し：sparse_semantic.py）
出力：G/logs/<log-dir>/まとめ.json と 太郎ごとの json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch

from ..boxes import box_class
from ..boxes.eyein import decode_ego, ego_map
from ..boxes.gaze import MATCH_COS
from ..boxes.objfile import GRID, bump_at
from ..boxes.semantic import RECALL_MIN  # noqa: F401  (noun_diag が使う)
from ..boxes.soundin import sound_tag
from ..formats import DTYPE, TAG_DIM, make_tag
from ..tagging import appearance_tag, noisy, word_tag
from ..world_eye import instances as ins
from ..world_eye import sounds as snd
from . import sparse_semantic as sps
from .voice_bed import SubBrain

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
TICK_S = 0.1
CENTRE = (GRID - 1) / 2.0
BOXES = ["kind", "objfile", "semantic", "soundin"]
N_VIEWS = 24
SCENE_S = 12.0
GAP_S = 3.0
N_SHOW = 3
SPACING = 8.0              # 物どうしの間（マス）。名簿のまとめる距離 NEAR_CELLS 4 より十分遠い
DWELL_S = (2.0, 4.0)
NAME_AT_S = 1.5
SOUND_AT_S = 0.5
P_NAME = 0.5               # 見ている物の名前を親が言う割合［切りのよい値］
P_SOUND = 0.5              # 音のある物を見ているとき、親が動かして鳴る割合［切りのよい値］
TURN_EVERY_S = 1.0         # 親が物を回す（向きが隣へ）
LOOK_S = 2.0               # 試し：語を聞いてから点を付ける時間（仕様書）
N_TRIALS = 10
STANDIN_SEED = 9
STANDIN_W = {"M": 1.0, "S": 1.0, "C": 1.0, "I": 0.5, "V": 0.5}

_ZTAG = torch.zeros(TAG_DIM, dtype=DTYPE)
_EGO0 = ego_map(0.0, 0.0)


# ---------------------------------------------------------------- 見た目の素
def make_standin(seed: int = STANDIN_SEED) -> dict:
    """代わりの表（DINOv2 ではない）。上の説明の式。app[20, 4, 24, 384]。"""
    g = np.random.default_rng(seed)
    D = 384
    unit = lambda: g.standard_normal(D)  # noqa: E731
    M = unit()
    S = {w: unit() for w in ins.WORDS}
    C = {c: unit() for c in ins.PALETTE}
    W = STANDIN_W
    app = np.zeros((len(ins.WORDS), ins.N_INST, N_VIEWS, D), dtype=np.float32)
    for i, w in enumerate(ins.WORDS):
        for k in range(ins.N_INST):
            inst = ins.instance(w, k)
            I, a, b = unit(), unit(), unit()
            for v in range(N_VIEWS):
                th = math.radians(15.0 * v)
                V = (math.cos(th) * a + math.sin(th) * b) / math.sqrt(1.0)
                app[i, k, v] = W["M"] * M + W["S"] * S[w] + W["C"] * C[inst["color"]] + W["I"] * I + W["V"] * V
    return {"app": app, "found": np.ones(app.shape[:3], dtype=bool), "kind": "代わりの表（DINOv2 ではない）",
            "weights": dict(W), "seed": seed}


def load_bank(spec: str) -> dict:
    if spec in ("代わり", "standin", ""):
        return make_standin()
    path = spec if os.path.isabs(spec) else os.path.join(REPO, spec)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if list(blob["words"]) != list(ins.WORDS):
        raise ValueError("表の語の並びが instances.WORDS と違う")
    return {"app": blob["app"].float().numpy(), "found": blob["found"].numpy(), "kind": "本物の表（DINOv2）",
            "meta": blob.get("meta", {})}


class Bank:
    def __init__(self, bank: dict) -> None:
        self.app = bank["app"]
        self.found = bank["found"]
        self.info = {k: v for k, v in bank.items() if k not in ("app", "found")}
        self._tag = {}

    def views(self, i: int, k: int) -> list[int]:
        return [v for v in range(N_VIEWS) if self.found[i, k, v]]

    def tag(self, i: int, k: int, v: int) -> torch.Tensor:
        key = (i, k, v)
        if key not in self._tag:
            self._tag[key] = appearance_tag(self.app[i, k, v].astype(np.float64))
        return self._tag[key]


# ---------------------------------------------------------------- 太郎
def make_boxes(seed: int, kind_learning: bool = True, semantic: str = "table", sound: str = "mean") -> dict:
    b = {bid: box_class(bid)() for bid in BOXES if bid != "kind"}
    b["kind"] = box_class("kind")(learning=kind_learning, seed=seed)
    if semantic == "sparse":      # 試し：まばらな語の表し方（tools/sparse_semantic.py。本物の太郎の係は替えていない）
        b["semantic"] = sps.SparseSemantic()
    if sound == "time":           # 試し：物の音の時間の形を残す入口
        b["soundin"] = sps.SoundinTime()
    b["semantic"].learning = True
    # 試験場には目を向ける反射・驚きの係が無いので、名簿の→orienteye・→surprise の列は作らない（速くするだけ）
    b["objfile"].outputs = [e for e in b["objfile"].outputs if e.dst in ("semantic", "kind")]
    return b


class NounBed:
    def __init__(self, seed: int, bank: Bank, kind_learning: bool = True, one_inst: bool = False,
                 semantic: str = "table", sound: str = "mean") -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        torch.manual_seed(seed)
        self.bank = bank
        self.sound = sound
        self.boxes = make_boxes(seed, kind_learning, semantic, sound)
        self.brain = SubBrain(self.boxes)
        self.train_inst = (0,) if one_inst else ins.TRAIN_INST
        self.stats = {"scenes": 0, "names": 0, "sounds": 0, "ticks": 0}

    def _tick(self, objs, ego=_EGO0, fire=None, word=None, sound=None):
        """objs：[{"i","k","v","r","c"}]。1回進める。"""
        eyein = [(noisy(self.bank.tag(o["i"], o["k"], o["v"])), bump_at(o["r"], o["c"], sigma=1.0)) for o in objs
                 if 0.0 <= o["c"] <= GRID - 1]
        eyein = eyein + [(_ZTAG, ego)]
        sal = torch.zeros(GRID, GRID, dtype=DTYPE)
        for o in objs:
            if 0.0 <= o["c"] <= GRID - 1:
                sal = torch.maximum(sal, bump_at(o["r"], o["c"]))
        inject = {("eyein", "kind"): eyein, ("prio", "objfile"): sal,
                  ("orienteye", "objfile"): fire if fire is not None else torch.zeros(2)}
        q = word_tag(word) if word else _ZTAG
        inject[("listen", "semantic")] = q
        inject[("listen", "kind")] = q
        ext = {"soundin": sound} if sound is not None else {}
        self.brain.tick(ext, inject)
        self.stats["ticks"] += 1

    def scene(self) -> None:
        rng = self.rng
        room = rng.choice(["食事", "お風呂", "遊び"])
        words = rng.sample([w for w in ins.WORDS if ins.ROOM_OF[w] == room], N_SHOW)
        cols = [CENTRE - SPACING, CENTRE, CENTRE + SPACING]
        rng.shuffle(cols)
        objs = []
        for w, c in zip(words, cols):
            i = ins.WORDS.index(w)
            k = rng.choice(self.train_inst)
            objs.append({"i": i, "k": k, "v": rng.choice(self.bank.views(i, k)), "r": CENTRE, "c": c, "w": w})
        # 最初に見る物を真ん中へ
        cur = rng.randrange(N_SHOW)
        self._look_at(objs, cur, first=True)
        t, end = 0, int(SCENE_S / TICK_S)
        while t < end:
            dwell = int(rng.uniform(*DWELL_S) / TICK_S)
            o = objs[cur]
            say = rng.random() < P_NAME
            ring = ins.KIND_OF[o["w"]] in snd.BASE_LEN and rng.random() < P_SOUND
            for d in range(dwell):
                if t >= end:
                    break
                if d > 0 and d % int(TURN_EVERY_S / TICK_S) == 0:
                    self._turn(o)
                word = o["w"] if (say and d == int(NAME_AT_S / TICK_S)) else None
                sound = None
                if ring and d == int(SOUND_AT_S / TICK_S):
                    sound = snd.synth(ins.KIND_OF[o["w"]], o["k"], seed=rng.randrange(10 ** 6))
                    self.stats["sounds"] += 1
                self.stats["names"] += word is not None
                self._tick(objs, word=word, sound=sound)
                t += 1
            nxt = rng.choice([j for j in range(N_SHOW) if j != cur])
            self._look_at(objs, nxt)
            t += 2
            cur = nxt
        for _ in range(int(GAP_S / TICK_S)):
            self._tick([])
        self.stats["scenes"] += 1

    def _turn(self, o) -> None:
        vs = self.bank.views(o["i"], o["k"])
        j = vs.index(o["v"]) if o["v"] in vs else 0
        o["v"] = vs[(j + self.rng.choice((-1, 1))) % len(vs)]

    def _look_at(self, objs, j: int, first: bool = False) -> None:
        """物 j へ目を移す：1回目に指令の写しを名簿へ、2回目に物を動かして目の動きの地図を渡す。"""
        dc = CENTRE - objs[j]["c"]
        if not first:
            fire = torch.tensor([(objs[j]["c"] - CENTRE) / CENTRE, 0.0], dtype=DTYPE)
            self._tick(objs, fire=fire)
        ego = ego_map(dc * (224.0 / GRID), 0.0)
        real = decode_ego(ego)[1]
        for o in objs:
            o["c"] += real
        self._tick(objs, ego=ego)

    def grow(self, n_scenes: int) -> None:
        for _ in range(n_scenes):
            self.scene()


# ---------------------------------------------------------------- 試し
def view_tags(bed: NounBed, kind, i: int, k: int, rng) -> torch.Tensor:
    v = rng.choice(bed.bank.views(i, k))
    x = noisy(bed.bank.tag(i, k, v)).reshape(1, TAG_DIM)
    return kind.transform(x)[0]


def look_trial(bed: NounBed, word: str, cands: list[tuple[int, int]], rng) -> int:
    """語を聞いて LOOK_S のあいだに、いちばん長く見た候補の番号。"""
    sem, kind = bed.boxes["semantic"], bed.boxes["kind"]
    r = sem.recall(word_tag(word))      # 今の表：W q（‖W q‖ ≥ RECALL_MIN で出す）。まばら版：細胞の見た目の予想の平均
    want = noisy(r) if r is not None else None
    counts = [0] * len(cands)
    for _ in range(int(LOOK_S / TICK_S)):
        tags = [view_tags(bed, kind, i, k, rng) for i, k in cands]
        if want is not None:
            cos = [float(t @ want) for t in tags]
            ok = [j for j, c in enumerate(cos) if c >= MATCH_COS]
            j = max(ok, key=lambda j: cos[j]) if ok else rng.randrange(len(cands))
        else:
            j = rng.randrange(len(cands))
        counts[j] += 1
    best = max(counts)
    return rng.choice([j for j, c in enumerate(counts) if c == best])


def similarity(bed: NounBed, rng, use_kind: bool = True) -> dict:
    """個体ごとに向きを全部通した荷札の平均どうしの cos（同じ種類の別の個体・別の種類・別の種類で同じ色）。"""
    kind = bed.boxes["kind"]
    T = {}
    for i, w in enumerate(ins.WORDS):
        for k in range(ins.N_INST):
            x = torch.stack([bed.bank.tag(i, k, v) for v in bed.bank.views(i, k)])
            y = kind.transform(x) if use_kind else x
            T[(i, k)] = make_tag(y.mean(0))
    col = {(i, k): ins.instance(w, k)["color"] for i, w in enumerate(ins.WORDS) for k in range(ins.N_INST)}
    sk, dk, sc = [], [], []
    keys = sorted(T)
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            p, q = keys[a], keys[b]
            c = float(T[p] @ T[q])
            if p[0] == q[0]:
                sk.append(c)
            else:
                dk.append(c)
                if col[p] == col[q]:
                    sc.append(c)
    return {"同じ種類の別の個体": round(float(np.mean(sk)), 3), "別の種類": round(float(np.mean(dk)), 3),
            "別の種類で同じ色": round(float(np.mean(sc)), 3)}


def probe(bed: NounBed, seed: int) -> dict:
    rng = random.Random(1000 + seed)
    words = ins.WORDS
    nW = len(words)
    out = {"A": {}, "B4": {}, "B2": {}, "C": {}}
    for w in words:
        bed.boxes["semantic"].recall(word_tag(w))
        out.setdefault("recall_norm", {})[w] = round(bed.boxes["semantic"].last_recall_norm, 2)
    for i, w in enumerate(words):
        a = b4 = b2 = 0
        ca = ins.kind_colors(w)[0]
        foils = [(j, m) for j in range(nW) if j != i for m in ins.TRAIN_INST
                 if ins.instance(words[j], m)["color"] == ca]
        for _ in range(N_TRIALS):
            others = rng.sample([j for j in range(nW) if j != i], 3)
            # A：覚えた個体（1語1個体の太郎は個体0）
            tk = rng.choice(bed.train_inst)
            cands = [(i, tk)] + [(j, rng.choice(ins.TRAIN_INST)) for j in others]
            order = list(range(4))
            rng.shuffle(order)
            a += look_trial(bed, w, [cands[o] for o in order], rng) == order.index(0)
            # B4：見せていない個体
            cands = [(i, ins.TEST_INST)] + [(j, rng.randrange(ins.N_INST)) for j in others]
            rng.shuffle(order)
            b4 += look_trial(bed, w, [cands[o] for o in order], rng) == order.index(0)
            # B2：形と色がぶつかる2択
            foil = rng.choice(foils) if foils else (others[0], 0)
            pair = [(i, ins.TEST_INST), foil]
            sw = rng.random() < 0.5
            pick = look_trial(bed, w, pair[::-1] if sw else pair, rng)
            b2 += pick == (1 if sw else 0)
        out["A"][w] = a / N_TRIALS
        out["B4"][w] = b4 / N_TRIALS
        out["B2"][w] = b2 / N_TRIALS
    # C：音だけ
    sem = bed.boxes["semantic"]
    wt = torch.stack([word_tag(w) for w in words])
    for w in ins.SOUND_WORDS:
        hit = 0
        for t in range(N_TRIALS):
            wave = snd.synth(ins.KIND_OF[w], rng.randrange(ins.N_INST), seed=10 ** 7 + t)
            n = sps.sound_tag_time(wave) if bed.sound == "time" else sound_tag(wave)
            score = sem.word_scores(wt, n=n) if hasattr(sem, "word_scores") else wt @ (sem.W.T @ n)
            hit += words[int(torch.argmax(score))] == w
        out["C"][w] = hit / N_TRIALS
    out["sim_before"] = similarity(bed, rng, use_kind=False)
    out["sim_after"] = similarity(bed, rng, use_kind=True)
    return out


def summarize_one(p: dict) -> dict:
    A, B4, B2, C = p["A"], p["B4"], p["B2"], p["C"]
    return {"A_words_70": sum(v >= 0.7 for v in A.values()), "A_mean": round(float(np.mean(list(A.values()))), 3),
            "B4_mean": round(float(np.mean(list(B4.values()))), 3), "B2_mean": round(float(np.mean(list(B2.values()))), 3),
            "B2_no_sound": round(float(np.mean([B2[w] for w in B2 if w not in ins.SOUND_WORDS])), 3),
            "C6_mean": round(float(np.mean([C[w] for w in ins.C_WORDS])), 3), "C_spoon": C.get("すぷうん")}


def run(seed: int, cond: str, bank: Bank, n_scenes: int, semantic: str = "table", sound: str = "mean") -> dict:
    t0 = time.time()
    bed = NounBed(seed, bank, kind_learning="kind_off" not in cond, one_inst=cond.startswith("one_inst"),
                  semantic=semantic, sound=sound)
    if cond.endswith("_negall"):      # 比べる用：種類の層の負の組を置き場の見え全部にする（kind.NEGATIVES の説明）
        bed.boxes["kind"].negatives = "all"
    bed.grow(n_scenes)
    k = bed.boxes["kind"]
    bed.boxes["semantic"].learning = False
    k.learning = False
    p = probe(bed, seed)
    s = summarize_one(p)
    return {"seed": seed, "cond": cond, "semantic": semantic, "sound": sound, "summary": s, "probe": p, "stats": bed.stats, "kind_train_steps": k.n_train,
            "kind_last_loss": k.last_loss, "n_learn": int(bed.boxes["semantic"].n_learn), "wall_s": round(time.time() - t0, 1)}


def summarize(results: list[dict]) -> dict:
    out = {}
    for cond in dict.fromkeys(r["cond"] for r in results):
        rs = [r["summary"] for r in results if r["cond"] == cond]
        m = lambda k: round(float(np.mean([r[k] for r in rs])), 3)  # noqa: E731
        A_pass_taros = sum(r["A_words_70"] >= 16 for r in rs)
        out[cond] = {"taros": len(rs), "A_words_70": [r["A_words_70"] for r in rs], "A_pass_taros": A_pass_taros,
                     "A_pass": A_pass_taros >= 4, "A_mean": m("A_mean"), "B4_mean": m("B4_mean"), "B4_pass": m("B4_mean") >= 0.6,
                     "B2_mean": m("B2_mean"), "B2_pass": m("B2_mean") >= 0.7, "B2_no_sound": m("B2_no_sound"),
                     "C6_mean": m("C6_mean"), "C_pass": m("C6_mean") >= 0.6, "C_spoon": m("C_spoon")}
        words = ins.WORDS
        pr = [r["probe"] for r in results if r["cond"] == cond]
        out[cond]["A_by_word"] = {w: round(float(np.mean([p["A"][w] for p in pr])), 2) for w in words}
        out[cond]["B4_by_word"] = {w: round(float(np.mean([p["B4"][w] for p in pr])), 2) for w in words}
        out[cond]["B2_by_word"] = {w: round(float(np.mean([p["B2"][w] for p in pr])), 2) for w in words}
        out[cond]["C_by_word"] = {w: round(float(np.mean([p["C"][w] for p in pr])), 2) for w in ins.SOUND_WORDS}
        out[cond]["sim_before"] = pr[0]["sim_before"]
        out[cond]["sim_after"] = {k: round(float(np.mean([p["sim_after"][k] for p in pr])), 3) for k in pr[0]["sim_after"]}
    return out


def parse_seeds(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main(argv=None):
    torch.set_num_threads(1)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--taros", default="0-4")
    ap.add_argument("--conds", default="full,kind_off,one_inst,one_inst_kind_off")
    ap.add_argument("--scenes", type=int, default=300)
    ap.add_argument("--bank", default="代わり")
    ap.add_argument("--log-dir", default="段階9_名詞")
    ap.add_argument("--semantic", default="table", choices=["table", "sparse"],
                    help="table＝今の 64×64 の表 W／sparse＝まばらな語の表し方（試し）")
    ap.add_argument("--sound", default="mean", choices=["mean", "time"],
                    help="mean＝今の物の音の荷札（帯の時間平均）／time＝時間の形を残す（試し）")
    a = ap.parse_args(argv)
    bank = Bank(load_bank(a.bank))
    log_dir = os.path.join(REPO, "G", "logs", a.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    print("見た目の素：", bank.info.get("kind"), flush=True)
    results = []
    for cond in a.conds.split(","):
        for seed in parse_seeds(a.taros):
            r = run(seed, cond, bank, a.scenes, a.semantic, a.sound)
            results.append(r)
            with open(os.path.join(log_dir, f"{cond}_seed{seed}.json"), "w", encoding="utf-8") as f:
                json.dump(r, f, ensure_ascii=False, indent=1, default=str)
            print(f"[{cond} 太郎{seed}] {json.dumps(r['summary'], ensure_ascii=False)} 学び{r['kind_train_steps']}歩 "
                  f"名づけ{r['stats']['names']} 音{r['stats']['sounds']} 覚えた{r['n_learn']} 似かた後 {r['probe']['sim_after']} "
                  f"（{r['wall_s']}秒）", flush=True)
    summary = summarize(results)
    summary["_bank"] = bank.info
    with open(os.path.join(log_dir, "まとめ.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=str)
    for cond, s in summary.items():
        if cond.startswith("_"):
            continue
        print(cond, {k: v for k, v in s.items() if not k.endswith("_by_word")}, flush=True)
    return summary


if __name__ == "__main__":
    main()
