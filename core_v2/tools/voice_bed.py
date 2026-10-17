"""段階7：体なしの声の試験場（仕様書 `2026-10-17_段階7_仕様書_言える.md` §2-5、移し方の文書 付録 D）。

耳の入口から声の出口までの係（earin・listen・semantic・wordplan・voice・voiceout）だけを本物にし、
脳と同じ二重バッファ（矢印1本ごとに1回の遅れ）で回す。体・カメラ・物の名簿は使わない。
名簿の代わりに、「いま見ている物」を名簿と同じ形（行ごとの荷札と山、今見ている行、名前を結び付ける行）で
意味の辞書へ手で渡す。物の見た目の荷札は、物ごとに決まった乱数（DINOv2 の特徴の代わり）。

親（環境。太郎へのほうびの計算ではない）：
* 話しかけ：遊びの時間に、平均 TALK_GAP_S 秒ごとに1つ（段階5の話しかけの文、parent_speech.make_corpus）。
  物の名前の入った文の P_JOINT は、その物を太郎に見せながら言う（名づけの場面：物が SCENE_S 秒見えていて、
  NAME_AT_S 秒に言う。段階4〜6の名づけと同じ形）。太郎が声を出している最中は話さない（終わるまで待つ）
* 応える（前の太郎 B `parent_sim_b.py:497-537` と同じ）：太郎の声が終わったら、確率 RESP_PROB×(0.5＋語らしさ)
  で気づき、RESP_DELAY_S 秒後に、いちばん近い語を1語だけ言い返す。語らしさ＝親の語のどれかとの似かたの最大

時間：遊びの時間（穏やかに起きていて親がいる時間）を1日 PLAY_H 時間だけ回す。眠っている時間などは回さない。
遊びの時間の中でも、何も起きていない間（太郎も親も黙っていて、意味の辞書の名残りも無い）は飛ばす。
月齢を縮めたり、1日の回数を増やしたりはしない（仕様書 §2-5）。

試し（PROBE_DAYS 日ごと。そのときの太郎を写して、覚えるのと喃語を止めて回す。もとの太郎は変えない）：
* わかる：「〇〇はどこ？」を聞かせ、意味の辞書が思い出した見た目が、いちばんその物に近いか
  （体ありの「見に行ける」の代わり。体なしなので目は動かさない）
* 言える（合格の線①）：物を PRESENT_S 秒見せ、BLANK_S 秒あける、を N_PRESENT 回。見せている間に太郎が出した声が、
  親の語のうちいちばんその物の名前に近く、しかも似かたが SIM_MIN（0.5、M が「似た声」とみなすのと同じ線）を超えるか。
  黙ったら外れ。「っ」は口から出ないので（ユーザー決定待ち、既定は「言えなくてよい」）、「こぷ」「こた」のような形も当たり
  （口の表に「っ」を足す案に替えるときは core_v2/mouth.py の TABLE に1行足すだけで、ここは変えない）

使い方：
  python -m core_v2.tools.voice_bed [--seed 0] [--days 730] [--start-day 0] [--probe-days 7]
                                    [--log-dir 段階7_声の試験場] [--save G/models/段階7_声_seed0.pt]
出力：G/logs/<log-dir>/seed<種>.jsonl（試しごとに1行）と seed<種>_まとめ.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
import random
import time

import numpy as np
import torch

from .. import mouth, wiring
from ..boxes import ENV, box_class
from ..boxes.semantic import TRACE_TICKS
from ..boxes.voice import SIM_MIN
from ..formats import GRID, DTYPE, check, make_tag, zero
from ..tagging import appearance_tag
from .parent_speech import FRAMES, FILLERS, P_ALONE, P_FILLER, WORDS

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
BOXES = ["earin", "listen", "semantic", "wordplan", "voice", "voiceout"]
TICK_S = 0.1
DAY_MONTHS = 1.0 / 30.4

PLAY_H = 4.5          # 1日の遊びの時間（前の太郎 B：45秒ごとの喃語が1日約360回＝約4.5時間）［Tier3・B の値］
TALK_GAP_S = 20.0     # 親の話しかけの平均の間隔（1時間に約180の発話。記憶ベース・要確認）［Tier3］
P_JOINT = 0.5         # 物の名前の入った文を、その物を見せながら言う割合［Tier3・仮置き］
SCENE_S = 5.0         # 名づけの場面で物が見えている秒数（段階4の名づけと同じくらい）
NAME_AT_S = 1.0       # 名づけの場面で、物が見えてから語を言うまで
RESP_PROB = 0.4       # 前の太郎 B の babble_response.prob（気づいて応える割合 30〜50% の中間）［Tier2］
RESP_DELAY_S = 1.0    # 太郎の声が終わってから親が応えるまで［Tier3・仮置き］
END_GAP_TICKS = 6     # 太郎の声がこれだけ途切れたら、1つの声が終わったとみなす（親の耳）
SETTLE_TICKS = 6      # 最後の入力から、係どうしの矢印に値が残らなくなるまで（耳から声の係までの遅れより長く）
PRESENT_S, BLANK_S, N_PRESENT = 5.0, 2.0, 20
PASS_N = 15
TEST_WORDS = ["ぼおる", "こっぷ"]


# ---------------------------------------------------------------- 6つの係だけの脳
class SubBrain:
    """脳（brain.Brain）と同じ二重バッファで、一部の係だけを回す。外の係から来る矢印はゼロ
    （inject で手で渡したものは除く）。"""

    def __init__(self, boxes: dict, check_formats: bool = False) -> None:
        self.boxes = boxes
        ids = set(boxes)
        self.edges = [e for e in wiring.EDGES if e.dst in ids]
        self.buffer = {e.key: zero(e.fmt, e.size) for e in self.edges}
        self.incoming = {bid: [(e.src, e.key) for e in self.edges if e.dst == bid] for bid in boxes}
        self.check_formats = check_formats

    def tick(self, external: dict | None = None, inject: dict | None = None) -> dict:
        external, inject = external or {}, inject or {}
        outs = {}
        for bid, box in self.boxes.items():
            inbox = {src: (inject[key] if key in inject else self.buffer[key]) for src, key in self.incoming[bid]}
            if bid in external:
                inbox[ENV] = external[bid]
            outs[bid] = box.step(inbox)
        to_env = {}
        for bid, out in outs.items():
            out = dict(out)
            if ENV in out:
                to_env[bid] = out.pop(ENV)
            for dst, v in out.items():
                if (bid, dst) in self.buffer:
                    if self.check_formats:
                        e = next(x for x in self.edges if x.key == (bid, dst))
                        check(e.fmt, v, e.size, name=f"{bid}→{dst}")
                    self.buffer[(bid, dst)] = v
        return to_env


def make_boxes(seed: int, learning: bool = True) -> dict:
    b = {bid: box_class(bid)() for bid in BOXES}
    b["listen"].learning = learning
    b["semantic"].learning = learning
    b["voice"].learning = learning
    b["voice"].babbling = learning
    b["voice"].rng = random.Random(seed)
    b["voice"].reset()
    return b


# ---------------------------------------------------------------- 物（名簿の代わり）
def object_tags(seed: int = 7) -> dict[str, torch.Tensor]:
    g = np.random.default_rng(seed)
    return {w: appearance_tag(g.standard_normal(384)) for w in WORDS}


_PEAK = torch.zeros(GRID, GRID, dtype=DTYPE)
_PEAK[GRID // 2, GRID // 2] = 1.0
_ZMAP = torch.zeros(GRID, GRID, dtype=DTYPE)


def roster(tag) -> list:
    """名簿と同じ形：見ている物の行（山の高さ1＝直前ずっとこれを見ていた）、今見ている行、名前を結び付ける行。"""
    if tag is None:
        return []
    return [(tag, _PEAK), (tag, _ZMAP), (tag, _ZMAP)]


# ---------------------------------------------------------------- 親
def closest_word(voice: str) -> tuple[str, float]:
    best, s = max(((w, mouth.similarity(w, voice)) for w in WORDS), key=lambda x: x[1])
    return best, s


def make_utterance(rng: random.Random) -> tuple[str, str | None]:
    r = rng.random()
    w = rng.choice(WORDS)
    if r < P_ALONE:
        return w, w
    if r < P_ALONE + P_FILLER:
        return rng.choice(FILLERS), None
    return rng.choice(FRAMES).format(w=w), w


class Bed:
    """1人の太郎と親。run_play(秒) で遊びの時間を回す。"""

    def __init__(self, seed: int, start_day: float = 0.0) -> None:
        self.seed = seed
        self.rng = random.Random(1000 + seed)
        torch.manual_seed(seed)
        self.boxes = make_boxes(seed)
        self.brain = SubBrain(self.boxes)
        self.objs = object_tags()
        self.day = start_day
        self.play_t = 0                  # 遊びの時間の中の回
        self.stats = {"babble": 0, "babble_word": 0, "own": 0, "resp": 0, "talk": 0, "scene": 0,
                      "named_by_taro": 0, "ticks": 0}
        self.own_log: list = []
        self._reset_play()

    def _reset_play(self):
        self.pending_say: list[tuple[int, str]] = []   # (言う回, 文)
        self.scene_until = -1
        self.scene_tag = None
        self.cur_voice = ""
        self.last_char_t = -10**9
        self.last_input_t = -10**9
        self.next_talk = self.play_t + self._gap()

    def _gap(self) -> int:
        return max(1, int(round(self.rng.expovariate(1.0 / TALK_GAP_S) / TICK_S)))

    @property
    def month(self) -> float:
        return self.day * DAY_MONTHS

    def _look(self, t):
        return self.scene_tag if self.scene_until >= t else None

    def _quiet(self) -> bool:
        """何も動いていない：太郎は黙っていて、最後の入力（聞いた音・見える物の替わり）から矢印の値が落ち着いている。
        見ている物がそのままなら、見ている間も飛ばしてよい（意味の辞書は skip で同じ物を見続けた分を足す）。"""
        v = self.boxes["voice"]
        return (not v.busy() and not self.cur_voice
                and self.play_t - self.last_char_t > END_GAP_TICKS
                and self.play_t - self.last_input_t > SETTLE_TICKS)

    def _next_event(self, end: int) -> int:
        v = self.boxes["voice"]
        cand = [self.next_talk, self.play_t + v.ticks_until_active(), end]
        cand += [ts for ts, _ in self.pending_say]
        if self.scene_until >= self.play_t:
            cand.append(self.scene_until + 1)
        return int(min(cand))

    def _may_speak(self) -> bool:
        """親は太郎の声にかぶせて話さない。太郎が声を出し始めるちょうどその回にもぶつけない
        （ぶつかると、親の語が自分の声の写しの間に耳へ届く。試験場の割り切り：親が太郎の喃語の待ちを見ている）。"""
        v = self.boxes["voice"]
        return not v.busy() and not self.cur_voice and v.ticks_until_active() > 2

    def step(self, heard_extra: str = ""):
        """1回進める。"""
        t = self.play_t
        v = self.boxes["voice"]
        text = heard_extra
        # 親の話しかけ（太郎が声を出している間は待つ）
        if t >= self.next_talk and self._may_speak() and self.scene_until < t:
            utt, w = make_utterance(self.rng)
            self.stats["talk"] += 1
            if w is not None and self.rng.random() < P_JOINT:
                self.scene_tag = self.objs[w]
                self.scene_until = t + int(SCENE_S / TICK_S)
                self.pending_say.append((t + int(NAME_AT_S / TICK_S), utt))
                self.stats["scene"] += 1
            else:
                self.pending_say.append((t, utt))
            self.next_talk = t + self._gap()
        due = [s for s in self.pending_say if s[0] <= t]
        if due and self._may_speak():
            self.pending_say = [s for s in self.pending_say if s[0] > t]
            text = " ".join([text] + [s for _, s in due]).strip()
        look = self._look(t)
        inject = {("objfile", "semantic"): roster(look)}
        heard = " ".join(x for x in (self._echo, text) if x)
        if heard or look is not self._prev_look:
            self.last_input_t = t
        self._prev_look = look
        to_env = self.brain.tick({"earin": heard, "voiceout": self.month}, inject)
        self._echo = ""
        ch = to_env.get("voiceout", "")
        if ch:
            self._echo = ch                 # 自分の声は次の回に耳へ戻る
            self.cur_voice += ch
            self.last_char_t = t
        elif self.cur_voice and t - self.last_char_t > END_GAP_TICKS:
            self._voice_done(self.cur_voice, look)
            self.cur_voice = ""
        self.play_t += 1
        self.stats["ticks"] += 1

    _echo = ""
    _prev_look = None
    skip_idle = True       # False：飛ばさず1回ずつ回す（飛ばしても結果が同じかの確かめ用）

    def _voice_done(self, voice: str, look):
        self.stats["own"] += 1
        w, s = closest_word(voice)
        if look is not None:
            self.stats["named_by_taro"] += 1
        if self.rng.random() < RESP_PROB * (0.5 + s):
            self.pending_say.append((self.play_t + int(RESP_DELAY_S / TICK_S), w))
            self.stats["resp"] += 1
        if len(self.own_log) < 200000:
            self.own_log.append((round(self.month, 3), voice, w if s > 0 else ""))

    def run_play(self, seconds: float):
        """遊びの時間を seconds 秒ぶん回す（何も起きない間は飛ばす）。"""
        end = self.play_t + int(seconds / TICK_S)
        v = self.boxes["voice"]
        while self.play_t < end:
            if self.skip_idle and self._quiet():
                n = self._next_event(end) - self.play_t - 1
                if n > 0:
                    v.skip(n)
                    self.boxes["semantic"].skip(n, roster(self._look(self.play_t)))
                    self.play_t += n
                    continue
            self.step()
        for e in v.log:
            if e["event"] == "babble":
                self.stats["babble"] += 1
                self.stats["babble_word"] += bool(e["word"])
        v.log.clear()

    def run_day(self):
        self._reset_play()
        self.run_play(PLAY_H * 3600)
        self.day += 1


# ---------------------------------------------------------------- 試し
def probe(bed: Bed) -> dict:
    b = copy.deepcopy(bed.boxes)
    for k in ("listen", "semantic", "voice"):
        b[k].learning = False
    b["voice"].babbling = False
    for x in b.values():
        x.reset()
    objs = bed.objs
    out = {"month": round(bed.month, 3), "day": bed.day}
    # わかる
    for w in TEST_WORDS:
        q = b["listen"].step({"earin": f"{w}はどこ？"})["semantic"]
        r = b["semantic"].recall(q) if float(torch.linalg.vector_norm(q)) > 0 else None
        ok = r is not None and max(objs, key=lambda o: float(objs[o] @ r)) == w
        out[f"know_{w}"] = bool(ok)
    # 言える
    brain = SubBrain(b)
    month = bed.month
    for w in TEST_WORDS:
        hits, said, voices = 0, 0, []
        for _ in range(N_PRESENT):
            voice = ""
            for k in range(int((PRESENT_S + BLANK_S) / TICK_S)):
                look = objs[w] if k < PRESENT_S / TICK_S else None
                env = brain.tick({"earin": "", "voiceout": month}, {("objfile", "semantic"): roster(look)})
                if look is not None:
                    voice += env.get("voiceout", "")
            if voice:
                said += 1
                cw, sim = closest_word(voice)
                hits += cw == w and sim > SIM_MIN
            voices.append(voice)
        out[f"say_{w}"] = hits
        out[f"said_{w}"] = said
        out[f"voices_{w}"] = sorted(set(voices))[:5]
        v = b["voice"]
        row = v.M.get(w, {})
        top = sorted(row.items(), key=lambda kv: -kv[1])[:3]
        out[f"M_{w}"] = [(k, round(s, 2)) for k, s in top]
    v = bed.boxes["voice"]
    out["notebook_sounds"] = len(v.known_sounds())
    out["has_sounds"] = {w: [c for c in w if c not in v.known_sounds()] for w in TEST_WORDS}
    out["known_words"] = sorted(bed.boxes["listen"].known())
    out["lexicon_top"] = v.lexicon.most_common(5)
    out["stats"] = dict(bed.stats)
    return out


def save_state(bed: Bed, path: str, meta: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"boxes": {k: bed.boxes[k].state_dict() for k in ("listen", "semantic", "wordplan", "voice")},
                "meta": meta}, path)


def export_voice(bed_or_boxes, path: str, meta: dict) -> None:
    """体ありの太郎へ持っていく分だけを JSON で書く：声の係（口の帳面・聞いた語の回数・M・喃語の回数）と
    言う語の係の語の一覧。意味の辞書は持っていかない（試験場の見た目の荷札は乱数で、体の目の荷札と違う）。"""
    boxes = bed_or_boxes.boxes if isinstance(bed_or_boxes, Bed) else bed_or_boxes
    blob = {"meta": meta, "voice": boxes["voice"].get_extra_state(),
            "wordplan": boxes["wordplan"].get_extra_state()}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(blob, f, ensure_ascii=False)


def import_voice(boxes: dict, path: str) -> dict:
    """export_voice で書いた JSON を、脳の声の係と言う語の係へ読む（体ありの太郎用。意味の辞書は読まない）。
    boxes は Brain.boxes でも、係の dict でもよい。返り値は meta（種・日・月齢）。"""
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    boxes["voice"].set_extra_state(blob["voice"])
    boxes["wordplan"].set_extra_state(blob["wordplan"])
    return blob.get("meta", {})


def _save_ckpt(path: str, state: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(state, f)
    os.replace(tmp, path)


def main(argv=None):
    torch.set_num_threads(1)   # 小さい計算ばかりなので1本の方が速い（4本：1日8.3秒→1本：6.2秒）
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--days", type=float, default=730)
    ap.add_argument("--start-day", type=float, default=0)
    ap.add_argument("--probe-days", type=float, default=7)
    ap.add_argument("--log-dir", default="段階7_声の試験場")
    ap.add_argument("--save", default=None)
    ap.add_argument("--checkpoint", default=None,
                    help="途中の太郎をここへ保存し（試しのたび）、あれば続きから回す（同じコマンドで再開できる）")
    ap.add_argument("--export", default=None, help="試しのたびに、体ありへ持っていく分を JSON で書く")
    a = ap.parse_args(argv)
    log_dir = os.path.join(REPO, "G", "logs", a.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"seed{a.seed}.jsonl")
    ckpt = (a.checkpoint if os.path.isabs(a.checkpoint) else os.path.join(REPO, a.checkpoint)) if a.checkpoint else None
    if ckpt and os.path.exists(ckpt):
        with open(ckpt, "rb") as f:
            st = pickle.load(f)
        bed, first, last_probe, wall0 = st["bed"], st["first"], st["last_probe"], st["wall_s"]
        torch.set_rng_state(st["torch_rng"])
        with open(path, encoding="utf-8") as f:       # 保存より後に書いた試しの行は捨てる（もう一度回るので）
            kept = [l for l in f if json.loads(l)["day"] <= bed.day]
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(kept)
        print(f"続きから：{bed.month:.2f}か月（{bed.day}日目）", flush=True)
        mode = "a"
    else:
        bed, first, last_probe, wall0, mode = Bed(a.seed, a.start_day), {}, -math.inf, 0.0, "w"
    t0 = time.time() - wall0
    with open(path, mode, encoding="utf-8") as f:
        while bed.day < a.start_day + a.days:
            bed.run_day()
            if bed.day - last_probe >= a.probe_days:
                last_probe = bed.day
                p = probe(bed)
                p["wall_s"] = round(time.time() - t0, 1)
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
                f.flush()
                for w in TEST_WORDS:
                    if p[f"know_{w}"] and f"know_{w}" not in first:
                        first[f"know_{w}"] = p["month"]
                    if p[f"say_{w}"] >= PASS_N and f"say_{w}" not in first:
                        first[f"say_{w}"] = p["month"]
                print(f"[{p['wall_s']:7.1f}s] {p['month']:5.2f}か月 わかる {[p['know_' + w] for w in TEST_WORDS]} "
                      f"言える {[p['say_' + w] for w in TEST_WORDS]} 帳面 {p['notebook_sounds']}音 "
                      f"足りない音 {p['has_sounds']} M {p['M_ぼおる'][:1]} {p['M_こっぷ'][:1]}", flush=True)
                if a.export:
                    export_voice(bed, a.export if os.path.isabs(a.export) else os.path.join(REPO, a.export),
                                 {"seed": a.seed, "day": bed.day, "month": round(bed.month, 3)})
                if ckpt:
                    bed.own_log = bed.own_log[-1000:]
                    _save_ckpt(ckpt, {"bed": bed, "first": first, "last_probe": last_probe,
                                      "wall_s": time.time() - t0, "torch_rng": torch.get_rng_state()})
    summary = {"seed": a.seed, "first_month": first, "stats": bed.stats, "wall_s": round(time.time() - t0, 1),
               "own_voices_sample": bed.own_log[-50:]}
    with open(os.path.join(log_dir, f"seed{a.seed}_まとめ.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    if a.save:
        save_state(bed, a.save if os.path.isabs(a.save) else os.path.join(REPO, a.save),
                   {"seed": a.seed, "day": bed.day, "first_month": first})
    print(json.dumps(first, ensure_ascii=False))


if __name__ == "__main__":
    main()
