"""段階8a：体なしの「ない」の試験場（仕様書 `2026-10-18_段階8_仕様書_ない.md` §2-5・付録）。

段階7の声の試験場（voice_bed）に、驚きの係（surprise）と、物の名簿（objfile）と、消える場面を足した。
* 本物の係：earin・listen・semantic・wordplan・voice・voiceout・surprise（SubBrain、二重バッファ）と、別に回す objfile
* 目の入口の代わりに、物の (荷札, 場所の山) と目の動きの地図と目立ちの地図を手で objfile へ渡す（カメラ・体は無い）。
  objfile の出力（→semantic・→surprise）をそのまま注ぎ込む。物の見た目は voice_bed と同じ種の乱数、
  でんしゃ・とけい（教えていない物）は別の種
* 物の場面の終わり方（親の場面。太郎へのほうびではない）：
  - 消える（P_VANISH）：太郎が見ている物が急に無くなる。VANISH_BEFORE_S 秒後に親が「○○ないね」、P_ALONE の割合で「ない」だけ
    （前の太郎の親 run/world/parent.py の vanish_template・vanish_gap_before_sec・vanish_gap_after_sec と同じ数）
  - 対照（親は「ない」と言わない）：親が見えるように動かして視野の外へ持っていく／太郎が目をそらして物が視野から出る（半々）
* 「ない」は何にも手で結び付けない。意味の辞書の学習（場面をまたいだ数え上げ）だけで結び付く

試し（試験場の太郎を写して、覚える・喃語を止めて回す。1回数秒）：
* A：知っている物が消える20回で驚きが立つ（u＞0）回数、対照20回（そのまま・目をそらす・持っていく）で立つ回数
* B：教えていない物（でんしゃ・とけい）が消えて1秒後（親が言うころ）、逆に引いた Wᵀ(a＋u·S) の一番近い知っている語が
  「ない」で、強さが RECALL_MIN 以上の回数（20回）。知っている物が消えたときに何が一番かも記録
* C（育ちの記録から）：太郎が言う語の係から「ない」を言った（M を使って言った）回のうち、物が消えてから3秒以内の割合。
  喃語の語の選び方（聞いた割合）で「ない」を言った回は別に数える
* 段階7の試し（voice_bed.probe：わかる・言える ぼおる・こっぷ）もそのまま回す（③）

使い方：
  python -m core_v2.tools.nai_bed [--seed 0] [--days 550] [--probe-days 14] [--surprise on|off] [--centre on|off] [--enc on|off]
                                  [--log-dir 段階8_ない] [--checkpoint …]
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

from ..boxes import box_class
from ..boxes.eyein import decode_ego, ego_map
from ..boxes.objfile import GRID, Objfile, bump_at
from ..boxes.semantic import RECALL_MIN
from ..boxes.voice import SIM_MIN
from ..formats import DTYPE, TAG_DIM
from .. import mouth
from ..tagging import appearance_tag, word_tag
from . import voice_bed as vb
from .parent_speech import P_ALONE, UNHEARD, WORDS

NAI = "ない"
BOXES = vb.BOXES + ["surprise"]
TICK_S = vb.TICK_S
CENTRE = (GRID - 1) / 2.0

P_VANISH = 0.1          # 物を見せる場面のうち、物が消えて終わる割合（切りのよい値。消える場面の回数の文献の値は無い）［Tier3・仮置き］
VANISH_BEFORE_S = 1.0   # 消えてから親が言うまで（run/world/parent.py の vanish_gap_before_sec）
VANISH_AFTER_S = 1.5    # 言ってから次まで（同 vanish_gap_after_sec）
MOVE_CELLS = 1.0        # 親が見えるように持っていく速さ（1回1マス＝約21°/秒。名簿が追える TRACK_CELLS 1.5 より遅い）
LOOK_AWAY_PX = -160.0   # 太郎が目をそらす跳び（画素。目の動きの地図で約19マス＝物が視野の外へ出る）
LOOK_AWAY_S = 2.0       # 目をそらしている間（そのあいだに親が物を片付け、目は戻る）
C_WINDOW_S = 3.0        # 試験 C：消えてからこの秒数以内（仕様書の合格の線）
N_TRIALS = 20

_Z2 = torch.zeros(2)
_ZTAG = torch.zeros(TAG_DIM, dtype=DTYPE)
_ZMAP = torch.zeros(GRID, GRID, dtype=DTYPE)
_EGO0 = ego_map(0.0, 0.0)
_EGO_AWAY = ego_map(LOOK_AWAY_PX, 0.0)
_EGO_BACK = ego_map(-LOOK_AWAY_PX, 0.0)
_AWAY_DC = decode_ego(_EGO_AWAY)[1]
_BACK_DC = decode_ego(_EGO_BACK)[1]


def untaught_tags(seed: int = 8) -> dict[str, torch.Tensor]:
    """教えていない物（でんしゃ・とけい）の見た目。知っている物（種7）と別の種。"""
    g = np.random.default_rng(seed)
    return {w: appearance_tag(g.standard_normal(384)) for w in UNHEARD}


def make_boxes(seed: int, learning: bool = True, surprise: bool = True, centre: bool = True, enc: bool = True) -> dict:
    b = vb.make_boxes(seed, learning)
    b["surprise"] = box_class("surprise")(enabled=surprise, use_centre=centre, use_enc=enc)
    return b


def closest_word(voice: str) -> tuple[str, float]:
    return max(((w, mouth.similarity(w, voice)) for w in WORDS + [NAI]), key=lambda x: x[1])


class Eye:
    """目の入口の代わり：物（荷札・場所・見えるか）から、objfile へ渡す入力を作り、objfile を1回進める。"""

    def __init__(self) -> None:
        self.of = Objfile()
        # 試験場には目を向ける反射が無いので、名簿の→orienteye の列は作らない（速くするだけ。中身は変わらない）
        self.of.outputs = [e for e in self.of.outputs if e.dst != "orienteye"]
        self.objs: list[dict] = []     # {"tag", "r", "c", "visible"}
        self.ego = _EGO0

    def busy(self) -> bool:
        return bool(self.objs or self.of.rows)

    def saccade(self, ego_m) -> None:
        self.ego = ego_m
        dc = decode_ego(ego_m)[1]
        for o in self.objs:
            o["c"] += dc

    def step(self) -> tuple[list, list]:
        if not self.busy():
            return [], []
        vis = [o for o in self.objs if o["visible"] and 0.0 <= o["c"] <= GRID - 1 and 0.0 <= o["r"] <= GRID - 1]
        eyein = [(o["tag"], bump_at(o["r"], o["c"], sigma=1.0)) for o in vis] + [(_ZTAG, self.ego)]
        sal = _ZMAP
        for o in vis:
            sal = torch.maximum(sal, bump_at(o["r"], o["c"]))
        out = self.of.step({"eyein": eyein, "orienteye": _Z2, "prio": sal})
        self.ego = _EGO0
        return out["semantic"], out["surprise"]


class NaiBed(vb.Bed):
    def __init__(self, seed: int, start_day: float = 0.0, surprise: bool = True, centre: bool = True,
                 enc: bool = True) -> None:
        super().__init__(seed, start_day)
        self.opts = {"surprise": surprise, "centre": centre, "enc": enc}
        torch.manual_seed(seed)
        self.boxes = make_boxes(seed, True, surprise, centre, enc)
        self.brain = vb.SubBrain(self.boxes)
        self.eye = Eye()
        self.scene = None
        self.stats.update({"vanish": 0, "move": 0, "look": 0, "nai_said": 0})
        self.vanish_vt: list[int] = []        # 消えた回（声の係の時計）
        self.events: list[tuple] = []         # (月齢, 種類 decide/babble, 語, 声, 直前の消えから何回)

    def _reset_play(self):
        super()._reset_play()
        if "eye" in self.__dict__:            # 1日の終わり（眠る）：見えていた物も驚きも持ち越さない
            self.scene = None
            self.eye = Eye()
            self.boxes["surprise"].reset()

    # ---- 場面 ----
    def _start_scene(self, t: int, w: str, utt: str) -> None:
        end = t + int(vb.SCENE_S / TICK_S)
        r = self.rng.random()
        kind = "vanish" if r < P_VANISH else ("move" if self.rng.random() < 0.5 else "look")
        self.eye.objs = [{"tag": self.objs[w], "r": CENTRE, "c": CENTRE, "visible": True}]
        self.pending_say.append((t + int(vb.NAME_AT_S / TICK_S), utt))
        self.scene = {"w": w, "kind": kind, "end": end, "phase": "show", "until": end}
        self.stats["scene"] += 1

    def _scene_step(self, t: int) -> None:
        sc = self.scene
        if sc is None or t < sc["until"] and sc["phase"] != "move":
            return
        o = self.eye.objs[0] if self.eye.objs else None
        if sc["phase"] == "show":
            if sc["kind"] == "vanish":
                o["visible"] = False
                self.vanish_vt.append(self.boxes["voice"].t)
                say = NAI if self.rng.random() < P_ALONE else f"{sc['w']}ないね"
                at = t + int(VANISH_BEFORE_S / TICK_S)
                self.pending_say.append((at, say))
                sc.update(phase="after", until=at + int(VANISH_AFTER_S / TICK_S))
            elif sc["kind"] == "move":
                sc.update(phase="move")
            else:
                self.eye.saccade(_EGO_AWAY)
                sc.update(phase="away", until=t + int(LOOK_AWAY_S / TICK_S))
            self.stats[sc["kind"]] += 1
        elif sc["phase"] == "move":
            o["c"] += MOVE_CELLS
            if o["c"] > GRID - 1:
                self.eye.objs = []
                sc.update(phase="after", until=t)
        elif sc["phase"] == "away":
            self.eye.objs = []                 # 目をそらしている間に親が片付けた
            self.eye.saccade(_EGO_BACK)
            sc.update(phase="after", until=t)
        elif sc["phase"] == "after":
            self.eye.objs = []
            self.scene = None

    # ---- 1回 ----
    def step(self, heard_extra: str = ""):
        t = self.play_t
        text = heard_extra
        if t >= self.next_talk and self._may_speak() and self.scene is None:
            utt, w = vb.make_utterance(self.rng)
            self.stats["talk"] += 1
            if w is not None and self.rng.random() < vb.P_JOINT:
                self._start_scene(t, w, utt)
            else:
                self.pending_say.append((t, utt))
            self.next_talk = t + self._gap()
        self._scene_step(t)
        due = [s for s in self.pending_say if s[0] <= t]
        if due and self._may_speak():
            self.pending_say = [s for s in self.pending_say if s[0] > t]
            text = " ".join([text] + [s for _, s in due]).strip()
        sem, sur = self.eye.step()
        inject = {("objfile", "semantic"): sem, ("objfile", "surprise"): sur}
        heard = " ".join(x for x in (self._echo, text) if x)
        if heard or sem:
            self.last_input_t = t
        to_env = self.brain.tick({"earin": heard, "voiceout": self.month}, inject)
        self._echo = ""
        ch = to_env.get("voiceout", "")
        if ch:
            self._echo = ch
            self.cur_voice += ch
            self.last_char_t = t
        elif self.cur_voice and t - self.last_char_t > vb.END_GAP_TICKS:
            self._voice_done(self.cur_voice, None)
            self.cur_voice = ""
        self.play_t += 1
        self.stats["ticks"] += 1

    def _voice_done(self, voice: str, look):
        """親が応える（段階7と同じ、物の名前の中で一番近い語を1語）。「ない」は応えの語に入れない
        （物が無いわけでもないのに、喃語に「ない」と応える親はいない。入れると短い喃語がみな「ない」に近く、
        1回目の確かめで3日目に「ああ」→「ない」の応えが M を315まで育てた）。"""
        self.stats["own"] += 1
        w, s = vb.closest_word(voice)
        if self.rng.random() < vb.RESP_PROB * (0.5 + s):
            self.pending_say.append((self.play_t + int(vb.RESP_DELAY_S / TICK_S), w))
            self.stats["resp"] += 1
        if len(self.own_log) < 200000:
            self.own_log.append((round(self.month, 3), voice, w if s > 0 else ""))

    def _quiet(self) -> bool:
        return (super()._quiet() and self.scene is None and not self.eye.busy()
                and not self.boxes["surprise"].busy())

    def run_play(self, seconds: float):
        end = self.play_t + int(seconds / TICK_S)
        v = self.boxes["voice"]
        while self.play_t < end:
            if self.skip_idle and self._quiet():
                n = self._next_event(end) - self.play_t - 1
                if n > 0:
                    v.skip(n)
                    self.boxes["semantic"].skip(n, [])
                    self.play_t += n
                    continue
            self.step()
        for e in v.log:
            if e["event"] == "babble":
                self.stats["babble"] += 1
                self.stats["babble_word"] += bool(e["word"])
                if e["word"] == NAI:
                    self._event("babble", NAI, "", e["t"])
            elif e["event"] == "decide" and e.get("say") and e["word"] == NAI:
                self._event("decide", NAI, e["voice"], e["t"])
        v.log.clear()
        self.vanish_vt = self.vanish_vt[-50:]

    def _event(self, kind, word, voice, vt):
        past = [x for x in self.vanish_vt if x <= vt]
        since = (vt - past[-1]) if past else None
        self.events.append((round(self.month, 3), kind, word, voice, since))
        if kind == "decide":
            self.stats["nai_said"] += 1


# ---------------------------------------------------------------- 試し
def _probe_copy(bed: NaiBed) -> dict:
    b = copy.deepcopy(bed.boxes)
    for k in ("listen", "semantic", "voice"):
        b[k].learning = False
    b["voice"].babbling = False
    for x in b.values():
        x.reset()
    return b


def run_trial(b: dict, tag, kind: str, month: float, show_s: float = vb.SCENE_S, after_s: float = 3.0,
              read_at_s: float = VANISH_BEFORE_S) -> dict:
    """1回の場面を、話しかけ無しで回す。kind：vanish・stay・move・look。
    戻り値：驚きの最大、消えて read_at_s 秒後の逆引き（語→cos）とその強さ、出た声。"""
    brain = vb.SubBrain(b)
    eye = Eye()
    eye.objs = [{"tag": tag, "r": CENTRE, "c": CENTRE, "visible": True}]
    n_show, n_after = int(show_s / TICK_S), int(after_s / TICK_S)
    sem_box, sur_box = b["semantic"], b["surprise"]
    out = {"u_max": 0.0, "voice": "", "recall": None, "recall_norm": 0.0}
    for k in range(n_show + n_after + 30):
        if k == n_show:
            if kind == "vanish":
                eye.objs[0]["visible"] = False
            elif kind == "look":
                eye.saccade(_EGO_AWAY)
        if k > n_show and kind == "move" and eye.objs:
            eye.objs[0]["c"] += MOVE_CELLS
            if eye.objs[0]["c"] > GRID - 1:
                eye.objs = []
        sem, sur = eye.step()
        env = brain.tick({"earin": "", "voiceout": month}, {("objfile", "semantic"): sem, ("objfile", "surprise"): sur})
        if n_show <= k < n_show + n_after:
            out["u_max"] = max(out["u_max"], sur_box.level)
            out["voice"] += env.get("voiceout", "")
        if k == n_show + int(read_at_s / TICK_S):
            x = sem_box._seen_vec(sem, sem_box._u)
            if x is not None:
                r = sem_box.W.T @ x
                out["recall_norm"] = float(torch.linalg.vector_norm(r))
                out["recall"] = r
    return out


def best_word(r, words) -> str:
    if r is None or not words:
        return ""
    return max(words, key=lambda w: float(word_tag(w) @ r))


def probe(bed: NaiBed) -> dict:
    month = bed.month
    out = {"month": round(month, 3), "day": bed.day, "opts": bed.opts}
    known = sorted(bed.boxes["listen"].known())
    out["nai_known"] = NAI in known
    # A：消える20回・対照20回
    b = _probe_copy(bed)
    van = [run_trial(b, bed.objs[WORDS[i % len(WORDS)]], "vanish", month) for i in range(N_TRIALS)]
    ctrl_kinds = ["stay", "look", "move"]
    ctl = [run_trial(b, bed.objs[WORDS[i % len(WORDS)]], ctrl_kinds[i % 3], month) for i in range(N_TRIALS)]
    out["A_vanish"] = sum(x["u_max"] > 0 for x in van)
    out["A_control"] = sum(x["u_max"] > 0 for x in ctl)
    out["A_u_mean"] = round(sum(x["u_max"] for x in van) / N_TRIALS, 3)
    # B：教えていない物が消える
    un = untaught_tags()
    bt = [run_trial(b, un[UNHEARD[i % 2]], "vanish", month) for i in range(N_TRIALS)]
    bw = [best_word(x["recall"], known) for x in bt]
    out["B_best_nai"] = sum(w == NAI for w in bw)
    out["B_pass_n"] = sum(w == NAI and x["recall_norm"] >= RECALL_MIN for w, x in zip(bw, bt))
    out["B_best_words"] = {w: bw.count(w) for w in set(bw)}
    # 知っている物が消えたとき、何が一番か（試験 C の見込み）
    kw = [best_word(x["recall"], known) for x in van]
    out["known_vanish_best"] = {w: kw.count(w) for w in set(kw)}
    # 消えたときに太郎が出した声（言う語の係から）
    vv = [x["voice"] for x in van + bt if x["voice"]]
    out["voices_on_vanish"] = sorted(set(vv))[:8]
    out["n_voice_on_vanish"] = len(vv)
    out["n_nai_voice_on_vanish"] = sum(closest_word(v)[0] == NAI and closest_word(v)[1] > SIM_MIN for v in vv)
    sv = [x["voice"] for x in ctl if x["voice"]]
    out["n_nai_voice_on_control"] = sum(closest_word(v)[0] == NAI and closest_word(v)[1] > SIM_MIN for v in sv)
    v = bed.boxes["voice"]
    out["M_nai"] = sorted(v.M.get(NAI, {}).items(), key=lambda kv: -kv[1])[:3]
    out["has_sounds_nai"] = [c for c in NAI if c not in v.known_sounds()]
    # 段階7の試し（③）
    p7 = vb.probe(bed)
    for k in ("know_ぼおる", "know_こっぷ", "say_ぼおる", "say_こっぷ", "voices_ぼおる"):
        out["s7_" + k] = p7[k]
    # C（育ちの記録、この試しまでの間）
    ev = bed.events
    dec = [e for e in ev if e[1] == "decide"]
    win = int(C_WINDOW_S / TICK_S)
    out["C_decide_nai"] = len(dec)
    out["C_decide_in_window"] = sum(e[4] is not None and e[4] <= win for e in dec)
    bab = [e for e in ev if e[1] == "babble"]
    out["C_babble_nai"] = len(bab)
    out["C_babble_in_window"] = sum(e[4] is not None and e[4] <= win for e in bab)
    bed.events = []
    out["stats"] = dict(bed.stats)
    return out


def _save_ckpt(path: str, state: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(state, f)
    os.replace(tmp, path)


def main(argv=None):
    torch.set_num_threads(1)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--days", type=float, default=550)
    ap.add_argument("--probe-days", type=float, default=14)
    ap.add_argument("--surprise", default="on")
    ap.add_argument("--centre", default="on")
    ap.add_argument("--enc", default="on")
    ap.add_argument("--log-dir", default="段階8_ない")
    ap.add_argument("--tag", default="")
    ap.add_argument("--checkpoint", default=None)
    a = ap.parse_args(argv)
    on = lambda s: s == "on"
    name = f"seed{a.seed}{a.tag}"
    log_dir = os.path.join(vb.REPO, "G", "logs", a.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{name}.jsonl")
    ckpt = (a.checkpoint if os.path.isabs(a.checkpoint) else os.path.join(vb.REPO, a.checkpoint)) if a.checkpoint else None
    if ckpt and os.path.exists(ckpt):
        with open(ckpt, "rb") as f:
            st = pickle.load(f)
        bed, last_probe, wall0 = st["bed"], st["last_probe"], st["wall_s"]
        torch.set_rng_state(st["torch_rng"])
        with open(path, encoding="utf-8") as f:
            kept = [l for l in f if json.loads(l)["day"] <= bed.day]
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(kept)
        mode = "a"
    else:
        bed = NaiBed(a.seed, surprise=on(a.surprise), centre=on(a.centre), enc=on(a.enc))
        last_probe, wall0, mode = -math.inf, 0.0, "w"
    t0 = time.time() - wall0
    with open(path, mode, encoding="utf-8") as f:
        while bed.day < a.days:
            bed.run_day()
            if bed.day - last_probe >= a.probe_days:
                last_probe = bed.day
                p = probe(bed)
                p["wall_s"] = round(time.time() - t0, 1)
                f.write(json.dumps(p, ensure_ascii=False, default=str) + "\n")
                f.flush()
                print(f"[{p['wall_s']:7.1f}s] {p['month']:5.2f}か月 ない知る {p['nai_known']} A {p['A_vanish']}/{p['A_control']} "
                      f"B {p['B_pass_n']}({p['B_best_nai']}) 知った物 {p['known_vanish_best']} "
                      f"C 言う {p['C_decide_in_window']}/{p['C_decide_nai']} 喃語 {p['C_babble_in_window']}/{p['C_babble_nai']} "
                      f"声 {p['voices_on_vanish'][:3]} 段階7 {p['s7_say_ぼおる']},{p['s7_say_こっぷ']}", flush=True)
                if ckpt:
                    bed.own_log = bed.own_log[-1000:]
                    _save_ckpt(ckpt, {"bed": bed, "last_probe": last_probe, "wall_s": time.time() - t0,
                                      "torch_rng": torch.get_rng_state()})


if __name__ == "__main__":
    main()
