"""家の場面の親：「見えない親の手」（follow-in。旧太郎のよい親の決まりを引き継ぐ）。

親は太郎の見ている物に合わせる（follow-in labeling：Tomasello・Farrar 1986。子が見ている物に親が名前を言うと、
子の注意を移させて言うより語がよく残る。旧太郎 parent.py の「見ている物に合わせて言う」と同じ考え）：
* 太郎がある物を BRING_AFTER_S（1秒）続けて見たら（目の真ん中に写っている物＝環境の正解 centre_object）、
  親がその物を手に取り、目の前 HOLD_DIST_M（0.3m）、そのときの視線の先へ持ってくる（TeleportObject、forceKinematic）
* 持っている間は、太郎から見て左右に ±SHAKE_M（3cm）、SHAKE_HZ（2回/秒）でやさしく振る
* 持ってきて NAME_AT_S（0.5秒）で名前を言う（P_NAME＝1.0：この世界では毎回言う）。言い方は段階5の話しかけと同じ
  （noun_scene.naming_utterance：1割は語だけ、残りは文の型）。そのあと太郎が見続けていれば REPEAT_GAP_S（2秒）ごとに
  最大 REPEAT（2）回言い直す（旧太郎の連呼 repeat_labels。noun_scene と同じ）
* HOLD_S（5秒）たったら元の場所・向きへ戻す。戻した物は COOLDOWN_S（3秒）は持ってこない（戻したとたんにまた見て、
  同じ物ばかり行き来しないため）［仮置き］
* 親の手そのものは描かない（AI2-THOR に手が無い。物だけが浮いて動く）。物の音はまだ無い（音の素が無い。object_sound の口だけある）

語の表（TYPE_WORD）：AI2-THOR の物の種類（objectType。unity/Assets/Scripts/SimObjType.cs で確かめた名前）→ 子どもの語。
台所（FloorPlan1〜30）と居間（FloorPlan201〜230）にある、拾える小さな物から10語。目だけの場面の20語と重なる語
（りんご・ぱん・こっぷ・おさら・すぷうん・ぼおる・くま）は同じかなで書く（耳がもう知っている語）。ほん・りもこん・まくらは新しい語。
くま（TeddyBear）・ぼおる（BasketBall）は寝室に多い（台所・居間に無ければ、その場面では出てこないだけ）。
Mug と Cup はどちらも「こっぷ」（子どもに向けては同じ語で呼ぶ）［Tier3・モデル上の選択］
"""

from __future__ import annotations

import math
import random

import numpy as np

TYPE_WORD = {
    "Apple": "りんご", "Bread": "ぱん", "Cup": "こっぷ", "Mug": "こっぷ", "Plate": "おさら", "Spoon": "すぷうん",
    "Book": "ほん", "RemoteControl": "りもこん", "Pillow": "まくら", "TeddyBear": "くま", "BasketBall": "ぼおる",
}
WORDS = sorted(set(TYPE_WORD.values()), key=list(TYPE_WORD.values()).index)

TICK_S = 0.1
BRING_AFTER_S = 1.0
HOLD_DIST_M = 0.3
SHAKE_M = 0.03
SHAKE_HZ = 2.0
NAME_AT_S = 0.5
P_NAME = 1.0
REPEAT = 2
REPEAT_GAP_S = 2.0
HOLD_S = 5.0
COOLDOWN_S = 3.0
MIN_Y_M = 0.05          # 持ってくる所が床より下にならないように


def naming_utterance(word: str, rng: random.Random) -> str:
    from ..world_eye.noun_scene import naming_utterance as nu
    return nu(word, rng)


class FollowInParent:
    """0.1秒ごとに tick(env) を呼ぶ親（上の説明）。"""

    def __init__(self, rng: random.Random, p_name: float = P_NAME, repeat: int = REPEAT) -> None:
        self.rng = rng
        self.p_name, self.repeat = p_name, repeat
        self.stats = {"brings": 0, "names": 0, "repeats": 0, "puts": 0}
        self.named: dict[str, int] = {}
        self.reset()

    def reset(self) -> None:
        self.looking = None        # 太郎が今見ている物（親から見た正解）
        self.look_ticks = 0
        self.held = None           # 持っている物の id
        self.d = 0                 # 持ってからの回数
        self.say = False
        self.base = None           # 振る真ん中
        self.axis = None           # 振る向き（太郎から見て左右）
        self.cool: dict[str, int] = {}
        self.word = ""             # 持っている物の語

    def tick(self, env) -> dict:
        """戻り値：その回に親がしたこと {"looking": 語, "said": 文, "brought": 語, "put": 語, "holding": 語}。"""
        oid = env.centre_object()
        if oid != self.looking:
            self.looking, self.look_ticks = oid, 0
        if oid is not None:
            self.look_ticks += 1
        did = {"looking": env.word_of(oid)}
        for k in list(self.cool):
            self.cool[k] -= 1
            if self.cool[k] <= 0:
                del self.cool[k]
        if self.held is None:
            if (oid is not None and oid not in self.cool
                    and self.look_ticks >= int(round(BRING_AFTER_S / TICK_S))):
                self._bring(env, oid)
                did["brought"] = env.word_of(oid)
            return did
        # 持っている間：振る・言う・戻す
        self.d += 1
        word = env.word_of(self.held)
        did["holding"] = word
        if self.d >= int(round(HOLD_S / TICK_S)):
            env.put_back(self.held, cause="parent_put")
            self.cool[self.held] = int(round(COOLDOWN_S / TICK_S))
            self.stats["puts"] += 1
            did["put"] = word
            self.held = None
            self.word = ""
            return did
        t = self.d * TICK_S
        env.hold(self.held, self.base + SHAKE_M * math.sin(2.0 * math.pi * SHAKE_HZ * t) * self.axis, cause="parent_shake")
        at, gap = int(round(NAME_AT_S / TICK_S)), int(round(REPEAT_GAP_S / TICK_S))
        k = (self.d - at) // gap
        if self.say and self.d >= at and (self.d - at) % gap == 0 and k <= self.repeat \
                and (k == 0 or oid == self.held):
            text = naming_utterance(word, self.rng)
            env.say(text, target=self.held, cause="parent_name")
            self.stats["names"] += 1
            self.stats["repeats"] += k > 0
            self.named[word] = self.named.get(word, 0) + 1
            did["said"] = text
        return did

    def _bring(self, env, oid: str) -> None:
        g = env.gaze_dir()
        base = np.asarray(env.eye_pos, dtype=float) + HOLD_DIST_M * g
        base[1] = max(float(base[1]), MIN_Y_M)
        from .env import cam_basis
        self.base = base
        self.axis = cam_basis(env.cam_yaw, env.cam_pitch)[1]
        self.held, self.d = oid, 0
        self.word = env.word_of(oid)
        self.say = self.rng.random() < self.p_name
        env.hold(oid, base, cause="parent_bring")
        env._log("bring", oid, self.word, cause="parent")
        self.stats["brings"] += 1


class LookLog:
    """太郎が物を見続けた時間（目の真ん中に同じ物が写っていた回の続き。環境の正解）。0.1秒ごとに add(id, 語)。"""

    def __init__(self) -> None:
        self.runs: list[tuple[str, float]] = []    # (語, 秒)
        self.cur = None
        self.n = 0
        self.word = ""

    def add(self, oid, word: str) -> None:
        if oid != self.cur:
            self.flush()
            self.cur, self.word, self.n = oid, word, 0
        if oid is not None:
            self.n += 1

    def flush(self) -> None:
        if self.cur is not None and self.n > 0:
            self.runs.append((self.word, round(self.n * TICK_S, 1)))
        self.cur, self.n = None, 0

    def summary(self) -> dict:
        d = [s for _w, s in self.runs]
        by: dict[str, list] = {}
        for w, s in self.runs:
            by.setdefault(w, []).append(s)
        return {"looks": len(d), "mean_s": round(float(np.mean(d)), 2) if d else 0.0,
                "median_s": round(float(np.median(d)), 2) if d else 0.0, "max_s": max(d) if d else 0.0,
                "n_ge_1s": sum(s >= 1.0 for s in d), "total_s": round(sum(d), 1),
                "by_word": {w: {"looks": len(v), "total_s": round(sum(v), 1), "max_s": max(v)} for w, v in by.items()}}
