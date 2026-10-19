"""盤面の世界で本体の脳（core_v2.brain.Brain、43係）を育てる。

1分ごとに：世界の様子を入口へ（体の内側の入口＝お腹、目の入口＝部屋にいる人。目は手書きの下駄）、
脳を rhythm.TICKS_PER_MIN 回進め、最後の回に声・体の出口から出た指令で世界を1分進める。
指令の「向けた相手の荷札」は、部屋にいる人の生まれつきの符号に一番似た人として世界へ渡す（下駄の出口側）。
下書き（core_v2/board/mind.py）と同じ集計ができるよう、board_stage1.grow と同じ形で返す。
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

import torch

from core_v2.board.world import Board, MINUTES_PER_DAY, stage1_adults, stage2_adults
from core_v2.boxes.need import HUNGRY_FROM
from core_v2.brain import Brain
from core_v2.rhythm import TICKS_PER_MIN
from core_v2.boxes.earin import normalize_kana
from core_v2.tagging import word_tag

SPEECH = os.path.join(os.path.dirname(__file__), os.pardir, "data", "盤面_親の言葉.json")


@lru_cache(maxsize=1)
def speech_data() -> dict:
    with open(SPEECH, encoding="utf-8") as f:
        return json.load(f)


def use_language(brain: Brain) -> None:
    """3段目：耳と口の下駄を入れ、言う語を組み立てる係を盤面の使い方にする。"""
    d = speech_data()
    brain.boxes["listen"].use_board_ear(d["ear"])
    brain.boxes["wordplan"].use_board()
    words = {normalize_kana(w) for ws in d["ear"].values() for w in ws}
    mouth = {normalize_kana(k): v for k, v in d["mouth"].items()}
    brain.boxes["voiceout"].use_board_mouth({w: mouth.get(w, w) for w in words})


def new_brain(seed: int, frozen: bool = False) -> Brain:
    torch.manual_seed(10_000 + seed)
    brain = Brain(check_formats=False)
    brain.boxes["worth"].frozen = frozen
    return brain


@lru_cache(maxsize=None)
def _base(name: str):
    """揺れのない生まれつきの符号（"obj:" で始まれば物、それ以外は人）。記録用。"""
    return word_tag(name if name.startswith("obj:") else "person:" + name)


def _who(j, names):
    return names[j] if 0 <= j < len(names) else None


OBJECTS = ["ボール", "コップ", "くつ", "わんわん", "くるま", "バナナ"]


def grow_core(rule: str, seed: int, days: int, frozen: bool = False, keep_log: bool = False,
              stage: int = 1, swap: bool = False, brain: Brain | None = None,
              understand_words: bool = True, answer_questions: bool = True) -> dict:
    """stage 1：ママだけ。2：ママとパパ（brain が無ければ1段目を同じ日数育ててから）。
    3：2段目の世界で大人が話し、言葉に応える（brain は2段目まで育った脳を渡す）。
    4：3段目の世界の目の前に物が置かれ、大人は問う語に名前で答える（brain は3段目まで育った脳）。
    4段目は、おなかがすいていない1分ごとに (分, 目の前の物, 名前を知っていたか, 行動) を "curious" に記録する。"""
    if stage == 2 and brain is None:
        brain = grow_core(rule, seed, days, frozen, stage=1)["brain"]
    adults = stage1_adults() if stage == 1 else stage2_adults(swap)
    board = Board(rule=rule, seed=seed + 500 * (stage - 1), adults=adults,
                  speech=speech_data()["lines"] if stage >= 3 else None, understand_words=understand_words,
                  objects=OBJECTS if stage >= 4 else None, answer_questions=answer_questions)
    if brain is None:
        brain = new_brain(seed, frozen)
    else:
        torch.manual_seed(20_000 + seed)
    if stage >= 3:
        use_language(brain)
    if stage >= 4:
        brain.boxes["semantic"].use_board()
    sem = brain.boxes["semantic"]
    curious = []
    brain.reset()
    stress = brain.boxes["stress"]
    episodes, cur, log = [], [], []
    fed, came, heard, spoke = False, None, None, None
    for t in range(days * MINUTES_PER_DAY):
        present = board.present()
        ext = {"gutin": {"fullness": board.fullness, "fed": fed},
               "eyein": {"people": [(n, n in (came, spoke)) for n in present], "object": board.object}}
        action, strength, target, word = "nothing", 0.0, None, None
        for k in range(TICKS_PER_MIN):
            if k == 0 and heard:
                ext["earin"] = heard        # 親の一言は1分に1回、最初の回だけ耳に届く
            elif "earin" in ext:
                del ext["earin"]
            to_env = brain.tick(ext)
        voice, body = to_env.get("voiceout"), to_env.get("bodyout")
        if isinstance(voice, dict) and "cry" in voice:
            action, strength, target = "cry", voice["cry"], _who(voice["to"], present)
        elif isinstance(voice, dict) and "say" in voice:
            action, strength, target, word = "say", voice["strength"], _who(voice["to"], present), voice["word"]
        elif isinstance(body, dict) and "reach" in body:
            action, strength, target = "reach", body["reach"], _who(body["to"], present)
        if target is None:
            action = "nothing"
        f0 = board.fullness
        obs = board.step(action, strength, target, word)
        if stage >= 4 and board.object and f0 >= HUNGRY_FROM:
            k = sem.knowing(_base("obj:" + board.object))
            curious.append((t, board.object, k >= 1.0, action if action != "say" else "say:" + word))
        fed, came, heard, spoke = obs["fed"], obs["came"], obs["heard"], obs["spoke"]
        if action != "nothing":
            cur.append((t, action if action != "say" else "say:" + word, strength, target, tuple(present)))
        if keep_log:
            log.append({"t": t, "fullness": round(f0, 3), "action": action, "strength": round(strength, 3),
                        "target": target, "present": present, "busy": obs["busy"], "fed": fed,
                        "stress": round(stress.level, 3), "hungry": f0 < HUNGRY_FROM, "heard": heard})
        if fed:
            if cur:
                episodes.append(cur)
            cur = []
    return {"episodes": episodes, "log": log, "brain": brain, "curious": curious}
