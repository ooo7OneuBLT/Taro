"""盤面の世界で本体の脳（core_v2.brain.Brain、43係）を育てる。

1分ごとに：世界の様子を入口へ（体の内側の入口＝お腹、目の入口＝部屋にいる人。目は手書きの下駄）、
脳を rhythm.TICKS_PER_MIN 回進め、最後の回に声・体の出口から出た指令で世界を1分進める。
指令の「向けた相手の荷札」は、部屋にいる人の生まれつきの符号に一番似た人として世界へ渡す（下駄の出口側）。
下書き（core_v2/board/mind.py）と同じ集計ができるよう、board_stage1.grow と同じ形で返す。
"""

from __future__ import annotations

from functools import lru_cache

import torch

from core_v2.board.world import Board, MINUTES_PER_DAY, stage1_adults, stage2_adults
from core_v2.boxes.need import HUNGRY_FROM
from core_v2.brain import Brain
from core_v2.rhythm import TICKS_PER_MIN
from core_v2.tagging import word_tag


def new_brain(seed: int, frozen: bool = False) -> Brain:
    torch.manual_seed(10_000 + seed)
    brain = Brain(check_formats=False)
    brain.boxes["worth"].frozen = frozen
    return brain


@lru_cache(maxsize=None)
def _base(name: str):
    return word_tag("person:" + name)


def _who(tag, names):
    return max(names, key=lambda n: float(_base(n) @ tag))


def grow_core(rule: str, seed: int, days: int, frozen: bool = False, keep_log: bool = False,
              stage: int = 1, swap: bool = False, brain: Brain | None = None) -> dict:
    if stage == 2 and brain is None:
        brain = grow_core(rule, seed, days, frozen, stage=1)["brain"]
    adults = stage1_adults() if stage == 1 else stage2_adults(swap)
    board = Board(rule=rule, seed=seed + (500 if stage == 2 else 0), adults=adults)
    if brain is None:
        brain = new_brain(seed, frozen)
    else:
        torch.manual_seed(20_000 + seed)
    brain.reset()
    want, stress = brain.boxes["want"], brain.boxes["stress"]
    episodes, cur, log = [], [], []
    fed, came = False, None
    for t in range(days * MINUTES_PER_DAY):
        present = board.present()
        ext = {"gutin": {"fullness": board.fullness, "fed": fed},
               "eyein": {"people": [(n, n == came) for n in present]}}
        action, strength, target = "nothing", 0.0, None
        for _ in range(TICKS_PER_MIN):
            to_env = brain.tick(ext)
        voice, body = to_env.get("voiceout"), to_env.get("bodyout")
        if isinstance(voice, dict) and "cry" in voice:
            action, strength, target = "cry", voice["cry"], _who(voice["to"], present)
        elif isinstance(body, dict) and "reach" in body:
            action, strength, target = "reach", body["reach"], _who(body["to"], present)
        f0 = board.fullness
        obs = board.step(action, strength, target)
        fed, came = obs["fed"], obs["came"]
        if action != "nothing":
            cur.append((t, action, strength, target, tuple(present)))
        if keep_log:
            log.append({"t": t, "fullness": round(f0, 3), "action": action, "strength": round(strength, 3),
                        "target": target, "present": present, "busy": obs["busy"], "fed": fed,
                        "stress": round(stress.level, 3), "hungry": f0 < HUNGRY_FROM})
        if fed:
            if cur:
                episodes.append(cur)
            cur = []
    return {"episodes": episodes, "log": log, "brain": brain}
