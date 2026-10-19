"""「欲しい」を行動へ押す係 〔DA・OXT〕（欲求）。

盤面の世界（2026-10-19、core_v2/board/mind.py の下書きから清書）：何をするか決める係。
* vars→want：不快（お腹がすいた強さ）。feel→want：いら立ち
* worth→want：行動ごとの見込み（行動の荷札, 地図の [r, j]＝j 番目の人に向けた時の、行 r の欲求の見込み。0＝選べない）
* semantic→want：目の前の物のわからなさ（4段目）。知りたい強さ＝CURIOSITY×わからなさ
* inner→want：いや（満腹なのにミルクが口に来る強さ、5段目）
* social→want：会いたかった人が戻ってきた強さ（5段目）。相手に向かいたい強さ＝SOCIAL×その強さ
* 不快＋いら立ち＋知りたい強さが高いほど何か行動しやすい（その割合で行動し、残りは何もしない）
* 行動するときは、まずどの欲求のために動くかを「強さ×その欲求で一番見込みのある行動の見込み」に比例して選び
  （ミルク：不快＋いら立ち、わかる：知りたい強さ、いや：いやの強さ、相手：相手に向かいたい強さ。見込みを掛けないと、
  3段目で人が部屋に戻るたびに「相手」の欲求で泣く・手を伸ばすを半々に選び、まんまを言う割合が下がった）、
  次に選択肢（行動×向ける相手。部屋にいる人だけ）を、その欲求の見込みに比例して選ぶ（確率マッチング。
  欲求ごとに見込みを持ち、体の内側の状態で使い分ける。Keramati & Gutkin 2014。知りたい強さが0なら3段目までと同じ）
* 合図の強さ＝BASE_STRENGTH＋0.5×不快＋いら立ち＋ミルク以外の欲求の強さ（1まで）
* want→cry・want→reach・want→wordplan（語を言う。3段目）：(行動の荷札, 地図の [0, j]＝強さ。j＝向ける人の場所)
* want→worth：選んだ行動の写し（行動の荷札, 地図の [0, j]＝1、[r, j]＝行 r の欲求のために選んだなら1（ミルクは無し）。
  その欲求の見込みだけを覚える）
1分に1回、rhythm.WANT_PHASE の回に選ぶ。乱数は torch の大域の乱数。生まれつき（学習は worth の側）。
"""

import torch

from ..formats import DTYPE, GRID
from ..rhythm import WANT_PHASE, phase
from ..tagging import word_tag
from .base import Box

BASE_STRENGTH = 0.3
CURIOSITY = 0.3                # わからなさ1のときの知りたい強さ（おなかの不快1より弱い）［仮置き］
SOCIAL = 1.0                   # 5段目：会いたかった人が戻ってきた時の、相手に向かいたい強さ（戻った直後1）［仮置き］
HEADS = ("food", "know", "stop", "social")   # worth.HEADS と同じ並び（地図の行）
# 生まれつきの行動の荷札（worth.ACT_TAGS と同じ符号）。それ以外の荷札は語として口へ
ACT_OUT = {"cry": word_tag("act:cry"), "reach": word_tag("act:reach")}


class Want(Box):
    BOX_ID = "want"

    def reset(self) -> None:
        self.n = 0
        self.last = {"action": "nothing", "strength": 0.0, "tag": None, "to": None}

    def step(self, inbox):
        out = self.passthrough()
        if phase(self.n) == WANT_PHASE:
            self.last = {"action": "nothing", "strength": 0.0, "tag": None, "to": None}
            disc = float(inbox.get("vars", 0.0))
            stress = float(inbox.get("feel", 0.0))
            curious = CURIOSITY * float(inbox.get("semantic", 0.0))
            drives = {"food": disc + stress, "know": curious,
                      "stop": float(inbox.get("inner", 0.0)), "social": SOCIAL * float(inbox.get("social", 0.0))}
            total = sum(drives.values())
            arousal = min(1.0, total)
            options = []
            for act, m in inbox.get("worth") or []:
                for j in range(GRID):
                    vs = [float(m[r, j]) for r in range(len(HEADS))]
                    if max(vs) > 0:
                        options.append((act, j, vs))
            if options and float(torch.rand(())) < arousal:
                # どの欲求のために動くか（ミルクの欲求しか無ければ乱数を引かない＝3段目までと同じ乱数の並び）
                head = "food"
                if total > drives["food"]:
                    # 欲求の強さ×その欲求で一番見込みのある行動の見込み（得られそうな分）に比例して選ぶ
                    pull = {h: drives[h] * max((vs[i] for _, _, vs in options), default=0.0)
                            for i, h in enumerate(HEADS)}
                    r = float(torch.rand(())) * sum(pull.values())
                    for h in HEADS:
                        r -= pull[h]
                        if r <= 0 and pull[h] > 0:
                            head = h
                            break
                row = HEADS.index(head)
                options = [(act, j, vs[row]) for act, j, vs in options if vs[row] > 0]   # その欲求で思い浮かぶ行動だけ
                r = float(torch.rand(())) * sum(o[2] for o in options)
                for act, j, v in options:
                    r -= v
                    if r <= 0:
                        break
                strength = min(1.0, BASE_STRENGTH + 0.5 * disc + stress + total - drives["food"])
                kind = next((k for k, t in ACT_OUT.items() if float(t @ act) > 0.99), "wordplan")
                self.last = {"action": kind if kind != "wordplan" else "say", "strength": strength, "tag": act, "to": j}
                m = torch.zeros(GRID, GRID, dtype=DTYPE)
                m[0, j] = strength
                out[kind] = [(act, m)]
                copy = torch.zeros(GRID, GRID, dtype=DTYPE)
                copy[0, j] = 1.0
                if row > 0:
                    copy[row, j] = 1.0
                out["worth"] = [(act, copy)]
        self.n += 1
        return out
