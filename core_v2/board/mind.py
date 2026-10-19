"""盤面の太郎の心：体・気持ち・何をするか決める係（1段目・2段目）。

下書きの約束（2026-10-19）：中の部品は core_v2 の空の係と同じ名前・同じ役にしておき、2段目が終わったら
係の中へ書き写す。誰かは64次元の荷札（core_v2.tagging の作り方。見るたびに少し揺れる）で表す。

生まれつき（手で書いた、育っても変わらない所）：
* need：お腹が HUNGRY_FROM より減ると不快が出る
* feel：不快＋いら立ちが高いほど何か行動しやすく、合図が強くなる
* stress：「この合図ならすぐミルクが来るはず」と期待していたのに来なかった分だけ、毎分いら立ちが積もる
  （予想の外れ）。満ちたら大きく下がり、何もなくても少しずつ下がる
* want：選択肢（合図×向ける相手。部屋にいる人だけ）を、worth の値に比例して選ぶ（確率マッチング）

覚える所（体験から変わる所）：
* worth：「この相手にこの合図を向けたら、すぐミルクが来るか」を1層のNNで覚える。入力は相手の荷札、
  合図ごとに重み64個と切片1つ。教える答えは「その合図のあと、何分でミルクが来たか」から作る手柄
  （次の1分で来たら1、1分遅れるごとに RECENCY 倍、WINDOW_MIN 分以内に来なければ0。
  別の人が来た時は、向けた相手は来なかったので0。誰が来たかは太郎に見えている）。
  どの合図が通じるか、誰がよく応えるかは教えない。ほうびはお腹が満ちることだけ
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import deque
from dataclasses import dataclass, field

import numpy as np

TAG_DIM = 64                   # core_v2.formats.TAG_DIM と同じ

SIGNALS = ("cry", "reach")
WINDOW_MIN = 10                # 合図とお腹が満ちることを結ぶ時間の幅
RECENCY = 0.3                  # ミルクが1分遅れるごとに手柄が何倍になるか（赤ちゃんが結び付けるのは数秒の近さまで、Watson 1979）
HUNGRY_FROM = 0.6
PRIOR = 0.1                    # 生まれた時の値（どの合図も・誰にも同じ）
LR = 0.2                       # worth の学び方の速さ
STRESS_DECAY = 0.8             # いら立ちが1分で残る割合
BASE_STRENGTH = 0.3
FACE_NOISE = 0.1               # 人の荷札の見るたびの揺れ


def word_tag(word: str) -> np.ndarray:
    """core_v2.tagging.word_tag と同じ作り方（torch なしで動くように写した）。"""
    seed = int.from_bytes(hashlib.sha256(word.encode("utf-8")).digest()[:8], "little")
    v = np.random.default_rng(seed).standard_normal(TAG_DIM).astype(np.float32).astype(np.float64)
    return v / np.linalg.norm(v)


def person_tag(name: str, rng: np.random.Generator) -> np.ndarray:
    """その人の荷札（生まれつきの符号）に、見るたびの揺れを足して長さ1に戻す。"""
    base = word_tag("person:" + name)
    v = base + rng.normal(0.0, FACE_NOISE, TAG_DIM)
    return v / np.linalg.norm(v)


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class Worth:
    """合図ごとの1層NN：相手の荷札 → すぐミルクが来る見込み（0〜1）。"""
    w: dict = field(default_factory=lambda: {s: np.zeros(TAG_DIM) for s in SIGNALS})
    b: dict = field(default_factory=lambda: {s: math.log(PRIOR / (1 - PRIOR)) for s in SIGNALS})
    frozen: bool = False

    def value(self, signal: str, tag: np.ndarray) -> float:
        return _sigmoid(float(self.w[signal] @ tag) + self.b[signal])

    def learn(self, signal: str, tag: np.ndarray, target: float):
        if self.frozen:
            return
        err = target - self.value(signal, tag)
        self.w[signal] += LR * err * tag
        self.b[signal] += LR * err


@dataclass
class Mind:
    seed: int = 0
    frozen: bool = False
    stress: float = 0.0
    t: int = 0
    worth: Worth = field(default_factory=Worth)
    recent: deque = field(default_factory=deque)    # (時刻, 合図, 向けた人, 荷札, 期待)
    last: dict = field(default_factory=dict)

    def __post_init__(self):
        self.rng = random.Random(10_000 + self.seed)
        self.np_rng = np.random.default_rng(20_000 + self.seed)
        self.worth.frozen = self.frozen

    @staticmethod
    def need(fullness: float) -> float:
        return max(0.0, (HUNGRY_FROM - fullness) / HUNGRY_FROM)

    def value(self, signal: str, who: str = "ママ") -> float:
        """外から見る用：その人の揺れのない荷札での値。"""
        base = word_tag("person:" + who)
        return self.worth.value(signal, base / np.linalg.norm(base))

    def act(self, fullness: float, present: list[str] | None = None) -> tuple[str, float, str | None]:
        present = present or ["ママ"]
        disc = self.need(fullness)
        arousal = min(1.0, disc + self.stress)
        if self.rng.random() >= arousal:
            self.last = {"discomfort": disc, "stress": self.stress, "action": "nothing",
                         "strength": 0.0, "target": None, "expected": 0.0}
            return "nothing", 0.0, None
        options = []
        for who in present:
            tag = person_tag(who, self.np_rng)
            for s in SIGNALS:
                options.append((s, who, tag, self.worth.value(s, tag)))
        r = self.rng.random() * sum(o[3] for o in options)
        for s, who, tag, v in options:
            r -= v
            if r <= 0:
                break
        strength = min(1.0, BASE_STRENGTH + 0.5 * disc + self.stress)
        self.recent.append((self.t, s, who, tag, v))
        self.last = {"discomfort": disc, "stress": self.stress, "action": s, "strength": strength,
                     "target": who, "expected": v}
        return s, strength, who

    def feel(self, obs: dict) -> dict:
        """1分の結果を受け取る。ミルクが来たら窓の中の合図に手柄、来なければ期待した分いら立つ。"""
        joy = 0.0
        if obs["fed"]:
            if self.recent:
                joy = 1.0 - self.recent[-1][4]          # 思ったより良かった分＝うれしい
            for t0, s, who, tag, _ in self.recent:
                # 来てくれた人に向けた合図だけ手柄。別の人に向けた合図は「その人は来なかった」
                self.worth.learn(s, tag, RECENCY ** (self.t - t0) if who == obs.get("came") else 0.0)
            self.recent.clear()
            self.stress *= 0.3
        else:
            for t0, s, _, tag, v in self.recent:
                if t0 == self.t:
                    self.stress += v                    # 今の合図ですぐ来るはずだった分
            while self.recent and self.t - self.recent[0][0] >= WINDOW_MIN:
                _, s, _, tag, _ = self.recent.popleft()
                self.worth.learn(s, tag, 0.0)
        self.stress = min(1.0, self.stress * STRESS_DECAY)
        self.t += 1
        self.last.update({"joy": joy})
        return self.last
