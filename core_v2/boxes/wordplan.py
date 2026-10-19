"""言う語を組み立てる係（3段目 目的を持ち続ける）。

**段階7（2026-10-17、ユーザー承認の仕様書 `2026-10-17_段階7_仕様書_言える.md` §2-2）**：
意味の辞書から、見ている物から逆に引いた語の荷札 Wᵀa が届く。これを、聞き分ける係から届いたことのある
「知っている語」の荷札のうち、いちばん似ているものに合わせ（1回ごとの雑音や、ほかの語の混ざりを落とす）、
届いている間ずっと声の係へ送り続ける（言いたい語を持ち続ける）。届かなくなったらゼロ。
知っている語をまだ1つも聞いていなければ何も送らない。

言うか黙るか（つながり M が十分強いか）は、M を持っている声の係が決める。この係は M を読まない
（係どうしは矢印以外で中身を見ない、base.py）。

受け取るもの：semantic（TAG：見ている物から引いた語）、listen（TAG：聞いた知っている語）。
出すもの：voice（TAG：言いたい語、言わないときはゼロ）、listen（使わない、ゼロ）。
生まれつき（しくみ）。経験で変わるのは「聞いたことのある語の荷札」の一覧だけ（reset で消えない）。

盤面の世界（2026-10-19、3段目「言葉で頼む」。use_board() で入る）：
* 聞いた語ごとに聞いた回数を数え、SAYABLE_AFTER 回以上聞いた語を「言える語」として wordplan→worth へ送る
  （名前の段の「3回で分かる」より多い10回。言えるようになるのは分かるより後）。1回に送れるのは MAX_SAYABLE 個までなので、
  言える語を順番に回して送る（どの語を選択肢にするかは、見込みを覚える係が見込みの高い順に決める）。
  4段目：今さっき聞いた言える語は先に送り、地図の [0, 0] に「まねしたさ」（聞いてすぐ1、1分ごとに ECHO_DECAY 倍）を
  載せる（赤ちゃんは聞いたばかりの語をまねして言いやすい。Masur 1995）
* 「選ぶ」係から言うと決めた語（want→wordplan）が届いたら、そのまま声の出口へ渡す（wordplan→voiceout）。
  5段目：話す行動（worth.TELL_TAGS）なら、思い出の帳面（episodic→wordplan）から向ける人に合う一番新しい物の語を選び、
  知っている語に合わせて声の出口へ。話したことを帳面へ返す（wordplan→episodic）。
  声の係を通して口の形にするのは、口の下駄を外す時
""" 

from __future__ import annotations

import torch

from ..formats import DTYPE, GRID, make_tag
from .base import Box

SAME_COS = 0.99   # これより似ていれば同じ語として一覧に足さない（荷札は語ごとに決まっているので同じ語なら1）
SAYABLE_AFTER = 10    # 盤面：何回聞いたら言える語になるか（言えるのは分かるより遅く、もっと多く聞いてから［仮置き］。
                      # 3回だと、LLMの耳が拾った一度きりに近い語まで40語以上が言える語になり、試す語が散らばった）
MAX_SAYABLE = 6       # 盤面：1回に送る言える語の数（泣く・手と合わせて MAX_OBJECTS 以内）
ECHO_DECAY = 0.5      # 盤面：まねしたさが1分ごとに何倍になるか
ECHO_MIN = 0.1        # これより小さいまねしたさは送らない


class Wordplan(Box):
    BOX_ID = "wordplan"

    def __init__(self) -> None:
        super().__init__()
        self.words: list[torch.Tensor] = []   # 聞いたことのある知っている語の荷札
        self.counts: list[int] = []           # 盤面：語ごとの聞いた回数（words と同じ並び）
        self.board = False
        self.reset()

    def use_board(self) -> None:
        self.board = True

    def reset(self) -> None:
        self.held = None

    def get_extra_state(self):
        return {"words": [w.tolist() for w in self.words], "counts": list(self.counts)}

    def set_extra_state(self, state) -> None:
        self.words = [make_tag(w) for w in state.get("words", [])]
        self.counts = list(state.get("counts", [0] * len(self.words)))

    def step(self, inbox):
        out = self.passthrough()
        q = inbox.get("listen")
        if q is not None and bool(q.any()):
            if not self.words or float((self._stack() @ q).max()) <= SAME_COS:
                self.words.append(q.clone())
                self.counts.append(0)
                self._stacked = None
            i = int((self._stack() @ q).argmax())
            self.counts[i] += 1
            self.__dict__.setdefault("_heard_at", {})[i] = self.__dict__.get("_n", 0)
        n = self.__dict__.get("_n", 0)
        self.__dict__["_n"] = n + 1
        if self.board:
            from ..rhythm import TICKS_PER_MIN
            sayable = [i for i in range(len(self.words)) if self.counts[i] >= SAYABLE_AFTER]
            echo = {i: ECHO_DECAY ** ((n - t) // TICKS_PER_MIN) for i, t in self.__dict__.get("_heard_at", {}).items()}
            echo = {i: e for i, e in echo.items() if e >= ECHO_MIN and self.counts[i] >= SAYABLE_AFTER}
            if sayable:
                k = self.__dict__.get("_turn", 0) % len(sayable)
                fresh = sorted(echo, key=echo.get, reverse=True)[:MAX_SAYABLE]
                pick = fresh + [i for i in (sayable[k:] + sayable[:k]) if i not in fresh][:MAX_SAYABLE - len(fresh)]
                self.__dict__["_turn"] = k + MAX_SAYABLE
                out["worth"] = []
                for i in pick:
                    m = torch.zeros(GRID, GRID, dtype=DTYPE)
                    m[0, 0] = echo.get(i, 0.0)
                    out["worth"].append((self.words[i], m))
            if inbox.get("want"):
                out["voiceout"] = []
                from .worth import TELL_TAGS
                for act, m in inbox["want"]:
                    kind = next((k for k, t in TELL_TAGS.items() if float(t @ act) > 0.99), None)
                    if kind is None:
                        out["voiceout"].append((act, m))
                        continue
                    # 5段目：話す。思い出の帳面の新しい順に、向ける人 j がいなかった時（tell_new）・いた時（tell_old）の物の語
                    j, row = int(m[0].argmax()), 0 if kind == "tell_new" else 1
                    ep = inbox.get("episodic")
                    word = next((w for w, em in (ep if isinstance(ep, list) else []) if float(em[row, j]) > 0), None)
                    if word is not None and self.words:
                        word = self.words[int((self._stack() @ word).argmax())]   # 知っている語に合わせる
                        out["voiceout"].append((word, m))
                        told = torch.zeros(GRID, GRID, dtype=DTYPE)
                        told[0, j] = 1.0
                        out["episodic"] = [(word, told)]
        r = inbox.get("semantic")
        self.held = None
        if r is not None and bool(r.any()) and self.words:
            self.held = self.words[int((self._stack() @ r).argmax())]
            out["voice"] = self.held
        return out

    def _stack(self) -> torch.Tensor:
        if self.__dict__.get("_stacked") is None or len(self._stacked) != len(self.words):
            self.__dict__["_stacked"] = torch.stack(self.words)
        return self._stacked
