"""種類の層（入口（感覚）、下側頭葉の役。段階9、2026-10-18、ユーザー承認の仕様書 `2026-10-18_段階9_仕様書_名詞を厚く.md` §3・付録）。

目の入口の物の荷札（DINOv2 の見た目を決まった乱数で 64 へ縮めたもの、tagging.appearance_tag）は、色が違うと別の物に
見えやすい。そこで目の入口から出る物の荷札を、全部この小さな NN に通してから名簿（objfile）と目を向ける先を選ぶ係（gaze）へ渡す。
見えの近さを「一緒に来た物」で育て、同じ種類の見えを近づける（Mayor・Plunkett 2010 の、名前を結ぶ前に見た目をまとめる層）。

* 中身：y = norm(x + MLP(x))、MLP は 64→128→64（ReLU）。最後の層は0で始めるので、生まれた時は足す分が0。
  足す分が全部0の間は x をそのまま渡す（正規化し直さないので、値は段階8までの太郎と1ビットも同じ）。
  ただし矢印が1本増えるので、物の荷札が名簿・gaze に届くのは段階8までより1回（0.1秒）遅い（逸脱ではなく配線の性質。仕様書の
  「1ビットも同じ」は値について）
* 学び（learning=True のときだけ。決まりは1つ、ほうびは使わない）：見え同士を当て合う（対照学習 InfoNCE、教師つき版の
  Khosla ら 2020 の形：正の組が何個あってもよい）。経験の置き場に見えを覚えておき、TRAIN_EVERY_S（1秒）ごとに Adam を1歩
  - 近づける（正の組）：①同じ行（名簿の同じ席に続けて座っている物）の 0.1〜1秒違いの見え
    ②同じ語、または同じ音と一緒に来た見え（語・音が届く前 LOOK_WINDOW_S＝3秒に見ていた行の見え。意味の辞書・名簿と同じ窓）。
    「同じ」は荷札の似かたが SAME_COS 以上（目的の係の「同じ探し物がもう一度来た」と同じ線 0.95。語の荷札は語ごとに
    決まった乱数なので、同じ語は1、違う語は約0）
  - 離す（負の組）：同じ回に名簿にいた別の行の見え（同じ場面にいた別の物）
  - 温度 TAU＝0.1（SimCLR・教師つき対照学習の定番の値、生まれつき）、学びの速さ LR＝1e-3（Adam の既定）、置き場 MEMORY＝512、
    見えを覚えるのは STORE_EVERY_S（0.5秒）ごと（512 で約2分ぶん。仕様書「最近2分の見えを覚えておいて1秒ごとに少し直す」）
* 学びの手がかりはどれも環境から来る事実（同じ物が続いて見えた・親が語を言った・物が鳴った）で、答え（種類の名前）はどこにも無い。
  形の特徴を手で作らない。語ごと・色ごとの線や重みも無い

受け取るもの：eyein（物ごとの (荷札, 地図)、最後に目の動きの地図）・objfile（objfile→surprise と同じ列：席ごとの
(行の荷札, 場所の山。高さ1＝見えた)、今見ている行の山、目立ちの下がり）・listen（語の荷札）・soundin（物の音の荷札）。
出すもの：objfile（eyein から来た列と同じ形、荷札だけ種類の層を通したもの）・gaze（物ごとの組）。
名簿の列は、種類の層が2回前に出した物から作られている（kind→objfile→kind で2回）ので、2回前に通した物と場所で突き合わせる。
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ..formats import DTYPE, GRID, MAX_OBJECTS, TAG_DIM, make_tag
from .base import Box

TICK_S = 0.1
HIDDEN = 128
TAU = 0.1
LR = 1e-3
MEMORY = 512
STORE_EVERY_S = 0.5
TRAIN_EVERY_S = 1.0
POS_MIN_S, POS_MAX_S = 0.1, 1.0      # 同じ行の見えを正の組にする時間の差（仕様書）
LOOK_WINDOW_S = 3.0                  # 語・音の前に見ていた行（意味の辞書・名簿と同じ。Yu・Smith 2012）
SAME_COS = 0.95                      # 語・音の荷札が「同じ」（goal.SAME_COS と同じ線）
NEGATIVES = "scene"                  # 負の組：仕様書どおり "scene"（同じ回の別の行）。"all"＝置き場のほかの見え全部（SimCLR・
                                     # 教師つき対照学習のふつうの形）。試験場で比べるための切り替え（段階9の途中経過を参照）
MATCH_CELLS = 1.5                    # 名簿の席と物を場所で突き合わせる距離（名簿の TRACK_CELLS と同じ）
SEATS = MAX_OBJECTS - 2              # objfile.SEATS と同じ
VISIBLE_MIN = 0.75                   # 席の山の高さ：1＝見えた・0.5＝見えない（マスの間に山の頂があると 0.9 台）の間

_YS, _XS = torch.meshgrid(torch.arange(GRID, dtype=DTYPE), torch.arange(GRID, dtype=DTYPE), indexing="ij")


def center(m: torch.Tensor) -> tuple[float, float] | None:
    mx = float(m.max())
    if mx <= 0.0:
        return None
    w = (m >= 0.5 * mx).to(DTYPE) * m
    t = float(w.sum())
    return float((w * _YS).sum()) / t, float((w * _XS).sum()) / t


class KindNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(TAG_DIM, HIDDEN)
        self.l2 = nn.Linear(HIDDEN, TAG_DIM)
        nn.init.zeros_(self.l2.weight)
        nn.init.zeros_(self.l2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.l2(torch.relu(self.l1(x)))
        return y / y.norm(dim=-1, keepdim=True).clamp_min(1e-8)


class Kind(Box):
    BOX_ID = "kind"

    def __init__(self, learning: bool = False, seed: int = 0) -> None:
        super().__init__()
        g = torch.random.get_rng_state()
        torch.manual_seed(seed)                       # 最初の層の重み（生まれつきのばらつき）を種で決める
        self.net = KindNet()
        torch.random.set_rng_state(g)
        self.learning = bool(learning)
        self.negatives = NEGATIVES
        self.opt = None
        self.memory: list[dict] = []                  # 経験の置き場（reset では消さない。約2分で入れ替わる）
        self.t = 0
        self.n_train = 0
        self.last_loss = float("nan")
        self.reset()

    # ---- 生まれた時と同じか ----
    def grown(self) -> bool:
        return bool(self.net.l2.weight.abs().sum() > 0 or self.net.l2.bias.abs().sum() > 0)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """(n, 64) の荷札を種類の層に通す。足す分が0の間はそのまま返す。"""
        if x.numel() == 0 or not self.grown():
            return x
        with torch.no_grad():
            return self.net(x)

    def reset(self) -> None:
        self._hist: list[tuple[int, list]] = []       # (回, [(生の荷札, 場所)]) 直近3回
        self._seat_row = [-1] * SEATS
        self._next_row = 0
        self._prev_q = None

    # ---- 1回 ----
    def step(self, inbox):
        out = self.passthrough()
        self.t += 1
        objs = inbox.get("eyein") or []
        things, ego = list(objs), None
        if things and float(torch.linalg.vector_norm(things[-1][0])) == 0.0:
            things, ego = things[:-1], things[-1]
        if things:
            x = torch.stack([tg for tg, _ in things])
            y = self.transform(x)
            ys = [make_tag(y[i]) if y is not x else things[i][0] for i in range(len(things))]
            pairs = [(ys[i], things[i][1]) for i in range(len(things))]
        else:
            pairs = []
        out["gaze"] = pairs
        out["objfile"] = pairs[:MAX_OBJECTS - 1] + ([ego] if ego is not None else [])
        if self.learning:
            self._hist.append((self.t, [(tg, center(m)) for tg, m in things]))
            del self._hist[:-3]
            self._learn(inbox)
        return out

    # ---- 学び ----
    def _learn(self, inbox) -> None:
        lst = inbox.get("objfile") or []
        att_seat = self._seats(lst)
        q = inbox.get("listen")
        if q is not None and float(torch.linalg.vector_norm(q)) > 0.0:
            if self._prev_q is None or float(self._prev_q @ q) < 0.99:
                self._label("word", q)
            self._prev_q = q
        else:
            self._prev_q = None
        n = inbox.get("soundin")
        if n is not None and float(torch.linalg.vector_norm(n)) > 0.0:
            self._label("sound", n)
        store = int(round(STORE_EVERY_S / TICK_S))
        if self.t % store == 0 and len(self._hist) >= 3:
            src = self._hist[-3][1]                 # 名簿の列が元にした、2回前に通した物
            for s, (tag, m) in enumerate(lst[:SEATS]):
                if self._seat_row[s] < 0 or float(m.max()) < VISIBLE_MIN:
                    continue                        # 空席・見えていない行
                c = center(m)
                cand = [(math.hypot(c[0] - p[0], c[1] - p[1]), tg) for tg, p in src if p is not None]
                if not cand:
                    continue
                d, tg = min(cand, key=lambda z: z[0])
                if d > MATCH_CELLS:
                    continue
                self.memory.append({"x": tg.detach().clone(), "row": self._seat_row[s], "t": self.t,
                                    "att": s == att_seat, "word": None, "sound": None})
            del self.memory[:-MEMORY]
        if self.t % int(round(TRAIN_EVERY_S / TICK_S)) == 0:
            self.train_step()

    def _seats(self, lst) -> int:
        """席ごとの行の番号を更新し、今見ている行の席を返す（無ければ −1）。"""
        if len(lst) < SEATS + 1:
            return -1
        for s in range(SEATS):
            filled = float(torch.linalg.vector_norm(lst[s][0])) > 0.0
            if not filled:
                self._seat_row[s] = -1
            elif self._seat_row[s] < 0:
                self._seat_row[s] = self._next_row
                self._next_row += 1
        hc = center(lst[SEATS][1])
        if hc is None:
            return -1
        best, bd = -1, MATCH_CELLS
        for s in range(SEATS):
            c = center(lst[s][1])
            if c is not None and self._seat_row[s] >= 0:
                d = math.hypot(c[0] - hc[0], c[1] - hc[1])
                if d <= bd:
                    best, bd = s, d
        return best

    def _label(self, key: str, tag: torch.Tensor) -> None:
        """語・音が届いた：直前 LOOK_WINDOW_S に見ていた行の見えに、その荷札を付ける。"""
        t0 = self.t - int(round(LOOK_WINDOW_S / TICK_S))
        for it in self.memory:
            if it["att"] and it["t"] >= t0 and it[key] is None:
                it[key] = tag.detach().clone()

    def train_step(self) -> float | None:
        """経験の置き場で InfoNCE を1歩（Adam）。正の組と負の組を両方持つ見えが無ければ何もしない。"""
        mem = self.memory
        if len(mem) < 2:
            return None
        X = torch.stack([m["x"] for m in mem])
        row = torch.tensor([m["row"] for m in mem])
        t = torch.tensor([m["t"] for m in mem], dtype=torch.float32) * TICK_S
        dt = (t[:, None] - t[None, :]).abs()
        same_row = row[:, None] == row[None, :]
        pos = same_row & (dt >= POS_MIN_S - 1e-6) & (dt <= POS_MAX_S + 1e-6)
        for key in ("word", "sound"):
            has = torch.tensor([m[key] is not None for m in mem])
            if has.sum() >= 2:
                C = torch.stack([m[key] if m[key] is not None else torch.zeros(TAG_DIM) for m in mem])
                pos |= ((C @ C.T) >= SAME_COS) & has[:, None] & has[None, :]
        eye = torch.eye(len(mem), dtype=torch.bool)
        pos &= ~eye
        neg = ((dt < 1e-6) & ~same_row) if self.negatives == "scene" else ~(pos | eye)
        anchors = pos.any(1) & neg.any(1)
        if not bool(anchors.any()):
            return None
        if self.opt is None:
            self.opt = torch.optim.Adam(self.net.parameters(), lr=LR)
        z = self.net(X)
        s = (z @ z.T) / TAU
        allowed = pos | neg
        s_masked = s.masked_fill(~allowed, float("-inf"))
        lse = torch.logsumexp(s_masked, dim=1, keepdim=True)
        logp = (s - lse).masked_fill(~pos, 0.0)
        per = -(logp.sum(1) / pos.sum(1).clamp_min(1))
        loss = per[anchors].mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.n_train += 1
        self.last_loss = float(loss.detach())
        return self.last_loss
