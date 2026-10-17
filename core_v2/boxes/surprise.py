"""予想外に気づく係（1段目 反射）。

**段階8a（2026-10-18、ユーザー決定「まずハードコードでいこうか」）：仮部品（逸脱リスト その57）。**
次の版（8b）では「消えた」の決まりを、次の回を予想する小さなNNの外れ（予測の誤差）に入れ替える。
そのとき入れ替えるのは中の計算だけで、出す形（意味の辞書へ AMOUNT＝驚きの強さ 0〜1）は変えない。

驚き（席 i）＝ 消えた_i × 覚え_i（enc）× 中心の重み_i。意味の辞書へは席の中でいちばん強いものを出す。
注意＝予測の誤差にかける重み（精度）という考え（Feldman・Friston 2010、Auksztulewicz・Friston 2015）を、
いちばん簡単な形にしたもの（調べ：`文献調査/2026-10-18_注意と予測_中心視野.md` の案A）。

* **消えた**（生まれつき）：物の名簿が席ごとに「見えている（山の高さ1）／視野の中にいるのに見えない（0.5）」を出す。
  - 中心（真ん中から FIX_CELLS 以内＝いま目を向けている物）の行が 1→0.5 になったら、消えた＝1
  - 周辺の行は、1→0.5 になり、**しかもその場所の目立ちが急に下がった**（名簿の最後の地図）ときだけ 1
    （周辺の「急に消えた」は上丘の速い道で安く知らせる。消えることは現れることより気づかれにくい：Boot ら 2005）
  - 1 は、その行がまた見えるか、名簿から消える（KEEP_S＝2秒）か、視野の外へ出るまで続く。
    目をそらして物が視野の外へ出たときは、山がゼロ（視野の外）になるので消えたにならない
* **覚え enc**（形は生まれつき、値は経験）：席の物を中心で見ている間 enc ← 1−(1−enc)·exp(−Δt/ENC_TAU_S) で育ち、
  それ以外は exp(−Δt/ENC_DECAY_S) でゆっくり薄れる。席に別の物が座ったら（席が空いた次の物）0 から。
  よく見て覚えた物ほど、無くなったときに強く驚く（慣らしの時間の効果：Hunter・Ames 1988、Richards の持続的注意）
* **中心の重み**：中心の物は1。周辺は、目の中心からの角度 E での視覚野の拡大率（Horton・Hoyt 1991：M(E)∝1/(E＋0.75°)）
  を、中心の縁を1として比べた値。周辺は小さいがゼロにならない。
  **太郎側の仮の判断：乳児の根拠なし（大人の周辺視の見落としのみ、Boot ら 2005）**。use_centre=False で切れる
* use_enc=False で enc を1にできる（覚えの掛け算を切る）。enabled=False で驚きを出さない（試験 B の「驚きを止めた太郎」）

「ない」という語と驚きのつながりは、この係には書かない。意味の辞書の学習（場面をまたいだ数え上げ）だけから生まれる。

数（どれも試験に合わせて動かしていない）：
* ENC_TAU_S＝5秒：Richards の心拍研究で、見せる時間 2.5〜5秒はなじみ好み（覚えきれていない）、10〜20秒で新しい物好み
  （覚えた）。5秒を時定数にすると 10秒で 0.86、20秒で 0.98［Tier2・文献の範囲の切りのよい値］
* ENC_DECAY_S＝60秒：慣れ（6秒）・もう見た（8秒）より1桁遅い「遅い記憶」。切りのよい値［Tier3・仮置き］
* DEG_PER_CELL＝60°/28：体の目の画角（run/world/toy_env.py の VISION_FOVY＝60）を地図の28マスで割ったもの
* E2_DEG＝0.75°：Horton・Hoyt 1991 の式の定数
* SEEN_H＝0.75：山の高さ 1 と 0.5 の真ん中（書式の読み取り。山の中心が升目の間にあると高さは 0.94〜1 になる）
"""

from __future__ import annotations

import math

import torch

from ..formats import DTYPE
from .base import Box
from .objfile import CENTER, FIX_CELLS, SEATS, TICK_S, object_center

ENC_TAU_S = 5.0
ENC_DECAY_S = 60.0
DEG_PER_CELL = 60.0 / 28.0
E2_DEG = 0.75
SEEN_H = 0.75

_GROW = math.exp(-TICK_S / ENC_TAU_S)
_DECAY = math.exp(-TICK_S / ENC_DECAY_S)


def centre_weight(d_cells: float) -> float:
    """目の中心からの距離（マス）→ 重み。中心（FIX_CELLS 以内）は1、外は視覚野の拡大率の比。"""
    if d_cells <= FIX_CELLS:
        return 1.0
    e_fix = FIX_CELLS * DEG_PER_CELL
    return (e_fix + E2_DEG) / (d_cells * DEG_PER_CELL + E2_DEG)


class _Seat:
    __slots__ = ("state", "enc", "vanish", "fov", "w")

    def __init__(self) -> None:
        self.state = "empty"    # empty / out（視野の外）/ seen / unseen
        self.enc = 0.0
        self.vanish = 0.0
        self.fov = False
        self.w = 1.0


class Surprise(Box):
    BOX_ID = "surprise"

    def __init__(self, enabled: bool = True, use_enc: bool = True, use_centre: bool = True) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.use_enc = bool(use_enc)
        self.use_centre = bool(use_centre)
        self.reset()

    def reset(self) -> None:
        self.seats = [_Seat() for _ in range(SEATS)]
        self.level = 0.0
        self.n_onset = 0          # 記録用：消えたが立った回数

    def get_extra_state(self):
        return {"enc": [s.enc for s in self.seats]}

    def set_extra_state(self, state) -> None:
        for s, e in zip(self.seats, state.get("enc", [])):
            s.enc = float(e)

    def busy(self) -> bool:
        """試験場が何も起きない時間を飛ばしてよいか（席に何か座っている間は飛ばさない）。"""
        return any(s.state != "empty" for s in self.seats)

    def step(self, inbox):
        out = self.passthrough()
        lst = inbox.get("objfile") or []
        rows, drop = (lst[:-1], lst[-1][1]) if lst else ([], None)
        best = 0.0
        for i, seat in enumerate(self.seats):
            tag, m = rows[i] if i < len(rows) else (None, None)
            if tag is None or not bool(tag.any()):
                if seat.state != "empty":
                    self.seats[i] = seat = _Seat()
                continue
            if seat.state == "empty":
                seat.enc = 0.0                       # 新しい物が席に座った
            h = float(m.max())
            if h <= 0.0:                             # 視野の外
                seat.state, seat.vanish = "out", 0.0
                seat.enc *= _DECAY
                continue
            r, c = object_center(m)
            d = math.hypot(r - CENTER, c - CENTER)
            seat.fov = d <= FIX_CELLS
            seat.w = centre_weight(d) if self.use_centre else 1.0
            if h > SEEN_H:
                seat.vanish = 0.0
                seat.enc = 1.0 - (1.0 - seat.enc) * _GROW if seat.fov else seat.enc * _DECAY
                seat.state = "seen"
            else:
                seat.enc *= _DECAY
                if seat.state == "seen":             # いま見えなくなった
                    dropped = drop is not None and float(drop[int(round(r)), int(round(c))]) >= 0.5
                    seat.vanish = 1.0 if (seat.fov or dropped) else 0.0
                    self.n_onset += seat.vanish > 0
                seat.state = "unseen"
            s = seat.vanish * (seat.enc if self.use_enc else 1.0) * seat.w
            best = max(best, s)
        self.level = best if self.enabled else 0.0
        out["semantic"] = torch.tensor(self.level, dtype=DTYPE)
        return out
