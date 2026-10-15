"""目を向ける先を選ぶ係（2段目 習慣）。

段階2・工程3の仮部品（仕様書 §3）：見えている物ごとに、探している物の荷札との似かた
cos を [0,1] に直し（負は0に切る）、その物の場所の山（地図）に掛けて足す＝「探し物らしさ」の地図。
視野の真ん中の最大値を「見つけた」とする。探し物が無ければどちらもゼロ。

真ん中＝行・列 12〜15 の 4×4 マス（中心 13.5 から ±1.5マス＝±12画素≒±3.5度）。
**仕様書は「3×3」**だが、28マスの中心は 13.5 でマスの境目にあり、3×3 は左右どちらかに寄る。
13〜15 の 3×3 で回したところ、ボールを 2.9度（マス 13,12）で見ているのに「見つけた」が
0.59 にしかならず（山の中心が窓の外）、棚が5秒で時間切れになった（2026-10-15、工程3の確かめ）。

cos の直し方：cos が MATCH_COS（0.7）未満の物は 0、それ以上はその cos のまま。
実測（2026-10-15、場面の20コマ、64へ縮めた雑音なしの荷札）で、登録した荷札と同じ物は 0.996〜0.999、
違う物は 0.28〜0.58、机のマスとは 0.47〜0.56。雑音（両側に σ=0.05）が入ると同じ物でも 0.86 ほど、
違う物は 0.5 前後になる。最初は max(cos, 0) にしたが、探し物が視野の外にあるとき、違う物（0.5〜0.6）が
地図の山になって、そこへ目が跳んだ（工程3の確かめ、「コップはどこ？」でコップが 33度＝視野の外）。
(cos+1)/2 はさらに差が縮むので使わない。
"""

import torch

from ..formats import GRID, DTYPE, Fmt, zero
from .base import Box

MATCH_COS = 0.7
C0, C1 = GRID // 2 - 2, GRID // 2 + 2     # 真ん中 4×4 マス（行・列 12〜15。中心は 13.5）


class Gaze(Box):
    BOX_ID = "gaze"

    def step(self, inbox):
        out = self.passthrough()
        want = inbox.get("goal")
        objs = inbox.get("eyein") or []
        if want is None or float(torch.linalg.vector_norm(want)) == 0.0 or not objs:
            return out
        seek = zero(Fmt.MAP)
        for tag, where in objs:
            c = float(tag @ want)
            if c >= MATCH_COS:
                seek = seek + c * where
        seek = seek.clamp(0.0, 1.0)
        out["prio"] = seek
        out["goal"] = seek[C0:C1, C0:C1].max().to(DTYPE)
        return out
