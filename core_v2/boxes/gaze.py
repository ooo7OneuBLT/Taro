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

**段階9（2026-10-18）：まちがえそうな物と比べる決まり（RULE="compete"、既定）**。目的の係から届くのは
(荷札, 地図) の列：先頭が探している物の荷札、続きが「まちがえそうな物」の荷札（意味の辞書が一緒に思い出す。物ふつうの見た目と、
似たほかの語の見た目）。見えている物は、探している物の荷札との cos が、まちがえそうな物のどの荷札との cos よりも大きいときだけ
探し物とし、その cos を地図に掛けて足す（線 0.7 は使わない）。どれも当たらなければ探し物は無し（自由に見る）。
本物の物（GSO・Objaverse）では同じ種類の別の個体どうしの似かたが 0.79（手作りの物は 0.90）で、線 0.7 に正しい物の半分ほどが
届かず、試験場の A が 73% に落ちた（2026-10-18、noun_diag --semantic sparse）。注意は、探している物の予想に合う物が
ほかと競って勝つことで決まる（Desimone・Duncan 1995、Reynolds・Heeger 2009）。決まった線ではなく、比べる相手で決まる。
まちがえそうな物が届かない（列の先頭だけ＝段階2の表・前の表 W）か、RULE="gate" なら前の決まり（線 0.7）。

目的を持っている間は、地図の全マスに SEEK_FLOOR（0.001）を敷く。prio は配線上「目的を持っている
強さ」を受け取れず、地図が空かどうかで重み w を決めているため（prio.py）。床が無いと、探し物の
切り出しが1コマ抜けただけで地図が空→ w=0 →目立ちだけの地図になり、orienteye の抑えをすり抜けて
目立つ所へ跳んだ（2026-10-15、妨害物から「コップはどこ？」：コップを1.1秒見たあと、7.2秒にボールへ）。
"""

import torch

from ..formats import GRID, DTYPE, Fmt, zero
from .base import Box

MATCH_COS = 0.7
SEEK_FLOOR = 1e-3
RULE = "compete"          # 段階9の既定。"gate"＝前の線 0.7（比べる用）
C0, C1 = GRID // 2 - 2, GRID // 2 + 2     # 真ん中 4×4 マス（行・列 12〜15。中心は 13.5）


def split_want(want):
    """目的の係から届いた値 → (探している物の荷札, [まちがえそうな物の荷札])。何も無ければ (None, [])。
    荷札1つ（前の形）でもよい。"""
    if want is None:
        return None, []
    if isinstance(want, torch.Tensor):
        return (want if float(torch.linalg.vector_norm(want)) > 0.0 else None), []
    if not want or float(torch.linalg.vector_norm(want[0][0])) == 0.0:
        return None, []
    return want[0][0], [t for t, _ in want[1:]]


def match(target, rivals, tags, rule: str | None = None) -> list:
    """見えている物ごとに、探し物なら探している物との cos、そうでなければ None。"""
    rule = rule or RULE
    out = []
    for tag in tags:
        c = float(tag @ target)
        if rule == "compete" and rivals:
            ok = all(c > float(tag @ r) for r in rivals)
        else:
            ok = c >= MATCH_COS
        out.append(c if ok else None)
    return out


class Gaze(Box):
    BOX_ID = "gaze"

    def __init__(self, rule: str | None = None) -> None:
        super().__init__()
        self.rule = rule or RULE

    def step(self, inbox):
        out = self.passthrough()
        target, rivals = split_want(inbox.get("goal"))
        objs = inbox.get("kind", inbox.get("eyein")) or []   # 段階9：種類の層を通した物（kind が無い試験場では目の入口から）
        if target is None or not objs:
            return out
        seek = zero(Fmt.MAP) + SEEK_FLOOR
        for (tag, where), c in zip(objs, match(target, rivals, [t for t, _ in objs], self.rule)):
            if c is not None:
                seek = seek + c * where
        seek = seek.clamp(0.0, 1.0)
        out["prio"] = seek
        out["goal"] = seek[C0:C1, C0:C1].max().to(DTYPE)
        return out
