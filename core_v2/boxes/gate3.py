"""目的の入れ替えを決める門番 〔DA〕（3段目 目的を持ち続ける）。

段階2・工程1：素通し（出ていく矢印すべてに書式のゼロを渡す）。
"""

from .base import Box


class Gate3(Box):
    BOX_ID = "gate3"
