"""体の動きを計画する係（3段目 目的を持ち続ける）。

段階2・工程1：素通し（出ていく矢印すべてに書式のゼロを渡す）。
"""

from .base import Box


class Bodyplan(Box):
    BOX_ID = "bodyplan"
