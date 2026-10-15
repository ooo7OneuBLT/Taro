"""ストレスの体の反応（全体に効く変数）。

段階2・工程1：素通し（出ていく矢印すべてに書式のゼロを渡す）。
"""

from .base import Box


class Stress(Box):
    BOX_ID = "stress"
