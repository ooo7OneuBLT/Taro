"""2語の意味を合わせる係（2段目 習慣）。

段階2・工程1：素通し（出ていく矢印すべてに書式のゼロを渡す）。
"""

from .base import Box


class Combine(Box):
    BOX_ID = "combine"
