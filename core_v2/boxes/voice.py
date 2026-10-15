"""声と口の動きを組み立てる係（2段目 習慣）。

段階2・工程1：素通し（出ていく矢印すべてに書式のゼロを渡す）。
"""

from .base import Box


class Voice(Box):
    BOX_ID = "voice"
