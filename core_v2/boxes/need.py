"""体の状態から「欲しい」を作る係（欲求）。

段階2・工程1：素通し（出ていく矢印すべてに書式のゼロを渡す）。
"""

from .base import Box


class Need(Box):
    BOX_ID = "need"
