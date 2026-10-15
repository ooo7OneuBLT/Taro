"""動きのずれを直す係（動きのずれを直す）。

段階2・工程1：素通し（出ていく矢印すべてに書式のゼロを渡す）。
"""

from .base import Box


class Cb(Box):
    BOX_ID = "cb"
