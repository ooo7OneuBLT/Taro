"""入口の音量つまみ（入口（感覚））。

段階2・工程1：素通し（出ていく矢印すべてに書式のゼロを渡す）。
"""

from .base import Box


class Volume(Box):
    BOX_ID = "volume"
