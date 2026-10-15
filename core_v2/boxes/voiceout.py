"""声の出口（出口（筋・内臓））。

段階2・工程1：素通し（出ていく矢印すべてに書式のゼロを渡す）。
"""

from .base import Box


class Voiceout(Box):
    BOX_ID = "voiceout"
