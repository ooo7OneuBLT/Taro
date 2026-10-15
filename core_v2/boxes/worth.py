"""どれくらい得かを見積もる係（価値と気持ち）。

段階2・工程1：素通し（出ていく矢印すべてに書式のゼロを渡す）。
"""

from .base import Box


class Worth(Box):
    BOX_ID = "worth"
