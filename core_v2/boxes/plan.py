"""目的をやりくりする係（4段目 考える）。

段階2・工程1：素通し（出ていく矢印すべてに書式のゼロを渡す）。
"""

from .base import Box


class Plan(Box):
    BOX_ID = "plan"
