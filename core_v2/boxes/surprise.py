"""予想外に気づいて注意を切り替える係（1段目 反射）。

段階2・工程1：素通し（出ていく矢印すべてに書式のゼロを渡す）。
"""

from .base import Box


class Surprise(Box):
    BOX_ID = "surprise"
