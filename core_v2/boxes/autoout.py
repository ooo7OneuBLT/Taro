"""内臓の出口（出口（筋・内臓））。

段階2・工程1：素通し（出ていく矢印すべてに書式のゼロを渡す）。
"""

from .base import Box


class Autoout(Box):
    BOX_ID = "autoout"
    PURE = True   # 出すものは届いた値だけで決まる（同じ値が届き続ける回は計算しない、brain.tick）
