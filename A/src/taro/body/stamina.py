"""
体力 — 赤ちゃんの口・喉・肺の未熟さ

【人間模倣・身体的制約】
赤ちゃんは長く発声できない。成長とともに体力が増える。
"""


class Stamina:

    def __init__(self, initial=3.0, growth_rate=0.001, max_stamina=15.0):
        self.value = initial
        self.growth_rate = growth_rate
        self.max_stamina = max_stamina

    def get(self):
        return self.value

    def grow(self):
        self.value = min(self.value + self.growth_rate, self.max_stamina)
