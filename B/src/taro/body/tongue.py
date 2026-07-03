"""
舌（Tongue） — 味覚センサー（体の器官）

【人間模倣】舌の味蕾（みらい）にある甘味受容体(T1R2/T1R3)が糖を検知し、「甘い」という
信号に変換する（＝センサー）。舌そのものは快楽物質を出さない＝**検知だけ**。快を作るのは
脳（hedonic.py の側坐核オピオイド系）。この分業をコードでも分ける（生物学的な物を係数に
隠さない、の原則）。

最小版：甘み(sweetness)のみ検知する。苦味・酸味・うま味・塩味は将来。
"""


class Tongue:
    """味覚センサー。食べ物の甘みを検知して味シグナル(0〜1)にする（検知のみ）。"""

    def __init__(self, sensitivity=1.0):
        self.sensitivity = sensitivity   # 受容体の感度
        self.last_taste = 0.0

    def taste(self, sweetness):
        """食べ物の甘みを検知して味シグナルを返す。"""
        self.last_taste = max(0.0, min(1.0, sweetness * self.sensitivity))
        return self.last_taste

    def get(self):
        return self.last_taste
