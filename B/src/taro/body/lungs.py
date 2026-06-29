"""
肺 — 発声の持続時間を決める身体構造

【人間模倣・身体シミュレーション】
旧stamina.pyの数式決め打ち（+0.001/ターン）を、
メカニズムベースに修正。

新生児の肺：容量が小さく、呼吸筋も弱い → 短い発声しかできない。
成長で肺活量が増え、呼吸筋が発達する → 長い発声が可能になる。

発声中は肺の空気を使い、息を吸って回復する。
1回の発声で出せるモーラ数は、肺の中の空気量で決まる。
"""


class Lungs:
    """
    太郎の肺。空気を吸って、発声で消費する。

    入力：呼吸（自動回復）、発声（消費）
    出力：発声可能なモーラ数（＝旧staminaの役割）
    """

    def __init__(self, capacity=3.0, air_per_mora=1.0,
                 recovery_rate=0.5, growth_rate=0.0001, max_capacity=15.0):
        """
        capacity: 肺活量（成長で増える）
        air_per_mora: 1モーラ発声に必要な空気量
        recovery_rate: 1tickあたりの空気回復量（呼吸）
        growth_rate: capacityの成長速度
        max_capacity: 肺活量の上限（成長しきった状態）
        """
        self.capacity = capacity
        self.air = capacity
        self.air_per_mora = air_per_mora
        self.recovery_rate = recovery_rate
        self.growth_rate = growth_rate
        self.max_capacity = max_capacity

    def tick(self):
        """1tick分の呼吸（空気が回復する）。"""
        self.air = min(self.capacity, self.air + self.recovery_rate)

    def get_max_mora(self):
        """現在の空気量で発声できる最大モーラ数。旧stamina.get()の代替。"""
        return int(self.air / self.air_per_mora)

    def consume(self, mora_count):
        """発声でmora_countモーラ分の空気を消費する。"""
        self.air = max(0.0, self.air - mora_count * self.air_per_mora)

    def grow(self):
        """身体の成熟で肺活量が増える。"""
        self.capacity = min(self.capacity + self.growth_rate, self.max_capacity)

    def get(self):
        """後方互換：旧Stamina.get()と同じインターフェース。"""
        return float(self.get_max_mora())
