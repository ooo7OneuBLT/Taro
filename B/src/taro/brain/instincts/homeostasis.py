"""
恒常性（Homeostasis） — 体の状態を快適値に保とうとする本能

【人間模倣】
ホメオスタシス（Cannon, 1932）。生得的な生物の基本機能。
体は各状態の快適値（set point）に向かって戻ろうとする。

快適値から離れる → つらい（arousal上昇）
快適値に戻る → ほっとする（arousal低下）→ 報酬

この「ほっとした分」が報酬 r_home。
「まんま」が大事な音になるのは、それが切実な状態の解消に関わるから。
"""


class Homeostasis:
    """
    恒常性の本能。arousalの変化から報酬を計算する。
    """

    def __init__(self):
        self.prev_arousal = 0.0

    def compute_reward(self, current_arousal):
        """
        arousalが下がった分だけ正の報酬。上がった分だけ負の報酬。

        r_home = prev_arousal - current_arousal
        下がった → 正（ほっとした）
        上がった → 負（つらくなった）
        変わらない → 0
        """
        r_home = self.prev_arousal - current_arousal
        self.prev_arousal = current_arousal
        return r_home
