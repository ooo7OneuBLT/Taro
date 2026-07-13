"""
精度重み付き知覚（Precision-Weighted Perception） — 感覚と期待を精度で混ぜる

【人間模倣（着想）】ベイズ脳仮説／予測符号化：知覚は「実際の感覚」と
「これまでの経験から形成された期待」を、精度（precision）で重み付けして
混ぜた結果として構成される（Friston系の枠組み）。精度が低いほど感覚に
忠実（実際の入力をそのまま知覚する）、精度が高いほど期待に引きずられる
（実際の変化を無視して"いつも通り"だと思い込む＝錯覚・硬直）。

【重要な設計上の注意（試行錯誤の記録）】
最初は「期待」として、今まさに評価している予測(predicted_next_sensory)
自身をdetachして使おうとした。しかし評価対象(predicted)と混ぜる相手
(predicted.detach())が数式上まったく同じ値になるため、
  perceived = precision*P.detach() + (1-precision)*A
  loss = MSE(P, perceived) = (1-precision)^2 * MSE(P, A)
という、ただの誤差スケーリングに退化してしまうことが実測で判明した
（precisionを0〜0.8まで振っても崩壊パターンに変化が出なかった）。

そこで「期待」は、評価対象の予測とは独立な、"最近の実際の感覚の移動平均"
として別途保持する（恒常性スケーリング(homeostatic_scaling.py)の
running_meanと同じ型だが、役割が違うため独立に持つ＝2つの本能を混同しない）。
"""

import torch


class PrecisionWeightedPerception:
    """
    最近の実際の感覚の移動平均を「期待(belief)」として保持し、
    precisionに応じて実際の感覚と混ぜた「知覚」を返す。
    """

    def __init__(self, dim, momentum=0.95):
        self.momentum = momentum
        self.belief = torch.zeros(dim)

    def perceive(self, actual, precision):
        """
        actual: 実際の感覚（今回の融合ベクトルなど）
        precision: 0〜1。高いほど期待（belief）寄りの知覚になる。

        戻り値: 知覚（perceived）。predicted_next_sensoryとの
        MSEをprediction_error()に渡すことを想定。
        """
        return precision * self.belief.detach() + (1.0 - precision) * actual

    def observe(self, actual):
        """今回の実際の感覚で「期待」を更新する（次回の知覚評価用）。"""
        with torch.no_grad():
            self.belief = (self.momentum * self.belief
                            + (1 - self.momentum) * actual.detach())
