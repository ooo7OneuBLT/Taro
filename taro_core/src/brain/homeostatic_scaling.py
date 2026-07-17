"""
恒常的シナプススケーリング（Homeostatic Synaptic Scaling） —
感覚エンコーダの出力が「崩壊」しないように保つ本能

【人間模倣】
神経細胞は活動量の基準値（set point）を持ち、そこから外れるとシナプス強度
全体を底上げ／底下げして基準値に戻す（Turrigiano & Nelson, 2004; Turrigiano,
2008。詳細は docs/参考文献リスト.md §1）。「予測誤差から学ぶ」ヘブ型の仕組み
（backprop）だけでは、活動が0に潰れるか暴走するかのどちらかに転びやすいことが
理論・実験の両面で知られており、この本能がその対抗機構として脳に実在する。

MIMo統合で、感覚エンコーダに感覚運動予測誤差だけでbackpropさせたところ、
「どんな入力が来てもほぼ同じ値を返す」崩壊が実際に起きた（研究日誌参照）。
太郎のNE・ドーパミンと同じ「本能」として、この対抗機構を追加する。

【人間模倣の解像度】
本物のシナプススケーリングは個々のシナプス強度を調整するが、ここでは
融合ベクトル（感覚エンコーダの最終出力）の水準で近似する。
target_varianceは新規の恣意的な定数ではなく、既存のLayerNorm（融合直後、
平均0・分散1に正規化）が想定する分散の基準（1.0）とmomentumはDopamineの
移動平均と同じ値（0.95）を流用し、新しい"決め打ち"を増やさない。
"""

import torch


class HomeostaticScaling:
    """
    融合ベクトルの「時間をまたいだ活動量」を移動平均で追跡し、それが
    基準値を下回ったら（＝どの入力でも似た値しか出さなくなったら）
    元に戻す方向の損失を返す。

    Dopamine.compute_rpe()と同じ形：まず"今の基準"を使って評価してから、
    基準を更新する（今回の値を先に基準に混ぜてしまうと、自分自身とだけ
    比較する空回りになるため）。
    """

    def __init__(self, dim, target_variance=1.0, momentum=0.95):
        self.target_variance = target_variance
        self.momentum = momentum
        self.running_mean = torch.zeros(dim)

    def homeostatic_loss(self, fused_vec):
        """
        今回の融合ベクトルが、最近の平均からどれだけ離れているか(二乗)を見る。
        離れが基準値未満（＝入力を区別できていない）なら、離れを大きくする
        方向の損失を返す。基準値以上ならペナルティなし（0）。
        """
        deviation_sq = (fused_vec - self.running_mean).pow(2)
        return torch.relu(self.target_variance - deviation_sq).mean()

    def observe(self, fused_vec):
        """今回の値を観測し、"最近の平均"を更新する（次回の評価用）。"""
        with torch.no_grad():
            self.running_mean = (self.momentum * self.running_mean
                                  + (1 - self.momentum) * fused_vec.detach())
