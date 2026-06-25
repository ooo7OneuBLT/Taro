"""
太郎の本能 — 報酬関数とドーパミン（報酬予測誤差）

4つのコア本能を実装する：
  1. 模倣衝動（r_imit）   — 親の発話に似るほど心地よい【人間模倣】
  2. 予測成功（r_pred）   — 次トークンの予測が当たるほど心地よい【人間模倣】
  3. ドーパミン（δ）      — 報酬予測誤差＝学習信号【人間模倣＝既存AI研究】
  4. 社会的報酬（r_social）— 親の笑顔の強さ【人間模倣】（環境から入力）

合成報酬: R = w_imit * r_imit + w_pred * r_pred + w_social * r_social
学習信号: δ = R - baseline （baseline = 報酬の移動平均 = 予想していた報酬）
"""

import torch


def compute_imitation_reward(parent_tokens, taro_tokens):
    """
    模倣衝動：親の発話と太郎の出力の類似度 → 内的報酬 [0, 1]

    【人間模倣】乳児は親の発声に似せようとする衝動を生まれつき持つ。
    似ているほど心地よい（連続的な報酬）。

    算出方法：位置ごとの文字一致率。長さが違う場合は長い方で割る。
    """
    if len(parent_tokens) == 0 and len(taro_tokens) == 0:
        return 1.0
    if len(parent_tokens) == 0 or len(taro_tokens) == 0:
        return 0.0

    max_len = max(len(parent_tokens), len(taro_tokens))
    matches = 0
    for i in range(min(len(parent_tokens), len(taro_tokens))):
        if parent_tokens[i] == taro_tokens[i]:
            matches += 1

    return matches / max_len


def compute_prediction_reward(prediction_probs, actual_tokens):
    """
    予測成功：次トークン予測の的中度 → 報酬 [0, 1]

    【人間模倣】脳は次に来るものを絶えず予測し、
    予測が当たると安定して心地よい（予測処理 / Friston）。

    prediction_probs: list of (vocab_size,) tensors — 各ステップの予測確率分布
    actual_tokens:    list of int — 実際に来たトークン列

    各ステップで「実際のトークンに割り当てた確率」の平均を返す。
    """
    if len(prediction_probs) == 0 or len(actual_tokens) == 0:
        return 0.0

    n = min(len(prediction_probs), len(actual_tokens))
    total = 0.0
    for i in range(n):
        prob = prediction_probs[i]
        token = actual_tokens[i]
        if token < len(prob):
            total += prob[token].item()
        # 未知トークンの場合は0（予測できなかった）
    return total / n


class Dopamine:
    """
    ドーパミン — 報酬予測誤差（RPE）を計算する。

    【人間模倣＝既存AI研究】
    ドーパミンニューロンは「もらえた報酬 − 予想した報酬」を発火する
    （Schultz, 1997）。これは強化学習のTD誤差と数式上一致。

    baseline = 報酬の移動平均 = 「普段どれくらい報酬がもらえるか」の予想。
    δ = R - baseline
      δ > 0 → 予想より良かった → この行動を増やす
      δ < 0 → 予想より悪かった → この行動を減らす
    """

    def __init__(self, momentum=0.95):
        self.momentum = momentum
        self.baseline = 0.0

    def compute_rpe(self, reward):
        """
        報酬予測誤差δを計算し、baselineを更新する。

        reward: 合成報酬R（スカラー）
        戻り値: δ（スカラー）
        """
        delta = reward - self.baseline
        self.baseline = self.momentum * self.baseline + (1 - self.momentum) * reward
        return delta

    def get_baseline(self):
        return self.baseline


def compute_total_reward(r_imit, r_pred, r_social, weights):
    """
    合成報酬 R = w_imit * r_imit + w_pred * r_pred + w_social * r_social

    weights: dict with keys 'w_imit', 'w_pred', 'w_social'
    """
    R = (weights["w_imit"] * r_imit
         + weights["w_pred"] * r_pred
         + weights["w_social"] * r_social)
    return R
