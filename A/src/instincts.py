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


def _edit_distance(a, b):
    """レーベンシュタイン距離（挿入・削除・置換の最小回数）を計算する。"""
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
    return dp[m]


def _weighted_edit_distance(a_tokens, b_tokens, vocab, vocal_tract):
    """
    声道パラメータ空間での重み付きedit distance。
    置換コストを「調音的な距離（0〜4のうち何個のパラメータが違うか）」で計算。
    似た口の動きで出せる音同士は置換コストが小さくなる。

    【人間模倣】赤ちゃんは自分の声と親の声の聴覚的類似度を感じ取っている。
    """
    idx2char = vocab.idx2char if hasattr(vocab, 'idx2char') else {}

    def sub_cost(t1, t2):
        if t1 == t2:
            return 0.0
        c1 = idx2char.get(t1, "")
        c2 = idx2char.get(t2, "")
        d = vocal_tract.param_distance(c1, c2)
        return d / 4.0  # 正規化：0〜1

    n, m = len(a_tokens), len(b_tokens)
    dp = [float(j) for j in range(m + 1)]
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], float(i)
        for j in range(1, m + 1):
            cost = sub_cost(a_tokens[i - 1], b_tokens[j - 1])
            prev, dp[j] = dp[j], min(dp[j] + 1.0, dp[j - 1] + 1.0, prev + cost)
    return dp[m]


def compute_imitation_reward(parent_tokens, taro_tokens, vocab=None, vocal_tract=None):
    """
    模倣衝動：親の発話と太郎の出力の類似度 → 内的報酬 [0, 1]

    【人間模倣】乳児は親の発声に似せようとする衝動を生まれつき持つ。
    似ているほど心地よい（連続的な報酬）。

    A2-3変更：声道パラメータ空間での重み付きedit distance。
    「ま」と「ば」は口の動きが近いので置換コストが小さい＝類似度が高い。
    赤ちゃんが自分の声と親の声の聴覚的類似度を感じ取ることの再現。
    """
    if len(parent_tokens) == 0 and len(taro_tokens) == 0:
        return 1.0
    if len(parent_tokens) == 0 or len(taro_tokens) == 0:
        return 0.0

    max_len = max(len(parent_tokens), len(taro_tokens))

    if vocab is not None and vocal_tract is not None:
        dist = _weighted_edit_distance(parent_tokens, taro_tokens, vocab, vocal_tract)
    else:
        dist = _edit_distance(parent_tokens, taro_tokens)

    return max(0.0, 1.0 - dist / max_len)


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


class Habituation:
    """
    馴化（飽き）— 同じ出力を繰り返すと報酬が減衰する。

    【人間模倣】胎児・新生児にも確認される最も基本的な神経メカニズム。
    ドーパミン系とは独立に存在する。

    同じ音を出し続けると「つまらなくなる」。
    新しい音を試すと「面白い」。
    これにより固着を防ぎ、探索を促す。
    """

    def __init__(self, history_size=20, decay_rate=0.05):
        self.history = []
        self.history_size = history_size
        self.decay_rate = decay_rate

    def compute_penalty(self, output_text):
        """
        直近の出力履歴と比べて、同じ出力の繰り返しにペナルティを与える。
        フレーズ全体の繰り返し＋文字レベルの繰り返しの両方を考慮。

        戻り値: 0.0（新しい出力）〜 -1.0（ずっと同じ出力）
        """
        if not self.history or not output_text:
            self._add(output_text)
            return 0.0

        # フレーズ全体の繰り返し
        phrase_repeats = sum(1 for h in self.history if h == output_text)
        phrase_penalty = -self.decay_rate * phrase_repeats

        # 文字レベル：出力の中で同じ文字が繰り返されていると飽きる
        if len(output_text) > 1:
            unique_ratio = len(set(output_text)) / len(output_text)
            monotony_penalty = -self.decay_rate * (1.0 - unique_ratio) * 3
        else:
            monotony_penalty = 0.0

        self._add(output_text)
        return max(-1.0, phrase_penalty + monotony_penalty)

    def _add(self, text):
        self.history.append(text)
        if len(self.history) > self.history_size:
            self.history.pop(0)


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


def compute_total_reward(r_imit, r_pred, r_social, r_habit, weights):
    """
    合成報酬 R = w_imit * r_imit + w_pred * r_pred + w_social * r_social + r_habit

    r_habit（馴化ペナルティ）は重みなしで直接加算。
    同じ出力の繰り返しで報酬が下がり、新しい出力で回復する。
    """
    R = (weights["w_imit"] * r_imit
         + weights["w_pred"] * r_pred
         + weights["w_social"] * r_social
         + r_habit)
    return max(0.0, R)
