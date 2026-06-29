"""
予測（予測誤差最小化） — 次トークンの予測が当たるほど心地よい

【人間模倣】脳は次に来るものを絶えず予測し、
予測が当たると安定して心地よい（予測処理 / Friston）。
"""


def compute_prediction_reward(prediction_probs, actual_tokens):
    """
    次トークン予測の的中度 → 報酬 [0, 1]
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
    return total / n
