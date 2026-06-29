"""
模倣衝動 — 親の発話に似るほど心地よい

【人間模倣】乳児は親の発声に似せようとする衝動を生まれつき持つ。
声道パラメータ空間での重み付きedit distanceで類似度を計算。
"""


def _edit_distance(a, b):
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
    return dp[m]


def _weighted_edit_distance(a_tokens, b_tokens, vocab, vocal_tract):
    idx2char = vocab.idx2char if hasattr(vocab, 'idx2char') else {}

    def sub_cost(t1, t2):
        if t1 == t2:
            return 0.0
        c1 = idx2char.get(t1, "")
        c2 = idx2char.get(t2, "")
        d = vocal_tract.param_distance(c1, c2)
        return d / 4.0

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
    親の発話と太郎の出力の類似度 → 内的報酬 [0, 1]
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
