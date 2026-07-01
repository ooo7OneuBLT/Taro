"""
模倣衝動 — 親の発話に似るほど心地よい

【人間模倣】乳児は親の発声に似せようとする衝動を生まれつき持つ。
声道パラメータ空間での重み付きedit distanceで類似度を計算。
"""


def _sub_cost(t1, t2, idx2char, vocal_tract):
    if t1 == t2:
        return 0.0
    c1 = idx2char.get(t1, "")
    c2 = idx2char.get(t2, "")
    d = vocal_tract.param_distance(c1, c2)
    return d / 4.0


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

    n, m = len(a_tokens), len(b_tokens)
    dp = [float(j) for j in range(m + 1)]
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], float(i)
        for j in range(1, m + 1):
            cost = _sub_cost(a_tokens[i - 1], b_tokens[j - 1], idx2char, vocal_tract)
            prev, dp[j] = dp[j], min(dp[j] + 1.0, dp[j - 1] + 1.0, prev + cost)
    return dp[m]


def compute_alignment_credit(target_tokens, generated_tokens, vocab, vocal_tract):
    """
    生成した各文字が目標語のどの位置にどれだけ一致していたかを求める。

    【人間模倣】DIVAモデル（Guenther）の運動学習は、発話全体が終わってから
    まとめて1つの誤差を評価するのではなく、聴覚フィードバックを継続的・
    フレーム単位で評価し、その場その場の調音を修正する。太郎の音韻バッファ
    （小脳の順/逆モデル）も同じ発想に基づいている。

    B2-2：REINFORCEが発話全体に単一のδしか与えないと、「ま」は良くて
    「み」は悪かった、という文字ごとの違いを学習できない（クレジット割り当て
    問題）。既に模倣報酬の計算に使っている重み付き編集距離のDPテーブルを
    バックトレースし、一致していた文字には高い信頼度、目標語にない余分な
    文字には低い信頼度を個別に与える。閾値のような決め打ちの数値を増やさず、
    既存の音声距離計算を再利用するだけで実現する。

    戻り値: generated_tokensと同じ長さのリスト。各要素はおおむね[-1, 1]で、
    1に近いほどその位置の文字が目標語に合っていたことを示す。
    """
    idx2char = vocab.idx2char if hasattr(vocab, 'idx2char') else {}
    n, m = len(generated_tokens), len(target_tokens)

    if n == 0:
        return []
    if m == 0:
        return [-1.0] * n

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = float(i)
    for j in range(m + 1):
        dp[0][j] = float(j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = _sub_cost(generated_tokens[i - 1], target_tokens[j - 1], idx2char, vocal_tract)
            dp[i][j] = min(dp[i - 1][j] + 1.0, dp[i][j - 1] + 1.0, dp[i - 1][j - 1] + cost)

    credit = [0.0] * n
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = _sub_cost(generated_tokens[i - 1], target_tokens[j - 1], idx2char, vocal_tract)
            if abs(dp[i][j] - (dp[i - 1][j - 1] + cost)) < 1e-9:
                credit[i - 1] = 1.0 - cost  # 一致度が高いほど+1に近い
                i -= 1
                j -= 1
                continue
        if i > 0 and abs(dp[i][j] - (dp[i - 1][j] + 1.0)) < 1e-9:
            credit[i - 1] = -1.0  # 目標語にない余分な音
            i -= 1
            continue
        j -= 1  # 目標語にあるが太郎が出していない音（生成側のcreditには反映しない）

    return credit


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
