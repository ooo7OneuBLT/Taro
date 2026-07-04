"""
原-辞書（Proto-Lexicon） — 聞いた発話から「単語の型」を切り出して蓄える

【人間模倣】統計的分節（Saffran, Aslin & Newport 1996）＋語テンプレート（Vihman）。
乳児は連続音声を「次の音の予測しやすさ」を手がかりに区切り、繰り返し出会う
予測しやすい並びを1つの単位（語の型）として記憶する。報酬（ドーパミン）は使わない
＝教師なし。予測の自信度と出会った頻度だけで単位が立ち上がる。

太郎では：知覚ヘッドが各音を直前から予測した自信度（＝遷移確率）を手がかりに、
「予測が谷になる所（＝単語の境目）」の間を1単位として切り出す。絶対的な決め打ち
閾値は使わず、隣同士の相対的な上下関係（局所的な谷）だけで境界を決める（Saffranの
「語の内部は遷移確率が高く、語の境目で下がる」に対応）。

B6-1b修正：初版は「発話内平均以上の連続run」で切っていたが、自信度は文脈が
積み上がるほど（＝語の後半ほど）上がりやすく、語の頭が切り落とされ後半だけが
単位化される偏りがあった（例：「まんま」でなく「んま」）。局所的な谷（隣より
低い点）で境界を引く方式に変えて、この偏りを避ける。
"""


class Lexicon:
    """
    原-辞書。聞いた発話から予測しやすい並びを切り出して頻度を数える。

    counts: tuple(tokens) → 出会った回数
    """

    def __init__(self, min_len=2):
        # min_len：単位として登録する最短の長さ（1音は語の型とみなさない）。
        # ⚠️構造的な下限であって調整用の恣意的定数ではない（1にすると全単音が語になる）。
        self.min_len = min_len
        self.counts = {}
        # B6-4：語↔内的状態の連合（cross-situational statistical learning, Smith & Yu）。
        # その語を聞いた時の動因状態[空腹,眠気,不快]を語ごとに累積し、平均を「その語が
        # 結びつく状態」とする。報酬なし・共起の統計だけ。視覚が無いので指示対象＝内的状態。
        self.state_sum = {}   # chunk -> [Σ空腹, Σ眠気, Σ不快, n]

    def segment(self, tokens, confidences):
        """
        発話を分節し、切り出した1単位（token列）を返す（無ければNone）。

        tokens: 実際に聞いた並び（BOS/EOS除く）
        confidences: 各tokenを直前の文脈から予測できた自信度[0-1]（tokensと同長）

        規準：位置iの自信度が両隣より低い「局所的な谷」を境界とする（相対値のみ、
        絶対閾値なし）。境界と境界の間（発話の端も境界とみなす）が1つの単位候補で、
        その中で最長のものを採用する＝語の内部（予測が当たり続ける）と境目（予測が
        落ちる）の相対関係だけで切り出す。
        """
        n = len(tokens)
        if n < self.min_len or len(confidences) != n:
            return None
        # 局所的な谷（両隣より低い点）を境界にする。端も境界とみなす。
        boundaries = {0, n}
        for j in range(1, n - 1):
            if confidences[j] < confidences[j - 1] and confidences[j] < confidences[j + 1]:
                boundaries.add(j)
        bs = sorted(boundaries)
        runs = [(bs[i], bs[i + 1]) for i in range(len(bs) - 1)]
        if not runs:
            return None
        s, e = max(runs, key=lambda r: r[1] - r[0])
        if e - s < self.min_len:
            return None
        return tuple(tokens[s:e])

    def observe(self, tokens, confidences, state=None):
        """
        発話を分節し、切り出した単位を辞書に登録（頻度+1）。切り出した単位を返す。

        state: その語を聞いた時の動因状態[空腹,眠気,不快]（B6-4）。渡すと語↔状態の連合を
        累積する（報酬でなく共起の統計）。
        """
        chunk = self.segment(tokens, confidences)
        if chunk is not None:
            self.counts[chunk] = self.counts.get(chunk, 0) + 1
            if state is not None and len(state) >= 3:
                acc = self.state_sum.get(chunk)
                if acc is None:
                    acc = [0.0, 0.0, 0.0, 0]
                    self.state_sum[chunk] = acc
                acc[0] += float(state[0]); acc[1] += float(state[1]); acc[2] += float(state[2])
                acc[3] += 1
        return chunk

    def assoc(self, chunk):
        """
        B6-4：その語が結びつく内的状態の平均[空腹,眠気,不快]を返す（無ければNone）。
        """
        acc = self.state_sum.get(chunk)
        if not acc or acc[3] == 0:
            return None
        n = acc[3]
        return (acc[0] / n, acc[1] / n, acc[2] / n)

    def top(self, n=10):
        """頻度上位の語の型を返す。"""
        return sorted(self.counts.items(), key=lambda kv: -kv[1])[:n]

    def known(self, min_count=1):
        """min_count回以上出会った語の型の集合を返す（産出計画で使う候補）。"""
        return {k for k, v in self.counts.items() if v >= min_count}
