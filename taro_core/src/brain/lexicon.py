"""原-辞書（Proto-Lexicon） — 聞いた発話から「単語の型」を切り出して蓄える

【出典】B/src/taro/brain/lexicon.py からの移植（2026-08-18）。原本はB側に残すが、
今後使うのはこちら（F2「耳の移植」）。学習則・分節ロジックは変えていない。

【F2での変更点（仕様書の指示どおり、1点のみ）】
B原本は状態ベクトルを[空腹,眠気,不快]の3次元に決め打ちしていた（B6-4）。
F2では「語を聞いた瞬間に見えているもの」＝視覚エンコーダ出力（64次元）を
stateとして渡えるようにするため、次元をコンストラクタ引数 state_dim として
一般化した。学習則（累積して平均する）自体はB原本のままで変えていない。

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

    def __init__(self, min_len=2, state_dim=3):
        # min_len：単位として登録する最短の長さ（1音は語の型とみなさない）。
        # 注意：構造的な下限であって調整用の恣意的定数ではない（1にすると全単音が語になる）。
        self.min_len = min_len
        self.counts = {}
        # B6-4：語↔内的状態の連合（cross-situational statistical learning, Smith & Yu）。
        # その語を聞いた時の状態ベクトル（B原本は[空腹,眠気,不快]の3次元固定）を語ごとに
        # 累積し、平均を「その語が結びつく状態」とする。報酬なし・共起の統計だけ。
        # F2：state_dim をコンストラクタ引数化（B原本は3固定）。視覚が無ければ
        # 内的状態、視覚があれば見えているものの特徴ベクトルなど、任意次元を渡せる。
        self.state_dim = state_dim
        self.state_sum = {}   # chunk -> [Σstate[0], ..., Σstate[state_dim-1], n]

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

        state: その語を聞いた時の状態ベクトル（B原本は[空腹,眠気,不快]の3次元。
        F2ではstate_dim次元の任意ベクトル）。渡すと語↔状態の連合を累積する
        （報酬でなく共起の統計）。
        """
        chunk = self.segment(tokens, confidences)
        if chunk is not None:
            self.counts[chunk] = self.counts.get(chunk, 0) + 1
            if state is not None and len(state) >= self.state_dim:
                acc = self.state_sum.get(chunk)
                if acc is None:
                    acc = [0.0] * self.state_dim + [0]
                    self.state_sum[chunk] = acc
                for i in range(self.state_dim):
                    acc[i] += float(state[i])
                acc[self.state_dim] += 1
        return chunk

    def assoc(self, chunk):
        """
        B6-4：その語が結びつく状態の平均（state_dim次元のtuple）を返す（無ければNone）。
        """
        acc = self.state_sum.get(chunk)
        if not acc or acc[self.state_dim] == 0:
            return None
        n = acc[self.state_dim]
        return tuple(acc[i] / n for i in range(self.state_dim))

    def top(self, n=10):
        """頻度上位の語の型を返す。"""
        return sorted(self.counts.items(), key=lambda kv: -kv[1])[:n]

    def known(self, min_count=1):
        """min_count回以上出会った語の型の集合を返す（産出計画で使う候補）。"""
        return {k for k, v in self.counts.items() if v >= min_count}
