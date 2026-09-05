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


def _cosine(a, b):
    """コサイン類似度。どちらかがゼロベクトルなら0.0を返す。

    【F2・2026-08-22】reverse_lookup（逆引き）専用のヘルパー。_unit()と役割は
    近いが、_unit()は「1つのベクトルを正規化して返す」のに対しこちらは
    「2つのベクトルの向きの近さを直接返す」。run/trainer.py の_cosine_sim()と
    同じ規約（ゼロノルムは0.0）だが、Lexicon側の入力はPythonのlist/tupleを
    素直に扱えるよう外部ライブラリ（numpy）に依存しない実装にした。
    """
    na = sum(float(x) * float(x) for x in a) ** 0.5
    nb = sum(float(x) * float(x) for x in b) ** 0.5
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    return dot / (na * nb)


def _unit(vec):
    """L2正規化したリストを返す（ゼロノルムならNone＝呼び出し側でスキップ）。

    【F1-5】対照学習は「向き」だけを扱う（引き寄せ・引き離しの式が内積的な
    差分更新のため、ノルムが揃っていないと語ごとに更新量が不公平になる）。
    """
    s = sum(float(x) * float(x) for x in vec) ** 0.5
    if s <= 0.0:
        return None
    return [float(x) / s for x in vec]


def _as_channels(state):
    """入力を「チャンネル名 -> ベクトル」の辞書に正規化する。

    【F2-12・2026-08-27】設計：F/docs/設計_F2-12_意味を感覚ごとに分けて持つ.md。
    observe / observe_view / reverse_lookup の3つの入口で使う。呼び出し側
    （run/trainer.py）は変えない＝これまでどおり生のリストを渡してくる。
    リストが来たら {"vision": リスト} と解釈することで、将来 {"vision": [...],
    "touch": [...]} のような辞書を渡す呼び出し側にもそのまま対応できる。
    """
    if state is None:
        return {}
    if isinstance(state, dict):
        return state
    return {Lexicon.DEFAULT_CHANNEL: state}


class Lexicon:
    """
    原-辞書。聞いた発話から予測しやすい並びを切り出して頻度を数える。

    counts: tuple(tokens) → 出会った回数
    """

    # 【F2-12・2026-08-27】意味を感覚ごとの引き出し（チャンネル）に分ける器。
    #   いまは "vision" チャンネルしか作らないので、計算結果はこれまでと
    #   完全に同一（設計の受け入れ条件1〜3）。
    DEFAULT_CHANNEL = "vision"

    def __init__(self, min_len=2, state_dim=3, mode="sum",
                 eta_pull=0.2, eta_push=0.1):
        # min_len：単位として登録する最短の長さ（1音は語の型とみなさない）。
        # 注意：構造的な下限であって調整用の恣意的定数ではない（1にすると全単音が語になる）。
        self.min_len = min_len
        self.counts = {}
        # 【分節第2案・2026-09-03】設計_分節（語の切れ目の発見）.md 第2案 第2部
        #   「発話まるごとの記録」節。単独で聞いたことのある発話全体を記録する
        #   （既知語を足がかりに切り出す segment_end_prob 専用。従来の counts/segment
        #   系には一切参照されない新設の器なので、ここに1行足すだけでは既存挙動は
        #   1ビットも変わらない）。
        self.utterance_counts = {}
        self.end_prob_sum = 0.0      # 【分節第2案】終わり確率の走行平均（発話末尾を除く全位置）
        self.end_prob_n = 0
        # B6-4：語↔内的状態の連合（cross-situational statistical learning, Smith & Yu）。
        # その語を聞いた時の状態ベクトル（B原本は[空腹,眠気,不快]の3次元固定）を語ごとに
        # 累積し、平均を「その語が結びつく状態」とする。報酬なし・共起の統計だけ。
        # F2：state_dim をコンストラクタ引数化（B原本は3固定）。視覚が無ければ
        # 内的状態、視覚があれば見えているものの特徴ベクトルなど、任意次元を渡せる。
        #
        # 【F2-12・2026-08-27】意味の入れ物を「感覚ごとの引き出し」に変えた。
        #   state_dim/proto/view_sum/view_n は、以前は素の属性だったが、いまは
        #   channels["vision"] を指すプロパティ（下に定義）。既存の読み出しコード
        #   （self.lexicon.proto など）はそのまま動く。中身の数字は1個も変えない。
        self.channels = {
            self.DEFAULT_CHANNEL: {
                "dim": state_dim,
                "proto": {},
                "view_sum": None,
                "view_n": 0,
            }
        }
        self.state_sum = {}   # chunk -> [Σstate[0], ..., Σstate[state_dim-1], n]
        # 【F1-5・2026-08-21】連合器の対照学習化（設計 F/docs/設計_F1-5_連合器の
        #   対照学習化.md）。mode="sum"（既定）は上のstate_sumによる単純平均のまま
        #   ＝1ビットも変わらない。mode="contrast"のときだけ、聞いた語の想像を
        #   いま見ている指紋へ近づけ（引き寄せ）、他の全語の想像をそこから遠ざける
        #   （引き離し）。背景等の共通成分は引き寄せと引き離しで相殺され、語ごとの
        #   「らしさ」の差だけが残る（CLIP型対照学習の最小オンライン版）。
        #   人間側対応：乳児の語彙学習は共起だけでなく非共起も使う（相互排他性の
        #   基盤）[Tier2・原理レベル]。
        self.mode = str(mode)
        self.eta_pull = float(eta_pull)
        self.eta_push = float(eta_push)
        # self.proto（contrast用: chunk -> [p_0..p_{state_dim-1}]）は
        # channels["vision"]["proto"] を指すプロパティ（下に定義）。
        # 【F2-8・2026-08-25】reverse_lookup（物→語）の両側引き算に使う
        #   「見慣れた景色」の平均。observe_view() で見るたびに足していく累積平均。
        #   なぜ要るか：見えのベクトルは4分の3ほどが全物体に共通で（実測：見え同士の
        #   コサインが0.72〜0.76）、そのまま比べると共通部分が勝負を決めてしまい
        #   「どの物を見ても同じ語」になる（実測：ぱぱの物を136回見せて正解0回）。
        #   なぜ「これから見る物の平均」ではなく走行平均か：未来に見る物の平均を
        #   使うのはカンニング（太郎がまだ見ていない物を知っていることになる）。
        #   人間側対応：乳児も「いつも見えている景色」に順応し、そこからの差に
        #   注目する[Tier2・原理レベル]。
        # self.view_sum/self.view_n（[Σv_0..Σv_{state_dim-1}]・観測回数）も
        # channels["vision"] を指すプロパティ（下に定義）。

    # ------------------------------------------------------------------
    # 【F2-12・2026-08-27】channels["vision"] を指すプロパティ群。
    #   既存の読み出しコード（self.lexicon.state_dim / .proto / .view_sum /
    #   .view_n）を1行も変えずに動かすための互換レイヤー。中身は全部
    #   channels["vision"] という同じ辞書を見ているだけで、実体は1つ。
    # ------------------------------------------------------------------
    @property
    def state_dim(self):
        return self.channels[self.DEFAULT_CHANNEL]["dim"]

    @state_dim.setter
    def state_dim(self, value):
        self.channels[self.DEFAULT_CHANNEL]["dim"] = value

    @property
    def proto(self):
        return self.channels[self.DEFAULT_CHANNEL]["proto"]

    @proto.setter
    def proto(self, value):
        self.channels[self.DEFAULT_CHANNEL]["proto"] = value

    @property
    def view_sum(self):
        return self.channels[self.DEFAULT_CHANNEL]["view_sum"]

    @view_sum.setter
    def view_sum(self, value):
        self.channels[self.DEFAULT_CHANNEL]["view_sum"] = value

    @property
    def view_n(self):
        return self.channels[self.DEFAULT_CHANNEL]["view_n"]

    @view_n.setter
    def view_n(self, value):
        self.channels[self.DEFAULT_CHANNEL]["view_n"] = value

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

    def segment_all(self, tokens, confidences):
        """発話を分節し、min_len以上の**全単位**をリストで返す（2026-08-31・二語文）。

        segment() は最長の1単位だけを返す（B6-1bの判断＝ノイズの谷で出る断片を
        捨てるため）。二語文への道では「わんわんいた」から「わんわん」と「いた」の
        **両方**を拾いたいので、全単位を返す入口を足した。
        自信度が全て同値（従来の1.0決め打ち）のときは谷が1つもできず単位は
        発話全体の1つだけ＝segment()と完全に同じ結果になる（後方互換）。
        """
        n = len(tokens)
        if n < self.min_len or len(confidences) != n:
            return []
        boundaries = {0, n}
        for j in range(1, n - 1):
            if confidences[j] < confidences[j - 1] and confidences[j] < confidences[j + 1]:
                boundaries.add(j)
        bs = sorted(boundaries)
        out = []
        for i in range(len(bs) - 1):
            s_, e_ = bs[i], bs[i + 1]
            if e_ - s_ >= self.min_len:
                out.append(tuple(tokens[s_:e_]))
        return out

    def segment_end_prob(self, tokens, confidences, end_probs):
        """発話を分節し、min_len以上の全単位をリストで返す（分節第2案・2026-09-03）。

        設計_分節（語の切れ目の発見）.md 第2案 第2部「segment_end_prob」節。
        谷の深さの閾値（第1案）は使わない。境界＝{0,n}
        ∪ 既知語（utterance_counts に単独発話として記録済みの列。長い順に、
          重ならないように探し、両端を境界にする）
        ∪ 終わり確率の切れ目（その発話の平均より高い位置。Christiansen 1998と同じ規準）。
        """
        n = len(tokens)
        if n < self.min_len:
            return []
        boundaries = {0, n}
        # 足がかり：既知語（長い順）を、重ならないように tokens の中で探す。
        # 【なぜ len(k) < n のときだけ、2026-09-03】observe() はこの関数を呼ぶ
        #   "前"に今回の発話まるごとを utterance_counts へ足す（設計の指定順）。
        #   そのため候補に自分自身（長さn）が必ず混ざる。自分自身は{0,n}という
        #   何も足さない境界しか作らないのに、長い順マッチで真っ先に採用されると
        #   tokens全域を「使用済み」にしてしまい、本来見つかるはずの短い既知語
        #   （「おわん」等）を覆い隠してしまう（実装時に机上で発覚）。境界の集合には
        #   影響しない自明な一致なので、footingの探索対象からは除く。
        known = sorted((k for k in self.utterance_counts
                        if self.min_len <= len(k) < n),
                       key=len, reverse=True)
        used = [False] * n
        for k in known:
            klen = len(k)
            for s in range(0, n - klen + 1):
                if any(used[s:s + klen]):
                    continue
                if tuple(tokens[s:s + klen]) == k:
                    for i in range(s, s + klen):
                        used[i] = True
                    boundaries.add(s)
                    boundaries.add(s + klen)
        # 終わり確率の切れ目（Christiansen, Allen & Seidenberg 1998と同じ規準：
        #   固定閾値ではなく、**これまで聞いた全位置の平均**より高い位置）。
        # 【2026-09-03・机上確認で修正】「その発話の平均」だと、単独発話（「がおー」）
        #   の内側で値がすべて微小でも最大値が平均を超えて切れ、断片（がお・コッ）が
        #   単独発話の回数ぶん出た。原法どおりコーパス平均（発話末尾を除く全位置の
        #   走行平均）と比べる。学習前は全位置 0 なので何も切れない。
        if end_probs and len(end_probs) == n:
            inner = end_probs[:-1]   # 最後のトークン直後（発話の末尾そのもの）は除く
            if inner:
                self.end_prob_sum += float(sum(inner))
                self.end_prob_n += len(inner)
                m = self.end_prob_sum / self.end_prob_n
                for i, p in enumerate(inner):
                    b = i + 1
                    # 【2026-09-03・F2-50bで発覚】知っている塊の内側では統計の切れ目を
                    #   使わない（足がかりの本来の意味＝既知の塊は丸ごと読む。CLASSIC-UB／
                    #   手がかり階層：語彙の知識＞音の統計・Mattys et al. 2005）。
                    #   「がおー」が「お」で終わるため「おさら」の頭の「お」の後で
                    #   終わり確率が上がり「お｜さら」と割れていた。両端は既に境界。
                    if used[b - 1] and used[b] and b not in boundaries:
                        continue
                    if p > m and p > 0:
                        boundaries.add(b)
        bs = sorted(boundaries)
        out = []
        for i in range(len(bs) - 1):
            s_, e_ = bs[i], bs[i + 1]
            if e_ - s_ >= self.min_len:
                out.append(tuple(tokens[s_:e_]))
        return out

    def _ensure_channel(self, name, dim):
        """そのチャンネルが無ければ作る（2026-08-28・F2-13）。

        中心窩(vision)以外の感覚を後から足せるようにするための入口。
        既存チャンネルには一切触らないので、1チャンネルだけの走行では
        この関数は最初の1回で何もせず戻る（計算は従来と同一）。
        """
        ch = self.channels.get(name)
        if ch is None:
            ch = {"dim": int(dim), "proto": {}, "view_sum": None, "view_n": 0}
            self.channels[name] = ch
        return ch

    def observe(self, tokens, confidences, state=None, end_probs=None, mode="valley"):
        """
        発話を分節し、切り出した単位を辞書に登録（頻度+1）。切り出した単位を返す。

        state: その語を聞いた時の状態ベクトル（B原本は[空腹,眠気,不快]の3次元。
        F2ではstate_dim次元の任意ベクトル）。渡すと語↔状態の連合を累積する
        （報酬でなく共起の統計）。

        【分節第2案・2026-09-03】設計_分節（語の切れ目の発見）.md 第2案 第2部
        「observe の拡張」節。end_probs/mode は既定値（None/"valley"）のままなら
        下の分岐で必ず従来の segment_all が呼ばれる＝1ビットも変わらない。
        """
        # 【分節第2案・2026-09-03】発話まるごとの記録（同節「発話まるごとの記録」）。
        #   utterance_counts は segment_end_prob 専用の新設の器で、従来の
        #   counts/segment系からは一切参照されないため、この1行だけでは
        #   既存挙動は変わらない。
        _utok = tuple(tokens)
        self.utterance_counts[_utok] = self.utterance_counts.get(_utok, 0) + 1
        # 【F2-12・2026-08-27】stateがリストなら {"vision": リスト} に正規化する
        #   （呼び出し側 run/trainer.py は変えない＝これまでどおりリストを渡す）。
        #   いまはvisionチャンネルしか無いので、以下の処理は正規化前と完全に同じ。
        states = _as_channels(state)
        state = states.get(self.DEFAULT_CHANNEL)
        # 【二語文・2026-08-31】最長の1つだけでなく全単位を登録する。
        #   自信度が全て同値（1.0決め打ち）なら単位は1つだけ＝従来と同一。
        #   返り値は従来どおり最長の単位（word_attention 等の互換のため）。
        # 【分節第2案・2026-09-03】mode=="end_prob" かつ end_probs が有効な長さの
        #   ときだけ新しい切り出し（segment_end_prob）を使う。それ以外（既定
        #   mode="valley"）は従来の segment_all のまま。
        if mode == "end_prob" and end_probs is not None and len(end_probs) == len(tokens):
            chunks = self.segment_end_prob(tokens, confidences, end_probs)
        else:
            chunks = self.segment_all(tokens, confidences)
        chunk = max(chunks, key=len) if chunks else None
        for chunk_ in chunks:
            self._register(chunk_, states, state)
        return chunk

    def _register(self, chunk, states, state):
        """切り出した1単位を辞書へ登録し、連合を1回ぶん学習する（observeの中身の切り出し）。"""
        if chunk is not None:
            self.counts[chunk] = self.counts.get(chunk, 0) + 1
            # 【F2-13・2026-08-28】vision 以外のチャンネル（例：周辺視）も同じ
            #   引き寄せ／引き離しで育てる。チャンネルが1つだけの走行では
            #   このループは空を回るだけで、計算は従来と1ビットも変わらない。
            for _name, _vec in states.items():
                if _name == self.DEFAULT_CHANNEL or _vec is None:
                    continue
                self._learn_channel(_name, chunk, _vec)
            if state is not None and len(state) >= self.state_dim:
                # sum系の累積は contrast モードでも並行して行う（counts/segment等の
                # 他機能の互換のため。二重帳簿だが数KBオーダーで無視できる。
                # F/docs/設計_F1-5…技術付録より）。
                acc = self.state_sum.get(chunk)
                if acc is None:
                    acc = [0.0] * self.state_dim + [0]
                    self.state_sum[chunk] = acc
                for i in range(self.state_dim):
                    acc[i] += float(state[i])
                acc[self.state_dim] += 1
                # 【F1-5】mode=="sum"ならここで終わり＝従来コードのみ実行、1ビットも
                #   変わらない。mode=="contrast"のときだけ引き寄せ＋引き離しを行う。
                if self.mode == "contrast":
                    v = _unit(state[:self.state_dim])
                    if v is not None:
                        p = self.proto.get(chunk)
                        if p is None:
                            # 初回はこの語の想像がまだ無いので、現在の見えをそのまま
                            # 初期値にする（引き寄せの式では前がゼロだと不安定なため）。
                            p = list(v)
                            self.proto[chunk] = p
                        else:
                            # 引き寄せ（EMA）：聞いた語の想像を今の指紋へ少し近づける。
                            for i in range(self.state_dim):
                                p[i] += self.eta_pull * (v[i] - p[i])
                        # 引き離し：他の全語の想像を今の指紋から遠ざける。
                        # q ← q − η⁻(v − q)。η⁻<η⁺とし、全語の瞬間に共通して写る
                        # 背景成分は引き寄せと引き離しで相殺される（設計2026-08-21）。
                        for other, q in self.proto.items():
                            if other == chunk:
                                continue
                            for i in range(self.state_dim):
                                q[i] -= self.eta_push * (v[i] - q[i])

    def _learn_channel(self, name, chunk, vec):
        """vision 以外のチャンネル1つ分の引き寄せ／引き離し（2026-08-28・F2-13）。

        式は vision と完全に同じ（observe 内の contrast 分岐をそのまま写した）。
        eta_push=0 のときは引き離しが no-op になる＝F2-11の判断（引き離しは外す）
        がそのままこのチャンネルにも効く。
        """
        if self.mode != "contrast":
            return
        ch = self._ensure_channel(name, len(vec))
        D = ch["dim"]
        if len(vec) < D:
            return
        v = _unit(list(vec)[:D])
        if v is None:
            return
        proto = ch["proto"]
        p = proto.get(chunk)
        if p is None:
            proto[chunk] = list(v)
        else:
            for i in range(D):
                p[i] += self.eta_pull * (v[i] - p[i])
        for other, q in proto.items():
            if other == chunk:
                continue
            for i in range(D):
                q[i] -= self.eta_push * (v[i] - q[i])

    def observe_view(self, vec):
        """いま見えているものを「見慣れた景色」の平均へ1つ足す。

        【F2-8・2026-08-25】reverse_lookup の引き算に使う平均を育てるだけで、
        語彙(counts)にも想像(proto)にも触らない。呼ばれなければ view_n=0 のまま
        ＝ reverse_lookup は見え側の引き算をしない（従来と同じ計算に戻る）。
        """
        # 【F2-12・2026-08-27】vecがリストなら {"vision": リスト} に正規化する
        #   （呼び出し側 run/trainer.py は変えない）。いまはvisionチャンネルしか
        #   無いので、以下の処理は正規化前と完全に同じ。
        vecs = _as_channels(vec)
        for name, v in vecs.items():
            if v is None:
                continue
            # 【F2-13・2026-08-28】チャンネルごとに「見慣れた景色」を持つ。
            #   vision だけの走行では従来と完全に同じ1本を育てる。
            ch = self._ensure_channel(name, len(v))
            D = ch["dim"]
            if len(v) < D:
                continue
            if ch["view_sum"] is None:
                ch["view_sum"] = [0.0] * D
            acc = ch["view_sum"]
            for i in range(D):
                acc[i] += float(v[i])
            ch["view_n"] = ch.get("view_n", 0) + 1

    def assoc(self, chunk):
        """
        B6-4：その語が結びつく状態を返す（無ければNone）。
        mode=="sum"（既定）：単純平均（state_dim次元のtuple）＝従来どおり。
        mode=="contrast"：対照学習で育てたproto（引き寄せ＋引き離し後のベクトル）。
        """
        if self.mode == "contrast":
            p = self.proto.get(chunk)
            return tuple(p) if p is not None else None
        acc = self.state_sum.get(chunk)
        if not acc or acc[self.state_dim] == 0:
            return None
        n = acc[self.state_dim]
        return tuple(acc[i] / n for i in range(self.state_dim))

    def reverse_lookup(self, vec):
        """物→語の逆引き（F2「見た物の名前を言う」、設計：F/docs/設計_F2_初語
        （見た物の名前を言う）.md 第2部）。

        いま見ている視覚ベクトル(vec, state_dim次元)を、self.proto の全chunk
        （対照学習で育った「その語の想像」）と総当たりでコサイン類似度を取り、
        最大のchunkとその類似度を返す。assoc()（語→物）のちょうど逆方向。

        mode!="contrast"（protoが育たない設定）や proto が空（まだ何も
        連合していない）ときは (None, 0.0) を返す（呼び出し側は「何も言わない」
        に倒す規約。run/trainer.py の_apply_word_production参照）。

        戻り値: (chunk, sim)。chunk は self.proto のキー（tuple）または None。
        """
        if self.mode != "contrast":
            return None, 0.0
        # 【F2-12・2026-08-27】vecがリストなら {"vision": リスト} に正規化する
        #   （呼び出し側 run/trainer.py は変えない）。チャンネルごとに
        #   _channel_sims でコサインを求め、束ねる。
        #   いまはvisionチャンネルしか proto が育たないので active は常に
        #   ["vision"] 以下＝単一チャンネル分岐しか実行されず、計算結果は
        #   正規化前と完全に同じ（設計の受け入れ条件1〜3）。
        channels_in = _as_channels(vec)
        active = [name for name in channels_in
                  if name in self.channels and self.channels[name]["proto"]]
        if not active:
            return None, 0.0
        if len(active) == 1:
            sims = self._channel_sims(active[0], channels_in[active[0]])
        else:
            # 複数チャンネル：暫定で「持っているチャンネルのコサインの単純平均」。
            # 【未決・F2-12】束ね方は2つ目の感覚を足すときに設計する。
            sums, counts = {}, {}
            for name in active:
                for chunk, sim in self._channel_sims(name, channels_in[name]).items():
                    sums[chunk] = sums.get(chunk, 0.0) + sim
                    counts[chunk] = counts.get(chunk, 0) + 1
            sims = {c: sums[c] / counts[c] for c in sums}
        if not sims:
            return None, 0.0
        best_chunk = max(sims, key=sims.get)
        return best_chunk, sims[best_chunk]

    def reverse_scores(self, vec):
        """逆引きの全候補の点数（chunk → コサイン）を返す（2026-08-30・文脈設計）。

        reverse_lookup と同じ計算で、1位だけでなく全chunkの点数を返す。
        文脈（GRU）の予測を点数に足し込むとき（設計_文脈（コンテキスト）.md 決定4）、
        呼び出し側が全候補を採点し直せるようにするための読み出し口。
        辞書が空・contrastモードでないときは空辞書。
        """
        if self.mode != "contrast":
            return {}
        channels_in = _as_channels(vec)
        active = [name for name in channels_in
                  if name in self.channels and self.channels[name]["proto"]]
        if not active:
            return {}
        if len(active) == 1:
            return dict(self._channel_sims(active[0], channels_in[active[0]]))
        sums, counts = {}, {}
        for name in active:
            for chunk, sim in self._channel_sims(name, channels_in[name]).items():
                sums[chunk] = sums.get(chunk, 0.0) + sim
                counts[chunk] = counts.get(chunk, 0) + 1
        return {c: sums[c] / counts[c] for c in sums}

    def _channel_sims(self, name, vec):
        """指定チャンネル1つ分の (chunk -> コサイン類似度) を返す。

        【F2-12・2026-08-27】reverse_lookup の中身を、旧来のvision専用計算
        （両側引き算＋総当たりコサイン）からチャンネル名を引数化しただけの形に
        切り出したもの。ロジック自体は1行も変えていない。
        """
        ch = self.channels[name]
        D = ch["dim"]
        # 【F2-8・2026-08-25】両側引き算。
        #   なぜ入れたか：ここは以前 _cosine(vec, p) を生のまま比べていたが、
        #   見えのベクトルも語の想像も4分の3ほどが共通成分で、そのまま比べると
        #   共通部分が勝負を決める。実測（F2-7、4語・551回の産出）では
        #   「どの物を見せても同じ語」になり、ぱぱの物は136回中0回しか当たらなかった。
        #   同じ4本のベクトルで引き算ありに直すと 2/4 → 3/4 になる（同日実測）。
        #   なお語→物の向きのテスト（F/scripts/f_wordreadout_nway.py の
        #   two_sided_scores_nway）は最初から両側引き算をしていた。同じ比較を
        #   2か所に別々に実装し、片方だけ引き算を入れ忘れていた＝二重実装の食い違い。
        #
        #   ⚠ 閾値のスケールが変わる：引き算前は 0.3〜0.9 に固まっていた値が、
        #   引き算後は −0.5〜+0.6 に広がる。実験ファイルの produce.threshold を
        #   0.8 のままにすると**一度も発話しない**。新しいスケールに合わせること。
        v = [float(x) for x in vec[:D]]
        if ch["view_n"] > 0:
            # 見え側：これまで見てきた景色の平均を引く（observe_view で育てた値）
            v = [v[i] - ch["view_sum"][i] / ch["view_n"] for i in range(D)]
        items = list(ch["proto"].items())
        if len(items) >= 2:
            # 語側：全語の想像の平均を引く（1語しか無いときは引くと零ベクトルになる）
            n = len(items)
            m = [sum(p[i] for _, p in items) / n for i in range(D)]
        else:
            m = None
        sims = {}
        for chunk, p in items:
            q = p if m is None else [p[i] - m[i] for i in range(D)]
            sims[chunk] = _cosine(v, q)
        return sims

    def top(self, n=10):
        """頻度上位の語の型を返す。"""
        return sorted(self.counts.items(), key=lambda kv: -kv[1])[:n]

    def known(self, min_count=1):
        """min_count回以上出会った語の型の集合を返す（産出計画で使う候補）。"""
        return {k for k, v in self.counts.items() if v >= min_count}
