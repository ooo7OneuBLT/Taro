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

    def observe(self, tokens, confidences, state=None):
        """
        発話を分節し、切り出した単位を辞書に登録（頻度+1）。切り出した単位を返す。

        state: その語を聞いた時の状態ベクトル（B原本は[空腹,眠気,不快]の3次元。
        F2ではstate_dim次元の任意ベクトル）。渡すと語↔状態の連合を累積する
        （報酬でなく共起の統計）。
        """
        # 【F2-12・2026-08-27】stateがリストなら {"vision": リスト} に正規化する
        #   （呼び出し側 run/trainer.py は変えない＝これまでどおりリストを渡す）。
        #   いまはvisionチャンネルしか無いので、以下の処理は正規化前と完全に同じ。
        state = _as_channels(state).get(self.DEFAULT_CHANNEL)
        chunk = self.segment(tokens, confidences)
        if chunk is not None:
            self.counts[chunk] = self.counts.get(chunk, 0) + 1
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
        return chunk

    def observe_view(self, vec):
        """いま見えているものを「見慣れた景色」の平均へ1つ足す。

        【F2-8・2026-08-25】reverse_lookup の引き算に使う平均を育てるだけで、
        語彙(counts)にも想像(proto)にも触らない。呼ばれなければ view_n=0 のまま
        ＝ reverse_lookup は見え側の引き算をしない（従来と同じ計算に戻る）。
        """
        # 【F2-12・2026-08-27】vecがリストなら {"vision": リスト} に正規化する
        #   （呼び出し側 run/trainer.py は変えない）。いまはvisionチャンネルしか
        #   無いので、以下の処理は正規化前と完全に同じ。
        vec = _as_channels(vec).get(self.DEFAULT_CHANNEL)
        if vec is None:
            return
        D = self.state_dim
        if self.view_sum is None:
            self.view_sum = [0.0] * D
        for i in range(D):
            self.view_sum[i] += float(vec[i])
        self.view_n += 1

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
