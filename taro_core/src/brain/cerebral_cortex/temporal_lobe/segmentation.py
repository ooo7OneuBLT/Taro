# -*- coding: utf-8 -*-
"""分節 — 聞いた音の列を「語の塊」に切り分ける（側頭葉）

【2026-09-15】`lexicon.py`（原-辞書）から**切り分けだけ**を取り出したもの。
元のファイルには「意味の表」（語ごとの見えの平均 `proto`・逆引き `reverse_lookup`）
が同居していたが、そちらは 2026-09-02 の段階2（`word_choice="gru_hippo"`）以降
発話の経路から外れており、誰も読んでいなかったため削除した。
経緯は `doc/状況整理/状況整理_語彙の仕組みをどうするか_2026-09-14.md` §11。
呼称は `現在地.md` の「呼称」節。

【人間模倣】統計的分節（Saffran, Aslin & Newport 1996）＋語テンプレート（Vihman）。
乳児は連続音声を「次の音の予測しやすさ」を手がかりに区切り、繰り返し出会う
予測しやすい並びを1つの単位（語の型）として記憶する。報酬（ドーパミン）は使わない
＝教師なし。予測の自信度と出会った頻度だけで単位が立ち上がる。

太郎では：本体GRUの知覚ヘッドが各音を直前から予測した自信度（＝遷移確率）を手がかりに、
「予測が谷になる所（＝単語の境目）」の間を1単位として切り出す。絶対的な決め打ち
閾値は使わず、隣同士の相対的な上下関係（局所的な谷）だけで境界を決める（Saffranの
「語の内部は遷移確率が高く、語の境目で下がる」に対応）。

B6-1b修正：初版は「発話内平均以上の連続run」で切っていたが、自信度は文脈が
積み上がるほど（＝語の後半ほど）上がりやすく、語の頭が切り落とされ後半だけが
単位化される偏りがあった（例：「まんま」でなく「んま」）。局所的な谷（隣より
低い点）で境界を引く方式に変えて、この偏りを避ける。

切り出した塊は `last_chunks` に置かれ、`chunk_vocab`（塊の名簿）を通じて
塊GRU・言語海馬へ流れる。
"""


class Segmenter:
    """聞いた発話を語の塊に切り分け、塊ごとの出会った回数を数える。"""

    def __init__(self, min_len=2):
        # min_len：単位として登録する最短の長さ（1音は語の型とみなさない）。
        # 注意：構造的な下限であって調整用の恣意的定数ではない（1にすると全単音が語になる）。
        self.min_len = min_len
        # 塊 -> 出会った回数（在庫）。
        self.counts = {}
        # 【塊レベル層・2026-09-07】仕様_M5_塊レベル層.md §1・§3。observe()が
        #   直近の分節結果（塊の列）を置く場所。まだ一度もobserveしていない
        #   状態でも呼び出し側が空リストとして扱えるよう空で初期化しておく。
        self.last_chunks = []
        # 【分節第2案・2026-09-03】設計_分節（語の切れ目の発見）.md 第2案 第2部
        #   「発話まるごとの記録」節。単独で聞いたことのある発話全体を記録する
        #   （既知語を足がかりに切り出す segment_end_prob 専用）。
        self.utterance_counts = {}
        self.end_prob_sum = 0.0      # 終わり確率の走行平均（発話末尾を除く全位置）
        self.end_prob_n = 0

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
        ∪ 終わり確率の切れ目（**これまでの全発話にわたる走行平均**より高い位置。
          Christiansen 1998と同じくコーパス平均と比べる。
          注意：「その発話の平均」ではない。2026-09-03に机上確認で直した。
          end_prob_sum/end_prob_n は初期化以降ずっと積み上がる）。
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

    def observe(self, tokens, confidences, end_probs=None, mode="valley"):
        """発話を分節し、切り出した塊を数えて、いちばん長い塊を返す。

        切り出した塊の列そのものは `self.last_chunks` に置く（塊GRU・言語海馬が使う）。

        【2026-09-15】以前はここで state（見えのベクトル）も受け取り、語ごとの
        「見えの平均」を育てていたが、それは誰も読んでいなかったため削除した。
        引数 state は無くなった。
        """
        # 【分節第2案・2026-09-03】発話まるごとの記録（同節「発話まるごとの記録」）。
        #   segment_end_prob 専用の器。
        _utok = tuple(tokens)
        self.utterance_counts[_utok] = self.utterance_counts.get(_utok, 0) + 1
        # 【二語文・2026-08-31】最長の1つだけでなく全単位を登録する。
        #   自信度が全て同値（1.0決め打ち）なら単位は1つだけ。
        # 【分節第2案・2026-09-03】mode=="end_prob" かつ end_probs が有効な長さの
        #   ときだけ新しい切り出し（segment_end_prob）を使う。
        if mode == "end_prob" and end_probs is not None and len(end_probs) == len(tokens):
            chunks = self.segment_end_prob(tokens, confidences, end_probs)
        else:
            chunks = self.segment_all(tokens, confidences)
        # 【塊レベル層・2026-09-07】仕様_M5_塊レベル層.md §1。空なら
        #   [tuple(tokens)] を置く（塊が1つも切り出せない＝発話全体が1つの塊、
        #   という仕様§3の規約）。
        self.last_chunks = list(chunks) if chunks else [tuple(tokens)]
        for chunk_ in chunks:
            if chunk_ is not None:
                self.counts[chunk_] = self.counts.get(chunk_, 0) + 1
        return max(chunks, key=len) if chunks else None

    def state_dict(self):
        """保存用（torch.save 可能な素の dict/list のみ）。"""
        return {
            "min_len": self.min_len,
            "counts": self.counts,
            "utterance_counts": self.utterance_counts,
            "end_prob_sum": self.end_prob_sum,
            "end_prob_n": self.end_prob_n,
        }

    def load_state_dict(self, d):
        """保存から復元する。古い `lexicon` 形式の blob もそのまま読める
        （意味の表のキーは無視する）。"""
        self.min_len = d.get("min_len", self.min_len)
        self.counts = d.get("counts", {})
        self.utterance_counts = d.get("utterance_counts", {})
        self.end_prob_sum = float(d.get("end_prob_sum", 0.0))
        self.end_prob_n = int(d.get("end_prob_n", 0))

    def top(self, n=10):
        """出会った回数の多い塊を n 個返す。"""
        return sorted(self.counts.items(), key=lambda kv: -kv[1])[:n]

    def known(self, chunk):
        """その塊を何回聞いたか。"""
        return self.counts.get(tuple(chunk), 0)
