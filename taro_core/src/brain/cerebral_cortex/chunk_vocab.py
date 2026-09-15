"""塊の名簿（ChunkVocab） — 塊（モーラID列）に番号を振る器。

【出典】仕様_M5_塊レベル層_2026-09-07.md 後半§1。音の名簿
（taro_core/src/senses/hearing.py の Vocabulary）と同じ流儀＝聞いて新しい
塊が立てば増える。人は整理しない（断片も1つの単位として登録される）。

塊＝タプル(int, int, ...)（音のIDの並び。Lexicon.segment_all/segment_end_prob
が切り出した1単位そのもの）。id空間は音の名簿（Vocabulary）とは別。
"""


class ChunkVocab:
    """塊(tuple[int]) <-> id の対応を動的に構築する。

    id: 0 <PAD>, 1 <BOS>, 2 <EOS>、以降は add_special で登録した特殊トークン
    （<PARENT>・<SELF>・<GONE>・<HERE>）と、encode_chunk で新規登録された塊。
    """

    def __init__(self):
        self.chunk2idx = {}
        self.idx2chunk = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>"}
        self.specials = {}
        self._next_id = 3

    @property
    def size(self):
        """現在登録されている塊・特殊トークンの総数（次に割り当てるid）を返す。引数は無い（self のみ）。"""
        return self._next_id

    def add_special(self, name):
        """複数文字の特殊トークン（"<PARENT>" 等）を1つのidとして登録する。

        既に登録済みならそのidを返す（音の名簿 Vocabulary.add_special と同じ流儀）。
        """
        if name in self.specials:
            return self.specials[name]
        idx = self._next_id
        self._next_id += 1
        self.specials[name] = idx
        self.idx2chunk[idx] = name
        return idx

    def encode_chunk(self, chunk):
        """塊(tuple[int])のidを返す。無ければ新規に割り当てる。"""
        chunk = tuple(int(t) for t in chunk)
        if chunk not in self.chunk2idx:
            idx = self._next_id
            self._next_id += 1
            self.chunk2idx[chunk] = idx
            self.idx2chunk[idx] = chunk
        return self.chunk2idx[chunk]

    def encode_chunks(self, chunks):
        """塊のリストをidのリストにする（encode_chunkを順に呼ぶだけ）。"""
        return [self.encode_chunk(c) for c in chunks]

    def chunk_string(self, idx, hearing_vocab):
        """idを文字列にする（特殊トークンは""、塊はhearing_vocab.decodeで文字列に）。"""
        val = self.idx2chunk.get(idx)
        if val is None:
            return ""
        if isinstance(val, str):
            # <PAD>/<BOS>/<EOS>/<PARENT>/<SELF>/<GONE>/<HERE> 等の特殊トークン
            return ""
        return hearing_vocab.decode(list(val))

    def state_dict(self):
        """保存用（torch.save可能な素のdict/listのみ。tupleはlist化）。"""
        return {
            "chunk2idx": [[list(k), v] for k, v in self.chunk2idx.items()],
            "specials": dict(self.specials),
            "next_id": self._next_id,
        }

    def load_state_dict(self, d):
        """state_dict形式の辞書 d を受け取り、chunk2idx・idx2chunk・specials・_next_id を復元する。返り値は無く、自身の状態を書き換える。
        """
        self.chunk2idx = {}
        self.idx2chunk = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>"}
        for k, v in d.get("chunk2idx", []):
            chunk = tuple(int(x) for x in k)
            v = int(v)
            self.chunk2idx[chunk] = v
            self.idx2chunk[v] = chunk
        self.specials = {}
        for name, v in d.get("specials", {}).items():
            v = int(v)
            self.specials[name] = v
            self.idx2chunk[v] = name
        self._next_id = int(d.get("next_id", 3))
