"""
太郎の脳 — 予測する単一の再帰ネットワーク

【人間模倣】脳は「次に何が来るか」を絶えず予測する機械である（Friston）。
太郎の脳はこれを1つの再帰ネットで実装する。
「翻訳装置（seq2seq）」のような人間にない構造は持たない。

オウム返しは脳に組み込むのではなく、本能（報酬）に導かれて
「行動」として創発する。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Vocabulary:
    """見た文字から動的に語彙を構築する。"""

    def __init__(self):
        self.char2idx = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2}
        self.idx2char = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>"}
        self.size = 3

    def encode(self, text):
        """文字列 → トークン列（未知文字は自動追加）"""
        indices = []
        for ch in text:
            if ch not in self.char2idx:
                self.char2idx[ch] = self.size
                self.idx2char[self.size] = ch
                self.size += 1
            indices.append(self.char2idx[ch])
        return indices

    def decode(self, indices):
        """トークン列 → 文字列（特殊トークンは除外）"""
        chars = []
        for idx in indices:
            ch = self.idx2char.get(idx, "?")
            if ch not in ("<PAD>", "<BOS>", "<EOS>"):
                chars.append(ch)
        return "".join(chars)


class TaroBrain(nn.Module):
    """
    太郎の脳。次のトークンを予測し続ける単一の再帰ネット。

    知覚：入力トークンを受け取り、次を予測（予測処理）
    行動：温度τでサンプリングし文字を産出（初期は喃語）
    """

    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128,
                 num_layers=1, temperature=2.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.temperature = temperature

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """
        x: (batch, seq_len) のトークン列
        戻り値: logits (batch, seq_len, vocab_size), hidden
        """
        emb = self.embedding(x)
        out, hidden = self.gru(emb, hidden)
        logits = self.output_layer(out)
        return logits, hidden

    def predict_next(self, token_idx, hidden=None):
        """
        1トークンを受け取り、次トークンの確率分布を返す（知覚）。

        戻り値: probs (vocab_size,), hidden
        """
        x = torch.tensor([[token_idx]], device=self._device())
        logits, hidden = self.forward(x, hidden)
        logits = logits[0, 0]
        probs = F.softmax(logits / max(self.temperature, 1e-8), dim=-1)
        return probs, hidden

    def generate(self, hidden, max_length, eos_idx):
        """
        太郎の番に文字を産出する（行動）。
        温度τでサンプリング＝初期は喃語（ばらつき大）、成長で安定。

        【人間模倣】喃語＝運動探索。τが高いと出力がランダム。

        戻り値: generated_indices, log_probs_list, hidden
        """
        generated = []
        log_probs = []
        bos = torch.tensor([[1]], device=self._device())  # <BOS>
        logits, hidden = self.forward(bos, hidden)

        for _ in range(max_length):
            logits_last = logits[0, -1]
            probs = F.softmax(logits_last / max(self.temperature, 1e-8), dim=-1)
            dist = torch.distributions.Categorical(probs)
            token = dist.sample()
            log_prob = dist.log_prob(token)

            if token.item() == eos_idx:
                break

            generated.append(token.item())
            log_probs.append(log_prob)

            token_input = token.unsqueeze(0).unsqueeze(0)
            logits, hidden = self.forward(token_input, hidden)

        return generated, log_probs, hidden

    def decay_temperature(self, decay_rate, min_temp):
        """温度τを減衰させる（発達＝発話の安定化）。"""
        self.temperature = max(self.temperature * decay_rate, min_temp)

    def resize_embedding(self, new_vocab_size):
        """語彙が増えたとき埋め込み層と出力層を拡張する。"""
        old_size = self.embedding.num_embeddings
        if new_vocab_size <= old_size:
            return

        old_emb_weight = self.embedding.weight.data
        new_emb = nn.Embedding(new_vocab_size, self.embedding.embedding_dim,
                               padding_idx=0)
        new_emb.weight.data[:old_size] = old_emb_weight
        self.embedding = new_emb

        old_out_weight = self.output_layer.weight.data
        old_out_bias = self.output_layer.bias.data
        new_out = nn.Linear(self.hidden_dim, new_vocab_size)
        new_out.weight.data[:old_size] = old_out_weight
        new_out.bias.data[:old_size] = old_out_bias
        self.output_layer = new_out

    def _device(self):
        return self.embedding.weight.device
