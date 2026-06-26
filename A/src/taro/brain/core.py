"""
太郎の脳（基本構造） — 予測する単一の再帰ネットワーク

【人間模倣】脳は「次に何が来るか」を絶えず予測する機械である（Friston）。
脳 →（口の動かし方）→ 声道シミュレータ → 文字
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from taro.body.vocal_tract import VocalTract, NUM_PLACE, NUM_MANNER, NUM_VOICING, NUM_VOWEL


class Vocabulary:
    """見た文字から動的に語彙を構築する。"""

    def __init__(self):
        self.char2idx = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2}
        self.idx2char = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>"}
        self.size = 3

    def encode(self, text):
        indices = []
        for ch in text:
            if ch not in self.char2idx:
                self.char2idx[ch] = self.size
                self.idx2char[self.size] = ch
                self.size += 1
            indices.append(self.char2idx[ch])
        return indices

    def decode(self, indices):
        chars = []
        for idx in indices:
            ch = self.idx2char.get(idx, "?")
            if ch not in ("<PAD>", "<BOS>", "<EOS>"):
                chars.append(ch)
        return "".join(chars)


class TaroBrain(nn.Module):
    """
    太郎の脳。

    知覚：入力トークン（文字）を受け取り、隠れ状態を更新（予測処理）
    行動：隠れ状態から口の4パラメータを出力 → 声道が文字に変換
    """

    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128,
                 num_layers=1, temperature=2.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.temperature = temperature
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)

        self.head_place = nn.Linear(hidden_dim, NUM_PLACE)
        self.head_manner = nn.Linear(hidden_dim, NUM_MANNER)
        self.head_voicing = nn.Linear(hidden_dim, NUM_VOICING)
        self.head_vowel = nn.Linear(hidden_dim, NUM_VOWEL)

        with torch.no_grad():
            self.head_place.bias.data[0] += 2.0
            self.head_manner.bias.data[0] += 2.0
            self.head_voicing.bias.data[1] += 1.0

        self.perception_head = nn.Linear(hidden_dim, vocab_size)

    def forward_hidden(self, x, hidden=None):
        emb = self.embedding(x)
        out, hidden = self.gru(emb, hidden)
        return out, hidden

    def forward_perception(self, x, hidden=None):
        out, hidden = self.forward_hidden(x, hidden)
        logits = self.perception_head(out)
        return logits, hidden

    def forward_articulation(self, gru_output):
        return (
            self.head_place(gru_output),
            self.head_manner(gru_output),
            self.head_voicing(gru_output),
            self.head_vowel(gru_output),
        )

    def generate(self, hidden, max_length, eos_idx, stamina=None, vocal_tract=None):
        if vocal_tract is None:
            vocal_tract = VocalTract()

        generated = []
        log_probs_all = []
        bos = torch.tensor([[1]], device=self._device())
        out, hidden = self.forward_hidden(bos, hidden)

        effective_max = max_length
        if stamina is not None:
            effective_max = min(max_length, int(stamina))

        allowed_place, allowed_manner, allowed_voicing, allowed_vowel = vocal_tract.get_allowed()

        for _ in range(effective_max):
            h_last = out[0, -1]
            pl, ml, vl, vol = self.forward_articulation(h_last)

            log_prob = torch.tensor(0.0, device=self._device())

            s_place, lp = self._sample_param(pl, allowed_place)
            log_prob = log_prob + lp
            s_manner, lp = self._sample_param(ml, allowed_manner)
            log_prob = log_prob + lp
            s_voicing, lp = self._sample_param(vl, allowed_voicing)
            log_prob = log_prob + lp
            s_vowel, lp = self._sample_param(vol, allowed_vowel)
            log_prob = log_prob + lp

            p = s_place if isinstance(s_place, int) else s_place.item()
            m = s_manner if isinstance(s_manner, int) else s_manner.item()
            v = s_voicing if isinstance(s_voicing, int) else s_voicing.item()
            w = s_vowel if isinstance(s_vowel, int) else s_vowel.item()
            char = vocal_tract.speak(p, m, v, w)

            if char in self._vocab_char2idx:
                token_idx = self._vocab_char2idx[char]
            else:
                break

            if token_idx == eos_idx:
                break

            generated.append(token_idx)
            log_probs_all.append(log_prob)

            token_input = torch.tensor([[token_idx]], device=self._device())
            out, hidden = self.forward_hidden(token_input, hidden)

        return generated, log_probs_all, hidden

    def _sample_param(self, logits, allowed_indices):
        if len(allowed_indices) == 1:
            return allowed_indices[0], torch.tensor(0.0, device=logits.device)

        mask = torch.full_like(logits, float("-inf"))
        for i in allowed_indices:
            mask[i] = 0.0
        masked_logits = logits + mask
        probs = F.softmax(masked_logits / max(self.temperature, 1e-8), dim=-1)
        dist = torch.distributions.Categorical(probs)
        sample = dist.sample()
        return sample.item(), dist.log_prob(sample)

    def set_vocab_mapping(self, char2idx):
        self._vocab_char2idx = char2idx

    def update_temperature(self, cumulative_r_imit, alpha, initial_temp, min_temp):
        self.temperature = max(initial_temp / (1.0 + alpha * cumulative_r_imit), min_temp)

    def resize_embedding(self, new_vocab_size):
        old_size = self.embedding.num_embeddings
        if new_vocab_size <= old_size:
            self.vocab_size = new_vocab_size
            return
        old_emb_weight = self.embedding.weight.data
        new_emb = nn.Embedding(new_vocab_size, self.embedding.embedding_dim, padding_idx=0)
        new_emb.weight.data[:old_size] = old_emb_weight
        self.embedding = new_emb
        old_out_weight = self.perception_head.weight.data
        old_out_bias = self.perception_head.bias.data
        new_out = nn.Linear(self.hidden_dim, new_vocab_size)
        new_out.weight.data[:old_size] = old_out_weight
        new_out.bias.data[:old_size] = old_out_bias
        self.perception_head = new_out
        self.vocab_size = new_vocab_size

    def _device(self):
        return self.embedding.weight.device
