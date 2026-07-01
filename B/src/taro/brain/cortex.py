"""
大脳皮質（Cortex） — 予測する単一の再帰ネットワーク

【人間模倣】大脳皮質は知覚・判断・運動指令の中枢。
「次に何が来るか」を絶えず予測する（Friston）。
口の動かし方の指令を出し、聞いた音を処理する。

A2-10：旧core.pyから改名。人間の脳の部品名に合わせた。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from taro.body import VocalTract, NUM_PLACE, NUM_MANNER, NUM_VOICING, NUM_VOWEL


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
                 num_layers=1, temperature=2.0, body_state_dim=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.temperature = temperature
        self.vocab_size = vocab_size
        self.body_state_dim = body_state_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        gru_input_dim = embedding_dim
        if body_state_dim > 0:
            from taro.brain.insula import Insula
            from taro.brain.instincts.critic import Critic
            self.insula = Insula(state_dim=body_state_dim, embedding_dim=embedding_dim)
            self.critic = Critic(state_dim=body_state_dim)
            gru_input_dim = embedding_dim + embedding_dim
        else:
            self.insula = None
            self.critic = None

        self.gru = nn.GRU(gru_input_dim, hidden_dim, num_layers, batch_first=True)

        self.head_place = nn.Linear(hidden_dim, NUM_PLACE)
        self.head_manner = nn.Linear(hidden_dim, NUM_MANNER)
        self.head_voicing = nn.Linear(hidden_dim, NUM_VOICING)
        self.head_vowel = nn.Linear(hidden_dim, NUM_VOWEL)

        with torch.no_grad():
            self.head_place.bias.data[0] += 2.0
            self.head_manner.bias.data[0] += 2.0
            self.head_voicing.bias.data[1] += 1.0

        self.perception_head = nn.Linear(hidden_dim, vocab_size)

    def forward_hidden(self, x, hidden=None, body_state=None):
        """
        入力トークンを処理して隠れ状態を更新する。

        body_state: 内部状態ベクトル（島皮質経由）。NoneならモデルA互換。
        """
        emb = self.embedding(x)
        if self.insula is not None:
            if body_state is not None:
                body_vec = self.insula(body_state)
            else:
                body_vec = torch.zeros(self.embedding.embedding_dim, device=emb.device)
            body_expanded = body_vec.unsqueeze(0).unsqueeze(0).expand(
                emb.size(0), emb.size(1), -1)
            emb = torch.cat([emb, body_expanded], dim=-1)
        out, hidden = self.gru(emb, hidden)
        return out, hidden

    def forward_perception(self, x, hidden=None, body_state=None):
        out, hidden = self.forward_hidden(x, hidden, body_state=body_state)
        logits = self.perception_head(out)
        return logits, hidden

    def forward_articulation(self, gru_output):
        return (
            self.head_place(gru_output),
            self.head_manner(gru_output),
            self.head_voicing(gru_output),
            self.head_vowel(gru_output),
        )

    def generate(self, hidden, max_length, eos_idx, stamina=None,
                 vocal_tract=None, ne_level=0.5, cerebellum=None,
                 speech_plan=None, body_state=None):
        """
        太郎の番に文字を産出する。

        A2-11：発話計画ベースに変更（GODIVAモデル）。
        1文字ずつ独立に生成するのではなく、
        「発話計画を立てる→バッファから順に実行→計画が終わったら止まる」。

        speech_plan: SpeechPlannerが作った計画（plan()済み）
        計画がなければ体力分だけ探索的に発声する（喃語期）
        """
        if vocal_tract is None:
            vocal_tract = VocalTract()

        generated = []
        log_probs_all = []
        bos = torch.tensor([[1]], device=self._device())
        out, hidden = self.forward_hidden(bos, hidden, body_state=body_state)

        allowed_place, allowed_manner, allowed_voicing, allowed_vowel = vocal_tract.get_allowed()

        # 発話計画がある場合：計画に沿って実行（計画が終わったら止まる）
        # 発話計画がない場合：体力分だけ探索的に発声（喃語）
        if speech_plan is not None and speech_plan.has_next():
            max_chars = min(speech_plan.get_plan_length(), int(stamina) if stamina is not None else max_length)
        else:
            max_chars = min(max_length, int(stamina) if stamina is not None else max_length)
            speech_plan = None

        for _ in range(max_chars):
            h_last = out[0, -1]
            pl, ml, vl, vol = self.forward_articulation(h_last)
            log_prob = torch.tensor(0.0, device=self._device())

            if speech_plan is not None and speech_plan.has_next():
                # 発話計画からモーラを取り出す
                item = speech_plan.next_motor()
                if item["motor"] is not None:
                    s_place, s_manner, s_voicing, s_vowel = item["motor"]
                    s_place = s_place if s_place in allowed_place else allowed_place[0]
                    s_voicing = s_voicing if s_voicing in allowed_voicing else allowed_voicing[0]
                    s_vowel = s_vowel if s_vowel in allowed_vowel else allowed_vowel[0]
                    lp = self._log_prob_of(pl, s_place, allowed_place)
                    log_prob = log_prob + lp
                else:
                    # 計画にあるが口の動きが不明 → 大脳皮質が自力で選ぶ
                    s_place, lp = self._choose_param(pl, allowed_place)
                    log_prob = log_prob + lp
                    if vocal_tract.is_coupled():
                        s_manner = vocal_tract.get_manner_for_place(s_place)
                    else:
                        s_manner, lp = self._choose_param(ml, allowed_manner)
                        log_prob = log_prob + lp
                    s_voicing, lp = self._choose_param(vl, allowed_voicing)
                    log_prob = log_prob + lp
                    s_vowel, lp = self._choose_param(vol, allowed_vowel)
                    log_prob = log_prob + lp
            else:
                # 発話計画なし（喃語期）→ 大脳皮質が自力で選ぶ
                s_place, lp = self._choose_param(pl, allowed_place)
                log_prob = log_prob + lp
                if vocal_tract.is_coupled():
                    s_manner = vocal_tract.get_manner_for_place(s_place)
                else:
                    s_manner, lp = self._choose_param(ml, allowed_manner)
                    log_prob = log_prob + lp
                s_voicing, lp = self._choose_param(vl, allowed_voicing)
                log_prob = log_prob + lp
                s_vowel, lp = self._choose_param(vol, allowed_vowel)
                log_prob = log_prob + lp

            # NEによる局所ノイズ注入（計画があっても少しずれる＝人間的な誤差）
            s_place = self._apply_ne_noise(s_place, ne_level, allowed_place)
            if not vocal_tract.is_coupled():
                s_manner = self._apply_ne_noise(s_manner, ne_level, allowed_manner)
            else:
                s_manner = vocal_tract.get_manner_for_place(s_place)
            s_voicing = self._apply_ne_noise(s_voicing, ne_level, allowed_voicing)
            s_vowel = self._apply_ne_noise(s_vowel, ne_level, allowed_vowel)

            char = vocal_tract.speak(s_place, s_manner, s_voicing, s_vowel)

            # 小脳に体験を記録
            if cerebellum is not None:
                cerebellum.learn_from_experience(s_place, s_manner, s_voicing, s_vowel, char)

            if char in self._vocab_char2idx:
                token_idx = self._vocab_char2idx[char]
            else:
                break

            if token_idx == eos_idx:
                break

            generated.append(token_idx)
            log_probs_all.append(log_prob)

            # 自己聴取
            token_input = torch.tensor([[token_idx]], device=self._device())
            out, hidden = self.forward_hidden(token_input, hidden, body_state=body_state)

        return generated, log_probs_all, hidden

    def _log_prob_of(self, logits, value, allowed_indices):
        """小脳の逆モデルが指定した値のlog確率を計算する。"""
        if len(allowed_indices) == 1:
            return torch.tensor(0.0, device=logits.device)
        mask = torch.full_like(logits, float("-inf"))
        for i in allowed_indices:
            mask[i] = 0.0
        masked_logits = logits + mask
        probs = F.softmax(masked_logits, dim=-1)
        if value < len(probs):
            return torch.log(probs[value] + 1e-10)
        return torch.tensor(0.0, device=logits.device)

    def _choose_param(self, logits, allowed_indices):
        """脳がパラメータを選ぶ（意図）。ノイズなし。"""
        if len(allowed_indices) == 1:
            return allowed_indices[0], torch.tensor(0.0, device=logits.device)

        mask = torch.full_like(logits, float("-inf"))
        for i in allowed_indices:
            mask[i] = 0.0
        masked_logits = logits + mask
        probs = F.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        sample = dist.sample()
        return sample.item(), dist.log_prob(sample)

    def _apply_ne_noise(self, value, ne_level, allowed):
        """
        NEによる局所ノイズ注入。

        【人間模倣】鳥のLMANが運動指令にノイズを注入するのと同じ原理。
        元の値から1つ隣にずれる確率がNEに比例する。
        NE=0 → ずれない。NE=1 → 高確率でずれる。ただし2つ以上は稀。
        """
        import random
        if len(allowed) <= 1:
            return value

        if random.random() < ne_level * 0.7:
            idx = allowed.index(value) if value in allowed else 0
            direction = random.choice([-1, 1])
            new_idx = max(0, min(len(allowed) - 1, idx + direction))
            return allowed[new_idx]

        return value

    def set_vocab_mapping(self, char2idx):
        self._vocab_char2idx = char2idx

    def receive_ne(self, ne_level):
        """
        アドレナリン受容体 — NEレベルを記録する。

        A2-9b：τを直接制御するのではなく、generate時に
        NEレベルに応じた局所ノイズとして反映する。
        """
        self.current_ne = ne_level

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
