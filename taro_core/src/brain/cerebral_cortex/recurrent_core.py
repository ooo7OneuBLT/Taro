# --- コピー元: Taro (github.com/ooo7OneuBLT/Taro) commit 3b976fc ---
# --- 元パス: B/src/taro/brain/cortex.py （unificationMIMoではimport文以外は無編集） ---

"""
大脳皮質（Cortex） — 予測する単一の再帰ネットワーク

【人間模倣】大脳皮質は知覚・判断・運動指令の中枢。
「次に何が来るか」を絶えず予測する（Friston）。
口の動かし方の指令を出し、聞いた音を処理する。

A2-10：旧core.pyから改名。人間の脳の部品名に合わせた。
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from vocal_tract import VocalTract, NUM_PLACE, NUM_MANNER, NUM_VOICING, NUM_VOWEL


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
            # 【人間模倣】満腹予期ヘッド（B2-10）。GRUの隠れ状態（聞いた音＋体の
            # 状態を統合した表現）から「もうすぐ満腹になる（授乳が来る）か」を予測する。
            # OFCの結果同定（何が来るか＝正体の予期）＋島皮質の内受容予測に相当し、
            # criticの「価値（良さ）」とは別に「特定の結果（ごはん）」を予期する。
            # これまで価値・予期系は体の状態しか見ておらず「聞いた音→体の未来」の
            # 経路が欠けていた（理解テストA案で実証）。その欠けた向きを配線する。
            self.satiety_head = nn.Linear(hidden_dim, 1)
            gru_input_dim = embedding_dim + embedding_dim
        else:
            self.insula = None
            self.critic = None
            self.satiety_head = None

        # 【M4e・2026-09-07・仕様_M4e_状態の線と驚きの書き込み】状態の線（提案2への
        #   第一歩）。0=なし、1=ある、2=消えた。padding_idx=0でindex 0は零ベクトル
        #   固定＝state_idを渡さない（None）呼び出しでは何も足さない＝既存実験は
        #   1ビットも変わらない。forward_hiddenのstate_id引数で使う。
        # 【重要・実測で発覚・2026-09-07】nn.Embedding()の初期化はtorchのグローバル
        #   乱数生成器を消費する。__init__の途中でここに新しい層を足すと、この後に
        #   作られるGRU等の初期値だけでなく、モデル構築より**後**（生成・探索noise
        #   ・海馬のサンプリング等）の乱数列まで全部ずれる（実測：produce.state_channel
        #   無しでも30step先頭の choice_gru 確信度が0.998→0.82等に変化＝既定不変①に
        #   違反）。torch.random.fork_rng() でこの層の初期化だけを乱数の「支流」に
        #   逃がし、グローバル生成器の状態をこの行の前後で不変に保つ。
        with torch.random.fork_rng(devices=[]):
            self.state_embedding = nn.Embedding(3, embedding_dim, padding_idx=0)

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

        # 【人間模倣】喃語の反復性（Frame/Content理論, MacNeilage 1998 BBS）。
        # 喃語は顎の開閉振動（フレーム）が生む音節で、舌・唇（コンテンツ）は
        # 最初ほとんど動かないため同じ音節が反復される（ままま等）。これを
        # 「直前に出した音の口の形を選び直しやすくする」調音の慣性バイアスとして
        # 近似する。声道の物理定数（口を動かすには前の形から動かす手間がかかる）
        # の一種であり、行動の閾値ではない。logit空間への加算。学習が進んで
        # コンテンツ制御が強まる（logitが大きくなる）ほど相対的に弱まるので、
        # 反復が徐々に変異へ移行する発達的傾向も自然に再現される。
        self.reduplication_bias = 2.0

        # 発声のたびに「許可された音だけ残すマスク」を作り直していたが、
        # 許可集合は声道の段階が変わるとき（年に数回）しか変わらない。
        # (ベクトル長, 許可indexの組) をキーにキャッシュし、生成の各文字での
        # マスク再構築（_choose_param・_log_prob_of・_normalized_entropy）を
        # 省く。計算結果は完全に同一で、速度だけ改善する。
        self._mask_cache = {}

    def predict_satiety(self, hidden_last):
        """
        【人間模倣】聞いた音＋体の状態（GRU隠れ状態）から「食べ物（授乳）が
        来る」ことの先取り＝referent anticipation を出す（B2-10）。

        重要（指標の意味づけ, 2026-07-02訂正）：これは「“今”満腹かどうかの報告」
        ではない。今の満腹度は体が内受容感覚で直接知っており、語で当てるのは順序が
        逆（＝hungerだけで解ける近道になる）。理解の証拠として読むべきは、hungerを
        固定して“聞いた語だけ”を変えたときの本出力の差（＝語の寄与）である。
        乳児研究で理解を「語→対象の先取り活性化（anticipatory looking）」で測るのに
        対応する（Bergelson & Swingley 2012 PNAS ほか, 参考文献§9）。ただし連合的・
        パブロフ的な先取りであり、完全な象徴的参照の証明ではない（初期理解の代理）。

        hidden_last: (hidden_dim,) の最終時刻表現。戻り値: 0〜1。
        """
        return torch.sigmoid(self.satiety_head(hidden_last)).squeeze(-1)

    def _get_mask(self, logits, allowed_indices):
        """許可index以外を-infにする加算マスク。段階ごとにキャッシュする。"""
        key = (logits.shape[-1], tuple(allowed_indices))
        m = self._mask_cache.get(key)
        if m is None or m.device != logits.device or m.dtype != logits.dtype:
            m = torch.full((logits.shape[-1],), float("-inf"),
                           device=logits.device, dtype=logits.dtype)
            for i in allowed_indices:
                m[i] = 0.0
            self._mask_cache[key] = m
        return m

    def forward_hidden(self, x, hidden=None, body_state=None, prefix_vec=None, state_id=None):
        """
        入力トークンを処理して隠れ状態を更新する。

        body_state: 内部状態ベクトル（島皮質経由）。NoneならモデルA互換。
        prefix_vec: 【V1・2026-08-31】視覚トークン（embedding次元のベクトル1個）。
          渡すとトークン列の頭に1トークンぶんとして連結する（設計_視覚とGRUの
          統合.md）。Noneなら従来と1ビットも変わらない。出力の系列長は+1になる
          ので、損失を取る呼び出し側は先頭位置を読み飛ばすこと。
        state_id: 【M4e・2026-09-07・仕様_M4e_状態の線と驚きの書き込み】今の状態
          （0=なし/1=ある/2=消えた）。渡すと、prefix_vec連結の**後**（島皮質の
          連結の**前**）に、全位置（prefixの位置にも）へ同じ埋め込みベクトルを
          足す。文の先頭だけでなく1音ごとに毎回同じ値が入り続ける＝人間側の
          「消えた」という知覚状態が言い終わるまで続いていることに対応する。
          Noneなら従来と1ビットも変わらない。
        """
        emb = self.embedding(x)
        if prefix_vec is not None:
            emb = torch.cat([prefix_vec.view(1, 1, -1).to(emb.dtype), emb], dim=1)
        if state_id is not None:
            _st = self.state_embedding(
                torch.tensor([state_id], device=emb.device)).view(1, 1, -1)
            emb = emb + _st
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

    def sequence_prob(self, token_ids, hidden=None):
        """トークン列の「文脈から見た言いやすさ」を返す（2026-08-30・文脈設計の決定4）。

        文脈（hidden）から始めて、BOS→列の各トークンを知覚ヘッドで予測したときの
        1音あたりの平均確率（logの平均のexp＝幾何平均）。0〜1のスケールなので、
        語彙のコサイン類似度とそのまま足し合わせられる。長さで正規化してあるため
        長い語が不利にならない。学習はしない（no_gradの読み出し専用）。

        token_ids が空なら None（採点不能。呼び出し側はコサインだけで判断する）。
        """
        if not token_ids:
            return None
        import torch.nn.functional as F
        dev = self._device()
        seq = [1] + list(token_ids)          # BOS から予測を始める
        x = torch.tensor([seq], dtype=torch.long, device=dev)
        with torch.no_grad():
            logits, _ = self.forward_perception(x, hidden=hidden)
            logp = F.log_softmax(logits[0, :-1, :], dim=-1)
            tgt = torch.tensor(seq[1:], dtype=torch.long, device=dev)
            picked = logp.gather(1, tgt.unsqueeze(1)).squeeze(1)
        return float(picked.mean().exp())

    def token_probs(self, token_ids, hidden=None):
        """列の各トークンを「直前までからどれだけ予測できたか」を返す（2026-08-31・二語文）。

        lexicon の統計的分節（自信度の谷で切る）へ渡す本物の自信度。
        BOS から始め、位置iの確率 = p(token_i | BOS, token_0..i-1, 文脈hidden)。
        読み出し専用（no_grad）。token_ids が空なら空リスト。
        """
        if not token_ids:
            return []
        import torch.nn.functional as F
        dev = self._device()
        seq = [1] + list(token_ids)
        x = torch.tensor([seq], dtype=torch.long, device=dev)
        with torch.no_grad():
            logits, _ = self.forward_perception(x, hidden=hidden)
            probs = F.softmax(logits[0, :-1, :], dim=-1)
            tgt = torch.tensor(seq[1:], dtype=torch.long, device=dev)
            picked = probs.gather(1, tgt.unsqueeze(1)).squeeze(1)
        return [float(v) for v in picked]

    def end_probs(self, token_ids, hidden=None, start=1):
        """列の各位置で「ここで発話が終わる確率」を返す（分節第2案・2026-09-03）。

        【分節第2案・2026-09-03】設計_分節（語の切れ目の発見）.md 第2案 第2部
        「end_probs」節。listen_eos（EOS=2を末尾に足す聞く学習）で育った
        p(EOS|文脈) を読み出す。token_probs と同じ枠組みで、gatherする対象を
        「実際に来たトークン」ではなく「固定インデックス2（EOS）」に変えただけ。

        token_ids を start（既定BOS=1。話者トークンを渡せば聞く学習と同じ頭になる）
        から forward_perception に通し、位置i（token_iを聞き終えた直後）の
        softmax(logits)[...,2] をリストで返す（長さ = len(token_ids)）。
        読み出し専用（no_grad）。token_ids が空なら空リスト。
        """
        if not token_ids:
            return []
        import torch.nn.functional as F
        dev = self._device()
        seq = [start] + list(token_ids)
        x = torch.tensor([seq], dtype=torch.long, device=dev)
        with torch.no_grad():
            logits, _ = self.forward_perception(x, hidden=hidden)
            probs = F.softmax(logits[0, 1:, :], dim=-1)
            picked = probs[:, 2]
        return [float(v) for v in picked]

    def forward_articulation(self, gru_output):
        return (
            self.head_place(gru_output),
            self.head_manner(gru_output),
            self.head_voicing(gru_output),
            self.head_vowel(gru_output),
        )

    def generate(self, hidden, max_length, eos_idx, stamina=None,
                 vocal_tract=None, ne_level=0.5, cerebellum=None,
                 speech_plan=None, body_state=None, jaw_cycles=None):
        """
        太郎の番に文字を産出する。

        A2-11：発話計画ベースに変更（GODIVAモデル）。
        1文字ずつ独立に生成するのではなく、
        「発話計画を立てる→バッファから順に実行→計画が終わったら止まる」。

        speech_plan: SpeechPlannerが作った計画（plan()済み）
        計画がなければ体力分だけ探索的に発声する（喃語期）

        jaw_cycles: 【F2-1・決定1】喃語モード専用。指定すると、体力
          （stamina）や max_length を無視して、この文字数（＝顎のサイクル
          数のつもり。日本語では1モーラ≒1文字なので厳密な音節数ではない）
          で打ち切る。設計：F/docs/設計_F2-1_喃語で口の内部モデルを作る.md
          第2部決定1（Fagan 2009：乳児の1発声は平均0.84音節、単一CV音節・
          単母音が大半）。speech_plan が与えられている場合はそちらを優先
          （speech_plan は None のときだけ意味を持つ）。
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
        elif jaw_cycles is not None:
            # 【F2-1・決定1・2026-08-23】喃語の長さは肺活量ではなく
            # 「顎のサイクル数」（1〜2から一様ランダムに引いた値、呼び出し側で
            # 決める）で決める。肺は「絶対に超えない上限」に戻す（文献に呼吸を
            # 主因とする裏付けが無いため。設計第2部決定1参照）。
            max_chars = int(jaw_cycles)
            speech_plan = None
        else:
            max_chars = min(max_length, int(stamina) if stamina is not None else max_length)
            speech_plan = None

        # 【F2-1・決定2・2026-08-23／実装作業⑥・想定外につき部分保留】
        # 設計は息切れ停止（breath_pressure/stop_prob）を「設定で戻せる形に
        # せず削除」と決めていたが、実測でこれを字義どおり削除すると
        # **既存のF2「見た物の名前を言う」word産出（mode未指定・jaw_cycles未指定）
        # の乱数列・出力が変わってしまう**ことが判明した
        # （既存呼び出しは speech_plan を渡さないため is_babble=True の
        #  この分岐を常に通っていた。run/tools/check_f2_1_babble_equivalence.py
        #  で確認：5トライアル中5トライアルとも生成文字列・log_probsが不一致）。
        # 「produce未設定／mode未指定の既存実験は乱数列を含め1ビットも変わらない
        # こと」という絶対条件（仕様「絶対に守ること」1）を優先し、
        # **jaw_cycles が指定されている（＝喃語モード）ときだけ息切れ停止を
        # 無効化**する形に変更した（jaw_cycles未指定＝既存呼び出しでは旧コード
        # と完全に同じ経路を通る）。設計の「設定で戻せる形にはせず削除」の
        # 方針そのものとは食い違うため、作業記録の「上に上げること」に記載する。
        is_babble = speech_plan is None and jaw_cycles is None

        # 【人間模倣】反復バイアス用：直前の音節の口の形を覚えておく（Frame/Content）。
        # generate()呼び出しごとにリセット（前の発話を引きずらない）。
        prev_place = prev_manner = prev_voicing = prev_vowel = None

        for t in range(max_chars):
            h_last = out[0, -1]
            pl, ml, vl, vol = self.forward_articulation(h_last)

            if is_babble and t > 0 and max_chars > 0:
                # B2-4→B2-5の息切れ停止（旧実装）。jaw_cycles指定時（喃語モード）
                # は上でmax_charsが1〜2に固定されるため、この停止機構は出番が
                # なく（決定1・2の意図どおり）、jaw_cycles未指定の既存呼び出し
                # 経路のみ、この停止機構を含めて完全に旧挙動を保つ。
                breath_pressure = t / max_chars
                interest = (self._normalized_entropy(pl, allowed_place)
                           + self._normalized_entropy(vl, allowed_voicing)
                           + self._normalized_entropy(vol, allowed_vowel)) / 3.0
                stop_prob = breath_pressure * (1.0 - interest)
                if random.random() < stop_prob:
                    break

            if speech_plan is not None and speech_plan.has_next():
                # 発話計画からモーラを取り出す
                item = speech_plan.next_motor()
                if item["motor"] is not None:
                    s_place, s_manner, s_voicing, s_vowel = item["motor"]
                    s_place = s_place if s_place in allowed_place else allowed_place[0]
                    s_voicing = s_voicing if s_voicing in allowed_voicing else allowed_voicing[0]
                    s_vowel = s_vowel if s_vowel in allowed_vowel else allowed_vowel[0]
                else:
                    # 計画にあるが口の動きが不明 → 大脳皮質が自力で選ぶ
                    s_place, _ = self._choose_param(pl, allowed_place)
                    if vocal_tract.is_coupled():
                        s_manner = vocal_tract.get_manner_for_place(s_place)
                    else:
                        s_manner, _ = self._choose_param(ml, allowed_manner)
                    s_voicing, _ = self._choose_param(vl, allowed_voicing)
                    s_vowel, _ = self._choose_param(vol, allowed_vowel)
            else:
                # 発話計画なし（喃語期）→ 大脳皮質が自力で選ぶ
                # 【人間模倣】反復バイアス：直前の音節の口の形へ戻りやすくする
                # （Frame/Content理論）。log_probは下でノイズ適用後の値を
                # 生のlogit（pl等）から計算するため、バイアスは選択のみに効き、
                # 学習対象（log_prob）は歪めない（B2-6の不変条件を維持）。
                s_place, _ = self._choose_param(
                    self._reduplicate_bias(pl, prev_place, allowed_place), allowed_place)
                if vocal_tract.is_coupled():
                    s_manner = vocal_tract.get_manner_for_place(s_place)
                else:
                    s_manner, _ = self._choose_param(
                        self._reduplicate_bias(ml, prev_manner, allowed_manner), allowed_manner)
                s_voicing, _ = self._choose_param(
                    self._reduplicate_bias(vl, prev_voicing, allowed_voicing), allowed_voicing)
                s_vowel, _ = self._choose_param(
                    self._reduplicate_bias(vol, prev_vowel, allowed_vowel), allowed_vowel)

            # B5-3：運動の自動化（VMS）。意図した口の動きがよく練習されているほど
            # 運動ノイズを抑えて安定して出す＝形の結晶化。報酬は足さず、練習回数だけで
            # 決まる（小脳の手続き的学習）。自動化度auto∈[0,1]でNEノイズを縮小する。
            eff_ne = ne_level
            # 【F2-1・決定4・2026-08-23】喃語モード（jaw_cycles指定）では
            #   automatization()を呼ばない。automatization_scale=2000は未検証の
            #   決め打ちのため、今回（喃語フェーズ）は使わない方針（設計第2部
            #   決定4）。automatization自体は削除せず、呼ばないだけにする
            #   （cerebellum.py冒頭コメント参照）。word産出モード（jaw_cycles
            #   未指定）は従来どおりautomatization()を呼ぶ＝1ビットも変えない。
            if cerebellum is not None and jaw_cycles is None:
                auto = cerebellum.automatization(s_place, s_manner, s_voicing, s_vowel)
                eff_ne = ne_level * (1.0 - auto)

            # NEによる局所ノイズ注入（計画があっても少しずれる＝人間的な誤差）
            s_place = self._apply_ne_noise(s_place, eff_ne, allowed_place)
            if not vocal_tract.is_coupled():
                s_manner = self._apply_ne_noise(s_manner, eff_ne, allowed_manner)
            else:
                s_manner = vocal_tract.get_manner_for_place(s_place)
            s_voicing = self._apply_ne_noise(s_voicing, eff_ne, allowed_voicing)
            s_vowel = self._apply_ne_noise(s_vowel, eff_ne, allowed_vowel)

            # 【人間模倣】反復バイアス用に、実際に発声された（ノイズ適用後の）
            # 口の形を次の音節へ引き継ぐ。慣性は「今いる口の位置」に働くため。
            prev_place, prev_manner = s_place, s_manner
            prev_voicing, prev_vowel = s_voicing, s_vowel

            # B2-6修正：log_probは実際に発声される（ノイズ適用後の）値から
            # 計算する。従来はノイズ適用前の意図した値のlog_probを学習対象に
            # していたため、太郎が「ま」を選んでも直後のNEノイズで「み」に
            # ずれた場合、その「み」への報酬・信用割り当てが「ま」の
            # log_probに誤って結びついていた（学習とその評価対象がズレていた）。
            log_prob = self._log_prob_of(pl, s_place, allowed_place)
            if not vocal_tract.is_coupled():
                log_prob = log_prob + self._log_prob_of(ml, s_manner, allowed_manner)
            log_prob = log_prob + self._log_prob_of(vl, s_voicing, allowed_voicing)
            log_prob = log_prob + self._log_prob_of(vol, s_vowel, allowed_vowel)

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
        mask = self._get_mask(logits, allowed_indices)
        masked_logits = logits + mask
        probs = F.softmax(masked_logits, dim=-1)
        if value < len(probs):
            return torch.log(probs[value] + 1e-10)
        return torch.tensor(0.0, device=logits.device)

    def _normalized_entropy(self, logits, allowed_indices):
        """
        許可された選択肢内でのカテゴリカル分布のエントロピーを0〜1に正規化する。

        【人間模倣＝既存AI研究】Oudeyer, Kaplan & Hafner (2007) の内発的動機付けロボティクス
        （novelty-seeking）：まだ結果が予測しにくい行動ほど「面白い」。
        1に近いほど「まだどれを選ぶか定まっていない＝興味深い」、
        0に近いほど「ほぼ決まっている＝予測可能で退屈」。
        """
        if len(allowed_indices) <= 1:
            return 0.0
        mask = self._get_mask(logits, allowed_indices)
        masked_logits = logits + mask
        probs = F.softmax(masked_logits, dim=-1)
        idx = torch.tensor(allowed_indices, device=logits.device)
        p = probs.index_select(0, idx).clamp(min=1e-10)
        entropy = -(p * torch.log(p)).sum()
        max_entropy = torch.log(torch.tensor(float(len(allowed_indices)), device=logits.device))
        return (entropy / max_entropy).item() if max_entropy > 0 else 0.0

    def _reduplicate_bias(self, logits, prev_idx, allowed_indices):
        """
        【人間模倣】直前に選んだ調音（口の形）へ戻りやすくする調音慣性バイアス。

        Frame/Content理論（MacNeilage 1998, BBS 21:499-546）：喃語期は顎（フレーム）が
        反復し、舌・唇（コンテンツ）が据え置かれるため同じ音節が繰り返される。
        prev_idxが許可集合にあるとき、その選択肢のlogitに定数を足し、
        次の音節が同じ口の形になりやすくする。閾値ではなく物理的な慣性の近似。
        """
        if prev_idx is None or len(allowed_indices) <= 1:
            return logits
        if prev_idx not in allowed_indices:
            return logits
        biased = logits.clone()
        biased[prev_idx] = biased[prev_idx] + self.reduplication_bias
        return biased

    def _choose_param(self, logits, allowed_indices):
        """脳がパラメータを選ぶ（意図）。ノイズなし。"""
        if len(allowed_indices) == 1:
            return allowed_indices[0], torch.tensor(0.0, device=logits.device)

        mask = self._get_mask(logits, allowed_indices)
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

        ⚠【要検討・2026-08-23 ユーザーの指摘により記録。まだ直していない】
        **「1つ隣にずれる」の"隣"が、音の近さと対応していない。**

            PLACES  = [なし, 両唇, 歯茎, 歯茎硬口蓋, 硬口蓋, 軟口蓋, 声門]
                        口の前 →→→→→→→→→→→→→→→→→→ 口の奥
                        ← ここだけは物理的に連続していて、隣＝近い

            MANNERS = [なし, 鼻音, 破裂音, 摩擦音, 破擦音, 弾き音, 半母音]
                        ← 鼻から抜く／口で破裂／隙間で擦る…は連続していない。
                          「鼻音の隣が破裂音」に物理的な意味は無い

            VOWELS  = [あ, い, う, え, お, （母音なし）]
                        ← 五十音順。舌の位置とは無関係。
                          「あ」の隣が「い」だが、舌の位置は最も遠い部類。
                          近いのは「お」（後舌）なのに配列上は4つ離れている

        そのため、人間の運動ノイズ（筋の出力がぶれる→舌の位置が少しずれる→
        音が少し曖昧になる＝連続的なずれ）ではなく、**記号がまるごと別の音に
        置き換わる**（離散的な置換）になっている。
        LMANが注入するのは連続的な運動指令へのノイズであり、原理が異なる。

        実測（F2-2・2026-08-23）：計画が正しく「わ・ん・わ・ん」でも、
        実走行では「わおざあ」「らあぱん」のように別の音へ飛ぶ。

        直すなら：素性空間の距離（`vocal_tract.param_distance` の考え方）や
        調音の物理的な近さで「隣」を定義し直す必要がある。
        目標B以来この形なので、直すと過去の実験の再現性に影響する。
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
