"""
本物のTaroBrain（taro_brain.py）に、運動性喃語（身体の感覚運動予測ループ）専用の
経路を追加する拡張。

【重要】既存のメソッド（forward_hidden / forward_perception / generate 等、音声関連の
すべて）は一切変更しない。追加するのは、5感覚の融合ベクトル(320次元)を専用のGRUに
流し込み、運動を出力する「新しい経路」だけ。

【人間模倣】音声用GRU(self.gru)とは別に、運動専用のGRU(self.motor_gru)を持つ。
言語（ブローカ野等）と運動（運動皮質・小脳等）は人間の脳でも別の領域が担っており、
同じ神経集団が両方を兼ねることはない。2026-07-10までは「将来の統合のため」という
工学的な都合だけでGRUを共有していたが、根拠ラベルが無いまま人間模倣から外れていた
ため分離した（詳細はdocs/人間模倣からの逸脱リスト.md）。
【解像度】発達初期の皮質は最初から領域特異的ではなく、共通の汎用回路から入力に
応じて後から専門分化する（interactive specialization）という知見もあり、本来は
「最初は共有・経験で分化」の方がより発達的に忠実な可能性がある。今回は分化の
仕組みまでは作らず、単純に分離するに留める（⚠️簡略化）。

現時点ではbody_state_dim=0でTaroBrainを構築する（insula/critic/satiety_headは
使わない）。内受容感覚は5感覚融合ベクトルの一部として既に含まれているため、
二重に持つ必要がないための判断。
"""

import torch
import torch.nn as nn

from taro_brain import TaroBrain
from predictive_coding_latent import PredictiveCodingLatent
from hippocampus import MotorHippocampus


class TaroBrainWithMotor(TaroBrain):
    """
    TaroBrainのサブクラス。運動性喃語用の`step_motor()`だけを追加する。
    """

    def __init__(self, vocab_size, sensory_dim=320, n_actuators=90,
                 embedding_dim=64, hidden_dim=128, proprio_dim=621, **kwargs):
        super().__init__(vocab_size=vocab_size, embedding_dim=embedding_dim,
                          hidden_dim=hidden_dim, body_state_dim=0, **kwargs)
        self.sensory_dim = sensory_dim
        self.n_actuators = n_actuators

        # 感覚融合ベクトル(320次元) → GRU入力次元(embedding_dim)に変換
        self.sensory_proj = nn.Linear(sensory_dim, embedding_dim)

        # 運動専用のGRU。音声用self.gruとは重みを共有しない（別々の神経集団に対応）。
        # パラメータ数は約74,000（触覚エンコーダ1個=約330万の1/44）で計算コストは軽微。
        self.motor_gru = nn.GRU(embedding_dim, hidden_dim, self.num_layers, batch_first=True)

        # 確率的な潜在変数＋推論時のその場調整（PV-RNNに着想、詳細はpredictive_coding_latent.py）
        self.latent_dim = 32
        self.pc_latent = PredictiveCodingLatent(hidden_dim, sensory_dim, latent_dim=self.latent_dim)

        # 感覚運動予測ループ：次に来る感覚を予測する。
        # 【人間模倣】予測は「今の状態z」だけでなく「これからする運動命令の写し
        # （遠心性コピー efference copy）」も手がかりにする。脳は運動命令を出すとき、
        # その写しを感覚予測系にも送り、「自分がこう動いたら次はこうなる」を前もって
        # 予測する（von Holst & Mittelstaedt, 1950 の遠心性コピー説）。行動を入力に
        # 取ることで初めて「行動→結果」を予測する順モデル（自己内部モデル）になる。
        # 2026-07-11以前は行動を入力に取っておらず、「今の状態→次の感覚」を予測する
        # だけで、順モデルの要件（行動条件づけ）を満たしていなかった（詳細は
        # docs/人間模倣からの逸脱リスト.md B6）。
        self.sensorimotor_prediction_head = nn.Linear(self.latent_dim + n_actuators, sensory_dim)

        # 運動性喃語：関節への命令を出す（潜在変数zから）
        self.motor_head = nn.Linear(self.latent_dim, n_actuators)

        # ─── 目標C1/C2で実証した改良自己モデル（step_motorの旧アーキを更新した版）───
        self.proprio_dim = proprio_dim
        # ① 遠心性コピーを再帰の中へ：[感覚, 前回行動] → GRU入力。step_motorのsensory_proj
        #    （感覚のみ）を置き換える改良（行動→感覚の結びつきを再帰で強める）。
        self.motor_input_proj = nn.Linear(sensory_dim + n_actuators, embedding_dim)
        # ② 予測ヘッドを非線形MLP＋LayerNorm。固有感覚の「変化量Δ」を予測（残差予測）。
        #    LayerNormは発散（ヘッドの出力爆発）を防ぐ標準的安定化（DreamerV3準拠、
        #    アブレーションで発散原因＝ヘッド増幅と確定したため）。
        self.forward_model_head = nn.Sequential(
            nn.Linear(self.latent_dim + n_actuators, hidden_dim), nn.SiLU(),
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, proprio_dim))
        # 海馬：睡眠リプレイで自己モデルを定着させる（C2で実証）。言語用海馬と同じCLS原理。
        self.hippocampus = MotorHippocampus()

    def init_motor_hidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_dim)

    def step_motor(self, sensory_vec, hidden, current_sensory_target=None,
                   ne_level=0.5, min_std=0.05, max_std=0.5):
        """
        運動性喃語の1ステップ。

        sensory_vec: (sensory_dim,) 5感覚を束ねたベクトル（学習中のオンライン
            エンコーダの出力。GRUに流し込む「今の入力」として使う）
        current_sensory_target: (sensory_dim,) PV-RNN風の再構成損失の"正解"。
            sensory_vecと同じ形だが、**独立した・学習させない別のエンコーダ**
            から作ること（呼び出し側の責任）。ここでsensory_vec自身を渡すと、
            「予測する側と正解を作る側が同じ感覚エンコーダ」という馴れ合いが
            起き、崩壊する（実測済み、docs/人間模倣からの逸脱リスト.md B4）。
            Noneの場合はsensory_vec.detach()で代用するが、これは崩壊リスクが
            残るデバッグ用の後方互換であり、通常は必ず指定すること。
        hidden: (num_layers, 1, hidden_dim) 前回の隠れ状態
        ne_level: 青斑核（LocusCoeruleus）が放出したNE水準（0〜1）。
            高いほど運動のゆらぎ（探索）を大きくする。太郎の既存の探索本能を
            そのまま運動選択のノイズ幅として流用する（新しい概念は導入しない）。

        運動命令は確定的な1点ではなく、正規分布からサンプリングする
        （方策勾配/REINFORCEで学習するには、選んだ行動の対数確率
        log_probが必要なため。basal_ganglia.pyのlearn_action()が
        既に対数確率＋δから汎用的に方策損失を計算できる形になっている
        ので、そこにそのまま渡せるlog_probをここで作る）。

        戻り値: (new_hidden, predicted_next_sensory, motor_action, log_prob, kl_loss, recon_loss)
        """
        if current_sensory_target is None:
            current_sensory_target = sensory_vec.detach()

        hidden_before = hidden[-1, 0]  # 今回の感覚を見る"前"の状態（事前の予想用）

        emb = self.sensory_proj(sensory_vec).unsqueeze(0).unsqueeze(0)  # (1, 1, embedding_dim)
        out, new_hidden = self.motor_gru(emb, hidden)
        h_last = out[0, -1]  # 今回の感覚を処理した"後"の状態（事後の判断用）

        # 確率的な潜在変数zを、事前の予想と事後の判断から、その場で調整して作る
        z, kl_loss, recon_loss = self.pc_latent.infer(hidden_before, h_last, current_sensory_target)

        # 先に運動命令を決める（予測が運動を手がかりにできるよう、順序を入れ替えた）。
        # レビューで判明したバグの修正：motor_headにzをそのまま渡すと、方策勾配
        # （REINFORCE、1サンプルで高分散）がstraight-through経由でposterior_net→
        # motor_gru→sensory_proj→感覚エンコーダという共有された幹まで逆伝播し、
        # 知覚学習の滑らかな勾配に高分散なノイズを混ぜてしまう。REINFORCEは本来
        # motor_head自身の重みへの勾配だけを要求する手法なので、ここでzを切り
        # 離しても数式上の正しさは失われない。
        mean = torch.tanh(self.motor_head(z.detach()))
        std = min_std + ne_level * (max_std - min_std)
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.sample()
        motor_action = torch.clamp(raw_action, -1.0, 1.0)

        # cortex.pyのB2-6と同型の修正：log_probは実際に使われる（クランプ後の）
        # 値から計算する。クランプ前のraw_actionで計算すると、範囲外に出て
        # クランプされた場合に「実際には取らなかった値」への信用割り当てに
        # なり、学習とその評価対象がズレる。
        log_prob = dist.log_prob(motor_action).sum()

        # 【人間模倣】遠心性コピー：これからする運動命令の写しを予測系に渡し、
        # 「今の状態z ＋ この運動 → 次の感覚」を予測する（＝行動を入力に取る順モデル）。
        # motor_action.detach()：予測誤差の勾配が方策(motor_head)側へ漏れないよう
        # 切り離す（H2の方針＝知覚学習と方策学習の勾配を混ぜない、と一貫）。予測ヘッド
        # 自身の重みは学習されるため、「この運動ならこうなる」の対応づけは獲得できる。
        efference_copy = motor_action.detach()
        predicted_next_sensory = self.sensorimotor_prediction_head(
            torch.cat([z, efference_copy], dim=-1))

        return new_hidden, predicted_next_sensory, motor_action, log_prob, kl_loss, recon_loss

    @staticmethod
    def prediction_error(predicted_next_sensory, actual_next_sensory):
        """
        感覚運動予測誤差。

        精度重み付き知覚（precision）を試す場合は、ここに渡す
        actual_next_sensoryを`PrecisionWeightedPerception.perceive()`の
        出力に差し替える（precision_perception.py参照）。過去に
        「predicted自身をdetachして混ぜる」実装を試したが、評価対象と
        混ぜる相手が数式上同一になり (1-precision)^2 倍の単純な
        スケーリングに退化することが判明したため、そちらのクラスでは
        評価対象と独立な"期待"を別途保持している。
        """
        return torch.nn.functional.mse_loss(predicted_next_sensory, actual_next_sensory)

    @staticmethod
    def sensorimotor_reward(prediction_error_value):
        """
        感覚運動予測誤差 → 報酬[0, 1]への変換。

        太郎の既存の型（imitation.pyのmax(0, 1 - dist/max_len)＝誤差が
        小さいほど1に近い報酬）と同じ発想だが、感覚のMSE誤差には
        edit distanceのmax_lenのような自然な上限が無いため、
        恣意的な正規化定数を持ち込まずに済む 1/(1+誤差) を使う
        （誤差0で報酬1、誤差が大きいほど0に漸近、常に(0, 1]に収まる）。
        """
        return 1.0 / (1.0 + prediction_error_value)

    # ─── 改良自己モデル（C1）＋睡眠リプレイ（C2）の本実装 ───

    def infer_latent(self, sensory_vec, prev_action, current_sensory_target, hidden):
        """[感覚, 前回行動] → GRU → 潜在z（遠心性コピーを再帰の中に入れた改良版）。
        戻り値: (z, kl_loss, recon_loss, new_hidden)。"""
        emb = self.motor_input_proj(
            torch.cat([sensory_vec, prev_action], dim=-1)).unsqueeze(0).unsqueeze(0)
        out, new_hidden = self.motor_gru(emb, hidden)
        z, kl_loss, recon_loss = self.pc_latent.infer(hidden[-1, 0], out[0, -1], current_sensory_target)
        return z, kl_loss, recon_loss, new_hidden

    def predict_proprio(self, z, action, current_proprio):
        """残差予測：現在の固有感覚 + Δ(z, 行動) = 次の固有感覚の予測。"""
        return current_proprio + self.forward_model_head(torch.cat([z, action], dim=-1))

    def consolidate(self, learner, n_batches=200, batch_size=128):
        """睡眠リプレイ：海馬に貯めた経験を再生し、順モデル（自己内部モデル）を定着させる。
        言語用海馬が core_b.consolidate に定着を任せているのと同じ分担で、運動用の定着
        ロジックをここに置く。覚醒時と同じ順モデル計算を、貯めた経験でバッチ再生する。"""
        N = len(self.hippocampus)
        if N < batch_size:
            return
        eps = self.hippocampus.replay()
        SV = torch.stack([e[0] for e in eps]); PA = torch.stack([e[1] for e in eps])
        AA = torch.stack([e[2] for e in eps]); CF = torch.stack([e[3] for e in eps])
        CLP = torch.stack([e[4] for e in eps]); NLP = torch.stack([e[5] for e in eps])
        H = torch.cat([e[6] for e in eps], dim=1)  # (num_layers, N, hidden_dim)
        mse = torch.nn.functional.mse_loss
        for _ in range(n_batches):
            idx = torch.randint(0, N, (batch_size,))
            hb = H[:, idx].contiguous()
            emb = self.motor_input_proj(
                torch.cat([SV[idx], PA[idx]], dim=-1)).unsqueeze(1)  # (bs,1,emb) batch_first
            out, _ = self.motor_gru(emb, hb)
            z, kl, rc = self.pc_latent.infer(hb[-1], out[:, 0], CF[idx])
            pred = CLP[idx] + self.forward_model_head(torch.cat([z, AA[idx]], dim=-1))
            loss = mse(pred, NLP[idx]) + kl + rc
            learner.optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(learner.brain.parameters(), learner.grad_clip)
            learner.optimizer.step()
