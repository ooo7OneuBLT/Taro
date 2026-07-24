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
from motor_cortex import MotorCortex
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

        # 【運動野】motor_gru・pc_latent・motor_head を motor_cortex.py にまとめた
        # （2026-07-23、解剖学的名称でのファイル分け）。外部コード(d_c5_motor_quality.py等
        # 25ファイル)は brain.motor_head 等の従来のアクセス方法のまま使えるよう、
        # 下の @property で self.motor_cortex.X に転送する（属性アクセスは互換、
        # state_dictキーだけ "motor_cortex.motor_head.weight" のように変わる＝
        # 旧チェックポイントは load_matching 側でキー読み替えが必要、C/scripts/参照）。
        self.latent_dim = 32
        self.motor_cortex = MotorCortex(embedding_dim, hidden_dim, self.num_layers,
                                         self.latent_dim, sensory_dim, n_actuators)

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

        # 脊髄CPG（運動性喃語の探索ノイズ源）。既定は未初期化＝explore()は白色ガウス＝従来と一致。
        # enable_spinal_babble() で色付きノイズ＋シナジーを太郎の中で有効化する。
        self.spinal_cpg = None
        self._babble_beta = 0.8
        self._babble_synergy = False
        self._babble_syn_w = 0.6

        # ─── 目標E（egomotion割引）：視覚版の順モデル ───
        # forward_model_head（固有感覚版）と全く同じ形。予測対象を視覚エンコーダの
        # 出力（fusion.pyのVisionEncoder、64次元embedding）に変える。
        # 【人間模倣】遠心性コピー説（von Holst & Mittelstaedt 1950）に基づく、
        # 「行動→視覚の結果」を予測する順モデル。太郎の実際の自己運動（首の回転）に
        # ついては、遠心性コピーがこの種の視覚変化の打ち消しに十分という文献あり
        # （V6野、neck efference copy）。予測誤差＝自分の動きで説明できない視覚変化
        # ＝egomotion割引の信号（この誤差自体をラベルとして使うのではなく、誤差が
        # 大きいこと自体が「他者・外界」の手がかりになる、という設計）。
        self.vision_dim = 64  # fusion.py: VisionEncoder(embedding_dim=64, ...)
        self.vision_forward_head = nn.Sequential(
            nn.Linear(self.latent_dim + n_actuators, hidden_dim), nn.SiLU(),
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, self.vision_dim))

    def predict_vision(self, z, action, current_vision):
        """残差予測：現在の視覚embedding + Δ(z, 行動) = 次の視覚embeddingの予測。
        predict_proprioと全く同じ形（対象が視覚か固有感覚かだけの違い）。"""
        return current_vision + self.vision_forward_head(torch.cat([z, action], dim=-1))

    # 【後方互換property、2026-07-23】motor_gru/pc_latent/motor_headはmotor_cortex.py
    # (MotorCortex)へ移した。外部25ファイルが brain.motor_head 等で直接アクセスしている
    # ため、属性アクセスだけは従来通り使えるよう転送する（state_dictキーは変わるので
    # チェックポイント側はload_matchingでの読み替えが別途必要）。
    @property
    def motor_gru(self):
        return self.motor_cortex.motor_gru

    @property
    def pc_latent(self):
        return self.motor_cortex.pc_latent

    @property
    def motor_head(self):
        return self.motor_cortex.motor_head

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

    # ─── 行動生成を太郎の中に一元化（2026-07-23）───
    # 【なぜ】これまで行動生成（運動野の精密制御→小脳の自動化ブレンド→探索サンプリング→
    # log_prob）は各目標の学習スクリプト側に書かれ、測定器も独自に再実装していた。運動性喃語
    # （＝どう探索するか）は太郎の身体機能であってスクリプトの機能ではない、という指摘を受け、
    # 行動生成の窓口を太郎の中に集約する。学習ループ・測定器は motor_drive → explore を
    # 呼ぶだけ。色付きノイズ・シナジー（脊髄CPG, spinal_cord/cpg.py）の統合は次段階でここに
    # 入れる（今は白色ガウス＝従来と数値完全一致）。
    def motor_drive(self, z, ne_level, cerebellum=None, min_std=0.05, max_std=0.5):
        """運動野の精密制御（motor_head）＋小脳の自動化ブレンドで、行動の中心(mean)と
        探索のゆらぎ幅(std)を作る。方策勾配の分散が知覚学習の幹へ漏れないよう z を detach
        する（従来と同一）。

        戻り値: (mean, std, cereb_w, cereb_err)
          cereb_w   … 小脳の自動化ブレンド率（小脳OFFなら None）
          cereb_err … 小脳のゲート誤差＝小脳の学習(observe)に使う（小脳OFFなら None）
        """
        policy_m = torch.tanh(self.motor_head(z.detach()))
        std = min_std + ne_level * (max_std - min_std)
        if cerebellum is None:
            return policy_m, std, None, None
        # 馴染んだ状態ほど小脳の滑らかな出力で置換＋探索ノイズ減（結晶化）。
        w_c, cere_a, e_c = cerebellum.gate(z.detach(), policy_m)
        mean = (1.0 - w_c) * policy_m + w_c * cere_a
        return mean, std * (1.0 - w_c), w_c, e_c

    def act_deterministic(self, z, cerebellum=None):
        """決定的な行動平均（評価・agency用、ノイズなし）。小脳ONなら自動化ブレンドを適用。
        motor_drive の探索抜き版（測定器はこれを呼ぶ）。"""
        pm = torch.tanh(self.motor_head(z))
        if cerebellum is None:
            return pm
        w, cere_a, _ = cerebellum.gate(z, pm)
        return (1.0 - w) * pm + w * cere_a

    def enable_spinal_babble(self, n_act, leg_r=(), leg_l=(), arm_r=(), arm_l=(),
                             beta=0.8, synergy=False, syn_w=0.6, seed=None):
        """脊髄CPG（運動性喃語の探索ノイズ源）を太郎の中で有効化する。以後 explore() は
        白色ガウスでなく、色付きノイズ（1/f^β）＋粗いシナジーで探索する。

        【なぜ太郎の中か】運動性喃語＝どう探索するかは太郎の身体機能（脊髄・脳幹の自律回路）。
        学習スクリプトや測定器が毎回組み立てるものではない。βやシナジーの有無は本来は発達段階
        （体の月齢）で太郎自身が決めるべきで、その配線は次段階（developmental_schedule）。今は
        呼び出し側が渡した値を太郎が保持する。leg/arm の関節indexはMIMo身体の配列（身体依存）。"""
        from spinal_cord.cpg import CPG
        self.spinal_cpg = CPG(n_act, leg_r, leg_l, arm_r, arm_l, seed=seed)
        self._babble_beta = beta
        self._babble_synergy = synergy
        self._babble_syn_w = syn_w

    def explore(self, mean, std):
        """探索：mean を中心に std のゆらぎでサンプルし、(行動, log_prob) を返す。
        ゆらぎの性質（白色か色付きか・シナジーの有無）を太郎自身が持つ（脊髄CPG）。

        - 脊髄CPG未初期化（既定）＝白色ガウス＝従来と数値完全一致。
        - enable_spinal_babble 済み＝色付きノイズ（1/f^β）＋シナジー。a = mean + std×色付き。
          色付きノイズは cpg.py で各時点が標準化されており周辺分布は N(0,1) なので、
          log_prob を Normal(mean, std) で計算するのは周辺尤度として正確（時間相関は分散を
          増やすが方策勾配にバイアスは生じない）。
        log_prob はクランプ後の値から計算（範囲外行動への誤った信用割り当てを防ぐ、発話側の
        運動選択 cortex.py と同型の扱い）。"""
        dist = torch.distributions.Normal(mean, std)
        if self.spinal_cpg is None:
            a = torch.clamp(dist.sample(), -1.0, 1.0)
        else:
            noise = self.spinal_cpg.sample(self._babble_beta, synergy=self._babble_synergy,
                                           syn_w=self._babble_syn_w)
            noise = torch.as_tensor(noise, dtype=mean.dtype)
            a = torch.clamp(mean + std * noise, -1.0, 1.0)
        return a, dist.log_prob(a).sum()

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
