"""
環境（親）⇄ 太郎 — 目標B用イベント駆動ループ

モデルAのcore.pyとの違い：
- 身体シミュレーション（胃・肺・内受容感覚）
- 島皮質（体の感覚→大脳皮質への入力線）
- 恒常性の本能（arousal低下→報酬）
- 泣き（自発的発声）
- イベント駆動時間（親がいない時間がある）
"""

import os
import math
import random
import torch
import torch.nn.functional as F
import yaml
from collections import deque
from taro.brain import (Vocabulary, TaroBrain, Cerebellum, Lexicon, BrocasArea, TaroLearner,
                        compute_imitation_reward, compute_prediction_reward,
                        Dopamine, Habituation, LocusCoeruleus, compute_total_reward,
                        Homeostasis, Hippocampus, compute_alignment_credit, Hedonic)
from taro.body import VocalTract, Stomach, Lungs, InternalState, BloodVessel, Adenosine, Tongue
from sim_clock import SimClock
from archive import Archive
from logger import Logger


def _project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


BODY_STATE_DIM = 4  # hunger, sleepiness, discomfort, arousal


class TaroEnvironmentB:
    """目標B用の環境。身体シミュレーション＋イベント駆動。"""

    def __init__(self, config_path=None, run_name=None):
        if config_path is None:
            config_path = os.path.join(_project_root(), "config", "config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        bc = self.cfg["brain"]
        device_str = bc.get("device", "auto")
        if device_str == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device_str

        self.vocab = Vocabulary()
        self.brain = TaroBrain(
            vocab_size=self.vocab.size,
            embedding_dim=bc["embedding_dim"],
            hidden_dim=bc["hidden_dim"],
            num_layers=bc["num_layers"],
            temperature=bc["initial_temperature"],
            body_state_dim=BODY_STATE_DIM,
        ).to(self.device)

        lc = self.cfg["learning"]
        self.learner = TaroLearner(self.brain, lr=lc["lr"], grad_clip=lc["grad_clip"])
        self.dopamine = Dopamine(momentum=lc["baseline_momentum"])
        self.clock = SimClock(seconds_per_turn=self.cfg["sim_clock"]["seconds_per_turn"])
        root = _project_root()
        self.archive = Archive(os.path.join(root, self.cfg["archive"]["archive_dir"]))
        self.logger = Logger(os.path.join(root, self.cfg["logging"]["log_dir"]),
                             run_name=run_name)
        self.weights = self.cfg["reward"]
        self.max_output_length = lc["max_output_length"]

        # 身体（configから読み込み）
        sc = self.cfg.get("stomach", {})
        self.stomach = Stomach(
            capacity=float(sc.get("capacity", 1.0)),
            digestion_rate=float(sc.get("digestion_rate", 0.0003)),
            initial_contents=float(sc.get("initial_contents", 0.7)),
            growth_rate=float(sc.get("growth_rate", 0.0001)),
        )
        lnc = self.cfg.get("lungs", {})
        self.lungs = Lungs(
            capacity=float(lnc.get("capacity", 3.0)),
            air_per_mora=float(lnc.get("air_per_mora", 1.0)),
            recovery_rate=float(lnc.get("recovery_rate", 0.5)),
            growth_rate=float(lnc.get("growth_rate", 0.0001)),
            max_capacity=float(lnc.get("max_capacity", 15.0)),
        )
        bv = self.cfg.get("blood_vessel", {})
        self.blood_vessel = BloodVessel(
            initial_glucose=float(bv.get("initial_glucose", 0.5)),
            # 【人間模倣・較正2026-07-02】0.0001だと約27回/日と人間(6〜12ヶ月は
            # 1日6〜10回, 2〜3時間おき)の3〜4倍速く空腹になっていた。calibrate_hunger.py
            # で振って 0.00003＝約7.5回/日(≈3.2時間おき)に合わせた。時間割授乳の前提。
            consumption_rate=float(bv.get("consumption_rate", 0.00003)),
        )
        self._glucose_efficiency = float(bv.get("glucose_efficiency", 3.0))
        ad = self.cfg.get("adenosine", {})
        self.adenosine = Adenosine(
            production_rate=float(ad.get("production_rate", 0.0001)),
            clearance_rate=float(ad.get("clearance_rate", 0.0003)),
        )
        self.internal_state = InternalState()
        self.vocal_tract = VocalTract()

        for ch in self.vocal_tract.get_all_chars():
            self.vocab.encode(ch)
        self.brain.resize_embedding(self.vocab.size)
        self.brain.to(self.device)
        self.brain.set_vocab_mapping(self.vocab.char2idx)

        # B3：満腹予期を教師ラベルでなく経験(TD)で学ぶか。True=イベント単位の時間割引TD。
        self.td_satiety = bool(self.cfg.get("td_satiety", True))
        self._now = 0                    # 直近tickの時刻(秒)。record_episodeの時刻に使う
        self._td_tau = 1800.0            # TDの時間割引の時定数(秒)＝授乳/消化のスケールに合わせる

        # B4：味覚。舌(検知)＝体の器官、快(liking)＝脳の本能（ドーパミンwantingとは別）。
        # 食べた瞬間の快を、満腹予期TDの強いUS（合図まんまへ後退させる燃料）にする。
        self.tongue = Tongue()                 # 甘みを検知するセンサー
        self.hedonic = Hedonic()               # 甘み→快(liking)。オピオイド的
        self._milk_sweetness = 0.8             # ⚠️未検証：まんま/母乳の甘み値（後で感度確認）
        self._pleasure_since = 0.0             # 直近エピソード以降に得た快の合計（TDのUSに渡す）

        # 脳の部品
        self.habituation = Habituation(history_size=20, decay_rate=0.05)
        self.locus_coeruleus = LocusCoeruleus()
        self.cerebellum = Cerebellum()
        self.brocas_area = BrocasArea()
        # B6-1：原-辞書（聞いた発話から予測しやすい並びを1単位として蓄える）
        self.lexicon = Lexicon()
        # B6-2：自発発声で「自由喃語」と「辞書の語」を頻度で競わせるときの、自由喃語側の
        # 引きの強さ。⚠️唯一の定数。除去テスト可（→∞で語を一切出さない＝B6-1までの挙動）。
        # 辞書頻度が累積で育つほど語が選ばれやすくなり、喃語→単語の移行がなだらかに創発する。
        self._k_babble = 1000.0
        # B6-4：語の選択を「今の内的状態と語の連合状態の一致」でどれだけ後押しするか。
        # ⚠️定数。除去テスト可（0で頻度のみ＝B6-2の挙動）。空腹時に「まんま(空腹と連合)」が
        # 選ばれやすくなる＝mand化。報酬でなく共起連合(Smith & Yu)由来。
        self._assoc_gain = 2.0
        self.homeostasis = Homeostasis()
        hc = self.cfg.get("hippocampus", {})
        self.hippocampus = Hippocampus(max_capacity=hc.get("max_capacity", 500))

        # 成功判定（N回中M回方式）
        succ = self.cfg.get("success", {})
        self.partial_threshold = succ.get("partial_threshold", 0.8)
        _window = succ.get("partial_window", 10)   # 直近N回を見る
        self.partial_success_target = succ.get("partial_success", 8)  # N回中M回で成功
        self.partial_window = deque(maxlen=_window)
        self.exact_streak = 0

        # 隠れ状態の保持（イベント間で維持）
        self._hidden = None

        # replayViewer用トレースログ（オプトイン。set_traceで有効化）
        self._trace = None

    def set_trace(self, trace_logger):
        """replayViewer用トレースログを有効化する（Noneで無効）。"""
        self._trace = trace_logger

    def trace_event(self, sim_seconds, kind, active, flows, utter="", say="", consolidated=0):
        """
        1イベントを trace.jsonl に書き出す（replayViewer用）。
        発火した部品(active)・流れ(flows)・そのときの数値(空腹/NE/ドーパミン/
        幸福度)を記録する。トレース無効時は何もしない。
        happiness＝1−arousal(つらさの裏返し)、dopamine＝報酬予測のbaseline を
        表示用スカラーとして使う（正確な生理値ではない近似）。

        consolidated: 直前のconsolidate()で実際に再生・定着した記憶の件数（sleepイベント用）。
        海馬の活動量は「記憶バッファの詰まり具合」ではなく「直前にどれだけ働いたか」で測るため、
        consolidate()がバッファを空にする"前"の仕事量をここで受け取る必要がある。
        """
        if self._trace is None:
            return
        arousal = self.internal_state.get_arousal()
        # 記録する数値はここに1行足すだけで増やせる（後処理の集約は数値を
        # 自動検出するので、足せば概観ログにも自動で反映される）。0〜1推奨。
        metrics = {
            "hunger":    float(self.internal_state.hunger),
            "ne":        float(self.locus_coeruleus.get_ne_level()),
            "dopamine":  max(0.0, min(1.0, float(self.dopamine.get_baseline()))),
            "happiness": max(0.0, min(1.0, 1.0 - float(arousal))),
            "pleasure":  max(0.0, min(1.0, float(self.hedonic.get()))),   # B4：味の快(liking)
            "sleepiness": max(0.0, min(1.0, float(self.internal_state.sleepiness))),
            "discomfort": max(0.0, min(1.0, float(self.internal_state.discomfort))),
        }
        rec = {"type": "event", "t": int(sim_seconds), "kind": kind,
               "modules": active, "flows": flows, "utter": utter}
        if say:                       # 親の発話（あれば）。replayViewerの会話表示用。
            rec["say"] = say
        fire = self._fire_indices()   # そのイベント時に発火したGRUノード番号（NN活性ビュー用）
        if fire is not None:
            rec["fire"] = fire
        # 各臓器の"実測"活動量[0-1]。従来のmodules（kind→部位の決め打ち表）と違い、実際に
        # その時点の内部変数を読んで数値化する。replayViewerはこれがあれば実測を優先表示する。
        rec["organs"] = {k: round(v, 3) for k, v in self._organ_activity(utter, consolidated).items()}
        rec.update({k: round(v, 3) for k, v in metrics.items()})
        self._trace.write_event(rec)

    def _organ_activity(self, utter="", consolidated=0):
        """
        各臓器の"実測"活動量[0-1]を返す（体ビューの実測表示用）。kindからの決め打ちでなく、
        その時点で実際にプログラムが保持している内部変数を読む。

        方針（fMRI的＝「そのイベントで実際に使った臓器だけ光る」）：各臓器は原則「何かした時
        だけ高く、平常時は低い」よう定義する。ただし脳(大脳皮質)・NE系(青斑核)は現実に常時
        働いており、正直に測ると常に活動が高い＝それはそのまま表示する（偽って0にはしない）。
        スケール定数は表示用の便宜（相対比較が目的で絶対値の生理的厳密さは求めない）。
        """
        out = {}
        # 大脳皮質：脳は毎イベント処理する＝常に働く。生の隠れ状態強度はほぼ一定なので、
        # 発火の"強さ"として控えめな定常値を出す（正直に「常に動いている」を表す）。
        try:
            h = self._hidden
            if h is not None:
                v = h.detach().reshape(-1).abs()
                out["cortex"] = float(max(0.0, min(1.0, v.mean().item() * 1.5)))
        except Exception:
            pass
        # 島皮質：内受容感覚の統合＝覚醒(つらさ)。空腹/眠気/不快が高い時に上がる。
        out["insula"] = float(max(0.0, min(1.0, self.internal_state.get_arousal())))
        # 胃：食事中は活動、そうでなければ中身の割合。空腹の赤ちゃんは普段ほぼ空なので、
        # 「食べている時に光る」よう is_feeding を優先する。
        try:
            if self.stomach.is_feeding():
                out["stomach"] = 1.0
            else:
                out["stomach"] = float(max(0.0, min(1.0, self.stomach.contents / max(1e-6, self.stomach.capacity))))
        except Exception:
            pass
        # 青斑核：ノルアドレナリン(NE)レベル。発達初期は本当に高く、育つと下がる（実測）。
        out["locus"] = float(max(0.0, min(1.0, self.locus_coeruleus.get_ne_level())))
        # 肺：発声・泣きで息を吐く時に働く（"空き具合"ではなく"今息を使っているか"）。
        lung = min(1.0, len(utter) / 6.0) if utter else 0.0
        try:
            if self.internal_state.is_crying():
                lung = max(lung, float(min(1.0, self.internal_state.cry_intensity)))
        except Exception:
            pass
        out["lungs"] = lung
        # クリティック：今の体の状態価値（安静に近いほど高い。低い＝つらい状態）。
        try:
            with torch.no_grad():
                cv = self.brain.critic(self._body_state_tensor())
                out["critic"] = float(max(0.0, min(1.0, torch.sigmoid(cv).mean().item())))
        except Exception:
            pass
        # 海馬：睡眠時は「直前にconsolidate()で実際に再生・定着させた件数」で活動を測る
        # （NREM睡眠中のシャープ波リプル＝この瞬間こそ海馬が最も働く）。consolidate()は
        # 処理後にバッファを空にするため、起きている間の「バッファの詰まり具合」を
        # そのまま使うと睡眠時にはもう空＝無活動に見えてしまうバグがあった（ユーザー指摘）。
        # 起きている間（consolidated=0）は従来通りバッファの詰まり具合（今後の仕事量の予兆）。
        try:
            if consolidated > 0:
                out["hippocampus"] = float(max(0.0, min(1.0, consolidated / 50.0)))  # ⚠️正規化定数、未検証
            else:
                out["hippocampus"] = float(max(0.0, min(1.0, len(self.hippocampus.episodes) / max(1, self.hippocampus.max_capacity))))
        except Exception:
            pass
        # 産出経路（発声した文字数で駆動）：声道・小脳(運動実行)・基底核(行動選択)。
        # 割る数6は「発話1語ぶんの目安の長さ」＝数文字でもはっきり点灯させる表示用の便宜。
        spoke = min(1.0, len(utter) / 6.0) if utter else 0.0
        out["vocal"] = spoke
        out["cerebellum"] = spoke
        out["basal_ganglia"] = 1.0 if utter else 0.0
        return out

    def _fire_indices(self, top_k=32):
        """直近の隠れ状態(self._hidden)から「発火した」GRUノードの番号リストを返す。
        活性の大きい上位top_k個を発火とする（常に一定数が光って賑やか＝かっこよさ用。
        どのノードが強いかはイベントで変わる）。replayViewerのNN活性ビューが使う。
        隠れ状態が無ければNone。"""
        h = self._hidden
        if h is None:
            return None
        try:
            vec = h.detach().reshape(-1)
            n = min(int(self.brain.hidden_dim), vec.shape[0])
            vec = vec[-n:].abs()                 # GRU最終層の隠れ状態の強さ
            k = min(top_k, n)
            return vec.topk(k).indices.reshape(-1).tolist()
        except Exception:
            return None

    def _body_state_tensor(self):
        """内部状態をテンソルに変換して脳に渡す。"""
        vec = self.internal_state.get_state_vector()
        return torch.tensor(vec, dtype=torch.float32, device=self.device)

    def tick_body(self, elapsed_seconds=1, sim_seconds=0):
        """
        身体シミュレーションを進める。親がいなくても毎tick呼ばれる。
        軽い計算のみ。

        胃の消化量 → 血管（血糖値）→ 空腹感 の順に更新。
        声道の成熟もここで進める（時間が経てば成熟する。親との会話は無関係）。
        """
        self._now = sim_seconds          # B3：イベント記録の時刻に使う（呼び出し側の変更不要）
        for _ in range(elapsed_seconds):
            self.stomach.tick()
            self.blood_vessel.receive_glucose(
                self.stomach.get_last_absorption() * self._glucose_efficiency)
            self.blood_vessel.tick()
            self.lungs.tick()
            self.internal_state.update_from_body(self.stomach, self.blood_vessel, self.lungs)
            self.internal_state.tick(adenosine=self.adenosine)
            self.stomach.grow()
            self.lungs.grow()
        vm = self.cfg.get("vocal_maturation", {})
        self.vocal_tract.update_stage(
            sim_seconds,
            vm.get("stage1_time", 300),
            vm.get("stage2_time", 900),
            vm.get("stage3_time", 1500),
            decouple_time=vm.get("decouple_time", 1200),
        )
        # 【人間模倣】探索の結晶化（B2-8）。声道と同じ月齢基準で、発達が
        # 進むほどNEの探索上限を下げる。完全成熟＝声道stage3の解禁時期
        # （12ヶ月）を基準にする。
        full_maturity = vm.get("stage3_time", 1500)
        if full_maturity > 0:
            self.locus_coeruleus.mature(sim_seconds / full_maturity)
        # B4：味の快は一過性（食べた瞬間に強く→数秒で消える）。表示用に減衰（学習は別変数）。
        self.hedonic.last_pleasure *= 0.5

    def check_cry(self):
        """
        泣いているかどうか。internal_stateの泣き管理を使う。

        戻り値: (泣いているか: bool, 泣きの強さ: float)
        """
        return self.internal_state.is_crying(), self.internal_state.cry_intensity

    def feed(self, amount=0.6):
        """授乳を開始する。一瞬ではなく、約30分かけて少しずつ胃に入る。"""
        self.stomach.start_feeding(amount)
        self.internal_state.on_feed(amount)   # B2-19：消化の下流（時間差で排泄→おむつ不快）へ
        # B4：食べた瞬間、舌が甘みを検知→脳が快(liking)を出す。これを満腹予期TDのUSに積む。
        taste = self.tongue.taste(self._milk_sweetness)
        self._pleasure_since += self.hedonic.evaluate(taste)

    def comfort(self, care_type="comfort"):
        """世話。discomfortやsleepinessを下げる。"""
        self.internal_state.apply_care(care_type)

    def step(self, parent_text, r_social=0.0, satiety_target=None):
        """
        親が話しかけたときの1ターン。モデルAのstep()と同じ構造だが、
        島皮質経由で体の感覚が脳に入る点と、恒常性の報酬が加わる点が異なる。

        satiety_target: B2-10。この発話の後に授乳が来るかの実際の結果
            （1.0=授乳が来る／0.0=来ない）。Noneなら満腹予期は学習しない。
            太郎はこの結果を教師に「聞いた音＋状態→満腹の到来」を予測できるよう学ぶ。
        """
        parent_tokens = self.vocab.encode(parent_text)
        self.brain.resize_embedding(self.vocab.size)
        self.brain.to(self.device)

        body_state = self._body_state_tensor()

        # 知覚学習
        full_tokens = [1] + parent_tokens + [2]
        p_loss, pred_probs, satiety_logit = self.learner.learn_perception(full_tokens, body_state=body_state)

        # B6-1：聞く側で分節して原-辞書に蓄積（Saffranの統計的分節）。
        # pred_probs[k] は full_tokens[k+1] を直前から予測した確率分布。parent_tokens[k]
        # ＝full_tokens[k+1] なので、その確率＝「その音を予測できた自信度」＝遷移確率。
        # 報酬は使わず、予測の自信度と頻度だけで「よく知っている並び」を単位化する。
        # 読点等は書き言葉の区切り記号で、赤ちゃんが実際に耳にする音ではない（無音の間に
        # 相当）ため、そこで発話を句に区切ってから句ごとに独立して分節する（単に取り除くと
        # 前後の句がくっついて誤った並びになるため）。
        if len(pred_probs) >= len(parent_tokens) and len(parent_tokens) >= 2:
            # B6-4：語↔内的状態の連合のため、聞いた瞬間の動因[空腹,眠気,不快]を渡す。
            heard_state = (float(self.internal_state.hunger),
                           float(self.internal_state.sleepiness),
                           float(self.internal_state.discomfort))
            PUNCT = "、。！？…・"
            clause_toks = []; clause_confs = []
            for k in range(len(parent_tokens)):
                if k < len(parent_text) and parent_text[k] in PUNCT:
                    if clause_toks:
                        self.lexicon.observe(clause_toks, clause_confs, state=heard_state)
                    clause_toks = []; clause_confs = []
                    continue
                clause_toks.append(parent_tokens[k])
                try:
                    clause_confs.append(float(pred_probs[k][parent_tokens[k]]))
                except Exception:
                    clause_confs.append(0.0)
            if clause_toks:
                self.lexicon.observe(clause_toks, clause_confs, state=heard_state)

        # 聞く（体の感覚も合流）
        listen_input = torch.tensor([full_tokens], device=self.device)
        with torch.no_grad():
            _, h = self.brain.forward_hidden(listen_input, body_state=body_state)
        self._hidden = h

        # B2-11：海馬に記録し、睡眠中に何度も反芻させる。太郎が起きている間に
        # 親と話す機会（年1万回程度）は自発喃語（年10万回以上）よりずっと少なく、
        # この頻度差が理解の学習を産出より大きく遅らせていた。睡眠リプレイで
        # 反芻回数を稼ぎ、実際の会話機会の少なさを補う。
        self.hippocampus.record_episode(full_tokens, body_state, event_time=self._now,
                                        reward_since=self._pleasure_since)
        self._pleasure_since = 0.0   # B4：快はこのエピソードに渡したのでリセット

        # 発話計画
        self.brocas_area.plan(parent_text, self.cerebellum, self.vocal_tract)

        # 発声
        ne_level = self.locus_coeruleus.get_ne_level()
        generated, log_probs, _ = self.brain.generate(
            hidden=h,
            max_length=self.max_output_length,
            eos_idx=2,
            stamina=self.lungs.get(),
            vocal_tract=self.vocal_tract,
            ne_level=ne_level,
            cerebellum=self.cerebellum,
            speech_plan=self.brocas_area,
            body_state=body_state,
        )
        taro_text = self.vocab.decode(generated)

        # 発声で肺の空気を消費
        self.lungs.consume(len(generated))

        # 報酬計算
        r_imit = compute_imitation_reward(parent_tokens, generated,
                                          vocab=self.vocab, vocal_tract=self.vocal_tract)
        r_pred = compute_prediction_reward(pred_probs, parent_tokens)
        r_habit = self.habituation.compute_penalty(taro_text)

        # 身体更新（世話の効果を反映してからarousalを取る）
        self.internal_state.update_from_body(self.stomach, self.blood_vessel)
        current_arousal = self.internal_state.get_arousal()
        r_home = self.homeostasis.compute_reward(current_arousal)

        R = compute_total_reward(r_imit, r_pred, r_social, r_habit, self.weights)
        R = max(0.0, R + self.weights.get("w_home", 0.3) * r_home)

        # B-11：状態依存クリティックでbaselineを取る（Dopamineのスカラー
        # 移動平均は空腹時と機嫌がいい時を区別できなかったため）
        value = self.brain.critic(body_state)
        delta = R - value.item()
        self.dopamine.compute_rpe(R)  # アーカイブ保存互換のため基準値のみ更新（学習には未使用）

        # 学習（B2-2：親の発話とのアライメントで文字ごとに信用割り当て）
        credits = compute_alignment_credit(parent_tokens, generated,
                                           vocab=self.vocab, vocal_tract=self.vocal_tract)
        a_loss = self.learner.learn_action(log_probs, delta, credits=credits)
        value_loss = self.learner.compute_value_loss(value, R)

        # 満腹予期の学習。B3：既定(td_satiety=True)では全知ラベルを使わず、
        # 睡眠中の consolidate で「経験した安腹→時間割引TD」で学ぶ（ここでは何もしない）。
        # td_satiety=False のときだけ従来の教師あり(B2-10)にフォールバックする（比較用）。
        satiety_loss = None
        if (not self.td_satiety) and satiety_target is not None and satiety_logit is not None:
            tgt = torch.tensor(float(satiety_target), device=satiety_logit.device,
                               dtype=satiety_logit.dtype)
            satiety_loss = F.binary_cross_entropy_with_logits(satiety_logit, tgt)

        pl, al = self.learner.update(p_loss, a_loss, value_loss, satiety_loss=satiety_loss)

        # 青斑核
        self.locus_coeruleus.observe_reward(R)
        self.locus_coeruleus.release_ne()
        self.brain.receive_ne(self.locus_coeruleus.get_ne_level())

        self.clock.tick(tokens_heard=len(parent_tokens))

        # 成功判定（N/M方式）
        exact_match = taro_text == parent_text
        partial_match = r_imit >= self.partial_threshold
        self.partial_window.append(1 if partial_match else 0)
        partial_score = sum(self.partial_window)
        partial_goal = (len(self.partial_window) == self.partial_window.maxlen
                        and partial_score >= self.partial_success_target)
        if exact_match:
            self.exact_streak += 1
        else:
            self.exact_streak = 0

        turn = self.clock.total_turns

        return {
            "turn": turn,
            "age": self.clock.age_str(),
            "parent": parent_text,
            "taro": taro_text,
            "r_imit": r_imit,
            "r_pred": r_pred,
            "r_social": r_social,
            "r_home": r_home,
            "R": R,
            "delta": delta,
            "p_loss": pl,
            "a_loss": al,
            "hunger": self.internal_state.hunger,
            "arousal": self.internal_state.get_arousal(),
            "sleepiness": self.internal_state.sleepiness,
            "stamina": self.lungs.get(),
            "partial_score": partial_score,
            "partial_goal": partial_goal,
            "exact_streak": self.exact_streak,
            "exact_match": exact_match,
            "partial_match": partial_match,
        }

    @staticmethod
    def _state_match(cur, assoc):
        """
        B6-4：今の動因状態curと、語が連合する状態assocの一致度（コサイン, 0〜1）。
        両方とも非負ベクトルなので0〜1。どちらかがほぼ0（動因なし/連合なし）なら0。
        """
        if assoc is None:
            return 0.0
        dot = sum(c * a for c, a in zip(cur, assoc))
        nc = math.sqrt(sum(c * c for c in cur))
        na = math.sqrt(sum(a * a for a in assoc))
        if nc < 1e-6 or na < 1e-6:
            return 0.0
        return max(0.0, min(1.0, dot / (nc * na)))

    def _pick_lexicon_word(self):
        """
        B6-2/B6-4：自発発声で「辞書の語を単位として出す」か「自由喃語」かを選ぶ。

        【人間模倣】語テンプレート（Vihman）が育つと産出がそれに導かれる。候補は
        「太郎が調音できる（＝喃語練習で口の動きを覚えた）辞書の語」だけ＝調音フィルタ
        （初語は"自分が産出に慣れている音"で構成される, Vihman）。

        B6-4：さらに「今の内的状態」と「語が連合する状態」の一致で重みを後押しする
        （空腹時に、空腹と連合した"まんま"が選ばれやすい＝mand化）。連合は共起の統計
        （Smith & Yu, cross-situational learning）で、報酬は使わない。

        重み＝辞書頻度 ×（1 + 連合ゲイン × 状態一致）。連合を切れば（ゲイン0）頻度のみ。

        戻り値：産出する語のテキスト（辞書の語が選ばれたとき）／None（自由喃語）。
        """
        if not self.lexicon.counts:
            return None
        # 今の動因状態[空腹,眠気,不快]
        cur = (float(self.internal_state.hunger),
               float(self.internal_state.sleepiness),
               float(self.internal_state.discomfort))
        words = []
        weights = []
        for toks, cnt in self.lexicon.counts.items():
            text = self.vocab.decode(list(toks))
            # 全文字を調音できる語だけ候補にする（調音フィルタ）
            if text and all(self.cerebellum.lookup_motor(ch) is not None for ch in text):
                match = self._state_match(cur, self.lexicon.assoc(toks))   # B6-4
                weights.append(cnt * (1.0 + self._assoc_gain * match))
                words.append(text)
        if not words:
            return None
        total = sum(weights) + self._k_babble
        r = random.random() * total
        if r < self._k_babble:            # 自由喃語が選ばれた
            return None
        r -= self._k_babble
        for w, wt in zip(words, weights):
            if r < wt:
                return w
            r -= wt
        return words[-1]

    def self_babble(self):
        """
        太郎が一人で喃語を出す。脳の現在の分布からサンプリング。

        【人間模倣】
        0〜6か月の乳児は穏やかな時間に発話計画なしで自発的に声を出す。
        脳が「今出しやすい音」を自由に試す → 海馬に記録 → 睡眠時に皮質へ定着。

        B6-2：辞書の語が育つと、自発発声の一部が「その語をブローカ野で区切って産出」
        （＝まとまりで出て止まる）に置き換わる。どの語が出るかは頻度×調音可能性で創発
        （まんま/だっこの個体差は保たれる）。語が選ばれなければ従来通り自由喃語。

        B-5変更：喃語のたびに学習しない。経験を海馬に蓄積し、
        睡眠移行時に consolidate() でまとめて大脳皮質を更新する。
        """
        body_state = self._body_state_tensor()

        # B6-2：辞書の語を単位として出すか、自由喃語か。語なら計画（ブローカ野）を使う。
        word_text = self._pick_lexicon_word()
        if word_text:
            self.brocas_area.plan(word_text, self.cerebellum, self.vocal_tract)
            speech_plan = self.brocas_area
        else:
            speech_plan = None            # 自由喃語（従来の喃語期パス）

        # 発声
        ne_level = self.locus_coeruleus.get_ne_level()
        generated, log_probs, _ = self.brain.generate(
            hidden=self._hidden,
            max_length=self.max_output_length,
            eos_idx=2,
            stamina=self.lungs.get(),
            vocal_tract=self.vocal_tract,
            ne_level=ne_level,
            cerebellum=self.cerebellum,
            speech_plan=speech_plan,
            body_state=body_state,
        )

        babble_text = self.vocab.decode(generated)
        self.lungs.consume(len(generated))

        if not generated:
            return {"taro": "", "R": 0.0, "r_pred": 0.0, "r_home": 0.0, "r_habit": 0.0,
                    "tokens": [], "log_probs": []}

        # 自分の声を聞く（hidden state更新）。学習はしない。
        full_tokens = [1] + generated + [2]
        listen_input = torch.tensor([full_tokens], device=self.device)
        with torch.no_grad():
            _, h = self.brain.forward_hidden(listen_input, body_state=body_state)
        self._hidden = h

        # 海馬に経験を記録（睡眠移行時にまとめて定着）
        self.hippocampus.record_episode(full_tokens, body_state, event_time=self._now,
                                        reward_since=self._pleasure_since)
        self._pleasure_since = 0.0

        # 馴化（飽き）：親との会話だけでなく自発喃語にも適用する
        # B-11修正：従来はstep()にしか適用されておらず、1日240〜380回起きる
        # 自発喃語には同じ音の繰り返しにペナルティが一切働いていなかった
        r_habit = self.habituation.compute_penalty(babble_text)

        # 恒常性報酬の参照値のみ計算（ログ用）
        self.internal_state.update_from_body(self.stomach, self.blood_vessel)
        current_arousal = self.internal_state.get_arousal()
        r_home = self.homeostasis.compute_reward(current_arousal)
        R = max(0.0, self.weights.get("w_home", 0.3) * r_home + r_habit)

        self.locus_coeruleus.observe_reward(0.0)
        self.locus_coeruleus.release_ne()
        self.brain.receive_ne(self.locus_coeruleus.get_ne_level())

        self.clock.tick(tokens_heard=0)

        return {
            "taro": babble_text,
            "r_pred": 0.0,  # 睡眠時に計算するため未算出
            "r_home": r_home,
            "r_habit": r_habit,
            "R": R,
            "tokens": generated,
            "log_probs": log_probs,
        }

    def word_similarity(self, generated_tokens, word):
        """
        太郎の発声が指定した語にどれだけ似ているか（連続値、0〜1）を返す。

        【人間模倣】Skinner (1957) の言語行動理論における「mand（要求）」：
        要求語は特定の欠乏状態（例：空腹）でのみ、その状態を解消する結果
        （例：食べ物）と結びつく。「まんま」が食べ物の要求として機能する
        条件は、(1) 実際に空腹であること、(2) 発声がその語に似ていることの
        両方。ここでは(2)の度合いだけを連続値で返す（(1)は呼び出し側で判定）。

        B2-3：以前はここで閾値による足切り（0.4未満なら0扱い）をしていたが、
        「似ているかどうか」を段階なしのオールオアナッシングで判定するのは
        人間にない仕組み。親は完璧に言えなくても「近い音」には連続的に
        反応の度合いを変える。足切りをやめ、呼び出し側で類似度そのものを
        確率として使う（近いほど気づかれやすい）よう変更した。
        """
        if not generated_tokens:
            return 0.0
        word_tokens = self.vocab.encode(word)
        return compute_imitation_reward(word_tokens, generated_tokens,
                                        vocab=self.vocab, vocal_tract=self.vocal_tract)

    def diagnostic_babble_at_hunger(self, hunger_value, target_word="まんま", n_samples=200):
        """
        診断専用：hungerを強制的に固定した状態で喃語をn_samples回生成し、
        target_wordとの平均類似度を返す。実際のinternal_stateは変更しない。

        自然に変動するhungerとの相関は、ノイズが大きく「学習されているが
        弱すぎて埋もれている」のか「そもそも学習されていない」のかを
        区別できない。ここでは空腹度だけを人工的に0または1に固定し、
        それ以外の内的状態は現在の値のまま揃えることで、hungerという
        1変数の影響だけを取り出して測定する（他の要因による交絡を排除）。

        戻り値: 類似度のリスト（空でない喃語のみ）
        """
        current_state = self.internal_state.get_state_vector()
        sleepiness, discomfort = current_state[1], current_state[2]
        arousal = max(hunger_value, sleepiness, discomfort)
        fake_state = [hunger_value, sleepiness, discomfort, arousal]
        body_state = torch.tensor(fake_state, dtype=torch.float32, device=self.device)

        ne_level = self.locus_coeruleus.get_ne_level()
        sims = []
        for _ in range(n_samples):
            generated, _, _ = self.brain.generate(
                hidden=self._hidden,
                max_length=self.max_output_length,
                eos_idx=2,
                stamina=self.lungs.get(),
                vocal_tract=self.vocal_tract,
                ne_level=ne_level,
                cerebellum=None,
                speech_plan=None,
                body_state=body_state,
            )
            if generated:
                sims.append(self.word_similarity(generated, target_word))

        return sims

    def comprehension_probe(self, heard_word, hunger_value, n_samples=100):
        """
        理解テスト（A案・産出でなく「聞く」側を測る／モデルは変更しない）。

        産出側（diagnostic_babble_at_hunger）は「空腹だとまんまと言う」を測るが、
        それは要求発声（conditioned mand）でも成立し、意味の理解を意味しない。
        こちらは逆に、太郎に heard_word を「聞かせた」あと、その瞬間の内部を読む：

        - critic_value：唯一の「予期」スカラー。ただし critic は body_state しか
          入力に取らないので、聞いた語によって変わらないはず（＝現アーキテクチャに
          "聞いた語→体の未来の予期" を表す場所が無いことの実証）。
        - hidden：聞いた直後のGRU隠れ状態。語ごとに区別できるか＝認識のRung1。
        - echoic_mama_sim：聞いた直後の隠れ状態から発声させ、「まんま」への
          平均類似度。聞く→言うの結び付き（まんまを聞くとまんまを言いやすいか）。

        hunger_value で空腹を固定し、他の内的状態は現在値に揃える
        （diagnostic_babble_at_hunger と同じdo介入）。実際の状態は変えない。
        """
        current_state = self.internal_state.get_state_vector()
        sleepiness, discomfort = current_state[1], current_state[2]
        arousal = max(hunger_value, sleepiness, discomfort)
        fake_state = [hunger_value, sleepiness, discomfort, arousal]
        body_state = torch.tensor(fake_state, dtype=torch.float32, device=self.device)

        heard_tokens = [1] + self.vocab.encode(heard_word) + [2]
        listen_input = torch.tensor([heard_tokens], device=self.device)

        ne_level = self.locus_coeruleus.get_ne_level()
        sims = []
        with torch.no_grad():
            out, hidden = self.brain.forward_hidden(listen_input, body_state=body_state)
            critic_value = self.brain.critic(body_state).item()
            # B2-10：聞いた語＋状態から「食べ物（授乳）が来る」先取りを読む。
            # 注意（指標訂正2026-07-02）：この生値そのものは理解の証拠ではない
            # （hungerで説明できる分を含む）。理解は run_comprehension_probe 側で
            # hungerを固定し“語だけ”変えたときの差（語の寄与）として読む。
            satiety = (self.brain.predict_satiety(out[0, -1]).item()
                       if getattr(self.brain, "satiety_head", None) is not None else None)
            hidden_vec = hidden.detach().reshape(-1).cpu().tolist()
            for _ in range(n_samples):
                generated, _, _ = self.brain.generate(
                    hidden=hidden,
                    max_length=self.max_output_length,
                    eos_idx=2,
                    stamina=self.lungs.get(),
                    vocal_tract=self.vocal_tract,
                    ne_level=ne_level,
                    cerebellum=None,
                    speech_plan=None,
                    body_state=body_state,
                )
                if generated:
                    sims.append(self.word_similarity(generated, "まんま"))

        return {
            "critic_value": critic_value,
            "satiety": satiety,
            "hidden": hidden_vec,
            "echoic_mama_sim": (sum(sims) / len(sims) if sims else 0.0),
            "n": len(sims),
        }

    def respond_to_babble(self, generated_tokens, log_probs, candidate_words, r_habit=0.0,
                          hunger=0.0, social=True, mand=False):
        """
        親が自発喃語に気づいて反応する（随伴的社会的フィードバック）。

        【人間模倣】Goldstein & Schwade (2008)：養育者は乳児の自発発声の
        約30〜50%に気づいて反応し、**言葉らしい発声ほど**反応をもらいやすい。
        反応をもらった発声パターンは乳児が再び自発的に発しやすくなる。

        B-9まで自発喃語（self_babble）は知覚学習（consolidate）のみで、
        発声を選ぶ4つのhead（head_place等）には報酬が一切届いていなかった。
        親との会話（step）と同じ経路（learn_action）をここでも使うことで、
        模倣と自発発話が同じ強化学習の仕組みを共有するようにする。

        B-11修正：B-10では「内的状態から先に正解ラベルを決め打ち」していた
        （空腹なら常に「まんま」が正解、という教師あり学習に近い設計）。
        これはGoldstein & Schwadeの趣旨（太郎の発声そのものが言葉らしいか
        どうかに養育者が反応する）とズレていた。太郎の発声を先に
        全候補語と比較し、最も近い語との類似度で判定するよう変更。
        内的状態は反応の対象選びには使わず、「太郎が実際に発した音」
        だけで判定する。

        B2-3修正：類似度による足切り（閾値0.4未満なら反応しない）を撤廃。
        6ヶ月で「ま」「ん」が解禁された後、類似度が平均0.32まで上がっても
        一度も0.4を超えなかったことが判明し（B2-2の分析）、「まだ下手だが
        惜しい」試みに一切報酬が発生しないため、改善の足がかり自体が
        存在しないという鶏と卵の状態になっていた。閾値を撤廃し、
        気づいた（呼び出し側の確率で決まる）以上は必ず、類似度に応じた
        連続的な大きさの報酬を与えるようにした。

        candidate_words: 反応しうる語の候補リスト（例：["まんま","よしよし","まま"]）
        """
        if not generated_tokens or not log_probs or not candidate_words:
            return None

        best_word = None
        best_r_imit = -1.0
        for word in candidate_words:
            word_tokens = self.vocab.encode(word)
            r = compute_imitation_reward(word_tokens, generated_tokens,
                                         vocab=self.vocab, vocal_tract=self.vocal_tract)
            if r > best_r_imit:
                best_r_imit = r
                best_word = word

        r_imit = best_r_imit
        r_social = 0.5  # 親が気づいて反応してくれたこと自体の報酬

        # 2つの報酬経路を合算して1回で学習する（同じlog_probsに2度backward
        # するとエラーになるため、更新はここ1か所に統一）。
        R = 0.0
        # 経路1：社会的反応（Goldstein & Schwade）。言葉らしさへの反応で、
        # 空腹とは無関係に起きる（＝これ単独だと無条件にまんまを言うよう学ぶ）。
        if social:
            R += self.weights["w_imit"] * r_imit + self.weights["w_social"] * r_social
        # 経路2：要求語（mandに着想・実装は⚠️逸脱）。B2-9追加。
        # 着想：空腹という動因状態でまんま様発声が結果（食べ物→解消）と結びつく
        # ＝Skinner(1957)のmand。満腹時は解消がないのでhungerに比例させ、閾値なしで
        # 「空腹時のみ報われる」ギャップを作る。
        # ⚠️逸脱：本来のmandは「実際に食べて解消した結果」で強化されるが、ここでは
        # 発声を認識した瞬間のhunger水準を即時報酬にしている（授乳の実行や後続の
        # arousal低下に依存しない）＝結果随伴ではなく動因直結の報酬シェーピング。
        # さらに恒常性報酬 r_home（実際の解消差分）と同じ重み w_home を流用しており、
        # 同じ「空腹→授乳→解消」の関係を機能的に二重計上している。よって
        # 「Skinnerのmandを忠実に実装した」とは言えず、mandに着想を得た近似に留まる。
        r_mand = 0.0
        if mand:
            r_mand = self.weights.get("w_home", 0.3) * max(0.0, hunger)
            R += r_mand
        R = max(0.0, R + r_habit)

        body_state = self._body_state_tensor()
        value = self.brain.critic(body_state)
        delta = R - value.item()
        self.dopamine.compute_rpe(R)  # アーカイブ保存互換のため基準値のみ更新（学習には未使用）

        # B2-2：最も近かった候補語とのアライメントで文字ごとに信用割り当て
        best_word_tokens = self.vocab.encode(best_word)
        credits = compute_alignment_credit(best_word_tokens, generated_tokens,
                                           vocab=self.vocab, vocal_tract=self.vocal_tract)
        a_loss = self.learner.learn_action(log_probs, delta, credits=credits)
        value_loss = self.learner.compute_value_loss(value, R)

        # 【学習材料は限定しない（2026-07-02, #4）】親の返事(best_word)を"知覚入力"として
        # 太郎に聞かせる。これまで返事は報酬(産出の強化)だけで、耳から言葉が入っておらず、
        # 理解(聞いて分かる)の学習機会が乏しかった（喃語は年10万回でも親の発話を聞く機会は
        # 少ない、という頻度差の主因のひとつ）。ここで親の語を知覚学習に通し、海馬にも記録して
        # 睡眠で反芻する。満腹予期の教師は付けない（この社会的返事の後に授乳が来るわけでは
        # ないため）＝純粋に「その語を聞いた」という知覚材料として扱う。副次的に、まんまを
        # 授乳以外の場面でも聞くことになり、語と空腹の相関がさらに緩む。
        heard_tokens = [1] + best_word_tokens + [2]
        p_loss_heard, _, _ = self.learner.learn_perception(heard_tokens, body_state=body_state)
        self.hippocampus.record_episode(heard_tokens, body_state, event_time=self._now,
                                        reward_since=self._pleasure_since)
        self._pleasure_since = 0.0
        p_arg = p_loss_heard if isinstance(p_loss_heard, torch.Tensor) \
            else torch.tensor(0.0, device=self.device)
        _, al = self.learner.update(p_arg, a_loss, value_loss)

        return {"r_imit": r_imit, "r_social": r_social, "r_mand": r_mand, "R": R,
                "delta": delta, "a_loss": al, "recognized_word": best_word}

    def consolidate(self):
        """
        睡眠移行時に海馬の経験を大脳皮質（GRU）に定着させる。

        【人間模倣】NREM睡眠中のシャープ波リプル（海馬→皮質の一方向転送）を模倣。
        覚醒中に蓄積した経験を順に再生し、予測モデルを強化する。
        行動学習（政策勾配）はここでは行わない（知覚定着＋B2-11で満腹予期の反芻）。

        B2-11：親との会話（satiety_targetあり）も海馬に記録されるようになった
        ため、ここで一緒に満腹予期（satiety_head）を反芻・強化する。実際に
        親と話す機会は少なくても、眠るたびに同じ記憶を何度も再生することで
        理解の学習量を稼ぐ（自発喃語が無制限に練習できるのと対称にする）。
        """
        experiences = self.hippocampus.replay()
        if not experiences:
            self.hippocampus.clear()
            return {"consolidated": 0, "p_loss": 0.0, "s_loss": 0.0}

        # B3/B4：満腹予期のTD学習。ブートストラップに使う「次イベントの予期V」は、
        # 本パスで出た satiety_logit を .detach() して流用する（consolidate中は重みを
        # 更新しないので、別に no_grad で測り直しても同じ値になる＝順伝播を半分に短縮）。
        total_p_loss = None
        logits = []   # 各経験の satiety_logit（勾配あり。無ければ None）
        Vs = []       # 各経験の予期 V=sigmoid(logit) を detach したもの（ブートストラップ用）
        for full_tokens, body_state, event_time, _reward_since in experiences:
            p_loss, _, satiety_logit = self.learner.learn_perception(full_tokens, body_state=body_state)
            if isinstance(p_loss, torch.Tensor):
                total_p_loss = p_loss if total_p_loss is None else total_p_loss + p_loss
            logits.append(satiety_logit)
            Vs.append(torch.sigmoid(satiety_logit).item() if satiety_logit is not None else 0.0)

        total_s_loss = None
        s_count = 0
        for i, (full_tokens, body_state, event_time, _reward_since) in enumerate(experiences):
            satiety_logit = logits[i]
            # B3/B4：満腹予期を「イベント単位の時間割引TD」で更新（全知ラベルなし）。
            # 目標 = 次イベントまでに得た報酬（空腹の低下＝経験した安腹 ＋ 味の快liking=US）
            #        ＋ γ^Δt × 次イベントの予期。＝図のドーパミンTD（予期が授乳→まんまへ後退）。
            if self.td_satiety and satiety_logit is not None and i + 1 < len(experiences):
                _tok_n, bs_n, t_n, taste_r = experiences[i + 1]
                try:
                    h_now = float(body_state[0]); h_next = float(bs_n[0])   # body_state[0]=空腹
                except Exception:
                    h_now = h_next = 0.0
                relief = max(0.0, h_now - h_next)                     # 空腹の低下＝食べ物が来た（経験）
                reward = relief + float(taste_r)                      # ＋味の快(US。食べた瞬間の強い報酬)
                dt = max(0.0, float(t_n) - float(event_time))
                discount = math.exp(-dt / self._td_tau)               # 時間割引 γ^Δt（SMDP）
                target = min(1.0, reward + discount * Vs[i + 1])
                tgt = torch.tensor(target, device=satiety_logit.device, dtype=satiety_logit.dtype)
                s_loss = F.binary_cross_entropy_with_logits(satiety_logit, tgt)
                total_s_loss = s_loss if total_s_loss is None else total_s_loss + s_loss
                s_count += 1

        combined_s_loss = (total_s_loss / s_count) if total_s_loss is not None else None
        if total_p_loss is not None:
            pl, _ = self.learner.update(total_p_loss / len(experiences), 0.0, satiety_loss=combined_s_loss)
        else:
            pl = 0.0

        s_val = combined_s_loss.item() if isinstance(combined_s_loss, torch.Tensor) else 0.0
        self.hippocampus.clear()
        return {"consolidated": len(experiences), "p_loss": pl, "s_loss": s_val, "s_count": s_count}

    def save(self, tag="manual"):
        return self.archive.save_snapshot(
            self.brain, self.vocab, self.dopamine, self.clock,
            self.cfg, tag=tag,
        )
