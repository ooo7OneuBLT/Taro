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
import torch
import yaml
from collections import deque
from taro.brain import (Vocabulary, TaroBrain, Cerebellum, BrocasArea, TaroLearner,
                        compute_imitation_reward, compute_prediction_reward,
                        Dopamine, Habituation, LocusCoeruleus, compute_total_reward,
                        Homeostasis)
from taro.body import VocalTract, Stomach, Lungs, InternalState, BloodVessel, Adenosine
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
            consumption_rate=float(bv.get("consumption_rate", 0.0001)),
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

        # 脳の部品
        self.habituation = Habituation(history_size=20, decay_rate=0.05)
        self.locus_coeruleus = LocusCoeruleus()
        self.cerebellum = Cerebellum()
        self.brocas_area = BrocasArea()
        self.homeostasis = Homeostasis()

        # 成功判定（N回中M回方式）
        succ = self.cfg.get("success", {})
        self.partial_threshold = succ.get("partial_threshold", 0.8)
        _window = succ.get("partial_window", 10)   # 直近N回を見る
        self.partial_success_target = succ.get("partial_success", 8)  # N回中M回で成功
        self.partial_window = deque(maxlen=_window)
        self.exact_streak = 0

        # 隠れ状態の保持（イベント間で維持）
        self._hidden = None

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

    def check_cry(self):
        """
        泣いているかどうか。internal_stateの泣き管理を使う。

        戻り値: (泣いているか: bool, 泣きの強さ: float)
        """
        return self.internal_state.is_crying(), self.internal_state.cry_intensity

    def feed(self, amount=0.6):
        """授乳を開始する。一瞬ではなく、約30分かけて少しずつ胃に入る。"""
        self.stomach.start_feeding(amount)

    def comfort(self, care_type="comfort"):
        """世話。discomfortやsleepinessを下げる。"""
        self.internal_state.apply_care(care_type)

    def step(self, parent_text, r_social=0.0):
        """
        親が話しかけたときの1ターン。モデルAのstep()と同じ構造だが、
        島皮質経由で体の感覚が脳に入る点と、恒常性の報酬が加わる点が異なる。
        """
        parent_tokens = self.vocab.encode(parent_text)
        self.brain.resize_embedding(self.vocab.size)
        self.brain.to(self.device)

        body_state = self._body_state_tensor()
        prev_arousal = self.internal_state.get_arousal()

        # 知覚学習
        full_tokens = [1] + parent_tokens + [2]
        p_loss, pred_probs = self.learner.learn_perception(full_tokens, body_state=body_state)

        # 聞く（体の感覚も合流）
        listen_input = torch.tensor([full_tokens], device=self.device)
        with torch.no_grad():
            _, h = self.brain.forward_hidden(listen_input, body_state=body_state)
        self._hidden = h

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
        delta = self.dopamine.compute_rpe(R)

        # 学習
        a_loss = self.learner.learn_action(log_probs, delta)
        pl, al = self.learner.update(p_loss, a_loss)

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

    def self_babble(self):
        """
        太郎が一人で喃語を出す。脳の現在の分布からサンプリング。

        【人間模倣】
        0〜6か月の乳児は穏やかな時間に発話計画なしで自発的に声を出す。
        脳が「今出しやすい音」を自由に試す → 自分の声を聞く → 強化される。
        発話計画（ブローカ野）なし = 喃語期のパス（cortex.pyのgenerate）。
        """
        body_state = self._body_state_tensor()

        # 発声（喃語）：speech_plan=None で喃語期のパスを使う
        ne_level = self.locus_coeruleus.get_ne_level()
        generated, log_probs, _ = self.brain.generate(
            hidden=self._hidden,
            max_length=self.max_output_length,
            eos_idx=2,
            stamina=self.lungs.get(),
            vocal_tract=self.vocal_tract,
            ne_level=ne_level,
            cerebellum=self.cerebellum,
            speech_plan=None,
        )

        babble_text = self.vocab.decode(generated)
        self.lungs.consume(len(generated))

        if not generated:
            return {"taro": "", "R": 0.0, "r_pred": 0.0, "r_home": 0.0}

        # 自分の声を聞く → 知覚学習（自己強化）
        full_tokens = [1] + generated + [2]
        p_loss, pred_probs = self.learner.learn_perception(full_tokens, body_state=body_state)

        listen_input = torch.tensor([full_tokens], device=self.device)
        with torch.no_grad():
            _, h = self.brain.forward_hidden(listen_input, body_state=body_state)
        self._hidden = h

        # 報酬（喃語 → 模倣・社会報酬なし。予測と恒常性のみ）
        r_pred = compute_prediction_reward(pred_probs, generated)
        r_habit = self.habituation.compute_penalty(babble_text)

        self.internal_state.update_from_body(self.stomach, self.blood_vessel)
        current_arousal = self.internal_state.get_arousal()
        r_home = self.homeostasis.compute_reward(current_arousal)

        R = compute_total_reward(0.0, r_pred, 0.0, r_habit, self.weights)
        R = max(0.0, R + self.weights.get("w_home", 0.3) * r_home)
        delta = self.dopamine.compute_rpe(R)

        a_loss = self.learner.learn_action(log_probs, delta)
        self.learner.update(p_loss, a_loss)

        self.locus_coeruleus.observe_reward(R)
        self.locus_coeruleus.release_ne()
        self.brain.receive_ne(self.locus_coeruleus.get_ne_level())

        self.clock.tick(tokens_heard=0)

        return {
            "taro": babble_text,
            "r_pred": r_pred,
            "r_home": r_home,
            "R": R,
        }

    def save(self, tag="manual"):
        return self.archive.save_snapshot(
            self.brain, self.vocab, self.dopamine, self.clock,
            self.cfg, tag=tag,
        )
