"""
環境（親）⇄ 太郎の汎用ループ

【既存AI研究】脳とI/Oを分離し「観測→行動→報酬」の
標準ループにする。CLI/GUI/親シミュレータを差し替え可能。
"""

import os
import torch
import yaml
from brain import Vocabulary, TaroBrain
from vocal_tract import VocalTract
from instincts import (compute_imitation_reward, compute_prediction_reward,
                        Dopamine, Habituation, compute_total_reward)
from learning import TaroLearner
from sim_clock import SimClock
from archive import Archive
from logger import Logger


def _project_root():
    """A/ ディレクトリのパスを返す。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TaroEnvironment:
    """太郎の全コンポーネントを束ねて1ターンのループを提供する。"""

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

        # A2追加：発声体力
        sc = self.cfg.get("stamina", {})
        self.stamina = float(sc.get("initial", 3))
        self.stamina_growth_rate = float(sc.get("growth_rate", 0.005))
        self.max_stamina = float(sc.get("max_stamina", 30))

        # A2-2追加：声道シミュレータ（太郎の口）
        self.vocal_tract = VocalTract()

        # 声道の全文字を語彙に事前登録
        for ch in self.vocal_tract.get_all_chars():
            self.vocab.encode(ch)
        self.brain.resize_embedding(self.vocab.size)
        self.brain.to(self.device)
        self.brain.set_vocab_mapping(self.vocab.char2idx)

        # A2-4追加：馴化（飽き）の本能
        self.habituation = Habituation(history_size=20, decay_rate=0.05)

        # A2追加：τの適応的減衰用の累積模倣報酬
        self.cumulative_r_imit = 0.0

        # A2変更：成功条件（部分一致＋完全一致の2段階）
        succ = self.cfg.get("success", {})
        self.partial_threshold = succ.get("partial_threshold", 0.8)
        self.partial_streak_target = succ.get("partial_streak", 10)
        self.exact_streak_target = succ.get("exact_streak", 10)
        self.partial_streak = 0
        self.exact_streak = 0

    def step(self, parent_text, r_social=0.0):
        """
        1ターンのループ。

        parent_text: 親の発話（文字列）
        r_social: 社会的報酬＝親の笑顔の強さ [0, 1]

        戻り値: dict（ターンの全情報）
        """
        parent_tokens = self.vocab.encode(parent_text)
        self.brain.resize_embedding(self.vocab.size)
        self.brain.to(self.device)

        full_tokens = [1] + parent_tokens + [2]
        p_loss, pred_probs = self.learner.learn_perception(full_tokens)

        # 親の発話を聞き、その直後の隠れ状態を引き継いで発話する
        # 【人間模倣】赤ちゃんは聞いた音の短期記憶を使って直後に真似る
        listen_input = torch.tensor([full_tokens], device=self.device)
        with torch.no_grad():
            _, h = self.brain.forward_hidden(listen_input)
        generated, log_probs, _ = self.brain.generate(
            hidden=h,
            max_length=self.max_output_length,
            eos_idx=2,
            stamina=self.stamina,
            vocal_tract=self.vocal_tract,
        )
        taro_text = self.vocab.decode(generated)

        r_imit = compute_imitation_reward(parent_tokens, generated,
                                          vocab=self.vocab, vocal_tract=self.vocal_tract)
        r_pred = compute_prediction_reward(pred_probs, parent_tokens)
        r_habit = self.habituation.compute_penalty(taro_text)
        R = compute_total_reward(r_imit, r_pred, r_social, r_habit, self.weights)
        delta = self.dopamine.compute_rpe(R)

        a_loss = self.learner.learn_action(log_probs, delta)
        pl, al = self.learner.update(p_loss, a_loss)

        # A2-3：声道の成熟ステージをsim時間で更新【人間模倣・身体的制約】
        vm = self.cfg.get("vocal_maturation", {})
        self.vocal_tract.update_stage(
            self.clock.total_seconds,
            vm.get("stage1_time", 300),
            vm.get("stage2_time", 900),
            vm.get("stage3_time", 1500),
        )

        # A2変更：τを成功体験に連動させる【人間模倣】
        self.cumulative_r_imit += r_imit
        bc = self.cfg["brain"]
        self.brain.update_temperature(
            self.cumulative_r_imit,
            bc.get("temperature_alpha", 0.02),
            bc["initial_temperature"],
            bc["min_temperature"],
        )

        # A2追加：発声体力の成長
        self.stamina = min(self.stamina + self.stamina_growth_rate, self.max_stamina)

        self.clock.tick(tokens_heard=len(parent_tokens))

        # A2変更：成功条件（部分一致＋完全一致の2段階）
        exact_match = taro_text == parent_text
        partial_match = r_imit >= self.partial_threshold

        if partial_match:
            self.partial_streak += 1
        else:
            self.partial_streak = 0

        if exact_match:
            self.exact_streak += 1
        else:
            self.exact_streak = 0

        turn = self.clock.total_turns
        self.logger.log_turn(
            turn, self.clock.total_seconds, parent_text, taro_text,
            r_imit, r_pred, r_social, R, delta, pl, al,
            self.brain.temperature,
        )

        if turn % self.cfg["archive"]["snapshot_interval_turns"] == 0:
            self.archive.save_snapshot(
                self.brain, self.vocab, self.dopamine, self.clock,
                self.cfg, tag=f"auto_t{turn}",
            )

        if turn % self.cfg["logging"]["plot_interval_turns"] == 0:
            self.logger.plot_learning_curve()

        return {
            "turn": turn,
            "age": self.clock.age_str(),
            "parent": parent_text,
            "taro": taro_text,
            "r_imit": r_imit,
            "r_pred": r_pred,
            "r_social": r_social,
            "R": R,
            "delta": delta,
            "p_loss": pl,
            "a_loss": al,
            "temperature": self.brain.temperature,
            "stamina": self.stamina,
            "partial_streak": self.partial_streak,
            "exact_streak": self.exact_streak,
            "exact_match": exact_match,
            "partial_match": partial_match,
        }

    def save(self, tag="manual"):
        return self.archive.save_snapshot(
            self.brain, self.vocab, self.dopamine, self.clock,
            self.cfg, tag=tag,
        )

    def load(self, path):
        return self.archive.load_snapshot(
            path, self.brain, self.vocab, self.dopamine, self.clock,
        )
