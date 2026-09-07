"""言語海馬（LanguageHippocampus） — 聞いた発話の鎖状連想メモリ＋睡眠リプレイ。

設計：F/docs/聞く学習の安定化/設計_言語海馬と睡眠リプレイ.md（段階1）。

【人間模倣】人間は聞いた瞬間、大脳皮質（脳の本体）を大きく書き換えるのではなく、
海馬が1回で素早く「仮の対応」を焼き付け、睡眠中にその経験を反芻することで
皮質がゆっくり育つ（相補学習系：McClelland et al. 1995・二次確認）。
乳児の実証：昼寝をした子だけが新語を般化できた（9〜16ヶ月児90名・一次確認）。

【逐語再生＝推定】「聞いた発話をそのまま睡眠中に再生している」ことを直接見た
研究は無い。睡眠中に特定記憶を狙って再活性化できる（TMR：Rasch 2007・Rudoy 2009、
メタ分析g=0.29でノンレム睡眠中のみ有効・二次確認）という間接証拠からの推定で
採用している。反証（思春期で行動成績に効果なし：Wilhelm 2020）もあり、
乳児対象のTMR研究自体は存在しない、という限界も残る。

【Tier3・工学較正】capacity（容量）・decay（減衰率）・replay_passes（睡眠1回の
再生周回数）・recent_ratio（直近偏重サンプルの比率）は、文献に数値根拠が無い
（皮質と海馬の学習速度比は原典にも記載なし＝調査で確認済み）ため、動作確認済みの
初期値をこちらで較正している。

保存方式：鎖状連想（生物実測：列を順番に圧縮再生する）＋パターン分離（1経験1
スロット。既存の分散重み＝lexicon/脳本体には混ぜない＝干渉しない）。
読み出し（recall）は診断専用（このファイル単体では行動に接続しない。段階2の課題）。
"""

import numpy as np
import torch


class LanguageHippocampus:
    """聞いた発話（視覚＋話者＋音列）をエピソード単位で焼き付け、睡眠でリプレイする。

    エピソード = {"key_vis": ndarray(64), "speaker": int, "tokens": list[int],
                  "strength": float, "written_at": int}
    """

    def __init__(self, cfg=None):
        cfg = cfg or {}
        # 【Tier3・初期値】設計「発注用の場所表」下の既定値をそのまま採用。
        self.capacity = int(cfg.get("capacity", 512))
        self.decay_rate = float(cfg.get("decay", 0.85))
        self.replay_passes = int(cfg.get("replay_passes", 3))
        self.recent_ratio = float(cfg.get("recent_ratio", 0.7))
        self.sleep_lr = float(cfg.get("sleep_lr", 0.005))
        self.episodes = []
        # 【塊レベル層・2026-09-07】仕様_M5_塊レベル層.md §2。tokensが「モーラID列」
        #   か「塊ID列」かの印。既定"mora"＝従来どおり。produce.chunk_levelが真の
        #   ときだけ run/taro_setup.py が"chunk"に切り替える。
        self.unit = str(cfg.get("unit", "mora"))

    # ------------------------------------------------------------ 書き込み
    def write(self, key_vis, speaker, tokens, step, strength=1.0):
        """1エピソードを焼き付ける（1回で覚える＝海馬の性質）。

        容量超過時はstrength最小のエピソードを捨てる（古さではなく弱さで捨てる。
        減衰で古いものは自然に弱くなるため、結果として古いものが捨てられやすい）。

        strength: 【M4e・2026-09-07・仕様_M4e_状態の線と驚きの書き込み】既定1.0。
          「消えたの印」が立っている間に聞いた発話は、呼び出し側が
          produce.hippocampus.gone_strength（既定1.0）を渡すことで大きくできる。
          人間側：予測が外れた出来事は強く符号化される（驚きの書き込み）。
        """
        ep = {
            "key_vis": np.asarray(key_vis, dtype=np.float32).copy(),
            "speaker": int(speaker),
            "tokens": [int(t) for t in tokens],
            "strength": float(strength),
            "written_at": int(step),
        }
        if len(self.episodes) >= self.capacity:
            weakest = min(range(len(self.episodes)),
                          key=lambda i: self.episodes[i]["strength"])
            self.episodes.pop(weakest)
        self.episodes.append(ep)

    # ------------------------------------------------------------ 減衰
    def decay(self):
        """睡眠1回ごとに全エピソードのstrengthを減衰させ、薄れきったものを消す。"""
        for ep in self.episodes:
            ep["strength"] *= self.decay_rate
        self.episodes = [ep for ep in self.episodes if ep["strength"] >= 0.05]

    # ------------------------------------------------------------ サンプル
    def sample(self, rng, n):
        """strengthを重みに、直近偏重でn個のエピソードを抽出する（重複あり）。

        recent_ratio：written_atが新しい上位半分から取る確率。乱数は torch の
        Generator のみ使用する（env.np_random は触らない＝決定性を壊さないため）。
        """
        m = len(self.episodes)
        if m == 0 or n <= 0:
            return []
        order = sorted(range(m), key=lambda i: self.episodes[i]["written_at"],
                        reverse=True)
        half = max(1, (m + 1) // 2)
        recent_idx = order[:half]
        old_idx = order[half:]

        def _weighted_pick(idx_pool):
            strengths = torch.tensor(
                [max(self.episodes[i]["strength"], 1e-6) for i in idx_pool],
                dtype=torch.float32)
            pick = torch.multinomial(strengths, 1, replacement=True,
                                      generator=rng).item()
            return idx_pool[pick]

        picked = []
        for _ in range(n):
            use_recent = old_idx == [] or (
                recent_idx != [] and
                torch.rand((), generator=rng).item() < self.recent_ratio)
            pool = recent_idx if use_recent else old_idx
            picked.append(self.episodes[_weighted_pick(pool)])
        return picked

    # ------------------------------------------------------------ 診断専用
    def recall(self, key_vis):
        """key_visに最も近い（コサイン類似度）エピソードのtokensを返す（測定専用）。

        段階1では行動には未接続。空なら空リストを返す。
        """
        if not self.episodes:
            return []
        q = np.asarray(key_vis, dtype=np.float32)
        qn = np.linalg.norm(q) + 1e-8
        best_i, best_sim = 0, -1e9
        for i, ep in enumerate(self.episodes):
            k = ep["key_vis"]
            kn = np.linalg.norm(k) + 1e-8
            sim = float(np.dot(q, k) / (qn * kn))
            if sim > best_sim:
                best_sim, best_i = sim, i
        return list(self.episodes[best_i]["tokens"])

    def recall_with_conf(self, key_vis):
        """【段階2・即答】key_visに最も近いエピソードの (tokens, 似ている度, 記憶の強さ) を返す。

        似ている度＝コサイン類似度（当たるとき≈0.92・外れるとき≈0.78の実測）、記憶の強さ＝
        睡眠ごとに減衰する strength。行動側はこの2つの積を海馬の自信として本体（GRU）と競わせる。
        空なら ([], 0.0, 0.0)。
        """
        if not self.episodes:
            return [], 0.0, 0.0
        q = np.asarray(key_vis, dtype=np.float32)
        qn = np.linalg.norm(q) + 1e-8
        best_i, best_sim = 0, -1e9
        for i, ep in enumerate(self.episodes):
            k = ep["key_vis"]
            sim = float(np.dot(q, k) / (qn * (np.linalg.norm(k) + 1e-8)))
            if sim > best_sim:
                best_sim, best_i = sim, i
        ep = self.episodes[best_i]
        return list(ep["tokens"]), best_sim, float(ep["strength"])

    # ------------------------------------------------------------ 保存復元
    def state_dict(self):
        """保存用（torch.save可能な素のdict/listのみ。ndarrayはリスト化）。"""
        return {
            "capacity": self.capacity,
            "decay": self.decay_rate,
            "replay_passes": self.replay_passes,
            "recent_ratio": self.recent_ratio,
            "sleep_lr": self.sleep_lr,
            "unit": self.unit,
            "episodes": [
                {
                    "key_vis": ep["key_vis"].tolist(),
                    "speaker": ep["speaker"],
                    "tokens": list(ep["tokens"]),
                    "strength": ep["strength"],
                    "written_at": ep["written_at"],
                }
                for ep in self.episodes
            ],
        }

    def load_state_dict(self, d):
        self.capacity = int(d.get("capacity", self.capacity))
        self.decay_rate = float(d.get("decay", self.decay_rate))
        self.replay_passes = int(d.get("replay_passes", self.replay_passes))
        self.recent_ratio = float(d.get("recent_ratio", self.recent_ratio))
        self.sleep_lr = float(d.get("sleep_lr", self.sleep_lr))
        self.unit = str(d.get("unit", "mora"))
        self.episodes = [
            {
                "key_vis": np.asarray(ep["key_vis"], dtype=np.float32),
                "speaker": int(ep["speaker"]),
                "tokens": [int(t) for t in ep["tokens"]],
                "strength": float(ep["strength"]),
                "written_at": int(ep["written_at"]),
            }
            for ep in d.get("episodes", [])
        ]

    def __len__(self):
        return len(self.episodes)
