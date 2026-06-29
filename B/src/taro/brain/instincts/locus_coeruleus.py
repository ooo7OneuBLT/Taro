"""
青斑核（Locus Coeruleus） — 脳全体の探索レベルを調整する中枢

【人間模倣】
青斑核は脳幹にある小さな核で、ノルエピネフリン（NE）を脳全体に放出する。
ドーパミン（報酬予測誤差）とは別の物質・別の回路。

役割：
- 報酬がもらえていない → NE放出を増やす → 脳がもっとバラバラに試す（探索モード）
- 報酬がもらえている → NE放出を減らす → 脳が今のやり方を続ける（搾取モード）

将来追加可能な機能：
- 覚醒レベルの調整（睡眠-覚醒サイクル）
- 注意の制御（集中↔広い注意）
- 記憶の強化（重要な出来事の定着）
"""


class LocusCoeruleus:
    """
    青斑核。最近の報酬を監視してノルエピネフリン（NE）量を決定する。

    NEは脳に直接作用するのではなく「放出」されるだけ。
    脳の各部位がアドレナリン受容体でNEを検出して反応する。
    """

    def __init__(self, reward_window=20, base_ne=0.5,
                 ne_increase_rate=0.05, ne_decrease_rate=0.03,
                 min_ne=0.1, max_ne=1.0):
        """
        reward_window: 最近何ターンの報酬を見るか
        base_ne: NEの基準レベル
        ne_increase_rate: 報酬ゼロ時のNE上昇速度
        ne_decrease_rate: 報酬あり時のNE低下速度
        """
        self.reward_history = []
        self.reward_window = reward_window
        self.ne_level = base_ne
        self.base_ne = base_ne
        self.ne_increase_rate = ne_increase_rate
        self.ne_decrease_rate = ne_decrease_rate
        self.min_ne = min_ne
        self.max_ne = max_ne

    def observe_reward(self, reward):
        """毎ターンの報酬を観測する。"""
        self.reward_history.append(reward)
        if len(self.reward_history) > self.reward_window:
            self.reward_history.pop(0)

    def release_ne(self):
        """
        ノルエピネフリンを放出する。

        最近の報酬の平均が低い → NEを増やす（もっと探索しろ）
        最近の報酬の平均が高い → NEを減らす（今のやり方を続けろ）

        戻り値: ne_level（0〜1の連続値。脳の受容体がこれを検出する）
        """
        if not self.reward_history:
            return self.ne_level

        recent_avg = sum(self.reward_history) / len(self.reward_history)

        if recent_avg < 0.1:
            # 報酬がほぼゼロ → NEを増やす（探索モードへ）
            self.ne_level = min(self.max_ne,
                                self.ne_level + self.ne_increase_rate)
        elif recent_avg > 0.3:
            # 報酬が十分ある → NEを減らす（搾取モードへ）
            self.ne_level = max(self.min_ne,
                                self.ne_level - self.ne_decrease_rate)
        else:
            # 中間 → 基準レベルに戻る
            if self.ne_level > self.base_ne:
                self.ne_level -= self.ne_decrease_rate * 0.5
            elif self.ne_level < self.base_ne:
                self.ne_level += self.ne_increase_rate * 0.5

        return self.ne_level

    def get_ne_level(self):
        return self.ne_level
