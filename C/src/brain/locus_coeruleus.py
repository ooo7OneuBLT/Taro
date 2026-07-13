# --- コピー元: Taro (github.com/ooo7OneuBLT/Taro) commit 3b976fc ---
# --- 元パス: B/src/taro/brain/instincts/locus_coeruleus.py （unificationMIMoでは無編集） ---

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
                 min_ne=0.1, max_ne=1.0, mature_max_ne=0.3):
        """
        reward_window: 最近何ターンの報酬を見るか
        base_ne: NEの基準レベル
        ne_increase_rate: 報酬ゼロ時のNE上昇速度
        ne_decrease_rate: 報酬あり時のNE低下速度
        mature_max_ne: 完全成熟時のNE上限（探索の天井）
        """
        self.reward_history = []
        self.reward_window = reward_window
        self.ne_level = base_ne
        self.base_ne = base_ne
        self.ne_increase_rate = ne_increase_rate
        self.ne_decrease_rate = ne_decrease_rate
        self.min_ne = min_ne
        self.max_ne = max_ne

        # 【着想は人間模倣・実装は⚠️逸脱】成熟による探索の結晶化（B2-8）。
        # 着想（人間模倣）：若い個体はゆらぎ（探索）が大きく、育つにつれてゆらぎが
        # 減り「うまくいったやり方」に固まる（結晶化）。人間ではシナプス刈り込み・
        # 臨界期の閉じ・前頭前野の成熟が、鳥ではLMANの運動系への影響低下が担う。
        # 実装（⚠️逸脱）：太郎では月齢に対して直線的にNE上限を max_ne →
        # mature_max_ne へ下げる（_ceiling参照）。これは活動履歴や刈り込みから
        # 内生的に下がるメカニズムではなく、カレンダー時刻に引いた決め打ちの
        # 線形カーブであり、プラン§4「数式でカーブを決め打ちしない」から外れる。
        # よって「探索→活用の縮小」という現象は人間模倣だが、その出し方は逸脱。
        # また複数の成熟機構をNEという1本の本能に集約した近似で、NE単独が
        # 人間の結晶化を駆動すると主張するものではない。
        self.mature_max_ne = mature_max_ne
        self.maturation = 0.0  # 0=新生児（探索大）, 1=完全成熟（結晶化）

    def mature(self, progress):
        """
        発達の進行度（0〜1）を受け取り、探索の天井を下げる。

        progress = 経過時間 / 完全成熟までの時間（声道stage3と同じ月齢基準）。
        """
        self.maturation = max(0.0, min(1.0, progress))

    def _ceiling(self):
        """現在の発達段階での探索の天井（NE上限）。成熟が進むほど低い。"""
        return self.max_ne - self.maturation * (self.max_ne - self.mature_max_ne)

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

        # 探索の天井は成熟で下がる（B2-8）。報酬ベースの調整（既存）と
        # 成熟ベースの上限（新規）の両方が同時に働く＝人間の2機構を再現。
        ceiling = self._ceiling()

        if recent_avg < 0.1:
            # 報酬がほぼゼロ → NEを増やす（探索モードへ）。ただし成熟した
            # 天井は超えられない（育つと際限なくは探索しない）。
            self.ne_level = min(ceiling,
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

        # 成熟した天井を超えていたら引き下げる（発達が進むと過去の高い
        # 探索水準はもう維持できない）。
        if self.ne_level > ceiling:
            self.ne_level = ceiling

        return self.ne_level

    def get_ne_level(self):
        return self.ne_level
