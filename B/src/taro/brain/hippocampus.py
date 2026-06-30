"""
海馬（Hippocampus） — 短期経験バッファ＆睡眠リプレイ

【人間模倣】海馬は覚醒中の経験を一時保存し、睡眠中のシャープ波リプルで
大脳皮質へ転送して長期記憶に定着させる（McClelland et al., 1995 CLS理論）。

B-5：スタブから実装へ。
- 覚醒中の喃語 → record_episode() で蓄積（forward onlyで軽い）
- 睡眠移行時  → core_b.consolidate() がこのバッファを使って皮質を更新
- 睡眠完了後  → clear() でリセット（海馬は短期間しか保持しない）

容量上限（max_capacity）: 人間の海馬のワーキングメモリ容量に相当。
溢れた場合は最古のエピソードを削除（FIFO）。
"""


class Hippocampus:
    """
    海馬。覚醒中の喃語経験を蓄積し、睡眠移行時にリプレイするバッファ。

    保存内容: (full_tokens, body_state) のペア
    リプレイ: TaroLearner.learn_perception で知覚予測を強化
    """

    def __init__(self, max_capacity=500):
        self.max_capacity = max_capacity
        self.episodes = []

    def record_episode(self, full_tokens, body_state):
        """喃語1回分の経験を記録する。容量超過時は最古のものを削除（FIFO）。"""
        if len(self.episodes) >= self.max_capacity:
            self.episodes.pop(0)
        self.episodes.append((full_tokens, body_state))

    def replay(self):
        """蓄積した全経験をリストで返す。"""
        return list(self.episodes)

    def clear(self):
        """睡眠後に海馬をクリアする（皮質への転送完了）。"""
        self.episodes.clear()

    def __len__(self):
        return len(self.episodes)
