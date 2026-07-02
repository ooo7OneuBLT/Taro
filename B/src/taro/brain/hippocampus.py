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

B2-11：これまで record_episode は太郎自身の自発喃語（self_babble）でしか
呼ばれておらず、親との会話（聞いた語→結果）は一度も海馬に記録されて
いなかった。太郎は自発喃語を1年で10万回以上練習できる一方、親との
やり取りは1年で1万回程度しかなく、この頻度差が「理解」の学習を
「産出」より大きく遅らせていた（人間は逆で、聞くだけの経験は運動発達を
待たずに無制限に積めるため理解が産出より先に育つ）。親との会話も
record_episode し、睡眠中に何度も反芻させることで、実際の会話機会の
少なさを「記憶の反芻」で補う（McClelland et al. 1995の海馬リプレイ理論）。
"""


class Hippocampus:
    """
    海馬。覚醒中の経験を蓄積し、睡眠移行時にリプレイするバッファ。

    保存内容: (full_tokens, body_state, satiety_target) のタプル。
    satiety_targetは自発喃語ではNone（教師なし＝知覚学習のみ）、
    親との会話では「この後に授乳が来たか」（B2-11、満腹予期の反芻用）。
    リプレイ: TaroLearner.learn_perception で知覚予測を、
    satiety_targetがあれば満腹予期も強化する。
    """

    def __init__(self, max_capacity=500):
        self.max_capacity = max_capacity
        self.episodes = []

    def record_episode(self, full_tokens, body_state, satiety_target=None):
        """
        経験を1件記録する。容量超過時は最古のものを削除（FIFO）。

        satiety_target: 親との会話の場合のみ指定（この発話の後に授乳が
            来たか、1.0/0.0）。自発喃語ではNoneのまま（教師データが無い）。
        """
        if len(self.episodes) >= self.max_capacity:
            self.episodes.pop(0)
        self.episodes.append((full_tokens, body_state, satiety_target))

    def replay(self):
        """蓄積した全経験をリストで返す。"""
        return list(self.episodes)

    def clear(self):
        """睡眠後に海馬をクリアする（皮質への転送完了）。"""
        self.episodes.clear()

    def __len__(self):
        return len(self.episodes)
