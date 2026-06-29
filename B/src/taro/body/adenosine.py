"""
アデノシン — 眠さの原因物質

【人間模倣】
脳が活動するとATPが消費され、副産物としてアデノシンが産生・蓄積される。
アデノシンが蓄積すると眠くなる。睡眠中に除去され、すっきり目覚める。
カフェインはアデノシン受容体をブロックして眠気を抑える（Borbély, 1982）。

他の臓器から信号を受け取れる設計（ノードとして機能）：
  現在：  覚醒中に arousal（脳活動）→ アデノシン産生
          睡眠中に              → アデノシン除去
  将来：  食後の副交感神経活性化 → アデノシン産生増加（食後の眠気）
          身体活動              → アデノシン産生増加（疲労性の眠気）
"""


class Adenosine:
    """アデノシン濃度の管理。sleepiness のソース。"""

    def __init__(self, production_rate=0.0001, clearance_rate=0.0003):
        self.level = 0.0
        self.production_rate = production_rate  # 覚醒中の基礎産生量（/秒）
        self.clearance_rate = clearance_rate    # 睡眠中の除去率（/秒）

    def tick_awake(self, arousal=0.0):
        """
        覚醒中の1秒分。脳活動（arousal）に比例して蓄積する。
        arousal が高いほど速く蓄積（忙しい脳ほど疲れる）。
        """
        self.level = min(1.0, self.level + self.production_rate * (1.0 + arousal))

    def tick_sleep(self):
        """
        睡眠中の1秒分。残量に比例して除去される（多いほど速く除去）。
        """
        self.level = max(0.0, self.level - self.clearance_rate * self.level)

    def get(self):
        """現在のアデノシン濃度（0〜1）。そのまま sleepiness として使う。"""
        return self.level
