"""
海馬（Hippocampus） — エピソード記憶（将来用）

【人間模倣】海馬は「さっき何があったか」を一時的に記憶し、
睡眠中に大脳皮質へリプレイして長期記憶に定着させる（CLS理論）。

A2-10：空の器として追加。機能の実装は将来のバージョンで行う。
"""


class Hippocampus:
    """海馬。現在は未実装。将来の睡眠リプレイ・エピソード記憶に使用。"""

    def __init__(self):
        self.episodes = []

    def record_episode(self, episode):
        """将来用：体験を記録する。"""
        pass

    def replay(self):
        """将来用：記憶をリプレイする（睡眠定着）。"""
        pass
