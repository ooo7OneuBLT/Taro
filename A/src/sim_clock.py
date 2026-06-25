"""
シミュレーション内時間（イベント駆動）

【人間模倣】太郎の発達年齢をsim時間で管理する。
1回のやり取りごとに「人間なら何秒かかるか」を加算する。
計算速度に依存せず再現性が高い。
"""


class SimClock:

    def __init__(self, seconds_per_turn=5.0):
        self.seconds_per_turn = seconds_per_turn
        self.total_seconds = 0.0
        self.total_turns = 0
        self.total_tokens_heard = 0

    def tick(self, tokens_heard=0):
        """1ターン経過。"""
        self.total_seconds += self.seconds_per_turn
        self.total_turns += 1
        self.total_tokens_heard += tokens_heard

    def age_str(self):
        """発達年齢を人間が読みやすい形式で返す。"""
        s = self.total_seconds
        if s < 60:
            return f"{s:.0f}秒"
        elif s < 3600:
            return f"{s / 60:.1f}分"
        elif s < 86400:
            return f"{s / 3600:.1f}時間"
        else:
            return f"{s / 86400:.1f}日"

    def state_dict(self):
        return {
            "seconds_per_turn": self.seconds_per_turn,
            "total_seconds": self.total_seconds,
            "total_turns": self.total_turns,
            "total_tokens_heard": self.total_tokens_heard,
        }

    def load_state_dict(self, d):
        self.seconds_per_turn = d["seconds_per_turn"]
        self.total_seconds = d["total_seconds"]
        self.total_turns = d["total_turns"]
        self.total_tokens_heard = d["total_tokens_heard"]
