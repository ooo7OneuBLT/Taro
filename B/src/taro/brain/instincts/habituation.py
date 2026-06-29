"""
馴化（飽き） — 同じ出力を繰り返すと報酬が減衰する

【人間模倣】胎児・新生児にも確認される最も基本的な神経メカニズム。
ドーパミン系とは独立に存在する。
同じ音を出し続けると「つまらなくなる」。新しい音を試すと「面白い」。
"""


class Habituation:

    def __init__(self, history_size=20, decay_rate=0.05):
        self.history = []
        self.history_size = history_size
        self.decay_rate = decay_rate

    def compute_penalty(self, output_text):
        """
        同じ出力の繰り返しにペナルティを与える。
        フレーズ全体＋文字レベルの単調さの両方を考慮。

        戻り値: 0.0（新しい出力）〜 -1.0（ずっと同じ出力）
        """
        if not self.history or not output_text:
            self._add(output_text)
            return 0.0

        phrase_repeats = sum(1 for h in self.history if h == output_text)
        phrase_penalty = -self.decay_rate * phrase_repeats

        if len(output_text) > 1:
            unique_ratio = len(set(output_text)) / len(output_text)
            monotony_penalty = -self.decay_rate * (1.0 - unique_ratio) * 3
        else:
            monotony_penalty = 0.0

        self._add(output_text)
        return max(-1.0, phrase_penalty + monotony_penalty)

    def _add(self, text):
        self.history.append(text)
        if len(self.history) > self.history_size:
            self.history.pop(0)
