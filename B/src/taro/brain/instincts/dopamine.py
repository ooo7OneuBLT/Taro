"""
ドーパミン（報酬予測誤差） — 学習を駆動するごほうび信号

【人間模倣＝既存AI研究】
ドーパミンニューロンは「もらえた報酬 − 予想した報酬」を発火する
（Schultz, 1997）。強化学習のTD誤差と数式上一致。
"""


class Dopamine:

    def __init__(self, momentum=0.95):
        self.momentum = momentum
        self.baseline = 0.0

    def compute_rpe(self, reward):
        """報酬予測誤差δを計算し、baselineを更新する。"""
        delta = reward - self.baseline
        self.baseline = self.momentum * self.baseline + (1 - self.momentum) * reward
        return delta

    def get_baseline(self):
        return self.baseline
