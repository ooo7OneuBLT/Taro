"""
快の生成（Hedonic / liking） — 味を生得的な快に変換する本能（脳）

【人間模倣】甘み→快、苦味/酸味→不快 は生得の反射（gustofacial reflex。脳幹レベルで、
無脳症児でも出る＝学習でない本能）。快そのもの(liking)は側坐核などのオピオイド系が担い、
ドーパミン（wanting＝意欲・報酬予測）とは**別の信号**（Berridge の liking/wanting 区別）。

役割：舌が検知した味シグナル → 即時の快(pleasure)。これは「食べた瞬間の強い報酬(US)」として
満腹予期のTD学習に使う（＝合図"まんま"へ後退させる燃料）。快は経験の答えを書くのではなく、
生得の報酬信号を出すだけ＝理解は依然TDで創発する。

⚠️ 快の強さ pleasure_gain は未検証の定数（人間模倣ではない。後で感度確認して調節）。
"""


class Hedonic:
    """味シグナル → 即時の快(liking)。ドーパミン(wanting)とは別の信号として持つ。"""

    def __init__(self, pleasure_gain=1.0):
        self.pleasure_gain = pleasure_gain   # ⚠️未検証：甘み→快の強さ
        self.last_pleasure = 0.0

    def evaluate(self, sweetness):
        """甘み → 生得的な快（甘いほど快、0〜1）。苦味/酸味の負の快は将来。"""
        self.last_pleasure = max(0.0, min(1.0, sweetness * self.pleasure_gain))
        return self.last_pleasure

    def get(self):
        return self.last_pleasure
