"""
血管（血中栄養素） — 胃と空腹感の間にあるバッファ

【人間模倣】
新生児の空腹感の主なシグナルは胃の空き具合ではなく、血糖値の低下。
胃が空でも血中に糖が残っていれば空腹感は低い。

B-3：「胃空 = 空腹MAX」の問題を解消するために追加。

  胃の消化 → 吸収量 → 血糖値（上がる）
  体の細胞が消費 → 血糖値（ゆっくり下がる）
  血糖値が低い → hunger（空腹感）

血管の詳細シミュレーション（肝臓・インスリン等）は将来。今は血糖値1つだけ。
"""


class BloodVessel:
    """
    血中栄養素のシンプルなバッファ。

    入力：胃の消化吸収量（tickごと）
    内部：blood_glucose が細胞消費でゆっくり減る
    出力：hunger（空腹度 0〜1）
    """

    def __init__(self, initial_glucose=0.7, consumption_rate=0.00005):
        """
        initial_glucose:   生まれた時の血糖値（0〜1）
        consumption_rate:  1秒あたりの体の細胞による消費量
                           0.00005 → 満腹から空腹まで約2.5〜3時間
        """
        self.blood_glucose = initial_glucose
        self.consumption_rate = consumption_rate

    def receive_glucose(self, absorption_amount):
        """
        胃の消化吸収量を受け取る。血糖値が上がる。
        absorption_amount: 胃から1tickで吸収された量
        """
        self.blood_glucose = min(1.0, self.blood_glucose + absorption_amount)

    def tick(self):
        """1秒分の消費。体の細胞がゆっくり血糖を使う。"""
        self.blood_glucose = max(0.0, self.blood_glucose - self.consumption_rate)

    def get_hunger(self):
        """空腹度（0〜1）。血糖値が低いほど1に近づく。"""
        return 1.0 - self.blood_glucose

    def get_blood_glucose(self):
        """血糖値（0〜1）。"""
        return self.blood_glucose
