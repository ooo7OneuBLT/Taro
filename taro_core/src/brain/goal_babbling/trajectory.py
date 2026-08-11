"""区分線形の軌道（GoalTrajectory）— Goal Babbling 段階1。

【人間模倣としての位置づけ】Rolf, Steil & Gienger 2011（Goal Babbling原典）は、
目標へ向かう探索の摂動が「ステップ的に切り替わるのでなく、ゆっくり連続的に
変化しなければならない」と明記している[Tier1]。太郎の既存の逸脱
（`goal_buf` から毎回目標をランダムに選び、`infer_goal_action` で一気にその目標へ
向かう行動を逆算する＝1判断で目標へ瞬間移動するのに等しい）は、この原典の要件に
反している。区分線形の軌道は、現在地から目標まで L ステップかけてゆっくり近づく
ことで、この要件を満たす（`doc/やることリスト.md` Goal Babbling段階1）。

対象空間（何を軌道でつなぐか）は問わない。座標でも、本設計の低次元感覚ベクトル
`g`（22次元、固有感覚＋自己接触）でも、原典の主張（摂動はゆっくり変化すべき）
自体は成立する（設計の判断、2026-08-02）。

【乱数を内部で消費しない】目標の選び方（goal_buf からのサンプル／ホーム回帰の
確率判定／陰性対照の生成）は、すべて呼び出し側（`run/trainer.py`）が既存の
`torch.rand`/`torch.randint` の流儀で行う。ここは「与えられた目標へ、いまの
姿勢から等分割で近づく」という決定的な計算だけを持つ。乱数の消費順序が
明確に1箇所（trainer.py）にまとまることで、再現性の追跡がしやすくなる。
"""


class GoalTrajectory:
    """現在地から目標まで、区分線形（等分割）で近づく中間目標を作る。

    L=25ステップかけて目標に到達する（Rolf 2011の実測パラメータ、Tier1）。
    学習パラメータは持たない（nn.Module ではない）。
    """

    def __init__(self, L=25):
        self.L = int(L)
        self.target = None
        self.remaining = 0

    def active(self):
        """いま軌道の途中か（新しい目標を選ぶ必要が無いか）。"""
        return self.target is not None

    def begin(self, target):
        """新しい目標を設定し、L ステップかけて近づく軌道を開始する。"""
        self.target = target.clone().detach()
        self.remaining = self.L

    def waypoint(self, current):
        """いま向かうべき中間目標を1つ返す（呼ぶたびに残りステップ数を1つ消費する）。

        残りステップ数で目標との差を割った分だけ進む。探索へ何度切り替わっても
        （goal_switch が頻繁にON/OFFする既存の仕組みでも）、再開のたびに
        「今の姿勢」から再計算されるので、途中経過が食い違わない。
        """
        step = (self.target - current) / max(self.remaining, 1)
        self.remaining = max(self.remaining - 1, 0)
        return current + step

    def done(self):
        """軌道を使い切ったか（次に active() を呼ぶと新しい目標が要る）。"""
        return self.remaining <= 0

    def reset(self):
        """軌道を打ち切る（探索へ切り替わったときに呼ぶ）。"""
        self.target = None
        self.remaining = 0
