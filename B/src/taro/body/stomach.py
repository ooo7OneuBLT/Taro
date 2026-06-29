"""
胃 — 食べ物を受け取り、消化で空になる容器

【人間模倣・身体シミュレーション】
声道が「4パラメータ→文字」だったように、
胃は「容器＋消化→空腹度」をメカニズムで再現する。

カーブは数式で決め打ちしない。
「胃の中身が消化で減る」仕組みから、空腹カーブが結果として出る。

新生児の胃：容量が小さく（~20-60mL）、消化が速い → 2-3時間おきに空腹になる。
成長で容量が増え、消化時間も延びる → 食事間隔が開く。

B-2：授乳は一瞬ではなく、30分（1800秒）かけて少しずつ胃に入る。
"""


class Stomach:
    """
    太郎の胃。容器として食べ物を受け取り、消化で中身が減る。

    入力：食事イベント（量）→ 授乳が始まる（一瞬ではなく期間がある）
    内部：contents が消化で時間とともに減る。授乳中は少しずつ増える
    出力：hunger（空腹度 0〜1）
    """

    def __init__(self, capacity=1.0, digestion_rate=0.0003,
                 initial_contents=0.7, growth_rate=0.0001,
                 feeding_duration=1800):
        """
        capacity: 胃の容量（成長で増える）
        digestion_rate: 1秒あたりの消化率（中身に比例して減る）
        initial_contents: 生まれた時の胃の中身
        growth_rate: 容量の成長速度（身体の成熟）
        feeding_duration: 1回の授乳にかかる秒数（新生児は約20-45分）
        """
        self.capacity = capacity
        self.contents = initial_contents
        self.digestion_rate = digestion_rate
        self.growth_rate = growth_rate
        self.feeding_duration = feeding_duration
        self._last_absorption = 0.0

        # 授乳の状態管理
        self.feeding = False          # 授乳中かどうか
        self._feed_remaining = 0      # 残りの授乳時間（秒）
        self._feed_per_tick = 0.0     # 1秒あたりの流入量

    def tick(self):
        """
        1秒分の消化を進める。
        授乳中なら、消化と同時に少しずつ胃に入ってくる。
        """
        # 授乳中：少しずつ胃に入る
        if self.feeding:
            self.contents = min(self.capacity, self.contents + self._feed_per_tick)
            self._feed_remaining -= 1
            if self._feed_remaining <= 0:
                self.feeding = False

        # 消化：中身の量に比例して減る
        if self.contents <= 0:
            self.contents = 0.0
            self._last_absorption = 0.0
            return

        digested = self.contents * self.digestion_rate
        self.contents = max(0.0, self.contents - digested)
        self._last_absorption = digested

    def start_feeding(self, amount=0.6):
        """
        授乳を開始する。一瞬ではなく、feeding_duration秒かけて少しずつ胃に入る。

        amount: 1回の授乳で最終的に胃に入る総量（0〜1）
        """
        self.feeding = True
        self._feed_remaining = self.feeding_duration
        self._feed_per_tick = amount / self.feeding_duration

    def stop_feeding(self):
        """授乳を中断する（赤ちゃんが拒否した等）。"""
        self.feeding = False
        self._feed_remaining = 0
        self._feed_per_tick = 0.0

    def is_feeding(self):
        """今、授乳中かどうか。"""
        return self.feeding

    def get_hunger(self):
        """空腹度（0〜1）。胃が空くほど1に近づく。"""
        if self.capacity <= 0:
            return 1.0
        return 1.0 - (self.contents / self.capacity)

    def get_last_absorption(self):
        """直前の1秒で吸収された量。胃→眠さの接続に使う。"""
        return self._last_absorption

    def grow(self):
        """身体の成熟で胃の容量が増える。"""
        self.capacity += self.growth_rate
