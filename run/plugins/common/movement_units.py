"""動きの区切り（movement units）を数える ── 手の動きが滑らかかどうかを測る。

【なぜ要るか、2026-07-31】太郎の速度を人間の乳児と比べたところ、
**平均は人間並みなのに、瞬間の最大だけが2倍以上高い**ことが分かった。

```
                     平均速度      最大速度
Thelen et al. 1993   23.6 cm/s    47.3 cm/s
Berthier & Keen 2006 24.0         49.0
太郎（触覚あり）      23.5        118.7    ← 最大だけ突出
```

つまり「ふだんは人間並みだが、時々とんでもなく速く振れる」＝滑らかでない。
速度の平均や最大では、この「ぎくしゃく」を数字にできない。

【何を測るか】
乳児のリーチは**1回の滑らかな動きではない**。加速と減速を何回も繰り返しながら
目標へ近づく。その1回分の山を **movement unit（動きの区切り）** と呼ぶ。

    von Hofsten (1979) が提唱。手先速度の1回の加速・減速のまとまり。
    速度プロファイルの**谷（極小）で区切る**。

発達すると回数が減る＝1回で届くようになる。**リーチングができたかどうかの指標**になる。

```
文献値（1リーチあたりの個数）
  von Hofsten 1991      4.02 → 2.44   （19〜31週）
  Konczak et al. 1995   5.2  → 1.5    （18〜156週）
  Berthier & Keen 2006  2.5（一定）    （14〜93週）
  Matthew & Cook 1990   5.6 → 3.1     （20〜34週）
  Konczak & Dichgans    2歳で75%の試行が「区切り1個」＝大人と同じ
```

出典：Berthier NE, Keen R (2006) "Development of reaching in infancy."
  *Exp Brain Res* 169:507-518. http://people.umass.edu/neb/papers/berthier2006.pdf
  （Table 4 に上記の各研究の値がまとまっている）
詳しい調査：doc/文献調査/リーチング/乳児の手の運動速度_2026-07-31.md

【太郎ではどう数えるか（文献との違い）】
文献は「1回のリーチ」を単位に数える。**太郎はまだリーチをしないので、
区切りとなる試行が無い。** そこで**1秒あたりの個数**で数える。

    人間の目安：区切りの間隔の中央値 **190ミリ秒**（中央50%は140〜270ミリ秒）
      [Tier1] Berthier & Keen 2006（原文で確認）
      ⇒ 1秒あたりに直すと **およそ 3.7〜7.1 個/秒**、中央値で 5.3 個/秒

    注意：これは「リーチしている最中」の値。何も狙わずに動いている自発運動の
      文献値は**見つからなかった**（調査ファイルの6章）。なので
      「人間はこの値」と断言はできない。**同じ物差しで条件を比べる**のが正しい使い方。

【手の速度は体幹から見た速度を使う】
文献の乳児は座位や仰臥位で体幹が固定されている。太郎は柵なしで体ごと動くので、
**腰から見た手の速度**を主指標にする（体ごと転がった分を手の動きと数えないため）。

注意：そのため、この道具が出す速度は `run/tools/check_speed.py` の値と**一致しない**。
  check_speed は地面から見た速度（文献と揃えるため）、こちらは腰から見た速度。
  実測（触覚あり seed0・4ヶ月・30秒）：
      地面から見た速度   平均 23.5 cm/s   最大 118.7 cm/s
      腰から見た速度     平均 19.7 cm/s   最大 125.8 cm/s（中央 15.8）

【2026-07-31 の初回の実測】
```
太郎（触覚あり seed0）   区切り 12.5 個/秒   間隔  80 ms
人間（リーチ中）         区切り 5.3 個/秒    間隔 190 ms（中央50%は140〜270ms）
```
**人間の2倍以上細かく振動している。** 平均速度は人間より遅いのに最大だけ突出、
という形と合わせると、太郎の動きは「細かく震えながら時々跳ねる」。
人間の乳児の「ゆっくり大きく振る」とは質が違う。

実験ファイルでの書き方:
    "plugins": {"movement_units": true}
    "plugins": {"movement_units": {"prominence": 0.02, "min_gap_ms": 20}}
"""
import numpy as np

from run.plugins.base import Plugin

# 谷とみなす深さのしきい値[m/s]。この分だけ速度が落ち込まないと「区切り」と数えない。
#   [Tier3・工学的判断] 文献に「どれくらい落ち込めば1区切りか」の定量的な定義は
#   見つからなかった（von Hofsten の原文は未入手）。細かい揺れを拾わない値として置く。
#   値を変えると個数が変わるので、**条件間の比較では必ず同じ値を使う**。
DEFAULT_PROMINENCE = 0.02

# 区切りとして数える最短の間隔[ミリ秒]。
#   注意：【2026-07-31 に 100 から 20 へ下げた】100ms にしていたところ、
#     太郎の実測が 134ms と**しきい値のすぐ上に張り付いていた**。
#     振ってみると 100ms→7.5個/秒、50ms→10.9、20ms→12.5、10ms→12.5 で、
#     **20ms で頭打ち＝それより細かい振動は無い**。
#     つまり 100ms のときの値は「太郎の細かさ」ではなく「しきい値の値」だった。
#   ⇒ 頭打ちになる 20ms を既定にする。これなら**測る側が答えを決めない**。
#     Berthier & Keen 2006 のリーチ中の間隔（中央値190ms・中央50%が140〜270ms）は
#     あくまで比較の目安であって、しきい値の根拠ではない。
DEFAULT_MIN_GAP_MS = 20


class MovementUnits(Plugin):
    name = "movement_units"

    def setup(self, ctx):
        self.prominence = float(self.config.get("prominence", DEFAULT_PROMINENCE))
        self.min_gap_ms = float(self.config.get("min_gap_ms", DEFAULT_MIN_GAP_MS))
        self._bind(ctx)
        # 通し（学習全体）の集計
        self.units = 0
        self.steps = 0
        self.peak = 0.0
        # 区間（記録の区切りごと）の集計
        self._seg_reset()
        # 谷を見つけるための状態
        self._prev_v = None
        self._rising = False          # いま速度が上がっている最中か
        self._valley_v = None         # 直近の谷の速度
        self._peak_since_valley = 0.0  # その谷のあとの山の高さ
        self._steps_since_unit = 10 ** 9

    def _bind(self, ctx):
        """体の id と 1ステップの秒数を引き直す。"""
        m = ctx.model
        self.hand_id = m.body("right_hand").id
        self.hip_id = m.body("hip").id
        # 1ステップの秒数。環境が持っていればそれを使う（frame_skip 込み）
        dt = getattr(getattr(ctx, "env", None), "dt", None)
        if dt is None:
            u = getattr(getattr(ctx, "env", None), "unwrapped", None)
            n = int(getattr(u, "frame_skip", 1) or 1) if u is not None else 1
            dt = float(m.opt.timestep) * n
        self.dt = float(dt)
        self.min_gap_steps = max(1, int(self.min_gap_ms / 1000.0 / self.dt))

    def on_body_change(self, ctx):
        # 注意：体を作り直すと body の id が変わりうる。累積した値は消さない。
        self._bind(ctx)

    def _seg_reset(self):
        self._seg_units = 0
        self._seg_steps = 0
        self._seg_peak = 0.0
        self._seg_vsum = 0.0

    def on_step(self, ctx):
        d = ctx.data
        # 腰から見た手の速度（体ごと転がった分を差し引く）
        self.feed(float(np.linalg.norm(
            d.cvel[self.hand_id][3:] - d.cvel[self.hip_id][3:])))

    def feed(self, v):
        """速度[m/s]を1つ受け取って数える。

        注意：ここを環境から切り離してあるのは、**答えの分かっている速度**
          （正弦波など）を流して数え方そのものを確かめられるようにするため。
          検証は run/tools/check_movement_units.py。
        """
        self.steps += 1
        self._seg_steps += 1
        self._seg_vsum += v
        self.peak = max(self.peak, v)
        self._seg_peak = max(self._seg_peak, v)
        self._steps_since_unit += 1

        prev = self._prev_v
        self._prev_v = v
        if prev is None:
            return

        if v > prev:
            # 上がっている。直前が谷だったなら、山の高さを更新していく
            if not self._rising:
                self._valley_v = prev          # ここが谷
                self._peak_since_valley = v
            self._rising = True
            self._peak_since_valley = max(self._peak_since_valley, v)
            # 谷から十分に持ち上がったら「区切り1つ」と数える
            if (self._valley_v is not None
                    and self._peak_since_valley - self._valley_v >= self.prominence
                    and self._steps_since_unit >= self.min_gap_steps):
                self.units += 1
                self._seg_units += 1
                self._steps_since_unit = 0
                self._valley_v = None          # この谷は数え終わり
        else:
            self._rising = False

    def _per_sec(self, units, steps):
        sec = steps * self.dt
        return (units / sec) if sec > 0 else 0.0

    def metrics(self, ctx):
        if not self._seg_steps:
            return None
        out = {
            # 1秒あたりの区切りの数。多い＝ぎくしゃく／少ない＝滑らか
            "mu_per_sec": round(self._per_sec(self._seg_units, self._seg_steps), 3),
            # 区切りの間隔[ミリ秒]。人間のリーチ中は中央値190ms（Berthier & Keen 2006）
            "mu_gap_ms": round(1000.0 / max(self._per_sec(self._seg_units,
                                                          self._seg_steps), 1e-9), 1),
            # 手の速度[cm/s]。文献と単位を揃える
            "hand_v_mean_cms": round(100.0 * self._seg_vsum / self._seg_steps, 2),
            "hand_v_peak_cms": round(100.0 * self._seg_peak, 2),
        }
        self._seg_reset()
        return out

    def line(self, ctx):
        if not self.steps:
            return None
        return (f"mu={self._per_sec(self.units, self.steps):.2f}/s "
                f"peak={100.0 * self.peak:.0f}cm/s")

    def report(self, ctx):
        if not self.steps:
            return None
        per_sec = self._per_sec(self.units, self.steps)
        gap = 1000.0 / per_sec if per_sec > 0 else float("inf")
        # 人間のリーチ中の間隔（Berthier & Keen 2006）と比べた位置づけ
        if gap < 140:
            judge = "人間のリーチ中より細かい（ぎくしゃくしている）"
        elif gap <= 270:
            judge = "人間のリーチ中の間隔（140〜270ms）の中に入っている"
        else:
            judge = "人間のリーチ中より粗い（ゆっくり大きく動いている）"
        return {
            "動きの区切り_毎秒": round(per_sec, 3),
            "区切りの間隔_ms": round(gap, 1),
            "人間との比較": judge,
            "手の速度_最大_cms": round(100.0 * self.peak, 2),
            "測ったステップ数": self.steps,
            "しきい値_深さ_ms": self.prominence,
            "しきい値_最短間隔_ms": self.min_gap_ms,
            "注意": "文献値はリーチ中の値。自発運動の文献値は見つかっていない"
                    "（doc/文献調査/リーチング/乳児の手の運動速度_2026-07-31.md）",
        }
