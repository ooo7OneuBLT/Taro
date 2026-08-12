"""触覚の順応（同じ場所を押され続けると感じ方が弱まる、人間の受容器が持つ性質）。

設計：作業記録（非公開）
指示：作業記録（非公開）

【どこに入るか】obs["touch"]（生の力ベクトル、点数×3次元）に、fusion.pyが
SomatosensoryCortexへ渡す直前で割り込む。obs["touch"]自体は書き換えず、新しい
キー obs["touch_percept"] にこのモジュールの出力を入れる（taro_setup.py側の仕事。
このファイルは「1点ずつの状態を持つ計算だけ」を担う）。

【粒度】1点ずつ（部位ごとにまとめない）。3-2節の見積もりの通り、計算・状態量の
コストは無視できる大きさ。

【速順応】時定数を持たず、差分そのものを使う（太郎の観測間隔が速順応の反応時間
より粗いため。差分をそのまま使う方が誠実という3案共通の判断）。

    fa_i(t) = fa_gain × |m_i(t) - m_i(t-1)|

【遅順応】末梢＋脳の二段階（脳はフラグでON/OFF、既定OFF）。

    接触中（m_i(t) > TOUCH_EPS）：
      g_peripheral_i(t) = g_peripheral_i(t-1) - (dt/tau_peripheral) × (g_peripheral_i(t-1) - floor)
      g_cortical_i(t)    = g_cortical_i(t-1)    - (dt/tau_cortical)    × (g_cortical_i(t-1)    - floor)
        include_cortical=False のとき g_cortical_i(t) は常に1.0固定（更新自体を行わない）
    非接触中（回復）：
      g_peripheral_i(t) = g_peripheral_i(t-1) + (dt/tau_recover) × (1.0 - g_peripheral_i(t-1))
      g_cortical_i(t)    = g_cortical_i(t-1)    + (dt/tau_recover) × (1.0 - g_cortical_i(t-1))

    sa_i(t) = m_i(t) × g_peripheral_i(t) × g_cortical_i(t)
    g_peripheral・g_cortical は常に [floor, 1.0] にクリップする。

【合成】速順応・遅順応をそれぞれ「知覚される力の量」として計算し、最後に足す。

    adapted_m_i(t) = sa_i(t) + fa_i(t)
    adapted_f_i(t) = adapted_m_i(t) × (f_i(t) / m_i(t))    （m_i(t)=0のときadapted_f_i(t)=0）

【逸脱として明記】人間の速順応型・遅順応型は別々の神経線維（別チャンネル）で脳へ
届くが、太郎は観測の次元を変えない絶対条件があるため、点ごとに1本の信号へ加算
合成してから渡す[Tier3・工学的妥協]。`doc/人間模倣からの逸脱リスト.md` に登録済み。

【成長時】体が育つと点の数・並びが変わる。`rebuild(n_points)` で全点リセット
（末梢・脳ゲインを1.0、前観測をNoneに戻す）。rebuild()を呼ばずに違う点数の
touch_flatを渡すとAssertionErrorで止まる（落とし穴チェックリスト項86と同型）。

【エピソード境界】太郎の学習プロセス全体では持続させる（既定）。
`touch_adapt_reset_on_episode=True`のときだけ、`reset_episode()`（rebuild()と
同じ初期化だが点数は変えない軽量版）をエピソード境界(env.reset())で呼ぶ。
呼ぶかどうかの判断は taro_setup.py 側（Taro.apply_touch_adaptation）が持つ。

根拠ラベル：
    速順応の差分ベース設計          [Tier1定性的]文献の定性的記述と整合
    遅順応の末梢時定数8.4秒          [Tier2・孫引き]
    遅順応の脳時定数15.0秒           [Tier2・孫引き、範囲5.7〜21秒の中間値]
    遅順応の残存率0.31               [Tier3・マウスのひげ受容器由来、種を跨ぐ流用]
    回復の時定数（暫定で減衰と同じ）  [Tier3・文献無し]
    速順応倍率fa_gain                [Tier3・工学的仮置き]
"""
import numpy as np

# 「接触あり」とみなす力の大きさのしきい値。ノイズと区別するための最小限の値
# （厳密なゼロ判定は浮動小数の丸め誤差で不安定になるため）。[Tier3・工学的判断]
TOUCH_EPS = 1e-8


class TouchAdaptation:
    """触覚の順応（速順応＋遅順応・末梢＋脳）の状態を持つ。1点ずつ独立に進む。

    メソッド：
        advance(touch_flat)   状態を1回だけ進める（新しい物理観測が生まれた
                               瞬間にだけ呼ぶこと。呼び出し側の責務）
        adapted()              順応済みの値を返す。何度呼んでも状態は変えない
        rebuild(n_points)      成長時：点数が変わったので全点リセット
        reset_episode()        エピソード境界用の軽量リセット（点数は不変）
    """

    def __init__(self, n_points, dt, *, fa_enabled=True, sa_enabled=True,
                 include_cortical=False, tau_peripheral_s=8.4, tau_cortical_s=15.0,
                 sa_floor=0.31, tau_recover_s=8.4, fa_gain=1.0):
        """
        Args:
            n_points: センサ点の数（touch_flatの長さはこの3倍）
            dt: advance()を1回呼ぶごとに経過したとみなす秒数
                （＝太郎の1判断＝K物理ステップぶんの時間。呼び出し側=taro_setup.pyが計算する）
            fa_enabled/sa_enabled: 速順応・遅順応の個別ON/OFF
            include_cortical: 遅順応に脳(体性感覚野)側の順応レイヤーを含めるか
            tau_peripheral_s/tau_cortical_s: 遅順応・末梢/脳の時定数[秒]
            sa_floor: 遅順応の残存率（末梢・脳共通）
            tau_recover_s: 順応から回復する時定数[秒]
            fa_gain: 速順応の倍率
        """
        self.dt = float(dt)
        self.fa_enabled = bool(fa_enabled)
        self.sa_enabled = bool(sa_enabled)
        self.include_cortical = bool(include_cortical)
        self.tau_peripheral_s = float(tau_peripheral_s)
        self.tau_cortical_s = float(tau_cortical_s)
        self.sa_floor = float(sa_floor)
        self.tau_recover_s = float(tau_recover_s)
        self.fa_gain = float(fa_gain)
        self.rebuild(n_points)

    @property
    def total_dim(self):
        return self.n_points * 3

    # ------------------------------------------------------------ リセット
    def rebuild(self, n_points):
        """体を作り直した（点の数・並びが変わった）ときに呼ぶ。全点リセット。"""
        self.n_points = int(n_points)
        self.g_peripheral = np.ones(self.n_points, dtype=np.float64)
        self.g_cortical = np.ones(self.n_points, dtype=np.float64)
        self.prev_m = None
        self._last_adapted = np.zeros(self.total_dim, dtype=np.float64)

    def reset_episode(self):
        """エピソード境界(env.reset())用の軽量リセット。点数は変えない。

        rebuild()と同じ初期化（ゲイン=1.0、前観測=None）を行うが、意味が違う
        （rebuild=体が変わった＝点の対応が失われた、reset_episode=同じ体のまま
        学習アルゴリズム側の都合でエピソードが区切られた）ので別名にしてある。
        """
        self.rebuild(self.n_points)

    # ------------------------------------------------------------ 前向き計算
    def advance(self, touch_flat):
        """新しい物理観測1個ぶん、状態を進める。戻り値は無い（adapted()で取る）。

        呼び出し側の責務：この呼び出しは「新しい物理観測が生まれた瞬間」に
        だけ行うこと（1つの物理観測に対し複数回呼ぶと、順応が実際より速く
        進む。設計2節・仕様4節）。
        """
        arr = np.asarray(touch_flat)
        if arr.shape[-1] != self.total_dim:
            raise AssertionError(
                f"触覚の順応：入力の次元{arr.shape[-1]} != 地図の次元{self.total_dim}。\n"
                "  体を作り直したなら rebuild(n_points) を呼ぶ必要がある"
                "（落とし穴チェックリスト 項86と同型）。")
        f = arr.reshape(self.n_points, 3).astype(np.float64, copy=False)
        m = np.linalg.norm(f, axis=-1)

        # ---- 速順応：時定数を持たない差分ベース ------------------------------
        if self.fa_enabled and self.prev_m is not None:
            fa = self.fa_gain * np.abs(m - self.prev_m)
        else:
            fa = np.zeros(self.n_points, dtype=np.float64)

        # ---- 遅順応：末梢＋脳の二段階 -----------------------------------------
        if self.sa_enabled:
            contact = m > TOUCH_EPS
            floor = self.sa_floor
            gp = self.g_peripheral
            gp = np.where(contact,
                         gp - (self.dt / self.tau_peripheral_s) * (gp - floor),
                         gp + (self.dt / self.tau_recover_s) * (1.0 - gp))
            self.g_peripheral = np.clip(gp, floor, 1.0)
            if self.include_cortical:
                gc = self.g_cortical
                gc = np.where(contact,
                             gc - (self.dt / self.tau_cortical_s) * (gc - floor),
                             gc + (self.dt / self.tau_recover_s) * (1.0 - gc))
                self.g_cortical = np.clip(gc, floor, 1.0)
            # include_cortical=False のときは g_cortical を一切更新しない
            #   （常に初期値1.0のまま＝設計3-4節の数式どおり「末梢のみの
            #   1段階モデル」に数値的に一致する）。
            sa = m * self.g_peripheral * self.g_cortical
        else:
            sa = m.copy()

        adapted_m = sa + fa

        # ---- 方向を戻す（力の大きさだけを順応させ、向きは保つ）----------------
        with np.errstate(divide="ignore", invalid="ignore"):
            unit = np.where(m[:, None] > TOUCH_EPS, f / m[:, None], 0.0)
        adapted_f = adapted_m[:, None] * unit

        self._last_adapted = adapted_f.reshape(-1)
        self.prev_m = m

    def adapted(self):
        """順応済みの値（touch_flatと同じ形）を返す。状態は変えない。"""
        return self._last_adapted
