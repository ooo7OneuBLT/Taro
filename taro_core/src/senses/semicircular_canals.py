"""三半規管（前庭迷路の一部）— 頭の回転（角速度）を感知する。

【対応】MIMoの前庭センサーは「加速度3＋角速度3」の6次元(`vestibular_acc`+`vestibular_gyro`、
`MIMo/mimoVestibular/vestibular.py`)。このうち**角速度3軸**が三半規管に相当する
（3本の半規管がそれぞれ別平面の回転を感知することに対応）。

★【2026-07-26】動特性（適応・時定数）を実装した。それまでは瞬間の角速度をそのまま
返すだけで [ARBITRARY・未実装] とラベルしていた。

【なぜ実装したか】未実装のままだと「回り続けても永久に感じ続ける」ことになる。
実際に VOR がこの出力を使うと、頭が12秒かけて40度傾いただけで眼球が可動域の限界
（±45度）まで回りきって張り付いた（ユーザーの目視「眼球が下がってきて視界から外れる」
→ 実測で確認、研究日誌 2026-07-26 続き6）。人間なら約6度で収まる。

【機構】半規管は「輪状の管の中の内リンパ液が、途中のクプラ（ゼリー状の膜）を押す」構造。
液の慣性でたわみ、クプラの弾性で戻る＝**高域通過フィルタ**として働く。
  Steinhausen のトルク振り子モデル（2次系）：
      δ(s)/ω(s) ≈ T1・T2・s / [(T1・s + 1)(T2・s + 1)]
      T1 ≈ 0.003〜0.01秒（短い方・粘性と慣性の比）
      T2 ≈ 4〜7秒       （長い方・粘性と弾性の比）
  ⚠️T1 が効くのは 16〜50Hz 以上で、太郎の物理は100Hz・頭部運動は数Hz以下なので
    寄与しない。**1次に簡略化する**（工学・ロボティクスでの標準的な簡略化）。

【時定数の値】
  ・末梢（クプラ）：成人 **4.2 ± 0.6秒**
    Dai M, Klein A, Cohen B, Raphan T (1999) "Model-based study of the human cupular
    time constant" J Vestib Res 9:293-301 [PMID 10472042]。従来「5〜6秒」とされた
    Fernandez & Goldberg (1971) J Neurophysiol 34:661-675 の値を再推定したもの。
  ・中枢の velocity storage が実効時定数を **15〜25秒**まで延ばす
    （Robinson 1977 のモデル、PMC5561016）
  ・★新生児は中枢側が短い：一次眼振の持続 **10秒**（生後2か月で15秒）
    Weissman BM, DiScenna AO, Leigh RJ (1989) Neurology 39:534 [PMID 2927678]
  ⚠️「新生児は時定数が成人の約半分」という記述は**一次典拠を特定できなかった**
    （2026-07-26 の調査）。しかも Eviatar (1983) は「**末梢**の時定数は加齢で変化しない」
    と報告しており、短いのは**中枢の延長機構**の方。以前のコメントは末梢と中枢を
    混同していた。

【velocity storage の扱い・簡略化】
  本来は脳幹の正帰還積分器が末梢の時定数を延長する（Robinson 1977）。
  ここでは**正帰還ループを組まず、時定数を延長後の値に置き換える**だけにする。
  入出力特性はほぼ等価だが、**機構を模倣したのではなく結果だけ合わせている**
  [Tier2・簡略化]。OKN との相互作用や両側性の前庭代償を扱うなら要再検討。

【3軸の扱い・簡略化】
  実際の3本の半規管は解剖学的に30〜45度傾いた3平面にあるが、
  **直交3軸で近似し、各軸に同じ1次フィルタを独立にかける**[標準的な簡略化]。
  MIMo の出力が既にx/y/z角速度なので、そのまま各軸に適用する。
"""
import numpy as np

# ---- 時定数 [秒] ----------------------------------------------------------
# ★実効時定数（末梢 + velocity storage をまとめた値）。
#   新生児：10秒（Weissman 1989 の一次眼振の持続時間）
#   成人  ：15〜25秒
# ⚠️[Tier2] 「一次眼振の持続時間」と「フィルタの時定数」は厳密には同じ量ではない。
#   オーダーが合う値として採用する。
TIME_CONSTANT_NEWBORN = 10.0
TIME_CONSTANT_ADULT = 20.0
# 時定数が成人値に達する月齢。⚠️[Tier3・ARBITRARY] Weissman 1989 は生後2か月で
# 10→15秒としか報告しておらず、成人値に達する時期の一次典拠は見つかっていない。
MATURE_AGE_MO = 24.0


def time_constant(age_months=0.0):
    """月齢に応じた実効時定数[秒]。新生児10秒 → 成人20秒へ線形に延びる。

    ⚠️線形補間は[Tier3・ARBITRARY]。向き（発達で延びる）だけが文献に一致
    （Ornitz らが「加齢の対数に対し時定数は増加、利得は減少」と報告）。
    """
    a = float(age_months)
    if a >= MATURE_AGE_MO:
        return TIME_CONSTANT_ADULT
    w = max(0.0, min(1.0, a / MATURE_AGE_MO))
    return (1.0 - w) * TIME_CONSTANT_NEWBORN + w * TIME_CONSTANT_ADULT


def read(vestibular_obs):
    """前庭ベクトル(6次元)から、三半規管に相当する角速度3軸を取り出す（生の値）。

    ⚠️動特性を通していない生の角速度。VOR など**反射に使うときは
    `SemicircularCanals` クラスを使うこと**（時定数が効かないと眼球が回りきる）。
    学習の入力としては、時系列の情報を脳側が持つのでこちらでも使える。
    """
    return np.asarray(vestibular_obs)[3:6]


class SemicircularCanals:
    """高域通過フィルタつきの三半規管。角速度を入れると「感じている回転」を返す。

    1次の高域通過を、数値的に安定な**相補フィルタ**（入力からその低域成分を引く）
    として実装する。微分を使わないので高周波ノイズを増幅しない。

        x[n+1] = x[n] + α・(ω[n] - x[n])      α = 1 - exp(-dt/T)
        出力   = ω - x

    x は「最近の角速度の平均」。ゆっくりした回転は x が追いつくので出力が消え、
    速い回転は追いつけないので出力に残る＝これが「回り続けると感じなくなる」性質。

    使い方:
        canals = SemicircularCanals(age_months=0.0)
        sensed = canals.update(omega, dt)    # omega は角速度3軸[rad/s]
    """

    def __init__(self, age_months=0.0, tau=None, n_axis=3):
        self.tau = float(time_constant(age_months) if tau is None else tau)
        self.n_axis = int(n_axis)
        self.baseline = np.zeros(self.n_axis, dtype=float)   # 追いかけている平均
        self.sensed = np.zeros(self.n_axis, dtype=float)     # 最後の出力

    def reset(self):
        self.baseline[:] = 0.0
        self.sensed[:] = 0.0

    def update(self, omega, dt):
        """角速度 omega[rad/s] を1ステップ入れて、感じている回転を返す。

        ⚠️厳密な離散化（1次系のゼロ次ホールド解）を使うので、dt がどんな値でも発散しない。
        """
        w = np.asarray(omega, dtype=float).reshape(-1)[: self.n_axis]
        alpha = 1.0 - float(np.exp(-float(dt) / max(self.tau, 1e-9)))
        self.baseline += alpha * (w - self.baseline)
        self.sensed = w - self.baseline
        return self.sensed.copy()
