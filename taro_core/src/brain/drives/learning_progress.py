"""内発的動機 — 太郎が「何を嬉しいと感じるか」。

【なぜ core にあるか、2026-07-25】学習進度（progress報酬）は各目標の学習スクリプトに
**3箇所コピペ重複**していた（`C/scripts/run_c_metrics_ac_lr.py` 781-784／
`D/scripts/d0_selftouch.py` 235-241／`E/scripts/e_growth_train.py` 748-752）。
時定数 0.9/0.99 もそれぞれにハードコードされ、1箇所で直せない状態だった。
一方で「予測しやすさ報酬」（`taro_brain_motor.sensorimotor_reward`）は core にあり、
**同じ"報酬"なのに置き場所が非対称**だった。
「何を嬉しいと感じるか」は**太郎の本能そのもの**なので core に集約する。
方針：[[feedback-core-vs-experiment-placement]]

【2つの報酬の違い（2026-07-25に実測で決着）】
| | predict（旧） | progress（本ファイル） |
|---|---|---|
| 何を報酬にするか | 誤差が**小さいこと** `1/(1+誤差)` | 誤差が**減ること** `pe_slow − pe_fast` |
| 落ちる穴 | **大行動バイアス**（大きく動くほど感覚変化がノイズを超えて予測しやすい＝暗い部屋問題の一種） | この穴を構造的に回避 |
| 人間の実測との一致 | ✕（誤差最小を好む証拠がない） | ○ **Goldilocks効果**（乳児は簡単すぎず難しすぎないものを選ぶ, Kidd et al. 2012） |

同一条件・同一シード・7000stepの対照実験（age=0・仰向け・筋肉モデル）：
predict は margin +55.5 だが**69.5%の時間うつ伏せ**、progress は margin +45.9 で
**うつ伏せ1.1%**（jerk も 1/2.5）。＝predict は指標を水増ししていた。

【根拠】Oudeyer & Kaplan の Learning Progress（内発的動機の計算論）。
「学べる余地があること」を報酬にするので、完全に予測できるもの（飽き）にも
まったく予測できないもの（避ける）にも報酬が出ない。
注意：時定数（0.9 / 0.99）は[Tier3・ARBITRARY]＝生物学的な裏づけはなく、
「速い平均と遅い平均の差を取る」という形だけが Learning Progress に由来する。

【surprise_bonus（2026-08-04追加、機構1）】レアな予測誤差の急上昇（驚き）が
progressを瞬間的に大きくマイナスへ振れさせ、罰になってしまう問題への対処。
Kakade & Dayan (2002) の二相性ドーパミン反応のうち、抑制側（基線以下への
落ち込み）は採用せず、一過性の増加を示す第一相のみを模す。Tier3・工学的近似
（具体的な検出式・時定数はドーパミン反応の数理モデルそのものではなく、
分散で正規化した増分を指数減衰させるという工学的な設計）。

【しきい値方式（超過量型・ヒンジ）への修正、2026-08-04】実装当初は
`max(dev,0)/sqrt(var)`（zがプラスなら全量を採用）だったが、実データで
平常時のtrace平均が2.1〜3.5（要件0.05未満）と大きく超過するバグが見つかった
（設計・案D、作業記録（非公開）3節）。
修正として`excess = max(z - k, 0)`（zがしきい値kを超えた分だけ採用するヒンジ）に
差し替えた。これはSchultz, Dayan & Montague (1997)の連続的RPE理論（報酬予測誤差の
大きさに応じて発火が連続的・段階的に変化するという理論）からの**追加の**逸脱に
あたる。しきい値未満では発火量が厳密にゼロという折れ線を導入することは、
生理学的な連続コーディングとは異なる新しい種類の近似を加えることになる。
ただし、これは平常時の発火頻度が既存の要件（0.05未満）を満たさなかったバグを
修正するための工学的な対処であり、人間らしさを高めるための選択ではない。
`doc\人間模倣からの逸脱リスト.md`項⑫に追記済み（2026-08-04）。
"""


class LearningProgress:
    """学習進度＝「予測誤差が減っていること」を報酬にする（Oudeyerの好奇心）。

    速い移動平均と遅い移動平均の差を取る。誤差が減り続けている間だけ正の値が出る。

    Args:
        tau_fast: 速い移動平均の保持率（大きいほど過去を引きずる）。
        tau_slow: 遅い移動平均の保持率。
        init: 移動平均の初期値。
        surprise_bonus: レアな予測誤差の急上昇（驚き）を検出し、一時的にprogressへ
            加算するボーナスのゲート兼利得（2026-08-04追加、機構1）。既定0.0＝OFF。
            `self_touch_interest_bonus`と同じ「1つの値がゲートかつ利得を兼ねる」
            パターン。式の中でtraceに足し込む係数（設計のnovelty_gainに相当する
            量）としてこの値を使う。
        surprise_decay: surprise_bonusのtraceの減衰率（Tier3・恣意的）。
        surprise_var_tau: surprise検出に使う分散の移動平均率（Tier3・恣意的）。
        surprise_threshold: zスコア（devを直近のばらつきの標準偏差で割った値）が
            何σを超えたら「レアな驚き」とみなすかのしきい値k（超過量型・ヒンジ、
            2026-08-04追加）。既定5.0[Tier3・工学的近似。実データ検証
            （run/tools/check_progress_surprise.py、reach_self・4シード・4500step、
            2026-08-04実測）の結果、既定候補だった4.0はnoneカテゴリtrace平均0.061
            （要件0.05未満に不合格）、4.5は4シード集計では0.045で合格するものの
            シード単位では4件中2件が0.05を超えていたため、5.0（集計0.034、
            シード単位でも4件中1件のみ僅かに超過）へ引き上げた]。

    注意：時定数は[Tier3・ARBITRARY]。3つの目標スクリプトで同じ値(0.9/0.99)が
    ハードコードされていたので、既定として引き継いだ（挙動を変えないため）。
    """

    def __init__(self, tau_fast=0.9, tau_slow=0.99, init=1.0,
                 rate_fast=None, rate_slow=None,
                 surprise_bonus=0.0, surprise_decay=0.9, surprise_var_tau=0.99,
                 surprise_threshold=5.0):
        self.tau_fast = float(tau_fast)
        self.tau_slow = float(tau_slow)
        # 注意：新しい値の重みは「1-tau」でなく**別に持つ**。理由：`1.0 - 0.9` は2進数で
        # 0.09999999999999998 になり、旧実装が書いていた `0.1` と厳密には一致しない
        # （差 2.8e-17）。移設で数値が1バイトも変わらないようにするため、旧実装と
        # 同じ係数をそのまま持つ。既定は 0.1 / 0.01（＝旧実装の3ファイルと同一）。
        self.rate_fast = (1.0 - self.tau_fast) if rate_fast is None else float(rate_fast)
        self.rate_slow = (1.0 - self.tau_slow) if rate_slow is None else float(rate_slow)
        if rate_fast is None and abs(self.tau_fast - 0.9) < 1e-12:
            self.rate_fast = 0.1      # 旧実装と厳密に同じ係数
        if rate_slow is None and abs(self.tau_slow - 0.99) < 1e-12:
            self.rate_slow = 0.01
        self.pe_fast = float(init)
        self.pe_slow = float(init)
        # ---- surprise_bonus（2026-08-04・機構1）用の内部状態 --------------------
        # 既定0.0のときはupdate()内でこれらに一切触れない（計算自体をスキップする）。
        self.surprise_bonus = float(surprise_bonus)
        self.surprise_decay = float(surprise_decay)
        self.surprise_var_tau = float(surprise_var_tau)
        self.surprise_threshold = float(surprise_threshold)
        self._surprise_var = 0.0
        self._surprise_trace = 0.0
        self._surprise_eps = 1e-6   # relative_surprise()の1e-6と揃える固定の内部定数
        # ---- resync()直後のfreeze（設計・案D発見のバグへの対処、2026-08-04）----
        #   resync直前のdevが小さいほどvarが極端に小さく初期化され、直後のごく
        #   普通の変動が異常なzを生む（resync()のdocstring参照）。freeze中は
        #   しきい値判定（excess）を強制的に0にし、varの更新だけは続ける。
        self._surprise_freeze_ticks = 100
        # 【2026-08-05発見・③設計検証中に判明】コンストラクタ直後も_surprise_varが
        #   0.0で、resync()直後と全く同じ機序でバグる：1回目のupdate()で
        #   _surprise_var=(1-var_tau)*dev**2 となり、z=dev/sqrt(_surprise_var)が
        #   devの大きさによらずおよそ1/sqrt(1-var_tau)（var_tau=0.99なら約10）に
        #   機械的に固定され、しきい値5.0を必ず超えてしまう。resync()と同じ
        #   freeze機構を起動時にも適用して防ぐ。
        self._surprise_freeze_remaining = self._surprise_freeze_ticks
        self._surprise_last_z = 0.0
        self._surprise_last_frozen = False

    def update(self, prediction_error):
        """予測誤差を1つ受け取り、移動平均を更新して学習進度を返す。

        Returns:
            pe_slow − pe_fast（正なら「最近うまくなっている」）。
            surprise_bonusが有効なときは、これに加えてsurprise_bonus_t
            （レアな驚きに対する一時的な加点）を足したものを返す。
        """
        pe = float(prediction_error)
        # 【なぜこの順序か】dev_tは「更新"前"のpe_slow」との差でなければならない
        # （更新後のpe_slowと比べると、その1tick分の変化そのものを打ち消して測る
        # ことになり、驚きを正しく検出できない）。
        pe_slow_prev = self.pe_slow
        self.pe_fast = self.tau_fast * self.pe_fast + self.rate_fast * pe
        self.pe_slow = self.tau_slow * self.pe_slow + self.rate_slow * pe
        progress = self.pe_slow - self.pe_fast
        if self.surprise_bonus:
            dev = pe - pe_slow_prev
            self._surprise_var = (self.surprise_var_tau * self._surprise_var
                                   + (1.0 - self.surprise_var_tau) * dev ** 2)
            # 増加方向だけを見る（Q3推奨。SAGG-RIACの両方向採用とは異なる、
            # Tier3・工学的判断・逸脱候補）。ここまでは旧式と同じ。
            z = dev / (self._surprise_var ** 0.5 + self._surprise_eps)
            self._surprise_last_z = z   # 検証スクリプト（_TraceRecorder）が読む
            if self._surprise_freeze_remaining > 0:
                # resync直後のfreeze中：varの更新は続けるが、しきい値判定は止める
                # （resync()のdocstring参照。案D発見のバグへの対処）。
                self._surprise_freeze_remaining -= 1
                self._surprise_last_frozen = True
                excess = 0.0
            else:
                self._surprise_last_frozen = False
                # 【しきい値方式（超過量型・ヒンジ）、2026-08-04】zがしきい値kを
                # 超えた分だけを採用する。k=0.0のとき旧式(max(z,0))と厳密に一致する
                # （モジュールdocstring参照）。
                excess = max(z - self.surprise_threshold, 0.0)
            # surprise_gain（設計のnovelty_gainに相当）としてsurprise_bonusの値を
            # そのまま使う（ゲートかつ利得を兼ねる。self_touch_interest_bonusと同じ流儀）。
            self._surprise_trace = (self.surprise_decay * self._surprise_trace
                                     + self.surprise_bonus * excess)
            surprise_bonus_t = self._surprise_trace
            progress = progress + surprise_bonus_t
        return progress

    def relative_surprise(self):
        """「いつもより驚いたか」を [0,1] で返す（0.5＝平常の驚き）。

        報酬の絶対値に固定閾値を当てると、値域の違う報酬関数を入れたときに壊れる
        （学習進度≒0.04は常に「報酬ゼロ」と誤認される）。長期基準線と比べる形にすると
        尺度に依存しない。探索/活用の切り替えや NE の相対化に使う。
        """
        return min(self.pe_fast / (self.pe_slow + 1e-6), 2.0) / 2.0

    def state(self):
        """ログ用（pe_fast, pe_slow）。"""
        return self.pe_fast, self.pe_slow

    def resync(self):
        """速い/遅い移動平均を今の値どうしで揃え、progressを0からスタートさせる。

        【なぜ、2026-08-03・Tier3・工学的判断】体が育つと固有感覚などの分布が変わり、
        次に来る生の予測誤差が急変しうる。この関数を呼ばずにいると、自己接触の瞬間と
        同じ機序（速い平均だけが跳ねてpe_fast > pe_slowになる）で、成長直後の予測しづらい
        期間が構造的に罰される恐れがある（調査報告2026-08-03 1-3節）。
        ここでは`pe_slow`を0にリセットしない。0にすると次の一歩で「急に良くなった」という
        誤った大きなプラスのprogressを一度だけ生んでしまう（pe_fastがまだ古い高い値のまま
        pe_slowだけ0から動き出すと、差がプラス側に不自然に開く）。代わりに
        **pe_slow の値を pe_fast へコピー**して、差＝0からスタートさせる
        （`self.pe_fast = self.pe_slow`。pe_slow のほうは変えない。
        向きに注意：pe_fast → 両方 ではない）。
        これは根拠のある確定的な修正ではなく、成長イベントでの罰を自己接触と同様に
        避けたいという設計判断（人間模倣からの逸脱ではなく実装上の工学的判断）。

        【surprise_bonus有効時の拡張、2026-08-04・Q6の推奨を具体化】pe_fastを
        書き換える"前"に、pe_fast−pe_slow（＝その時点でのdevに相当する量）を
        _surprise_varの初期値にする。成長直後、pe_fastとpe_slowの差は直近の
        予測誤差の振れ幅の代理量であり、これをvarの初期値にすることで、
        resync直後にsurpriseが異常値になることを防ぐ（0初期化のままだと、
        次tickのdevをほぼ0の分散で割ることになりsurpriseが異常に大きくなる
        リスクがあるため）。traceは0からやり直す。

        【resync()のvar初期化バグの修正（freeze方式）、2026-08-04・設計・案D発見】
        上記の対処だけでは不十分だった。resync直前のdev（pe_fast−pe_slow）が
        小さい（＝直近の予測が安定していた）ほど、_surprise_varが極端に小さい値
        （例：dev=0.001ならvar=1e-6）に初期化され、直後のごく普通の変動
        （dev=0.05〜0.3程度）が実効z=数十〜数万という異常値として誤検出される
        （設計・案D 3節で数値検算済み）。対処として、resync()が呼ばれた直後、
        一定tick数（_surprise_freeze_ticks＝100、var_tau=0.99の実効平均窓
        1/(1-var_tau)=100tickに合わせた目安）は、しきい値判定自体を止める
        （update()内でexcessを強制的に0.0にする）。varの更新は通常どおり続ける
        ——これにより、freeze期間が終わる頃には、varが実際のtickのdevの2乗を
        十分に取り込んで、より妥当な値に収束している。
        """
        if self.surprise_bonus:
            dev = self.pe_fast - self.pe_slow   # 書き換える"前"の値を使う
            self._surprise_var = dev ** 2
            self._surprise_trace = 0.0
            self._surprise_freeze_remaining = self._surprise_freeze_ticks
        self.pe_fast = self.pe_slow


def smoothness_cost(action, prev_action):
    """行動の**急変**にかかるコスト（CAPS）＝「さっきと違うことを急にするのは損」。

    `a_t − a_{t-1}` の各成分を二乗した**平均**を返す（合計ではない。理由は下）。
    呼び出し側で `報酬 − λ×smoothness_cost(...)` の形で引く。

    【根拠】CAPS（Mysore et al. 2021, *Regularizing Action Policies for Smooth Control
    with Reinforcement Learning*, http://ai.bu.edu/caps/）[Tier1]。
    強化学習の方策は1stepごとに独立にノイズをサンプルするため出力が高周波になりやすく、
    実機ではモータを傷める・シミュレーションでは**力積が打ち消し合って体が動かない**。

    【太郎ではなぜ要るか】制御頻度を1Hz→10Hzに上げると、探索ノイズも10倍の頻度で
    サンプルされる＝**高周波化して打ち消し合う**。過去にK=10（10Hz）を試して失敗した
    （経験量を10倍に揃えてもK=100に全指標で届かなかった）原因の最有力候補がこれで、
    CAPSはその症状への直接対策になる。
    副次効果として「**何もしない**（＝前回と同じ行動を続ける）」が学習で選べるようになる。
    人間は「座りたい」と1秒ごとに考え直さない。

    **合計でなく平均**を返す。理由：`‖a_t−a_{t-1}‖²` は**行動の次元数に比例**するので、
    合計にすると λ の意味が「関節90次元か筋180次元か」で変わってしまう。実測すると
    合計では 11.9 になり、progress報酬（≒0.04）に対して λ=0.01 でも**ペナルティが報酬の3倍**
    という状態だった。次元数という無関係な量で寄与が決まるのを止める、という
    太郎の既存方針（感覚ごとの重み 段階1・`block_pe`）と同じ扱いに揃える。

    注意：λ（ペナルティの重さ）は**[Tier3・ARBITRARY]**。強すぎると動きが縮んで固まる
    （拮抗筋の共収縮で実際に起きた失敗と同じ罠）。必ず複数値で振って確かめること。
    平均にした上での目安：progress報酬≒0.04・本コスト≒0.13 なので、
    λ=0.01 でペナルティは報酬の約3%、λ=0.1 で約30%。
    注意：`effort_cost`（大きな力を出すのが損）とは**別物**。こちらは「大きさ」でなく
    「**変化量**」に効く＝ゆっくり大きく動くのは咎めない。

    Args:
        action: 今の行動（torch.Tensor または ndarray）。
        prev_action: 1つ前の行動。同じ形。

    Returns:
        float のコスト（次元あたりの平均）。
    """
    d = action - prev_action
    return float((d ** 2).mean())


def effort_cost(action, muscle_weights):
    """努力コスト（代謝コスト）＝「大きな力を出すのは損」。

    activation² × 筋サイズ の合計。筋力（最大トルク）が大きい筋ほど動かすとコストが高い、
    という代謝の標準形。

    注意：[Tier2〜3] 二乗形（Σu²）は工学的な近似で、実際の代謝式は非二乗（Umberger 等）。
    注意：MIMoの `cost()` はトルク単位を二乗して実質トルク³になり桁が狂うため使わない
    （2026-07-19 に確認）。正規化された行動 a∈[-1,1] を使う。

    Args:
        action: 行動（torch.Tensor または ndarray）。
        muscle_weights: 筋ごとの重み（合計1に正規化しておくと effort∈[0,1] で扱いやすい）。

    Returns:
        float のコスト。呼び出し側で `報酬 − λ×effort_cost(...)` の形で引く。
    """
    return float((action ** 2 * muscle_weights).sum())
