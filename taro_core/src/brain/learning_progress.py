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
"""


class LearningProgress:
    """学習進度＝「予測誤差が減っていること」を報酬にする（Oudeyerの好奇心）。

    速い移動平均と遅い移動平均の差を取る。誤差が減り続けている間だけ正の値が出る。

    Args:
        tau_fast: 速い移動平均の保持率（大きいほど過去を引きずる）。
        tau_slow: 遅い移動平均の保持率。
        init: 移動平均の初期値。

    注意：時定数は[Tier3・ARBITRARY]。3つの目標スクリプトで同じ値(0.9/0.99)が
    ハードコードされていたので、既定として引き継いだ（挙動を変えないため）。
    """

    def __init__(self, tau_fast=0.9, tau_slow=0.99, init=1.0,
                 rate_fast=None, rate_slow=None):
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

    def update(self, prediction_error):
        """予測誤差を1つ受け取り、移動平均を更新して学習進度を返す。

        Returns:
            pe_slow − pe_fast（正なら「最近うまくなっている」）
        """
        pe = float(prediction_error)
        self.pe_fast = self.tau_fast * self.pe_fast + self.rate_fast * pe
        self.pe_slow = self.tau_slow * self.pe_slow + self.rate_slow * pe
        return self.pe_slow - self.pe_fast

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


def smoothness_cost(action, prev_action):
    """行動の**急変**にかかるコスト（CAPS）＝「さっきと違うことを急にするのは損」。

    `‖a_t − a_{t-1}‖²` を返す。呼び出し側で `報酬 − λ×smoothness_cost(...)` の形で引く。

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
