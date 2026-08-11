"""脊髄・脳幹相当の上位層：周期・振幅が揺らぐ振動子による共通駆動。

新しい駆動モジュール（伸張反射＋揺らぐ振動子の共通駆動）の上位層。下位層
（`stretch_reflex.py`）が力を出す一方、ここは「基準長を動かす」役割を担う。

【2026-08-11・ユーザーからの修正指示（変更1・変更2）】当初案（案B・案Cの具体化）は
上位層の中身を「独立成分と共通成分をsqrt(1-ρ)・sqrt(ρ)で混ぜる相関ノイズ」として
実装しており、振動子（リズム）という要素が数式上どこにも残っていなかった。これは
`spinal_cord/cpg.py` の `ColoredNoiseGenerator`（1/f^βの色付きノイズ、周期性を持たない）
と同じ発想へのすり替わりであり、ユーザーから指摘されて直した。この改訂で、上位層は
「位相・周波数・振幅を持ち、周波数と振幅がゆっくり揺らぐ振動子」として実装する。

【cpg.py とは無関係】このファイルは `spinal_cord/cpg.py` を一切importしない
（`ColoredNoiseGenerator` を含め、すべて新規に独立実装する）。クラス名にも
「CPG」という文字を使わない（cpg.pyという名前と中身の不一致が実際に混乱を
生んだことの再発防止。変更2）。

【根拠ラベル】
    リズム（振動子）にすること自体          [人間模倣の要請、Tier2〜3]
        ・1回目の設計（白紙設計）で設計者3人が独立に到達した結論
        ・森・國吉2010の参照実装がBVP振動子（工学的近似だが、リズムという
          発想自体は先行研究が採用している）
        ・ユーザーが本物の乳児のGMA動画を見て「手足の伸縮しか起こらない」と観察
    揺らがせること自体                    [人間模倣の要請、Tier1〜2]
        ・Prechtlのwrithingの定義に周期性は入っていない（「数秒〜数分続き、強度・
          速度が漸増漸減し、緩やかに始まり緩やかに終わる」）
        ・胎児の実測（Sival）ではバースト間隔（onset-onset interval）は不規則
        ・太郎は過去に「反復が無いのは原理的欠陥」と主張して撤回している
          （doc/人間模倣からの逸脱リスト.md「2026-07-26：自発運動の生成方式そのものの
          位置づけ」節）。きっちりした周期にすると、この撤回した主張を裏返しで
          再現することになる。
    具体的な数式・パラメータの値              [Tier3・工学的判断]
        太郎が実装しやすい単純な形（sin波の位相・周波数・振幅にノイズでランダム
        ウォークをかける）を自分で設計した。BVP振動子（van der Pol型方程式、
        Kawai 2017 PLOS ONE）という特定の工学的近似式は、パラメータが公開論文に
        載っておらず、かつ生理学的実測から出た式でもないため、人間の根拠としては
        使わない（ユーザー指示のとおり。BVP振動子を人間の根拠として引用していない）。

（設計：作業記録（非公開）
7-2節の擬似コードをそのまま実装する。数式・定数の並びは同節と同一）
"""
import numpy as np


class WanderingOscillator:
    """位相φ・周波数f・振幅Aを持つ、1本の揺らぐ振動子。

    f と A はそれぞれ、中心値(f0, A0)の周りをオルンシュタイン=ウーレンベック過程
    （平均に戻ろうとするランダムウォーク）でゆっくり揺らす。
    [Tier3・工学的判断。数式・定数は自分で設計した]

    「だいたい2秒で往復するが毎回少し違う」という揺らぎ方は、本物の心拍・呼吸の
    ように「繰り返しに見えるが厳密な周期ではない」実測（Sival、バースト間隔が
    不規則）と矛盾しない（統合版7-2節「両立の考え方」）。
    """

    def __init__(self, f0=0.5, A0=1.0, tau_f=2.0, sigma_f=None, tau_A=2.0, sigma_A=None,
                 f_min=None, f_max=None, A_min=0.0, A_max=None, seed=None):
        self.f0, self.A0 = float(f0), float(A0)
        self.tau_f = float(tau_f)
        # sigma_f/sigma_A の既定＝中心値の20〜30%目安（統合版7-2節「初期パラメータの
        # 推奨値」）。25%を代表値として採用（[Tier3・感度分析対象、測定が振る]）。
        self.sigma_f = float(sigma_f) if sigma_f is not None else 0.25 * self.f0
        self.tau_A = float(tau_A)
        self.sigma_A = float(sigma_A) if sigma_A is not None else 0.25 * self.A0
        # f_min/f_max ＝ f0の0.5倍〜2倍程度（統合版7-2節の推奨値）
        self.f_min = float(f_min) if f_min is not None else 0.5 * self.f0
        self.f_max = float(f_max) if f_max is not None else 2.0 * self.f0
        # A_min/A_max ＝ 0〜A0の2倍程度（統合版7-2節の推奨値）
        self.A_min = float(A_min)
        self.A_max = float(A_max) if A_max is not None else 2.0 * self.A0
        self.f = self.f0
        self.A = self.A0
        self.phase = 0.0
        self.rng = np.random.default_rng(seed)

    def step(self, dt):
        """周波数・振幅を、平均に戻ろうとするランダムウォーク（オルンシュタイン=
        ウーレンベック過程のオイラー=丸山離散化）で更新し、位相を進めて
        A*sin(phase) を返す。"""
        decay_f = np.exp(-dt / self.tau_f) if self.tau_f > 0 else 0.0
        noise_f = self.sigma_f * np.sqrt(max(1.0 - decay_f ** 2, 0.0)) * self.rng.normal()
        self.f = float(np.clip(self.f0 + (self.f - self.f0) * decay_f + noise_f,
                                self.f_min, self.f_max))

        decay_A = np.exp(-dt / self.tau_A) if self.tau_A > 0 else 0.0
        noise_A = self.sigma_A * np.sqrt(max(1.0 - decay_A ** 2, 0.0)) * self.rng.normal()
        self.A = float(np.clip(self.A0 + (self.A - self.A0) * decay_A + noise_A,
                                self.A_min, self.A_max))

        self.phase = float((self.phase + 2.0 * np.pi * self.f * dt) % (2.0 * np.pi))
        return self.A * np.sin(self.phase)


class RhythmicCommonDriveGroup:
    """関節グループごとに、独立成分と共通成分の2種類のWanderingOscillatorを持ち、
    sqrt(1-rho)・sqrt(rho)で混ぜて関節空間の信号 c_j(t) を作る。

    独立成分・共通成分は同じ分布のパラメータ(f0, A0等)を使うことを前提とする
    （でなければsqrt(1-rho)・sqrt(rho)の混ぜ方が想定する分散のつり合いが崩れる。
    検証1で数値的に確かめる）。

    joint_names：グループに属する関節のラベル（任意のhashable。行動配列の
        関節indexそのものを渡してよい）。1要素だけのグループにすると、
        その関節専用の「共通」成分が他のどの関節とも共有されないため、
        結果として関節間の相関は生まれない（設計7-4節 grouping="none" と
        数値的に等価。grouping側の組み立てはこのクラスの外、呼び出し側
        （taro_core外＝run側の配線）が担う）。
    """

    def __init__(self, joint_names, rho, oscillator_kwargs=None, seed=None):
        self.joint_names = list(joint_names)
        self.rho = float(rho)
        kw = dict(oscillator_kwargs or {})
        # cpg.py と同じ「seed + k」方式（乱数系統を関節ごとに分ける）。
        # 注意：Python組み込みの hash() は文字列に対して既定でプロセスごとに
        #   ランダム化される（PYTHONHASHSEED未設定時）ため、シード用に使うと
        #   同じシードでも再現しない。整数のオフセットだけを使う。
        s = (lambda k: seed + k) if seed is not None else (lambda k: None)
        self.indep = {j: WanderingOscillator(seed=s(1 + i), **kw)
                      for i, j in enumerate(self.joint_names)}
        self.common = WanderingOscillator(seed=s(0), **kw)

    def step(self, dt):
        """{関節名: c_j(t)}（基準長へ加える前の、無次元の信号）を返す。"""
        common_x = self.common.step(dt)
        w_indep = float(np.sqrt(max(1.0 - self.rho, 0.0)))
        w_common = float(np.sqrt(max(self.rho, 0.0)))
        out = {}
        for j, osc in self.indep.items():
            indep_x = osc.step(dt)
            out[j] = w_indep * indep_x + w_common * common_x
        return out


def joint_signal_to_L0_offset(c_j, moment_1_j, moment_2_j, amp):
    """関節jの無次元共通駆動信号c_j(t)を、拮抗筋2本(neg/pos)の基準長オフセットへ
    変換する（案C 4-2節・統合版7-1節、moment arm経由の変換）。

    moment_1_j・moment_2_j はMuscleModel構築時に確定した、拮抗筋2本への符号込みの
    換算係数（読み取るだけで書き換えない）。moment_1は常に正・moment_2は常に負
    （`MIMo/mimoActuation/muscle.py` の `_compute_parametrization` で確認済み）
    なので、同じ1つの信号にこの2つを掛けるだけで、拮抗筋2本には自動的に符号が
    逆のオフセットが乗る（write_joint_commandの罠＝片方の筋しか書かれない、を
    構造的に踏まない理由。設計4節）。

    Returns:
        (delta_L0_neg, delta_L0_pos)
    """
    a = float(amp)
    return a * moment_1_j * c_j, a * moment_2_j * c_j
