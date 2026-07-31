# --- コピー元: Taro (github.com/ooo7OneuBLT/Taro) commit 3b976fc ---
# --- 元パス: B/src/taro/... （unificationMIMoでは無編集のまま流用） ---

"""
恒常性（Homeostasis） — 体の状態を快適値に保とうとする本能

【人間模倣】
ホメオスタシス（Cannon, 1932）。生得的な生物の基本機能。
体は各状態の快適値（set point）に向かって戻ろうとする。

快適値から離れる → つらい（arousal上昇）
快適値に戻る → ほっとする（arousal低下）→ 報酬

この「ほっとした分」が報酬 r_home。
「まんま」が大事な音になるのは、それが切実な状態の解消に関わるから。

注意：【Tier3・2026-07-30 に判明】この式は先行研究の**縮退したケース**である。

    太郎     r = max(空腹, 眠気, 不快)_{t-1} − max(...)_t
    先行研究  r = β·(D_t − D_{t+1})、  D(s) = ‖s − s*‖ⁿ （通常 n=2）
             （Keramati & Gutkin のドライブ低減理論。
               Yoshida, Daikoku, Nagai, Kuniyoshi 2021/2024 が使用）

    先行研究の具体形（Yoshida, Arikawa, Kanazawa, Kuniyoshi 2024,
      PNAS Nexus 3(12): pgae540 ── 2026-07-30 に全文を確認）：

        R    = α(d_t − d_{t+1}) − C          α = 100
        d    = s_青² + s_赤²                  （目標点は原点・二乗距離）
        C    = k_p‖姿勢 − 目標姿勢‖² + k_u‖トルク‖²
               k_p = 0.005（姿勢のコスト）、k_u = 0.0005（トルクのコスト）

      重みを振るときの形（そのまま太郎に使える）：
        d = (2w_赤/(w_青+w_赤))·s_赤² + (2w_青/(w_青+w_赤))·s_青²
        比率 1:1 / 2:1 / 4:1 / 8:1 / 16:1 の5条件で行動が連続的に変わる
      注意：太郎にはコスト項 C（姿勢・トルク）が無い＝「無理な姿勢・動きすぎ」が罰されない

    違い（どれも太郎が意図して選んだものではなく、単に単純に作った結果）：
      ・集約が **max**（最も辛い1つが支配＝ボトルネック型・非可微分）
        先行研究は**二乗距離**（全次元が同時に効く・滑らか）
      ・目標点（set point）が暗黙の0で**片側だけ**
        ＝「多すぎ」（食べ過ぎ・寝過ぎ）のコストが表現できない
      ・スケーリング定数（β）が無い（係数1に固定）
        ＝恒常性報酬と学習進度報酬の相対的な重みを振れない
      ・複数のドライブの**重みを操作できない**
        先行研究では重みを振ると行動が連続的に変わることが主要な結果になっている
        （Yoshida et al. 2024, PNAS Nexus 3(12)：栄養バイアスを1:1〜16:1で振る）

    ⇒ 宿題：D を「重み付き二乗距離＋set point」に一般化する。
      実装コストは低く、「空腹と眠気のどちらを優先するか」を振る実験ができるようになる。
      → doc/やることリスト.md ／ doc/人間模倣からの逸脱リスト.md
"""


class Homeostasis:
    """
    恒常性の本能。arousalの変化から報酬を計算する。
    """

    def __init__(self):
        self.prev_arousal = 0.0

    def compute_reward(self, current_arousal):
        """
        arousalが下がった分だけ正の報酬。上がった分だけ負の報酬。

        r_home = prev_arousal - current_arousal
        下がった → 正（ほっとした）
        上がった → 負（つらくなった）
        変わらない → 0
        """
        r_home = self.prev_arousal - current_arousal
        self.prev_arousal = current_arousal
        return r_home
