"""立ち直り反射・層2＝角速度ダンパー（VCR風の連続的バイアス）。

【なぜこの形か、2026-08-15】
2026-07-28に一度「立ち直り反射」（傾いた角度をゆっくり戻し続ける位置サーボ）を
検討し保留にした（`doc/人間模倣からの逸脱リスト.md` その10）。理由：
    Goldberg & Cullen (2011) Exp Brain Res 210:331-345（成人ヒトVCRの実測）
        1Hz未満（ゆっくりした傾き）  利得ほぼゼロ
        1〜3Hz（速い揺れ）           ここだけダンパーとして効く
        3Hz以上                      頭自体の慣性で安定
＝「傾いた角度に比例して戻す」位置サーボは実測と逆向き。
その10の結論：「独立回路のON/OFFではなく、体格・筋力・支持面・課題に応じて
強度が変わる連続的なバイアスとして設計すべき」。

【今回どう満たしたか＝位置サーボになっていないことの確認】
このクラスの入力は角度ではなく**角速度**（head_angular_velocity）のみで、
角度・傾き量は一切受け取らない。式は
    bias = clip(k_d0 * (1 - support_fraction) * head_angular_velocity, -bias_max, bias_max)
であり、角速度に比例するダンパーである。
    ・ゆっくりした傾き（角速度が小さい）→ bias も自然に小さい
      （Goldberg & Cullenの「1Hz未満はゲインほぼゼロ」と定性的に整合する近似）
    ・速い揺れ（角速度が大きい）→ bias が大きく効く
      （「1〜3Hzだけダンパーとして効く」と定性的に整合する近似）
    ・support_fraction（体幹の外部支持割合）が高いほど (1 - support_fraction) が
      小さくなり、支えが多いほど反射が弱まる＝「連続的なバイアス」というその10の
      要求を満たす（独立回路のON/OFFではない。support_fractionは0/1でなく
      連続値として扱える設計）。

【既知の限界（コメントと報告の両方に明記する指示）】
Goldberg & Cullen (2011) の「3Hz以上は頭自体の慣性で安定する」という高周波側の
飽和は、この式では**表現されない**。角速度に比例するダンパーは角速度が
大きいほど際限なく（bias_maxまで）増え続けるので、高周波領域での頭部慣性による
自然な頭打ちを近似できていない。bias_maxによるクリップは「暴走を防ぐための
工学的な上限」であり、「3Hz以上での慣性由来の安定化」を模した機構ではない。
このクラス単体でこの限界を解消することはできない（上位で角速度に周波数依存の
フィルタをかけるなどの拡張が必要）。今回のスコープでは、この限界を明示した
うえで単純な角速度比例ダンパーとして実装する。

【前庭覚由来の頭角速度の取得方法】
`taro_core/src/senses/semicircular_canals.py` の `SemicircularCanals` が、
三半規管の高域通過特性（ゆっくりした回転は「感じなくなる」）を経た角速度
`sensed`（rad/s、3軸）を返す実装として既に存在する（`E/scripts/e_vor.py` の
VORクラスも同じ経路を使っている）。本クラスの `compute()` は
`head_angular_velocity`（deg/s、スカラー・対象1軸ぶん）を引数として受け取る
だけで、`SemicircularCanals` 自体には依存しない（副作用なし・疎結合を保つ
ため。呼び出し側が `SemicircularCanals.update()` の出力から対象軸を
取り出し、rad/s→deg/sに変換してから渡す想定）。
    deg/s への変換：head_angular_velocity_dps = sensed_axis_rad_s * 180.0 / pi
なぜ三半規管（高域通過後）の値を使うべきか：三半規管を通さない生の角速度を
使うと、頭がゆっくり傾き続けても「回り続けても永久に感じ続ける」ことになり
（`semicircular_canals.py` のdocstring参照）、その10が問題にした「ゆっくりした
傾きに反応する」という誤りを別の場所で再現してしまう。三半規管の高域通過を
経た値を使うことで、ゆっくりした傾きは自然に減衰し、位置サーボ的な挙動を
避けられる。

【support_fractionの取得方法】
`run/scene_tools/e_scene.py` の `constraint_summary(scene)`（副作用なし）は
    {"root_pinned": bool, "pinned_groups": [...], "free_groups": [...], ...}
を返す。`pinned_groups` は「椅子・実験者の手などで固定されている体の部位」
（`_repin()` のコメント「首は実験者の手が担当」「体幹・脚も完全固定する」を
参照）に対応するため、支持の割合として妥当な代理指標になる。
taro_core は run 配下に依存しない設計（呼び出し方向を run→taro_core に保つ）
なので、本ファイルは `e_scene.py` を import しない。代わりに
`support_fraction_from_pinned_groups()` という純粋関数を用意し、呼び出し側が
`constraint_summary(scene)["pinned_groups"]` をそのまま渡せるようにする。

【根拠ラベル】
    VCR（前庭頸反射）の周波数特性そのもの        [Tier1・Goldberg & Cullen 2011]
    「角速度比例ダンパー」という近似の妥当性      [Tier3・工学的判断。上記の
                                                     限界（高周波飽和なし）を持つ近似]
    k_d0（ゲイン）・deadzone・bias_max の定量値   [Tier3・文献値なし、恣意的]
"""
import numpy as np

# ---- Tier3・恣意的パラメータ ------------------------------------------------
# 【なぜこの値か】新生児の立ち直りダンパーの定量的なゲインは文献に存在しない
# （その10の調査で迷路性/視性立ち直り反射のヒト一次文献にも到達できなかった）。
# stretch_reflex.py と同じ方針＝「較正の余地を残す」だけを目標に、出発点として
# 小さめの値を置く。感度分析はこのファイル内の _demo_sensitivity_sweep() で
# 単体で確認する（run/main.pyへの統合前）。実測較正はしていない。
DEFAULT_K_D0 = 0.02          # [Tier3] bias(無次元, [0,1]の行動配列に加算) / (deg/s)
DEFAULT_DEADZONE_DPS = 5.0   # [Tier3] 不感帯。三半規管通過後でも残る小さな揺れを無視する
DEFAULT_BIAS_MAX = 0.3       # [Tier3] 出力上限。stretch_reflexのr_max=1.0より控えめに
                              # している＝反射単独で行動配列を支配しないようにする安全側の値


ALL_SUPPORT_GROUPS = ("arm", "finger", "leg", "trunk", "head")


def support_fraction_from_pinned_groups(pinned_groups, all_groups=ALL_SUPPORT_GROUPS):
    """constraint_summary(scene)["pinned_groups"] から support_fraction[0,1] を作る。

    副作用なし。e_scene.py を import しない（taro_core は run 配下に依存しない）。
    体の部位（arm/finger/leg/trunk/head の5グループ）のうち、外部で固定されて
    いる（＝椅子・実験者の手で支えられている）割合をそのまま support_fraction
    とする。全く固定が無ければ0.0（反射が最大限効く）、全部固定なら1.0
    （反射がほぼ効かない＝ (1 - support_fraction) が0に近づく）。

    注意[Tier3・簡略化]：5グループを等しい重みで数えているだけで、
    「体幹の支持だけが本来重要」といった重み付けはしていない。
    """
    groups = set(all_groups)
    pinned = set(pinned_groups) & groups
    if not groups:
        return 0.0
    return len(pinned) / len(groups)


class RightingDamper:
    """VCR（前庭頸反射）の周波数特性に近似した、角速度比例の立ち直りダンパー。

    位置（傾き角）は一切扱わない。角速度のみを入力に取ることで、その10で
    問題になった「ゆっくりした傾きへの位置サーボ」を避けている。

    入力（毎tick）：
        head_angular_velocity   頭の角速度[deg/s]。三半規管（高域通過後）由来を推奨。
        support_fraction        体幹の外部支持割合[0,1]。
                                 support_fraction_from_pinned_groups() 参照。

    計算式（仕様どおり、変更しない）：
        bias = clip(k_d0 * (1 - support_fraction) * head_angular_velocity,
                     -bias_max, bias_max)
        （ただし |head_angular_velocity| < deadzone_dps のときは bias = 0）

    出力：
        探索側の命令（行動配列）に**加算**する（stretch_reflex.py と同じ
        「副作用なしで新しい配列を返す」構造）。層1（postural_gate、ゲート方式）
        とは異なり、層2は反射自体の強さが人間でもある程度固定的なダンパーとして
        働く性質のため、加算方式を採る（仕様の設計判断）。
    """

    def __init__(self, k_d0=DEFAULT_K_D0, deadzone_dps=DEFAULT_DEADZONE_DPS,
                 bias_max=DEFAULT_BIAS_MAX):
        self.k_d0 = float(k_d0)
        self.deadzone_dps = float(deadzone_dps)
        self.bias_max = float(bias_max)

    def compute(self, head_angular_velocity, support_fraction):
        """bias（スカラー、正負の向きを持つ）を返す。副作用なし。

        head_angular_velocityの符号がそのままbiasの符号になる＝「傾いている
        向きと逆に戻す」という向きの決定は呼び出し側（apply()、またはapply()を
        使わず直接この値を目的の筋へ配線する側）が担う。このクラス自体は
        「どちらの筋が伸筋でどちらが屈筋か」を知らない。
        """
        omega = float(head_angular_velocity)
        sf = float(np.clip(support_fraction, 0.0, 1.0))
        if abs(omega) < self.deadzone_dps:
            return 0.0
        bias = self.k_d0 * (1.0 - sf) * omega
        return float(np.clip(bias, -self.bias_max, self.bias_max))

    def apply(self, action, neg_indices, pos_indices, head_angular_velocity,
              support_fraction):
        """行動配列(1次元、[0,1]の要素を持つ)に bias を加算し、クリップして返す。

        neg_indices／pos_indices：頭・首を動かす筋のうち、負方向／正方向に
        働く筋のインデックス列（MuscleModelの拮抗筋の並びに対応。
        stretch_reflex.py の neg=[:n]/pos=[n:] と同じ考え方）。
        biasが正なら正方向筋（pos_indices）に、負なら負方向筋（neg_indices）に
        その絶対値を加算する（拮抗筋の片方だけを働かせる＝relu的な配分。
        stretch_reflex.py の書式には無いが、単一スカラーのbiasを2方向の
        筋へ配る必要があるための本ファイル独自の変換）。

        呼び出し側の配列を書き換えず、新しい配列を返す（副作用なし）。
        """
        bias = self.compute(head_angular_velocity, support_fraction)
        a = np.array(action, dtype=np.float64)
        pos_bias = max(bias, 0.0)
        neg_bias = max(-bias, 0.0)
        if pos_bias > 0.0:
            for idx in pos_indices:
                a[idx] = np.clip(a[idx] + pos_bias, 0.0, 1.0)
        if neg_bias > 0.0:
            for idx in neg_indices:
                a[idx] = np.clip(a[idx] + neg_bias, 0.0, 1.0)
        return a


def _demo_sensitivity_sweep():
    """k_d0・head_angular_velocity を複数水準で振り、compute()の出力biasを一覧化する。

    e_toy_env.py への統合前に、このファイル単体（run/main.pyを起動せずに）で
    感度を確認するための手順（ユーザーからの必須指示）。
    `python righting_damper.py` で直接実行できる。
    """
    k_d0_values = [0.01, 0.02, 0.05, 0.1]
    omega_values_dps = [0.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0, -30.0]
    support_fraction_values = [0.0, 0.5, 1.0]

    print("=== RightingDamper 感度確認（deadzone_dps=%.1f, bias_max=%.2f 固定） ==="
          % (DEFAULT_DEADZONE_DPS, DEFAULT_BIAS_MAX))
    header = "k_d0".rjust(8) + "support_frac".rjust(14)
    for w in omega_values_dps:
        header += ("omega=%.0f" % w).rjust(12)
    print(header)
    for k_d0 in k_d0_values:
        damper = RightingDamper(k_d0=k_d0)
        for sf in support_fraction_values:
            row = ("%.3f" % k_d0).rjust(8) + ("%.2f" % sf).rjust(14)
            for w in omega_values_dps:
                bias = damper.compute(w, sf)
                row += ("%.4f" % bias).rjust(12)
            print(row)
    print()
    print("=== deadzone(不感帯) の効きを確認する ===")
    for dz in [1.0, 5.0, 10.0, 20.0]:
        damper = RightingDamper(k_d0=0.02, deadzone_dps=dz)
        biases = [damper.compute(w, 0.0) for w in omega_values_dps]
        print(("deadzone_dps=%6.1f -> " % dz)
              + ", ".join("%.4f" % b for b in biases))


if __name__ == "__main__":
    _demo_sensitivity_sweep()
