"""伸張反射（脊髄反射弓）。新しい駆動モジュール（伸張反射＋揺らぐ振動子の共通駆動）の下位層。

【なぜ、2026-08-11】既存の自発運動生成（`spinal_cord/cpg.py` の `ColoredNoiseGenerator`）は
色付きノイズを関節トルクに直接流す近似で、人間の実際の機構（脊髄反射＋リズム生成の2層）を
飛ばしている、という課題が設計（白紙設計・案①）で指摘された。ここはその下位層＝
伸張反射（腱反射・筋紡錘を介した脊髄反射弓）の実装。上位層（リズム生成・基準長を動かす）は
`common_drive.py`。

【cpg.py とは無関係】このファイルは `spinal_cord/cpg.py` を一切importしない
（`ColoredNoiseGenerator`・`CPG`・`antagonist_map`・`write_joint_command`・
`is_antagonist_action` のいずれも使わない、独立実装。設計のユーザーからの修正指示
「変更3」）。

【根拠ラベル】
    伸張反射の存在そのもの                 [Tier1・確立した生理学]
    動的成分（速度に応じた反射）の存在      [Tier1・教科書的に確立]
    新生児での定量的なゲイン(k_s, k_v)      文献に存在しない[Tier3・工学的判断]
（設計：作業記録（非公開） 4節）

【qpos_springの二重利用の罠を踏まない】基準長L0は `model.qpos_spring` を一切読み書きせず、
このクラスが**独立にPython側の配列として**持つ（案①4-4節の罠の回避策。必ず守る）。
"""
import numpy as np

# ゲインの初期値（すべて [Tier3・工学的判断、恣意的]）。
#
# 【なぜこの値か】新生児での伸張反射の定量的なゲインは文献に存在しない
# （案①2-1節が既に調査済み）。`tone_from_gravity`（`infant_limbs.py`）と同種の考え方
# ＝「機能から逆算する」ほうが勘で決めるより検証しやすい、という方針を踏まえたいが、
# 今回のスコープでは「較正の余地を残す」だけで十分（実装への仕様より）。
# 感度分析（実際に値を振って達成条件がどう変わるかを見る）は測定が担当する。
#
# 単位の目安：muscle_lengths（lce）はMuscleModelの仮想筋長でおおむね[0.75, 1.05]の
# 範囲（`MuscleModel.lce_min/lce_max`）。deadzoneを超える伸びに対して活性化が
# 緩やかに立ち上がるよう、k_sはこの範囲の1割程度の伸びで活性化がおおよそ0.5になる
# 大きさ（=1/0.1=10のオーダー）を出発点として、やや控えめにDEFAULT_K_S=5.0とした。
# k_vは肘の反跳（e_arm_recoil_test.py）程度の動的速度（lce_dotでおおむね数割/秒）で
# 過大な急停止を起こさない程度にDEFAULT_K_V=0.5とした。いずれも実測較正はしていない。
DEFAULT_K_S = 5.0
DEFAULT_K_V = 0.5
DEFAULT_DEADZONE = 0.005
DEFAULT_R_MAX = 1.0


class StretchReflex:
    """拮抗筋2本（neg=[:n]・pos=[n:]）それぞれについて、伸びている量・速さに応じた
    活性化を計算し、探索側が出した命令（行動配列）に加算する反射弓。

    入力（毎tick）：
        muscle_lengths      lce、2n次元。env.unwrapped.actuation_model.muscle_lengths
        muscle_velocities   lce_dot、2n次元。同 .muscle_velocities
        基準長 L0（2n次元）。このクラスが内部のPython配列として独立に持つ
            （model.qpos_spring は一切読み書きしない）。

    計算式（案①4-3節、そのまま）：
        r_m = clip(k_s * relu(lce_m - L0_m - deadzone) + k_v * relu(lce_dot_m), 0, r_max)
        ＝伸びている方向にだけ反応する（relu）。静的成分(k_s)＋動的成分(k_v)。

    出力：
        探索側の命令（行動配列、2n次元）に**加算**する：
        action_m' = clip(action_m + r_m, 0, 1)

    拮抗筋の対応：muscle_lengthsの並び（neg=[:n], pos=[n:]）は行動配列の並びと
    一致している（is_antagonist_action済みの2n次元配列を前提にしてよい。
    cpg.pyのdocstring参照。ただしこのクラス自体はcpg.pyをimportしない）。
    """

    def __init__(self, n_joint, k_s=DEFAULT_K_S, k_v=DEFAULT_K_V,
                 deadzone=DEFAULT_DEADZONE, r_max=DEFAULT_R_MAX):
        self.n_joint = int(n_joint)          # 関節数(=n)。行動配列は2n次元
        self.k_s = float(k_s)
        self.k_v = float(k_v)
        self.deadzone = float(deadzone)
        self.r_max = float(r_max)
        self.L0 = None                       # (2n,) 未初期化。init_reference/set_referenceで与える

    def init_reference(self, muscle_lengths):
        """env.reset()直後のmuscle_lengthsを、そのままL0として採用する（案①4-2節・既定）。"""
        arr = np.asarray(muscle_lengths, dtype=np.float64)
        if arr.shape[0] != 2 * self.n_joint:
            raise ValueError(
                f"StretchReflex.init_reference: muscle_lengthsの次元が合わない "
                f"({arr.shape[0]} != {2 * self.n_joint})")
        self.L0 = arr.copy()

    def set_reference(self, new_L0):
        """基準長L0を書き換える（上位層＝共通駆動が毎tick呼ぶことでL0を動かす）。"""
        arr = np.asarray(new_L0, dtype=np.float64)
        if arr.shape[0] != 2 * self.n_joint:
            raise ValueError(
                f"StretchReflex.set_reference: new_L0の次元が合わない "
                f"({arr.shape[0]} != {2 * self.n_joint})")
        self.L0 = arr

    def compute(self, muscle_lengths, muscle_velocities):
        """反射の活性化 r（2n次元）を計算する（行動配列にはまだ加算しない）。"""
        if self.L0 is None:
            raise RuntimeError(
                "StretchReflex.L0が未初期化。init_reference()かset_reference()を先に呼ぶこと。")
        lce = np.asarray(muscle_lengths, dtype=np.float64)
        lce_dot = np.asarray(muscle_velocities, dtype=np.float64)
        static = self.k_s * np.maximum(lce - self.L0 - self.deadzone, 0.0)
        dynamic = self.k_v * np.maximum(lce_dot, 0.0)
        return np.clip(static + dynamic, 0.0, self.r_max)

    def apply(self, action, muscle_lengths, muscle_velocities):
        """反射出力を行動配列(2n次元、[0,1])に加算し、クリップして返す（案①4-3節）。

        呼び出し側の配列を書き換えず、新しい配列を返す（副作用なし）。
        """
        r = self.compute(muscle_lengths, muscle_velocities)
        a = np.asarray(action, dtype=np.float64) + r
        return np.clip(a, 0.0, 1.0)
