"""把握反射（palmar grasp reflex）＝脊髄反射弓。

【解剖学的位置づけ、研究日誌続き16〜17】運動野(motor_cortex.py)でも脊髄CPG(cpg.py)でも
なく、第三の回路＝反射弓。学習しない・自発的でもない、刺激(接触)への固定配線の応答。

【文献根拠】
- 触れると指が屈曲する反射は新生児期から存在する[Tier1]（StatPearls "Grasp Reflex"；
  Cleveland Clinic "Newborn Reflexes"；Wikipedia "Palmar grasp reflex"）
- 生後3-6ヶ月にかけて消えていく理由＝皮質脊髄路の成熟による抑制、という一般原理[Tier1〜2]
  （原始反射は回路が壊れるのでなく、皮質からの抑制が強まって隠れる）
- 誘発刺激は本来「撫でる」動きに近いとの記述もあるが[要出典]、太郎では静的接触の閾値判定で
  簡略化[Tier3・ARBITRARY]

【消え方の設計】新しい独立の発達スケジュールは作らず、既存の corticospinal.py の
w_mean（developmental_schedule.py）に**逆連動**させる：
    reflex_strength = 1 - w_mean(age)
w_meanが0(writhing)のとき反射は最大、w_meanが上がる(fidgety〜E1)につれて反射は自然に弱まる。
これは「皮質脊髄路の成熟が反射を抑制する」という同じ生物学的原理の再利用であり、
GENAMPのような根拠ゼロの独立パラメータとは異なる。

【恣意的な部分】TOUCH_THRESHOLD・GRASP_TARGETの具体的な数値に文献根拠なし[Tier3・ARBITRARY]。

【トリガー範囲の訂正、2026-07-23続き7】当初は手のひら+全指+親指を接触判定に含めていたが、
文献で「反射は手のひら刺激への空間的特異性を示す」「注目すべきは、親指はこの反射の対象外」
と判明[Tier1]（ScienceDirect Topics "Palmar Grasp Reflex"、PMC3384944）。
  - トリガー(接触判定) = 手のひら(hand)のみ。指は含めない。
  - 反応(曲げる指)     = 4指(ff/mf/rf/lf)のみ。親指(thumb)は除外。
機序：手のひらの腱刺激→正中神経・尺骨神経経由の脊髄反射。「掴み続ける(clinging)」は
指の腱への軽い牽引(traction)への固有感覚反応という2段階構造だが、太郎では簡略化し
1段階（接触→屈曲）のみ実装[Tier3・簡略化]。
"""
import numpy as np

TOUCH_THRESHOLD = 0.05   # [Tier3・ARBITRARY] 接触判定の閾値。文献に具体的な力の値なし
GRASP_TARGET = 0.9       # [Tier3・ARBITRARY] 指を曲げる方向への目標値(action space[-1,1])


class GraspReflex:
    """model から手のひらのbody id・4指(親指除く)のアクチュエータidxを名前ベースで検索し、
    毎stepの触覚出力から「握る」命令を計算する。"""

    def __init__(self, model):
        self.touch_bodies = {"right": [], "left": []}   # トリガー＝手のひらのみ
        for i in range(model.nbody):
            name = model.body(i).name
            for side in ("right", "left"):
                if name == f"{side}_hand":
                    self.touch_bodies[side].append(i)

        self.flex_actuators = {"right": [], "left": []}   # 反応＝4指のみ、親指は除外
        for i in range(model.nu):
            name = model.actuator(i).name  # 例: "act:right_ff_knuckle"
            for side in ("right", "left"):
                if f"act:{side}_" in name and any(
                        f"_{finger}_{joint}" in name
                        for finger in ("ff", "mf", "rf", "lf")   # thumbは対象外[Tier1]
                        for joint in ("knuckle", "middle", "distal")):
                    self.flex_actuators[side].append(i)

    def _contact(self, sensor_outputs, side):
        total = 0.0
        for bid in self.touch_bodies[side]:
            arr = sensor_outputs.get(bid)
            if arr is not None and arr.size:
                total += float(np.abs(arr).sum())
        return total > TOUCH_THRESHOLD

    def apply(self, action, sensor_outputs, reflex_strength):
        """action(numpy, shape=(n_act,))に反射を重ねて返す。reflex_strength<=0なら無変更
        （E_GRASP_REFLEX=0の既定では呼ばれないので、常時1バイト差なしが保証される）。"""
        if reflex_strength <= 0:
            return action
        a = action.copy()
        for side in ("right", "left"):
            if self._contact(sensor_outputs, side):
                for i in self.flex_actuators[side]:
                    a[i] = (1 - reflex_strength) * a[i] + reflex_strength * GRASP_TARGET
        return a
