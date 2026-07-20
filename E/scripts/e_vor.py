"""VOR（前庭動眼反射）：眼球を方策から切り離し、頭の動きを打ち消して視線を安定させる。

【なぜ作るか＝実測で判明した構造的な逸脱】
太郎は**眼球の可動域をほぼ100%振り回していた**：
    left_eye_horizontal  使用92.8° / 可動域90°   |w|mean 0.37 rad/s
    left_eye_vertical    使用82.3° / 可動域80°
    右眼も同様。**角速度は頭(0.32)より速い(0.39 rad/s)**
    視線方向の変化 = 物理1ステップあたり平均0.545°
原因＝方策の探索ノイズが**90アクチュエータ全部（眼球6個を含む）に一律**にかかっていたこと。

【人間ではどうか（文献）】
 ・General Movements（＝運動性喃語）の定義は「**首・腕・体幹・脚**の運動」＝**眼球は含まれない**。
 ・眼球運動は別系統で、すべて機能を持つ（VOR／サッケード／追視／固視）。
 ・**VORは三ニューロン弓**（前庭神経節→前庭核→動眼・滑車・外転神経核）＝**脳幹で完結し
   大脳皮質を経由しない**。「随意眼球運動とVORは最終的に同じ外眼筋運動核（final common
   pathway）に収束するが、**異なる皮質・皮質下システムに由来し、初期処理には別個の脳幹
   前運動回路を使う**」。
 → **方策（＝皮質相当）が眼球を動かす太郎の構造は、解剖学的に誤り**。方策から外すのが正しい。

【利得の根拠＝実測値がある（恣意的な決め打ちではない）】
   Finocchio, Preston & Fuchs (1991) *Vision Research* 31(10):1717-1730（PMID 1767493）:
   「暗所のVOR利得の平均は **1〜4ヶ月児で 1.03 ± 0.014**、**成人で 0.59 ± 0.03**」
   ＝**乳児のVORは成人の約1.7倍強い**。さらに同論文は「視覚刺激は成人のVORを増強したが
   乳児の利得には効果が無く、**乳児は視覚補償でなく生来の高いVORに強く依存している**」と
   報告している＝新生児にVORを入れる根拠は強い。生後4ヶ月間で有意な変化なし。

【E1では入れないもの（根拠が無いため）】
 ・サッケード／固視：**視覚処理（対象の定位）が前提** → 視覚を扱う段階で検討。
 ・スムーズパージュート：**6週から発達＝新生児には無い**（von Hofsten）。

【実装形式＝速度サーボ（第1版の誤りと訂正）】
第1版は ctrl に「目標角度」を書いたが**効かなかった**（実測：眼球の可動域使用 92.8°→91.6° と
ほぼ不変、視線の揺れ 0.5446°→0.5632°/step とむしろ悪化、最大角速度 4.15→11.24 rad/s と増加）。
原因は MIMo の `SpringDamperModel` が**トルクモーター**であること：
  「MIMoの筋はトルクモーターで表現され、**制御入力1が最大トルクを表す**」
  `simulation_torque() = actuator_gear × control_input`
＝ctrl は角度ではなく**最大トルクに対する割合(−1〜1)**。角度を書いたので「最大トルクの0.8倍で
眼を回し続ける」動作になっていた。
→ 本版は**速度フィードバック**でトルクを作る：
     ω_desired = −gain × ω_head(当該軸)      ← VORの本体
     ctrl = clip( Kv × (ω_desired − ω_eye) , −1, 1 )
生理的にも、VORは前庭の**速度信号**が外眼筋を駆動する速度制御であって位置指定ではない。

【⚠️簡略化・恣意的な部分（ラベリング対象）】
 ・**Kv（速度フィードバックゲイン）は恣意的**。VOR利得1.03（実測値）とは別物で、トルクと
   速度をつなぐ単位変換に相当する。速度追従が成立する十分な大きさに置いている。
 ・新生児で報告される「前庭の時定数が成人の約1/2」「急速相(quick phase)が稀」「低周波での
   位相ずれ>20°」は**再現していない**＝**利得のみの実装**。
 ・半規管のモデル化は省略し、**頭部bodyの角速度を直接**用いる（前庭器官の動特性は無視）。
"""
import numpy as np

VOR_GAIN = 1.03            # 1〜4ヶ月児の実測値（Finocchio et al. 1991。成人は0.59）
# Kv＝速度誤差→トルク割合の変換ゲイン。**スイープの実測で選んだ**（勘や目視ではない）：
#   Kv    実効利得  視線の揺れ/step  眼球|w|max
#   1.0    0.597     0.2923          15.46
#   2.0    0.583     0.2426           2.37
#   5.0    0.641     0.2536           3.05   ← 採用（利得が最大かつ暴れない）
#  10.0    0.638     0.2406           3.86
#  20.0    0.549     0.2204           9.68
#  50.0    0.593     0.2267           2.27
# ⚠️どのKvでも実効利得は0.55〜0.64で頭打ちし、**目標1.03に届かない**。
#   原因は眼筋の弱さではなく**トルク飽和**（必要な制御入力が|ctrl|≥1になるのが29.7%のtick、
#   平均1.388・p95 5.89）。さらにその飽和の原因は**太郎の頭が動きすぎていること**
#   （頭部角速度 mean 0.32〜0.41 rad/s ＝18〜23°/秒で動き続ける）。
#   実際の新生児は「安静時は頭を片側に向けたまま・限られた振幅・遅い速度」なので、
#   人間の眼筋でも同じ勢いで頭が振れ続ければ飽和する。
#   → **眼筋を強化して1.03に合わせるのは原因を隠す対症療法なので採らない**。
#   頭の運動が人間的になれば利得は自然に1.03へ近づくはず＝そのとき測り直す。
VOR_KV = 5.0
EYE_KEY = "eye"            # 眼球のジョイント/アクチュエータ名に含まれる語


class VOR:
    """頭部角速度を打ち消す向きに眼球を動かし、方策の眼球出力を無効化する。

    ⚠️方策の出力次元は変えない（受け取るが眼球には流さない）。次元を変えると
    C5の学習済みモデル（n_actuators=90）が読めなくなるため。
    """

    def __init__(self, model, data, gain=VOR_GAIN, kv=VOR_KV):
        self.gain = float(gain)
        self.kv = float(kv)
        self.head_bid = int(model.body("head").id)
        self.units = []          # (actuator_id, joint_id, eye_body_id, axis, ctrl_lo, ctrl_hi)
        for i in range(model.nu):
            name = model.actuator(i).name
            if EYE_KEY not in name:
                continue
            jid = int(model.actuator_trnid[i, 0])
            if jid < 0:
                continue
            self.units.append(dict(
                aid=i, jid=jid,
                bid=int(model.jnt_bodyid[jid]),
                axis=np.array(model.jnt_axis[jid], dtype=float),
                dofadr=int(model.jnt_dofadr[jid]),
                lo=float(model.actuator_ctrlrange[i, 0]),
                hi=float(model.actuator_ctrlrange[i, 1]),
                jlo=float(model.jnt_range[jid, 0]),
                jhi=float(model.jnt_range[jid, 1]),
                qposadr=int(model.jnt_qposadr[jid]),
            ))

    def reset(self):
        pass

    def _head_omega_world(self, data):
        """頭部の角速度（ワールド基準）。cvelは[角速度3, 線速度3]の順。"""
        return np.array(data.cvel[self.head_bid][:3], dtype=float)

    def override(self, action, model, data, dt):
        """方策の行動から眼球ぶんを取り除き、VORの指令に差し替えて返す。"""
        if not self.units:
            return action
        out = np.array(action, dtype=float).copy()
        w_world = self._head_omega_world(data)
        for u in self.units:
            # 眼球ボディのローカル軸へ頭部角速度を投影
            R = np.array(data.xmat[u["bid"]], dtype=float).reshape(3, 3)
            w_axis = float(np.dot(R.T @ w_world, u["axis"]))
            w_des = -self.gain * w_axis            # 頭と逆向き・同じ速さ＝視線を空間に固定
            w_eye = float(data.qvel[u["dofadr"]])  # 今の眼球角速度
            torque_ratio = self.kv * (w_des - w_eye)
            # 可動域の端では戻す向きにだけ力を出す（端に押し付け続けない）
            ang = float(data.qpos[u["qposadr"]])
            if ang >= u["jhi"] and torque_ratio > 0:
                torque_ratio = 0.0
            elif ang <= u["jlo"] and torque_ratio < 0:
                torque_ratio = 0.0
            out[u["aid"]] = float(np.clip(torque_ratio, u["lo"], u["hi"]))
        return out

    def gaze_axis(self, model, data, camera="eye_left"):
        """今の視線方向（カメラのz軸）。揺れの測定用。"""
        cid = int(model.camera(camera).id)
        return np.array(data.cam_xmat[cid], dtype=float).reshape(3, 3)[:, 2].copy()
