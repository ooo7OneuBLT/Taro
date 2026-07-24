"""口まわりの原始反射：探索反射（rooting）と吸啜反射（sucking）。

【なぜ作るか＝発達順の是正】
E1は当初 hand regard（手を見つめる）を目標にしたが、文献調査で**その手前に
「手-口協応」がある**と判明した（[逸脱リスト 2026-07-20 その5]）：
  妊娠10週で手が顔・口へ／19週で「手が来る前に口が開く＝予期」／視覚的誘導を必要としない。
＝太郎は「手を口に運べもしない段階」で「手を見つめさせよう」としていた（目標Dと同じ
  発達順の飛ばし）。手-口協応は **視覚が要らない・触覚で信号が立つ・予測の成立が
  行動で見える** ので、いま詰まっている「視覚信号が弱い」問題を回避できる。

【2つの反射（いずれも確立した原始反射。恣意的に足した本能ではない）】
 探索反射 rooting  ：口・頬まわりに触れると、その方向へ顔を向けて口を刺激源に近づける。
 吸啜反射 sucking  ：口に何か入ると吸う。吸啜そのものが**栄養と無関係に自律神経を鎮める**
   （疑核＝nucleus ambiguus が吸啜運動と心臓の迷走神経支配を共有＝解剖学的事実。
    非栄養的吸啜(non-nutritive sucking)がNICUで鎮静に使われる。[参考文献リスト §目標E-21]）。

【⚠️MIMoの身体的制約による重大な簡略化＝逸脱】
 MIMoには**口・顎・唇の関節が存在しない**（頭は swivel/tilt/tilt_side の3自由度だけ。
 実測で確認）。＝「吸う」という律動運動を物理的に出せない。
 → **吸啜反射は運動としては再現せず、「口が接触している＝吸っている状態」という
   内部状態として扱う**。その状態が discomfort（不快）を下げる。運動としての吸啜律動、
   舌・顎の動き、CPG(吸啜中枢パターン発生器)は再現しない。[逸脱リストに記録]

【探索反射の実装形式】
 VOR（e_vor.py）と同じく **action override**（方策の首出力を反射で差し替える）。
 反射弓は脳幹で完結し皮質（方策）を経由しない、という解剖学に沿う。
 口中心から接触点がずれた向き(y=左右, z=顎-頭頂)に応じて、首(swivel/tilt)へトルクを出す。
 ⚠️ゲイン ROOT_GAIN は恣意的（速度でなくトルク割合。接触方向へ向く十分な大きさに置く）。

【口センサーの決め方】
 MIMoの頭には触覚センサーが約300点あり、各点は頭ローカル座標を持つ。そのうち
 **顔前面・目より下・中央** の領域を「口周辺(perioral)」とする（可視化で確認済み：
 目の直下から顎にかけての中央領域、23点前後）。インデックス決め打ちでなく**座標条件で
 実行時に選ぶ**ので、体型補正(E_LEG_SCALE等)で点数が変わっても追従する。
 ⚠️唇だけを分離する解剖学的マーカーはMIMoに無いので、頬〜口の広い perioral 野になる。
   rooting は頬刺激でも起こるので機能的には妥当だが、「口」より広いことは明記する。
"""
import numpy as np


# --- 口周辺センサーの座標条件（頭ローカル。顔前面=+x, 左右=y, 顎<->頭頂=z）---
# 目は頭ローカルで (x=0.052, |y|=0.018, z=0.050)。口はその下・顔前面・中央。
MOUTH_X_MIN = 0.025        # これより前面（顔の表側）
MOUTH_Z_MAX = 0.045        # 目(z=0.05)より下
MOUTH_Z_MIN = -0.005       # 顎より上（首まで含めない）
MOUTH_Y_ABS = 0.028        # 中央（左右の頬の外側まで行かない）

ROOT_GAIN = 8.0            # 探索反射：口中心からのずれ(m)→首トルク割合。⚠️恣意的
SUCK_DISCOMFORT_DROP = 0.02   # 吸啜1秒あたりの discomfort 低下量。⚠️恣意的（迷走神経鎮静の代理）
CONTACT_FORCE_MIN = 1e-3   # これ以上の力で「接触」とみなす（ノイズ床）


def find_mouth_sensor_idx(model, data, touch):
    """頭の触覚センサーのうち、口周辺(perioral)に該当する点のインデックスを返す。

    座標条件で選ぶので体型が変わっても追従する（インデックス決め打ちをしない）。
    戻り値: head の sensor_positions/sensor_outputs 内の行インデックス配列。
    """
    hb = int(model.body("head").id)
    sp = touch.sensor_positions.get(hb)
    if sp is None or sp.shape[0] == 0:
        return np.zeros(0, dtype=int), hb
    idx = np.where(
        (sp[:, 0] > MOUTH_X_MIN)
        & (sp[:, 2] < MOUTH_Z_MAX) & (sp[:, 2] > MOUTH_Z_MIN)
        & (np.abs(sp[:, 1]) < MOUTH_Y_ABS)
    )[0]
    return idx, hb


class OralSystem:
    """口まわりの反射をまとめて扱う。HybridEnv から optional に注入して使う。

    毎ステップ：
      1. mouth_contact() で口周辺の接触力と接触点の重心を読む
      2. rooting_override(action) で首出力を接触方向へ差し替える（VORの後・step前）
      3. sucking_discomfort_drop() で「吸っている間 discomfort を下げる量」を返す
         （HybridEnv が internal_state.discomfort に適用）
    """

    def __init__(self, model, data, touch):
        self.mouth_idx, self.head_bid = find_mouth_sensor_idx(model, data, touch)
        self.touch = touch
        # ⚠️2026-07-21修正：内臓(discomfort)は100物理stepに1回しか進まない
        # （HybridEnv.STEPS_PER_BODY_SECOND）。sucking_discomfort_drop()をその瞬間だけ
        # 呼ぶと、99step接触して100step目に離れただけで**接触ごと見逃す**（hand regardの
        # 視覚信号が1〜2tickで消えて埋もれた問題と同じ構造）。
        # → poll()を**毎物理step**呼び、1秒の間に一度でも接触したかをORで貯める。
        self._contact_this_window = False
        # 口の中心（頭ローカル）＝口周辺センサーの重心。rooting のずれ基準に使う。
        sp = touch.sensor_positions.get(self.head_bid)
        self.mouth_center = (sp[self.mouth_idx].mean(axis=0)
                             if len(self.mouth_idx) else np.zeros(3))
        # 首アクチュエータ id（swivel=左右, tilt=上下）
        self.act_swivel = self._find_act(model, "head_swivel")
        self.act_tilt = self._find_act(model, "head_tilt")

    @staticmethod
    def _find_act(model, key):
        for i in range(model.nu):
            if key in model.actuator(i).name:
                return i
        return None

    # --- 接触の読み取り -------------------------------------------------
    def mouth_contact(self):
        """口周辺の (接触しているか, 力の合計, 接触点の重心[頭ローカル])。"""
        if len(self.mouth_idx) == 0:
            return False, 0.0, self.mouth_center
        forces = self.touch.sensor_outputs.get(self.head_bid)   # (N,3)
        if forces is None:
            return False, 0.0, self.mouth_center
        f = np.linalg.norm(np.asarray(forces)[self.mouth_idx], axis=1)   # 各点の力の大きさ
        total = float(f.sum())
        if total < CONTACT_FORCE_MIN:
            return False, total, self.mouth_center
        # 力で重み付けした接触重心（どこに当たっているか）
        sp = self.touch.sensor_positions[self.head_bid][self.mouth_idx]
        w = f / (f.sum() + 1e-12)
        centroid = (sp * w[:, None]).sum(axis=0)
        return True, total, centroid

    # --- 探索反射（rooting）：首を接触方向へ ---------------------------
    def rooting_override(self, action):
        """接触点が口中心からずれた向きへ首を回す指令を action に上書きして返す。

        接触が無ければ首出力はそのまま（方策の値を残す）。
        ⚠️符号（どちらのトルクで顔がどちらを向くか）は実測で合わせる前提。
        """
        contact, _, centroid = self.mouth_contact()
        out = np.array(action, dtype=float).copy()
        if not contact:
            return out
        dy = float(centroid[1] - self.mouth_center[1])   # 左右のずれ
        dz = float(centroid[2] - self.mouth_center[2])   # 顎<->頭頂のずれ
        if self.act_swivel is not None:
            out[self.act_swivel] = float(np.clip(ROOT_GAIN * dy, -1.0, 1.0))
        if self.act_tilt is not None:
            out[self.act_tilt] = float(np.clip(ROOT_GAIN * dz, -1.0, 1.0))
        return out

    # --- 吸啜反射（sucking）：接触中は discomfort を下げる -------------
    def poll(self):
        """毎物理stepごとに呼ぶ。1秒(=100step)の間に一度でも接触したかをORで記録する。

        瞬間的な接触（1〜2stepだけ触れてすぐ離れる）を、100stepに1回のサンプリングで
        取りこぼさないための仕組み。rooting_override とは別に、discomfort用にここで
        独立して監視する（rootingは"今この瞬間"の接触方向に首を向けるだけでよいが、
        discomfortは"この1秒間に起きたか"を漏れなく知る必要があるため）。
        """
        contact, _, _ = self.mouth_contact()
        if contact:
            self._contact_this_window = True

    def sucking_discomfort_drop(self):
        """直近1秒の間に一度でも口が接触した（＝吸っていた）なら下げる discomfort 量（>=0）。

        MIMoに口関節が無いため運動は出さず、接触を吸啜状態とみなす（逸脱）。
        poll()で貯めたフラグを消費してリセットする（次の1秒分をまた貯め直す）。
        """
        occurred = self._contact_this_window
        self._contact_this_window = False
        return SUCK_DISCOMFORT_DROP if occurred else 0.0
