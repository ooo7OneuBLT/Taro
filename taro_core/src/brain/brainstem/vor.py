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

【2026-07-26・上記の理解は半分だけ正しかった】
「トルクモーター」というのは MuscleModel が**内部でトルクモーターを流用している**という話
（`muscle.py:345` "We apply our muscle torque by reusing the torque motors in the simulation"）で、
**外から渡す行動配列の形式とは別物**だった。実際の形式は：
    MuscleModel … 行動 n_joint×2 次元・各要素 [0, 1]
                  前半 n_joint = 負方向筋 / 後半 n_joint = 正方向筋
つまり**符号で向きを表せない**。−1〜1 を1本の筋に書いていたので負の指令が消え、
**眼は片方向（下）にしか動けなかった**。実測で左目の上下角が1秒で下限 −47度に張り付き、
重力を切っても同じだった（`e_eye_drift_probe.py`）。
→ 現在は `taro_core/src/brain/spinal_cord/cpg.py` の `write_joint_command` を通している。
   中の実装の説明と、呼び出し側との約束を取り違えたのが原因（落とし穴 項50）。
→ 本版は**速度フィードバック**でトルクを作る：
     ω_desired = −gain × ω_head(当該軸)      ← VORの本体
     ctrl = clip( Kv × (ω_desired − ω_eye) , −1, 1 )
生理的にも、VORは前庭の**速度信号**が外眼筋を駆動する速度制御であって位置指定ではない。

【注意簡略化・恣意的な部分（ラベリング対象）】
 ・**Kv（速度フィードバックゲイン）は恣意的**。VOR利得1.03（実測値）とは別物で、トルクと
   速度をつなぐ単位変換に相当する。速度追従が成立する十分な大きさに置いている。
 ・新生児で報告される「前庭の時定数が成人の約1/2」「急速相(quick phase)が稀」「低周波での
   位相ずれ>20°」は**再現していない**＝**利得のみの実装**。
 ・半規管のモデル化は省略し、**頭部bodyの角速度を直接**用いる（前庭器官の動特性は無視）。
"""
import numpy as np

VOR_GAIN = 1.03            # 1〜4ヶ月児の実測値（Finocchio et al. 1991。成人は0.59）
# 【2026-07-26・以下のスイープ表と診断はすべて無効】
#   この測定は**眼が片方向にしか動けない状態**で行われた。反射が MuscleModel の形式で
#   行動を書いていなかったため、負の指令が clip(action, 0, 1) で消えていた
#   （落とし穴チェックリスト 項50、研究日誌 2026-07-26 続き7）。
#   ・「どのKvでも実効利得は 0.55〜0.64 で頭打ち」   → 無効。測り直すこと
#   ・「原因はトルク飽和（29.7%のtickで |ctrl|≥1）」 → 誤診の可能性が高い。
#       飽和ではなく、**指令の半分が捨てられていた**だけかもしれない
#   ・「眼筋を強化して1.03に合わせるのは対症療法だから採らない」
#       → この判断の土台（飽和の数字）が消えた。利得を測り直してから再検討する
#   Kv=5.0 は暫定的に残しているだけで、根拠は現時点で無い。
# ------------------------------------------------------------------------
# Kv＝速度誤差→トルク割合の変換ゲイン。**スイープの実測で選んだ**（勘や目視ではない）：
#   Kv    実効利得  視線の揺れ/step  眼球|w|max
#   1.0    0.597     0.2923          15.46
#   2.0    0.583     0.2426           2.37
#   5.0    0.641     0.2536           3.05   ← 採用（利得が最大かつ暴れない）
#  10.0    0.638     0.2406           3.86
#  20.0    0.549     0.2204           9.68
#  50.0    0.593     0.2267           2.27
# 注意：どのKvでも実効利得は0.55〜0.64で頭打ちし、**目標1.03に届かない**。
#   原因は眼筋の弱さではなく**トルク飽和**（必要な制御入力が|ctrl|≥1になるのが29.7%のtick、
#   平均1.388・p95 5.89）。さらにその飽和の原因は**太郎の頭が動きすぎていること**
#   （頭部角速度 mean 0.32〜0.41 rad/s ＝18〜23°/秒で動き続ける）。
#   実際の新生児は「安静時は頭を片側に向けたまま・限られた振幅・遅い速度」なので、
#   人間の眼筋でも同じ勢いで頭が振れ続ければ飽和する。
#   → **眼筋を強化して1.03に合わせるのは原因を隠す対症療法なので採らない**。
#   頭の運動が人間的になれば利得は自然に1.03へ近づくはず＝そのとき測り直す。
# 2026-07-27 に測り直した（`e_vor_gain_probe.py`）。指令を
#   kv*(目標速度 − 今の眼球速度) から kv*目標速度 に変えたので、上の表は全部無効。
#   首を 0.5Hz で振り、実効利得＝−(眼球角速度)/(頭角速度) を回帰で求めた：
#
#     頭の速さ    Kv=0.5  1.0   1.5   2.0   3.0   5.0     飽和(Kv=1.5)
#     14度/秒     0.647  0.640 0.523 0.615 0.623 0.697      10%
#     31度/秒     0.424  0.615 1.170 1.295 1.334 1.320      28%
#     41度/秒     0.499  0.861 1.031 1.170 1.211 1.263      47%
#
#   Kv=1.5 を採る。新生児の頭の速さ（過去の実測 18〜23度/秒）より少し速い
#   31〜41度/秒 の帯で 1.03〜1.17 になり、目標 1.03 を含む。
#
# 注意：残る問題：**利得が頭の速さで大きく変わる**（0.42〜1.33）。人間のVORは
#   広い速度範囲でほぼ一定（線形）。太郎が非線形なのは、指令が [-1,1] を
#   超えて飽和することと、眼球の粘性・慣性が低速で相対的に効くため。
#   ＝逸脱として記録し、深追いはしない（視線誘導反射の本筋ではない）。
VOR_KV = 1.5
EYE_KEY = "eye"            # 眼球のジョイント/アクチュエータ名に含まれる語

# 耳石器の経路（静的な傾きへの反応）を速度指令に変換するP制御のゲイン。
# 耳石器が出すのは「目標の眼球角度」なので、既存の速度サーボに混ぜるために
#   ω = kp × (目標角 - 現在角)
# で速度指令へ直す。注意[Tier3・ARBITRARY] 文献に値はない。
# 大きすぎると目標角へ跳ねるように動く（人間の counter-roll は緩やか）ので控えめに取る。
KP_OCR = 2.0


class VOR:
    """頭部角速度を打ち消す向きに眼球を動かし、方策の眼球出力を無効化する。

    注意：方策の出力次元は変えない（受け取るが眼球には流さない）。次元を変えると
    C5の学習済みモデル（n_actuators=90）が読めなくなるため。
    """

    def __init__(self, model, data, gain=VOR_GAIN, kv=VOR_KV,
                 age_months=0.0, use_canals=True, use_otolith=True,
                 kp_ocr=KP_OCR):
        self.gain = float(gain)
        self.kv = float(kv)
        self.kp_ocr = float(kp_ocr)
        self.head_bid = int(model.body("head").id)
        self.n_actuator = int(model.nu)
        # 2026-07-26：感覚器を経由するようにした。それまでは
        #   data.cvel[head] から物理量を直読みしており、三半規管も耳石器も通っていなかった
        #   （「semicircular」「canal」の参照がこのファイルに0件だった）。
        #   その結果、頭がゆっくり傾き続けると眼球が可動域の限界に張り付いた。
        self.canals = None
        self.otolith = None
        import sys as _s, os as _o
        _core = _o.path.abspath(_o.path.join(
            _o.path.dirname(_o.path.abspath(__file__)), _o.pardir, _o.pardir,
            "taro_core", "src", "brain"))
        if _core not in _s.path:
            _s.path.insert(0, _core)
        from spinal_cord.cpg import write_joint_command
        self._write = write_joint_command
        if use_canals or use_otolith:
            _b = _o.path.join(_o.path.dirname(_o.path.abspath(__file__)),
                              _o.pardir, _o.pardir, "taro_core", "src", "senses")
            _b = _o.path.abspath(_b)
            if _b not in _s.path:
                _s.path.insert(0, _b)
            if use_canals:
                from semicircular_canals import SemicircularCanals
                self.canals = SemicircularCanals(age_months=age_months)
            if use_otolith:
                from otolith_organs import OtolithOrgans
                self.otolith = OtolithOrgans()
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
        """半規管モジュール(self.canals)と耳石器モジュール(self.otolith)が設定されていれば、それぞれのreset()を呼んで内部状態を初期化する。引数・返り値は無い。
        """
        if self.canals is not None:
            self.canals.reset()
        if self.otolith is not None:
            self.otolith.reset()

    def _head_omega_world(self, data):
        """頭部の角速度（ワールド基準）。cvelは[角速度3, 線速度3]の順。"""
        return np.array(data.cvel[self.head_bid][:3], dtype=float)

    def override(self, action, model, data, dt, suppress=1.0):
        """方策の行動から眼球ぶんを取り除き、VORの指令に差し替えて返す。

        Args:
            suppress: VORの出力に掛ける係数。1.0でそのまま（従来）、0.0で完全に止める。
                サッケードの最中だけ下げるために使う。

        【なぜ・2026-09-11】人間ではVORは**サッケード開始時に利得が下がり、
          終了前に元に戻る**（能動的な抑制であって単純加算ではない。
          Daye, Roberts, Zee & Optican 2015, J Neurosci【原文確認】）。
          太郎にはこれが無く、サッケード中もVORが同じ強さで働き続けていた。
          実測（F2-122pre・190発）：サッケード中のVORの指令とサッケードの指令を
          比べると、**1度未満の命令では90%の発でVORの方が大きい**
          （VOR 0.77 対 サッケード 0.24）。命令が4度以上だとVORが勝つのは0%。
          これは向きの一致率と完全に対応する（1度未満0%・1〜2度78%・
          2〜4度86%・4〜8度100%・8度以上97%）。
          ＝**小さいサッケードがVORに埋もれていた。**
          調査：doc/文献調査/二語文/2026-09-11_VORとサッケードの合成_人間側.md
        """
        if not self.units:
            return action
        # 抑制が 0 でも**早期 return してはいけない**。override の役目は
        #   「方策（皮質）の眼球出力を捨てる」ことなので、抜けると方策の生の
        #   眼球指令が残ってしまう。0 を書き込む形で下のループを通す。
        suppress = float(suppress)
        out = np.array(action, dtype=float).copy()
        w_world = self._head_omega_world(data)

        # ①三半規管を通す：ゆっくりした回転は「感じなくなる」（高域通過）。
        #   通さないと「回り続けても永久に感じ続ける」＝眼球が回りきって張り付く。
        if self.canals is not None:
            w_world = self.canals.update(w_world, float(dt))

        # ②耳石器を通す：頭が傾いたまま静止したときの「着地点」を決める。
        #   比力（重力＋運動加速度）から重力方向を推定し、傾き角 × 0.15 を目標にする。
        ocr_target = 0.0
        if self.otolith is not None:
            # 頭部座標系での比力。MuJoCo は重力を含んだ加速度を直接持たないので、
            # 頭の姿勢から重力方向を頭部座標へ回して比力の主成分とする。
            # 注意：[ARBITRARY・簡略化] 本来は MIMo の vestibular_acc（耳石器の入力）を
            #   使うべきだが、VOR は物理ステップの途中で呼ばれ観測が未生成のため、
            #   ここでは重力方向だけを使う。運動加速度ぶんは含まれない＝
            #   「重力慣性のあいまいさ」を回避してしまっている点は人間より有利。
            Rh = np.array(data.xmat[self.head_bid], dtype=float).reshape(3, 3)
            g_head = Rh.T @ np.array([0.0, 0.0, -9.81])
            self.otolith.update(g_head, float(dt))
            ocr_target = float(self.otolith.counter_roll())

        for u in self.units:
            # 眼球ボディのローカル軸へ頭部角速度を投影
            R = np.array(data.xmat[u["bid"]], dtype=float).reshape(3, 3)
            w_axis = float(np.dot(R.T @ w_world, u["axis"]))
            w_des = -self.gain * w_axis            # 頭と逆向き・同じ速さ＝視線を空間に固定
            # 耳石器の経路を足す（角度指令をP制御で速度指令に変換してから加算）。
            #   ロール軸（torsional）にのみ効かせる＝OCR はロール専用の現象。
            if self.otolith is not None and "torsional" in model.actuator(u["aid"]).name:
                ang_now = float(data.qpos[u["qposadr"]])
                w_des += self.kp_ocr * (ocr_target - ang_now)
            # 2026-07-27：`kv * (w_des - w_eye)` から `kv * w_des` に変えた。
            #
            # 【何が起きていたか】眼球の今の角速度 w_eye を引く形は、
            #   **眼球の速度を目標値に保つフィードバック制御**になる。頭が静止すると
            #   目標速度が0なので、VORが「眼球の速度を0に保て」と働き続け、
            #   視線誘導反射のサッケードを完全に打ち消していた。
            #     実測：同じ指令で VOR OFF なら1秒で25.6度動くのに、
            #           VOR ON では -1.4度＝まったく動かない。
            #
            # 【人間はどうか】VORは半規管 → 前庭神経核 → 外眼筋運動核 → 筋 の
            #   **三ニューロン弓**で、経路にフィードバックループを含まない。
            #   頭が静止していればVORの出力はゼロ＝何もしない。眼球を静止位置に
            #   保っているのはVORではなく別の仕組み（脳幹の積分器と眼球まわりの
            #   粘弾性）。太郎はその役割までVORにやらせていた＝逸脱。
            #
            # 注意：これでVORの利得が変わるので、gain の較正はやり直しが要る。
            #   （人間でもVORの利得較正は小脳が担う別の仕組み）
            torque_ratio = self.kv * w_des * suppress
            # 可動域の端では戻す向きにだけ力を出す（端に押し付け続けない）
            ang = float(data.qpos[u["qposadr"]])
            if ang >= u["jhi"] and torque_ratio > 0:
                torque_ratio = 0.0
            elif ang <= u["jlo"] and torque_ratio < 0:
                torque_ratio = 0.0
            # 2026-07-26：ここで `out[aid] = torque_ratio` と直接書いていたのが誤り。
            #   MuscleModel では1関節が2本の筋（前半＝負方向筋・後半＝正方向筋）で駆動され、
            #   各要素は [0, 1] に切り捨てられる。負の指令は消え、正の指令はいつも同じ
            #   向き（負方向）に眼を動かした＝**眼は下にしか動けなかった**。
            #   実測：VOR ON で左目の上下角が1秒で下限 −47度に張り付き戻らない。
            #   共通の写像 write_joint_command を通す（眼球は相反神経支配なので共収縮は0）。
            self._write(out, u["aid"], torque_ratio, self.n_actuator,
                        co_activation=0.0)
        return out

    def gaze_axis(self, model, data, camera="eye_left"):
        """今の視線方向（カメラのz軸）。揺れの測定用。"""
        cid = int(model.camera(camera).id)
        return np.array(data.cam_xmat[cid], dtype=float).reshape(3, 3)[:, 2].copy()
