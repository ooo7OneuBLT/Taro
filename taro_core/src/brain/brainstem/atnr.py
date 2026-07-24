"""非対称性緊張性頸反射（ATNR, asymmetric tonic neck reflex）＝脳幹反射弓。

【解剖学的位置づけ、2026-07-23】把握反射(`spinal_cord/grasp_reflex.py`)は脊髄だけで完結する
反射弓だが、ATNRは違う。入力は①頸部の固有感覚（首の捻れ）②内耳の前庭器官（頭の回転）の両方、
中枢は**脳幹**（網様体＋前庭神経核）で統合され、そこから脊髄運動ニューロンへ「顔側は伸筋・
後頭側は屈筋」の緊張を下行させる。脊髄だけで閉じる反射ではないため`spinal_cord/`でなく
`brainstem/`に新設して置く[Tier1]（StatPearls "Tonic Neck Reflex"；ScienceDirect Topics
"Tonic Neck Reflex"）。

【トリガーに前庭を使わない理由】文献上は頸部固有感覚と前庭の両方が入力だが、現実の頭部回転では
この2つは常に同時に発火する（頭が回れば首も内耳も同時に動く）。太郎には前庭感覚
(`senses/otolith_organs.py`, `senses/semicircular_canals.py`)を実装済みだが、今回は把握反射が
接触センサを直読したのと同じ簡略化方針で、**首関節(`head_swivel`)の角度を直接読む**（前庭経由の
新しい配線は追加しない）[Tier3・簡略化、根拠：2信号は冗長で同じ方向を指すため省略しても結果は
歪みにくいという判断。ユーザー合意済み]。

【なぜ実装するか(経緯)】文献的な必要性(拘束仮説)は弱いとユーザーへ報告した(§Fの通り一次実証は
弱い)。それでも文献で「顔側の腕が伸びて視野に入り、hand regard・目と手の協調を促す」という
別の機能仮説が見つかり(Harkla、Documenting Hope等、いずれも一次研究でなく臨床教育サイト)、
ユーザー判断で実装することにした[ユーザー明示的合意、根拠は弱いと承知の上での実装]。

【反応の設計】「フェンサーの姿勢」の核＝肘の伸展/屈曲に加え、**顔側の肩を外転**させることで
実際に手が視野に入るようにする(ユーザー指示：「視界に手が入るようにすればいい」)。後頭側は
視野と無関係なので肘の屈曲のみに留める(肩は動かさない、簡略化)。脚も本来ATNRに含まれるという
記述があるが、今回の目的(手を視野に入れる)に直接関係しないため対象外[Tier3・簡略化]。

【符号の実機検証、2026-07-23】MIMoの`act:{side}_elbow`と`act:{side}_shoulder_abduction`は
左右ともctrl>0で関節角が正方向（伸展／外転）に動くことをシミュレータで直接確認済み
（`atnr_signcheck.py`、使い捨て検証スクリプト）。`head_swivel`はctrl>0でqpos正方向に動くことも
確認済み。正負がどちらの向き(体の左/右)に対応するかは、肩関節の局所座標オフセット
（右肩y=-0.024・左肩y=+0.024、モデルXMLより）から幾何学的に導出：head_swivel qpos>0は
顔が体の**左**を向く方向[幾何学的導出、実測ではない]。

【消え方の設計】把握反射と同じ`reflex_strength = 1 - w_mean(age)`を再利用する。これは単に
似た形の曲線を借りるのではなく、太郎の`w_mean`(`corticospinal.py`)が「皮質からの随意指令が
脊髄CPGの自発出力に対してどれだけ優勢か」＝皮質の成熟度そのものを表す変数であり、
「皮質が成熟すると皮質-網様体路/皮質-前庭路を通じて脳幹の反射回路を抑制する」という
ATNRの消失メカニズムと構造的に一致するため[Tier1〜2、一般原理の転用]。ただし抑制の解剖学的な
行き先は把握反射(皮質脊髄路→脊髄)とATNR(皮質-網様体路等→脳幹)で異なる[正直にラベリング]。

【恣意的な部分】NECK_YAW_THRESHOLD_DEG・各TARGET値に文献根拠なし[Tier3・ARBITRARY]。
"""
import numpy as np

NECK_YAW_THRESHOLD_DEG = 15.0   # [Tier3・ARBITRARY] 反射が発動する首の振れ角。文献に具体値なし
ELBOW_EXTEND_TARGET = 0.9       # [Tier3・ARBITRARY] 顔側の肘＝伸展方向(ctrl>0、実機検証済み)
ELBOW_FLEX_TARGET = -0.9        # [Tier3・ARBITRARY] 後頭側の肘＝屈曲方向(ctrl<0)
SHOULDER_ABDUCT_TARGET = 0.9    # [Tier3・ARBITRARY] 顔側の肩＝外転方向(ctrl>0、実機検証済み)


class ATNR:
    """model から首(head_swivel)のqpos位置・肘/肩外転アクチュエータidxを名前ベースで検索し、
    毎stepの首の向きから「顔側は伸展、後頭側は屈曲」の命令を計算する。"""

    def __init__(self, model):
        jid = model.joint("robot:head_swivel").id
        self.neck_qposadr = model.jnt_qposadr[jid]

        self.elbow = {}
        self.shoulder_abduct = {}
        for i in range(model.nu):
            name = model.actuator(i).name
            for side in ("right", "left"):
                if name == f"act:{side}_elbow":
                    self.elbow[side] = i
                elif name == f"act:{side}_shoulder_abduction":
                    self.shoulder_abduct[side] = i

    def apply(self, action, qpos, reflex_strength):
        """action(numpy, shape=(n_act,))に反射を重ねて返す。reflex_strength<=0または首が
        閾値未満なら無変更（E_ATNR_REFLEX=0の既定では呼ばれないので、常時1バイト差なしが保証される）。"""
        if reflex_strength <= 0:
            return action
        neck_deg = float(np.degrees(qpos[self.neck_qposadr]))
        if abs(neck_deg) < NECK_YAW_THRESHOLD_DEG:
            return action
        face_side = "left" if neck_deg > 0 else "right"
        occiput_side = "right" if face_side == "left" else "left"

        a = action.copy()
        i = self.elbow[face_side]
        a[i] = (1 - reflex_strength) * a[i] + reflex_strength * ELBOW_EXTEND_TARGET
        i = self.shoulder_abduct[face_side]
        a[i] = (1 - reflex_strength) * a[i] + reflex_strength * SHOULDER_ABDUCT_TARGET
        i = self.elbow[occiput_side]
        a[i] = (1 - reflex_strength) * a[i] + reflex_strength * ELBOW_FLEX_TARGET
        return a
