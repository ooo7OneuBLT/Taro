"""姿勢制御反射（層1）＝座位で体幹が前後に傾いたとき、傾きと逆側の筋だけを通す
ゲート（門）。太郎の座位保持の学習の下位層。

仕様：作業記録（非公開）
設計：作業記録（非公開）

【役割・なぜゲート方式か】決定事項「学ばせるのは加減（どれだけ強く筋を使うか）だけ」を
徹底するため、生得側（このモジュール）は「向き」を100%決め、「強さ」は一切決めない。
`stretch_reflex.py` の `StretchReflex` は行動配列に**加算**する方式だが（生得側が
足す"量"も持つ）、このモジュールは**ゲート（マスク）**方式＝傾きと同じ側の筋の指令を
問答無用で0にクリップし、逆側はそのまま探索側の指令を通す。量は一切足さない。
（設計1節(e)：3案とも最終的にこの使い分けに合意）

【入力】
    固有感覚   hip_bend系の関節角度（`robot:hip_bend1`・`robot:hip_bend2`）。
              MIMoのXMLを実際に読んで確認した構造（下記【関節構造の実測】）。
    骨盤触覚   `somatosensory_cortex.body_names_of_group("hip")` == ["hip","lower_body"]
              に対応するbodyの触覚点。**TouchMapを経由せず**、
              `touch.sensor_outputs`/`touch.sensor_positions`（bodyローカル座標、
              TrimeshTouchではキーがbody_id）を直接読む（設計1節(d)の指摘：
              TouchMap.positionsはワールド座標由来の正規化座標で、リクライニング
              角度が変わると「前後」の意味がずれる可能性があるため）。
    ソース：`taro_core/src/senses/somatosensory_cortex.py` を実際に読んで確認済み
    （229行目付近のdocstring「TrimeshTouch … キーはbody_id」）。

【関節構造の実測、2026-08-15】`MIMo/mimoEnv/assets/benchmarkv2_scene_temp_*.xml`
（`scene_io.build`が実行時に生成する成長済みXMLと同じテンプレート）を実際に読んで
確認した。

    robot:hip_bend1  hip(骨盤)と lower_body(腰)の間。axis="0 1 0"（Y軸）。可動域[-17, 30.5]度
    robot:hip_bend2  lower_bodyとupper_bodyの間。axis="0 1 0"（同じY軸）。可動域[-17, 30.5]度
                     ＝ hip_bend1・hip_bend2は"limited"な固定腱（tendon, coef 1/-1,
                       range[-.01,.01]rad≈0.57度）で強く連動しており、事実上一体で動く。
    アクチュエータ  "act:hip_bend"が1本だけ存在し、robot:hip_bend1を駆動する
                    （hip_bend2に専用アクチュエータは無い。連動腱で追従する）。
    robot:chest_lean axis="1 0 0"（X軸）＝**前後(Y軸)ではなく左右(X軸)の側屈**。
                    robot:hip_lean1/2（axis="1 0 0"、明示的に「lean」＝側屈）と
                    同じ軸であり、chest_leanは名前に反して前後の曲げではない。

【想定外・逸脱】設計は「hip_bend系・chest_lean系の拮抗筋2本」をゲート対象として
指示していたが、上記の実測により **chest_lean は前後(sagittal)の曲げではなく
左右(lateral)の側屈用の関節**と判明した（`head_tilt`（前後・Y軸）と`head_tilt_side`
（左右・X軸）の対比、および反射・接触実験（`atnr.py`）や過去のreach基準物体配置
（`benchmarkv2_scene`のtest_object1が世界座標x=+1に置かれ、reach方向＝MIMoの前方と
一致）から、このモデル全体でX軸=左右、Y軸=前後、Z軸=上下という軸の意味が一貫している
ことも確認した。またchestには前後方向の可動関節がそもそも存在しない（hip_bend2が
lower_body-upper_body間で既に前後の曲げを担っており、upper_body-chest間は
lean(側屈)とrot(捻り)の2自由度のみ）。
    ⇒ **chest_lean は座位の前後ゲート対象から除外**し、実際に前後の曲げを担う
      hip_bend（1本のアクチュエータ、hip_bend1・hip_bend2を連動して駆動）だけを
      対象にした。この判断は実装時点の実測に基づく修正であり、設計の意図
      （前傾で背中の伸展側を通し腹側を塞ぐ、という機能そのもの）は
      hip_bendだけで完全に実現できる（座位の胴体で前後に曲がる関節はここしかない
      ため、対象を広げる必要も無い）。上に報告・想定外として明記する。

【拮抗筋のneg/pos対応、2026-08-15実測で確定】
    `run/tools/check_common_drive.py`（verify3、2026-08-11）が既に
    「moment_1は全アクチュエータで常に正・moment_2は常に負」を実測済みで、
    `run/tools/check_joint_compliance.py`（2026-08-13）も別関節(shoulder_ad_ab)で
    独立に「pos チャンネル(index+n_actuator)がqposのhi方向（増加方向）に対応する」
    ことを実測している。MuscleModelの構成
    （`lce_1=(qpos-qpos_spring)*moment_1+lce_1_ref`、`moment_1>0`常に、
      `lce_2=(qpos-qpos_spring)*moment_2+lce_2_ref`、`moment_2<0`常に）から、
    「pos側(index+n_actuator)を活性化するとqposが増える、neg側(index)を活性化すると
    qposが減る」は**アクチュエータごとの解剖学的な向きに依存しない、MuscleModelの
    構成上の一般的性質**（moment_1/moment_2の符号が全アクチュエータで揃っているため）
    だと判断できる。項57・項106（「片方の軸の符号をもう片方に流用しない」
    「共通信号配布で符号が揃う保証はない」）はここでは当てはまらない
    （揃っているのは軸ごとの解剖学的な意味ではなく、moment_1/2という数式内部の
    符号であり、実測でその普遍性そのものが確認済みのため）。
    それでも本モジュール自身のスコープ（hip_bend）で**個別に実測して確定させる**
    （`run/tools/check_posture_reflex.py`）。結果：pos側(index+n_actuator)を
    活性化すると hip_bend1 の qpos が増加（＝前屈側、30.5度の方向）、neg側(index)を
    活性化すると qpos が減少（＝後屈側、-17度の方向）ことを実機で確認した
    （検証ログは作業記録参照）。

【骨盤触覚の前後の符号、2026-08-15実測で確定・想定外あり】
    まず「+x=前方(腹側)」という**幾何学的な事前予想**を立てた：
    (1) `MIMo/mimoEnv/assets/benchmarkv2_scene_temp_*.xml`のreach用基準物体
    （test_object1・test_object2）が世界座標x=+1（MIMo正面）に置かれている。
    (2) `D/scripts/d_beta_sitting_env.py`のコメント（2026-07-18較正済み）
    「太郎は…回転させない限り太郎と同じ『+X方向を向く』姿勢を引き継ぐ」との
    既存の確認記録。この2点から、hip_bend系のqpos>0（【拮抗筋のneg/pos対応】の
    実測で確定した"pos側チャンネルが増やす方向"）は頭・胸が world +x
    （前方）へ動く「前屈」であることも別途実測で確認した
    （`run/tools/check_posture_reflex.py`の診断：hip_bend1=+30.5度で
    head.xposがx=0.0→0.20へ前方移動することを確認）。

    次に本モジュール実装時に自作した検証（`run/tools/check_posture_reflex.py`の
    verify_b_touch_front_back_sign、重力オフ・"hip"body溶接・hip_bend1のみ
    qfrc_appliedで駆動・前後(world x=±0.06)にリジッドな壁を置いた最小限の
    MuJoCo環境）で、実際にhip・lower_bodyの触覚点に生じる力を実測したところ、
    **前屈方向（hip_bend1が正、qfrc>0、実測到達角+13.27度）で接触が生じ、
    その力は body-local **x<0** 側の点に集中していた**（diff=−10.3、
    mag=10.8）。これは上記の幾何学的な事前予想（"local x>0=前方"）と**符号が逆**
    だった。

    【想定外】事前予想と実測が食い違った理由は、実装時点では完全には特定できて
    いない（lower_bodyの関節オフセット・複合回転の効果までは解析的に追えたが、
    実際にどの部位がどの壁に接触したかまでは特定していない。前輪駆動で
    chest/headまで前方へ大きく振れる一方、lower_body自身の局所的な前面が
    壁に届く前に、別の面が接触した可能性がある）。**実測を優先し**、
    「diffの符号を反転して使えば、前屈でdiff>0・後屈でdiff<0という固有感覚と
    一致する信号になる」という実測結果に忠実な設定（`FRONT_IS_POSITIVE_LOCAL_X
    = False`）を採用する。ただし後屈方向の対照実験ではこの壁配置で接触が
    再現できなかった（mag=0、壁に届かなかった）ため、**この符号確定は
    前屈方向1条件のみの実測に基づく[Tier3・実測範囲が限定的]**。
    座位保持の本番シーン（座面・背もたれのある実際の接触）ができた時点で
    再検証することを推奨する（作業記録「上に上げること」参照）。

【根拠ラベル】
    姿勢制御反射（脊髄反射弓）の存在そのもの     [Tier1・確立した生理学]
    骨盤触覚と固有感覚を使い前庭を使わない簡略化  [Tier3・工学的判断、ATNR
                                                (`brainstem/atnr.py`)と同じ
                                                「常に同時に発火する冗長信号を
                                                1つに絞る」考え方を踏襲]
    傾き判定の閾値（tilt_deadzone_deg等）        [Tier3・文献に定量値なし、
                                                調査が探しても見つからなかった]
    骨盤触覚の前後差を「ゲートのfallback」として  [Tier3・工学的判断。固有感覚
    重み付けする設計                             (hip_bend角度)の方が誤差なく
                                                exactに向きを判定できるため
                                                主とし、触覚は固有感覚が不感帯内
                                                （ほぼ直立）のときの早期補助信号
                                                として使う設計にした。設計は
                                                「両方から判定」とのみ指示し
                                                優先順位までは決めていなかった
                                                （想定外・作業記録に明記）]
"""
import numpy as np

# 傾きの判定閾値（固有感覚：hip_bend系qposの平均値、度）。
# [Tier3・ARBITRARY] 新生児の座位保持で「どれだけ傾いたら立て直し反射が要るか」の
# 定量的な閾値は文献に見つからなかった（調査確認済み、設計の前提を踏襲）。
# 可動域[-17,30.5]度の中間的な値として、感度確認をしやすい5.0度を初期値にした。
# 学習前の感度確認（run/tools/check_posture_reflex.pyで振って確認）を経て、
# 実験ファイル側で上書きする想定。
DEFAULT_TILT_DEADZONE_DEG = 5.0

# 触覚のfallback判定：前後差(diff)が「触覚点の力の合計(mag)」に対してこの比率を
# 超えたら触覚由来の判定を採用する。[Tier3・ARBITRARY] 触覚点数・力の単位系に
# 依存する相対値にすることで、体が成長して点数が変わっても閾値の意味が大きくは
# ぶれないようにした（工学的判断、定量根拠なし）。
DEFAULT_TOUCH_DEADZONE_FRAC = 0.1

# 【2026-08-15実測で確定】MuscleModelの構成上、pos側（行動配列のindex+n_actuator）を
# 活性化するとqposが増加方向（hi側）へ、neg側（index）を活性化すると減少方向（lo側）へ
# 動く。run/tools/check_posture_reflex.py の verify_muscle_channel_sign で
# 「act:hip_bend」について実測して確認した（docstring【拮抗筋のneg/pos対応】参照）。
POS_CHANNEL_INCREASES_QPOS = True

# 【2026-08-15実測で確定・Tier3（範囲限定・想定外あり）】幾何学的な事前予想
# （+xが前方）とは符号が逆の実測結果が出た。実測（前屈方向1条件のみ）を優先し、
# diffを反転して使う設定にした。run/tools/check_posture_reflex.py の
# verify_b_touch_front_back_sign で確認した（docstring【骨盤触覚の前後の符号】参照。
# 想定外として作業記録に明記済み。座位保持の本番シーンができたら再検証を推奨）。
FRONT_IS_POSITIVE_LOCAL_X = False

# ゲート対象の関節名パターン（アクチュエータ名にこの文字列を含むものを対象にする）。
# 【想定外・2026-08-15】設計の「hip_bend系・chest_lean系」からchest_leanを除外した
# （docstring【想定外・逸脱】参照：chest_leanは実測で左右の側屈用関節と判明したため）。
GATE_ACTUATOR_PATTERNS = ("hip_bend",)

# 固有感覚として読む関節名（"robot:"は付けずに渡す）。
PROPRIOCEPTIVE_JOINT_NAMES = ("hip_bend1", "hip_bend2")


class PosturalGate:
    """座位での体幹前後の傾きに応じて、傾きと同じ側の体幹筋の指令を0にクリップする
    ゲート（門）。傾きと逆側の指令はそのまま通す（副作用なし、新しい配列を返す）。

    `stretch_reflex.StretchReflex` とコード構造は同じ（コンストラクタでパラメータ、
    毎tickの計算とapply()の分離、副作用なし）だが、`apply()` は「加算」ではなく
    「ゲート（マスク）」である点が異なる（docstring冒頭【役割・なぜゲート方式か】）。

    入力（毎tick）：
        qpos    MuJoCoの関節角度配列（env.unwrapped.data.qpos）。
        touch   env.unwrapped.touch（mimoTouch.touch.TrimeshTouch想定）。
                Noneを渡せば固有感覚だけで判定する（touch未接続でも動く）。

    出力：
        探索側が出した行動配列（2n_actuator次元、MuscleModel前提）に
        ゲートを適用した新しい配列。
    """

    def __init__(self, model,
                 actuator_patterns=GATE_ACTUATOR_PATTERNS,
                 proprioceptive_joint_names=PROPRIOCEPTIVE_JOINT_NAMES,
                 tilt_deadzone_deg=DEFAULT_TILT_DEADZONE_DEG,
                 touch_deadzone_frac=DEFAULT_TOUCH_DEADZONE_FRAC,
                 pos_channel_increases_qpos=POS_CHANNEL_INCREASES_QPOS,
                 front_is_positive_local_x=FRONT_IS_POSITIVE_LOCAL_X):
        self.n_actuator = int(model.nu)

        # ---- ゲート対象アクチュエータのindex（行動配列のneg側index。pos側はindex+n） --
        self.actuator_idx = []
        for i in range(self.n_actuator):
            name = model.actuator(i).name
            if any(p in name for p in actuator_patterns):
                self.actuator_idx.append(i)
        if not self.actuator_idx:
            raise ValueError(
                f"PosturalGate: actuator_patterns={actuator_patterns} に一致する"
                f"アクチュエータが見つからない。model.actuator(i).nameを確認すること。")

        # ---- 固有感覚として読む関節のqposアドレス ------------------------------------
        self.joint_qposadr = []
        missing = []
        for jn in proprioceptive_joint_names:
            try:
                jid = model.joint("robot:" + jn).id
                self.joint_qposadr.append(int(model.jnt_qposadr[jid]))
            except Exception:
                missing.append(jn)
        if not self.joint_qposadr:
            raise ValueError(
                f"PosturalGate: proprioceptive_joint_names={proprioceptive_joint_names} の"
                f"どれもmodelに見つからない。")
        # 一部だけ見つからないのは(将来モデルが変わって片方が無くなる等)静かに許容しない。
        # 少なくとも1つ見つかれば動作はするが、見つからなかった分は明示的に記録しておく。
        self._missing_proprioceptive_joints = missing

        # ---- 骨盤触覚のbody_id ------------------------------------------------------
        # 【なぜ、2026-08-15】設計1節(d)の指摘どおりTouchMapを経由しない。
        # body_names_of_group("hip") == ["hip","lower_body"] のbody_idを直接引く。
        from somatosensory_cortex import body_names_of_group
        hip_body_names = body_names_of_group("hip")
        self.hip_body_ids = []
        _missing = []
        for nm in hip_body_names:
            try:
                self.hip_body_ids.append(int(model.body(nm).id))
            except Exception:
                # 【2026-09-12】全滅なら下で raise するが、**一部だけ見つからない**
                #   場合はここを黙って通り、骨盤触覚が半分しか効かない状態で走れる。
                #   それを警告として残す（エラー.logに出る）。詳細は run/log_setup.py
                _missing.append(nm)
        if _missing and self.hip_body_ids:
            # 太郎の脳は単体でも import されるので `run.log_setup` に依存しない
            # （doc/ログの使い方.md の決まり）。標準の logging だけ使う
            import logging
            logging.getLogger(__name__).warning(
                "骨盤触覚のbody %s がmodelに無く、%s だけで骨盤を見ています",
                _missing, [nm for nm in hip_body_names if nm not in _missing])
        if not self.hip_body_ids:
            raise ValueError(
                f"PosturalGate: body_names_of_group('hip')={hip_body_names} の"
                f"どのbodyもmodelに見つからない。")

        self.tilt_deadzone_deg = float(tilt_deadzone_deg)
        self.touch_deadzone_frac = float(touch_deadzone_frac)
        self.pos_channel_increases_qpos = bool(pos_channel_increases_qpos)
        self.front_is_positive_local_x = bool(front_is_positive_local_x)

    # ------------------------------------------------------------ 入力の読み取り
    def proprioceptive_tilt_deg(self, qpos):
        """hip_bend系qposの平均を度で返す。正=前屈（前傾）、負=後屈（後傾）
        （docstring【拮抗筋のneg/pos対応】・【関節構造の実測】参照。
        hip_bend1・hip_bend2は連動腱でほぼ同じ値を取るので平均で十分）。"""
        vals = [float(qpos[a]) for a in self.joint_qposadr]
        return float(np.degrees(np.mean(vals)))

    def touch_tilt_signal(self, touch):
        """骨盤(hip・lower_body)の触覚から、前後差(diff)と力の合計(mag)を返す。

        diff = Σ(力の大きさ * sign(local x)) 。front_is_positive_local_x=Trueなら
        diffが正＝前方(local x>0)側に力が集中＝前傾（hip_bend>0）と同じ向き
        （docstring【骨盤触覚の前後の符号】、実測で確認済み・ただし範囲限定Tier3）。
        touchがNone、または対象bodyに触覚点が無ければ (0.0, 0.0) を返す。
        """
        if touch is None:
            return 0.0, 0.0
        diff = 0.0
        mag_sum = 0.0
        sensor_positions = getattr(touch, "sensor_positions", None)
        sensor_outputs = getattr(touch, "sensor_outputs", None)
        if sensor_positions is None or sensor_outputs is None:
            return 0.0, 0.0
        for bid in self.hip_body_ids:
            pos = sensor_positions.get(bid)
            force = sensor_outputs.get(bid)
            if pos is None or force is None:
                continue
            pos = np.asarray(pos, dtype=np.float64)
            force = np.asarray(force, dtype=np.float64)
            if pos.shape[0] == 0 or force.shape[0] == 0:
                continue
            mag = np.linalg.norm(force, axis=-1)
            x_sign = np.sign(pos[:, 0])
            diff += float(np.sum(mag * x_sign))
            mag_sum += float(np.sum(mag))
        if not self.front_is_positive_local_x:
            diff = -diff
        return diff, mag_sum

    def tilt_direction(self, qpos, touch=None):
        """傾きの向きを +1（前傾）・-1（後傾）・0（不感帯内、判定しない）で返す。

        固有感覚(hip_bend角度)を主に使う（誤差なくexactに向きが分かるため）。
        固有感覚が不感帯(tilt_deadzone_deg)以内のときだけ、触覚の前後差を
        fallbackとして使う（【根拠ラベル】参照、設計は優先順位まで指定していない
        ため実装時に決めた判断。想定外として報告済み）。
        """
        prop_deg = self.proprioceptive_tilt_deg(qpos)
        if abs(prop_deg) >= self.tilt_deadzone_deg:
            return 1 if prop_deg > 0 else -1
        diff, mag = self.touch_tilt_signal(touch)
        if mag > 1e-9 and abs(diff) >= self.touch_deadzone_frac * mag:
            return 1 if diff > 0 else -1
        return 0

    # ------------------------------------------------------------ ゲートの適用
    def compute(self, qpos, touch=None):
        """傾きの向き(+1/-1/0)を計算する（行動配列にはまだ適用しない）。"""
        return self.tilt_direction(qpos, touch)

    def apply(self, action, qpos, touch=None):
        """傾きと同じ側の体幹筋の指令を0にクリップして返す（ゲート、加算ではない）。

        呼び出し側の配列を書き換えず、新しい配列を返す（副作用なし、
        stretch_reflex.StretchReflex.applyと同じ流儀）。

        direction=+1（前傾）のとき、qposをさらに増やす側（前傾を強める側）の
        チャンネルを0にクリップし、qposを減らす側（立て直す側）を通す。
        direction=-1（後傾）のときはその逆。direction=0（不感帯内）なら無変更。
        """
        a = np.asarray(action, dtype=np.float64).copy()
        direction = self.tilt_direction(qpos, touch)
        if direction == 0:
            return a
        n = self.n_actuator
        for i in self.actuator_idx:
            # pos_channel_increases_qpos=True の場合：
            #   index+n（pos側）を活性化するとqposが増える(前傾が強まる)
            #   index（neg側）を活性化するとqposが減る(後傾が強まる)
            if self.pos_channel_increases_qpos:
                increasing_idx, decreasing_idx = i + n, i
            else:
                increasing_idx, decreasing_idx = i, i + n
            if direction > 0:
                suppress_idx = increasing_idx    # 前傾を強める側を塞ぐ
            else:
                suppress_idx = decreasing_idx    # 後傾を強める側を塞ぐ
            a[suppress_idx] = 0.0
        return a
