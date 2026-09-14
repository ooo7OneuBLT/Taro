"""実験の測定条件 — 実験者が乳児の頭を抑える（視線誘導反射を測るため）。

注意：【置き場所について、2026-07-28】最初 `taro_core/src/caregiver/hands.py` に置いたが、
ユーザーの指摘「頭を支えるのって実験だからCoreじゃなくない？」で目標Eへ移した。
区別すべきものが2つある：

    日常の世話（core）    抱っこする、うつ伏せから戻す、泣いたら抱く
                          ＝太郎の環境そのもの。まだ未実装
                          （やることリスト「世話をする他者が環境に居ない」）
    実験の測定条件（ここ） 実験者が頭を抑えて視線を測る
                          ＝人間の実験でもそうしているから同じにする

このファイルは後者。方針＝[[feedback-core-vs-experiment-placement]]（実験は目標フォルダ）。


【なぜ要るか、2026-07-28】視線誘導反射の実験が成立していなかった。太郎の首が倒れて
しまい、対象を視野に捉え続けられなかったためである。当初は「首がすわってから測る」
＝立ち直り反射を実装する方針だったが、**人間の実験を読み直したら前提が違っていた**：

    Hunter, S. K., & Richards, J. E. (2003)
    5週齢       … 頭の両側に枕を置く
    8〜14週齢   … 親が手で頭を抑える
    （中心を見ていない試行は除外）

＝**人間の乳児実験でも、赤ちゃんは自分で頭を支えていない**。実験者が支えている。
新生児のリーチ実験（von Hofsten 1982）も同様で、乳児は椅子や実験者の手で
頭と体を支えられた状態で行われている。
⇒ 太郎に必要なのは「首がすわること」ではなく「**頭を支えられること**」だった。

【なぜバネで実装するか（枕でも外力でもなく）】
  ①枕（geomを置く）… 摩擦係数・接触の硬さ・位置という**根拠のない数値が3つ増える**。
    しかも滑る・跳ねる・隙間ができるといった失敗が実行するまで分からない。
  ②毎ステップ外力を加える（`qfrc_applied`）… env.step は frame_skip 回ぶん物理を進めるので、
    **その間ずっと同じ力がかかり続ける**（目標とのずれが更新されない）＝行き過ぎる。
  ③関節のバネ（`jnt_stiffness` + `qpos_spring`）… MuJoCo が**物理ステップごとに解く**ので
    行き過ぎない。決めるパラメータは「支える強さ」1つだけ。← 採用

  目標角を**支え始めた時点の角度**にするので、静止しているあいだは力がゼロになる。
    ＝「触れているが押していない」という手の振る舞いに一致する。

【B-1 と B-2】ユーザーとの合意（2026-07-28）で、まず B-1 から始めて後で B-2 へ進む。
    B-1  ほぼ完全に固定する（HOLD_STIFFNESS_FIRM）
         まず「頭が固定されていれば視線誘導反射が測れるのか」を確かめる。
         柔らかさの調整を後回しにするのは、うまくいかなかったときに
         「支えが甘いのか反射が悪いのか」を切り分けられるようにするため。
    B-2  柔らかく支える（HOLD_STIFFNESS_SOFT）
         力を入れれば動くが放っておけば戻る＝枕で支えられた乳児に近い。

注意：**これは太郎の身体ではなく実験の設定**。太郎自身は「頭が支えられている」ことを知らない。

【日常の世話は別に作る】やることリストの「世話をする他者が環境に居ない」
（うつ伏せから戻す・泣いたら抱く）は太郎の環境そのものなので core 側に作る。
このファイルはあくまで**測定のために頭を抑える**もので、両者を混ぜないこと。
"""
import numpy as np

# 支える強さ [N·m/rad]。
#
# 注意：[Tier3・ARBITRARY] 親の手の剛性を測った文献は無い。次の物理から決めた目安：
#   新生児の頭の重力モーメントは約 0.38 N·m（仰向け・中立）。
#   バネが k のとき、ずれ θ で釣り合うので θ = 0.38 / k。
#       k =  21.7 → 1.0度ずれて釣り合う
#       k = 217   → 0.1度
#       k =  10   → 2.2度
#       k =   2   → 11度
# 人間の実験で許される頭のずれの実測値は無い（Hunter & Richards 2003 は
#   「中心を見ていない試行は除外」とだけ書いてあり、角度の閾値が無い）。
HOLD_STIFFNESS_FIRM = 200.0   # B-1：ほぼ完全固定（0.1度程度のずれ）
HOLD_STIFFNESS_SOFT = 10.0    # B-2：柔らかく支える（2度程度のずれ）

# 支える首の関節（":" の後ろの名前）。頭の3軸すべて。
HOLD_JOINTS = ("head_swivel", "head_tilt", "head_tilt_side")


# ============================================================================
# 月齢に応じて支える強さを大きくする（2026-08-15 追加）
# ----------------------------------------------------------------------------
# 【なぜ要るか】月齢が上がると頭の質量・重力モーメントが増える。バネの復元力
# k・ずれ角θの釣り合いは k・θ ≈ τ（頭の重力モーメント）なので、同じ k=200 の
# ままだと θ=τ/k が月齢とともに拡大し、「支えているのに頭が前傾する」状態に
# なりうる。MIMoの筋肉モデルは筋力(FMAX)を月齢で変えない
# （2026-07-28に判明・2026-08-15に`mimoActuation/muscle.py`を再確認し今も
# 同じと確認した）ので、月齢別の「支える力の発達」は身体側では作れない。
# 実験条件（このファイル）側でスケーリングする。
#
# 【想定外（2026-08-15、実装）】この機能を作る指示の根拠になった実測値
# 「支えたまま3秒静止での傾きが4ヶ月0.01度／6ヶ月7.71度／7ヶ月13.87度」は、
# 現在のコードでは**再現できなかった**（実測：k=200のままで4ヶ月0.14度・
# 6ヶ月0.16度・7ヶ月0.18度。月齢差はごくわずか）。原因と見られるのは
# `CaregiverHands._critical_damping`（2026-07-29、コミットc43b9c9で追加）
# ＝支えのバネに臨界減衰を入れたことで、支えの見た目のずれが既に
# 大きく改善されていたと考えられる。仕様の数値は臨界減衰を入れる前の
# 版で測られたものと推測される（実装担当へ報告済み。捏造ではなく実測値）。
#   ⇒ とはいえ「月齢が上がると頭の重力モーメントが増え、支える強さも
#     それに応じて大きくする必要がある」という物理的な理屈自体は変わらない
#     （実測でも6・7ヶ月の重力モーメントは4ヶ月よりqfrc_biasで13〜18%大きい）。
#     指示された仕組みは実装しておき、現在の実測値を基準にスケーリングする。
#
# 【やり方】文献値が無いので[Tier3・ARBITRARY]。目標は
# 「4ヶ月・k=200のときの実際のずれ角と同水準に、6・7ヶ月のずれ角を揃える」。
# θ=τ/k（釣り合い）なので、k_age = k_base × (τ_age / τ_4mo)。
# τ（頭を支える関節にかかる重力等のバイアス力）は文献値ではなく、
# その場のモデルから mujoco の `qfrc_bias`（重力・コリオリ等、バネの復元力を
# 含まない）で**実測**する。τ_4mo は基準として、2026-08-15に
# `e_body_config.body_kwargs_from_env(4.0)` + flexion=True の構成で
# head_tilt 関節を実測した値（0.739 N·m）を丸めて使う。
HOLD_AGE_SCALE_FROM_MO = 4.0            # これ以下の月齢では従来どおり固定（変更しない）
HOLD_REF_GRAV_MOMENT_4MO = 0.74         # [Tier3・実測] 4ヶ月時の head_tilt の重力等バイアス力[N・m]
                                         #   （2026-08-15計測。infant_bodyの体型が変わったら要再計測）


# ============================================================================
# 体を支える範囲を部位で選ぶ（2026-07-29 追加）
# ----------------------------------------------------------------------------
# 【なぜ要るか】ユーザーの提案：
#   「肩から手先までと目は自由に動かせるようにして、それ以外の関節は固定する」
#
# 【人間の実験ではどうか】リーチングの実験は、乳児を**椅子に固定し頭を支えた**
# 状態で行う。von Hofsten (1982) の新生児リーチ実験、Carvalho et al. (2007) の
# ベビーチェア（水平から70度）いずれも体幹は支持されている。
# 乳児用チェアは股ベルトの装着が安全基準で義務づけられてもいる。
#   ⇒ 首・体幹を支えるのは**人間の実験条件をそのまま写したもの**。
#
# 注意：【逸脱】脚の固定は人間からの逸脱。現実の乳児は椅子に座っても脚は自由に動く。
#   交絡（脚の動きで姿勢が崩れる・視野に入って反射が誤発火する）を減らすための
#   **実験の測定条件**であって、太郎の身体の性質ではない。
#   逸脱リストに登録すること。
#
# 注意：指は「肩から手先まで」に含める（握るのに要る）。
JOINT_GROUPS = {
    "head":   ("head_swivel", "head_tilt", "head_tilt_side"),
    "trunk":  ("hip_bend1", "hip_bend2", "hip_lean1", "hip_lean2",
               "hip_rot1", "hip_rot2", "chest_lean", "chest_rot"),
    "arm":    ("shoulder_horizontal", "shoulder_ad_ab", "shoulder_rotation",
               "elbow", "hand1", "hand2", "hand3"),
    "leg":    ("hip1", "hip2", "hip3", "knee",
               "foot1", "foot2", "foot3", "toes", "big_toe"),
}
# 日本語名（表示・記録用）
GROUP_JP = {"head": "首", "trunk": "体幹", "arm": "肩から手先", "leg": "脚",
            "finger": "指", "eye": "眼球"}


def joint_group(short_name):
    """関節名（"robot:" と左右の接頭辞を外した名前）から部位を判定する。"""
    nm = short_name
    for pre in ("right_", "left_"):
        if nm.startswith(pre):
            nm = nm[len(pre):]
            break
    if "eye" in nm:
        return "eye"
    for g, names in JOINT_GROUPS.items():
        if nm in names:
            return g
    # 指（ff/mf/rf/lf/th で始まる細かい関節）
    if nm.split("_")[0] in ("ff", "mf", "rf", "lf", "th"):
        return "finger"
    return "other"


def joints_to_support(model, free=("arm", "finger", "eye")):
    """「自由にする部位」以外の、動かせる関節の名前を返す。

    Args:
        free: 自由にしておく部位（"arm" / "finger" / "eye" / "head" / "trunk" / "leg"）
    Returns:
        list[str]: 支える関節の名前（":" の後ろ）
    """
    import mujoco as _mj
    free = set(free or ())
    out = []
    for j in range(model.njnt):
        if int(model.jnt_type[j]) != int(_mj.mjtJoint.mjJNT_HINGE):
            continue          # 自由関節・ボール関節はバネで支えられない
        nm = (model.joint(j).name or "").split(":")[-1]
        if not nm:
            continue
        g = joint_group(nm)
        if g in free or g == "eye":
            continue          # 注意眼球は反射が動かすので、常に支えない
        out.append(nm)
    return out


class CaregiverHands:
    """親が手で頭を支える。関節のバネとして表現する。

    使い方::

        hands = CaregiverHands(model, data)
        hands.hold()          # リセット直後に呼ぶ（今の角度で支え始める）
        ...                   # 実験を回す
        hands.release()       # 支えるのをやめる（元のバネに戻る）

    注意：`hold()` は**モデルの値を書き換えるだけ**なので毎ステップ呼ぶ必要はない。
      毎ステップ呼んでも害はないが、目標角が更新されるので支えの意味が変わる
      （＝「今いる場所で支え続ける」＝頭がゆっくり流れていってしまう）。
    """

    def __init__(self, model, data, joints=None, stiffness=None, damping=None, age=None):
        """
        Args:
            stiffness: 支える強さ [N·m/rad]。None なら B-1（ほぼ完全固定）。
                明示的に指定した場合は、`age` によるスケーリングより優先される
                （実験ファイルで明示された値を勝手に変えないため）。
            damping: 減衰。None なら臨界減衰（ちょうど振動しない値）を計算する。
            age: 月齢。None または `HOLD_AGE_SCALE_FROM_MO`（4ヶ月）以下なら
                `stiffness` をそのまま使う（4ヶ月の挙動を変えないため）。
                それより月齢が高く、かつ**頭の3関節（HOLD_JOINTS）を支える
                構成のとき**だけ、月齢に応じて自動的に強さを大きくする
                （体幹・脚などhead以外を支える用途には適用しない＝
                この基準値`HOLD_REF_GRAV_MOMENT_4MO`は頭専用に実測した値のため）。
        """
        self.model = model
        self.data = data
        self.joints = tuple(HOLD_JOINTS if joints is None else joints)
        self.stiffness = float(HOLD_STIFFNESS_FIRM if stiffness is None else stiffness)
        self._stiffness_explicit = stiffness is not None
        self.age = age
        self._damping = damping
        self._holding = False
        self._saved = {}      # 元の (stiffness, spring, damping) を関節ごとに保存

        self._ids = []
        for j in range(model.njnt):
            if model.joint(j).name.split(":")[-1] in self.joints:
                self._ids.append(dict(jid=j,
                                      name=model.joint(j).name.split(":")[-1],
                                      qadr=int(model.jnt_qposadr[j]),
                                      dof=int(model.jnt_dofadr[j])))

    # ------------------------------------------------------------------
    def _critical_damping(self, k):
        """臨界減衰（ちょうど振動しない減衰）。首まわりの頭の慣性から計算する。

        c = 2√(k·I)。注意文献値ではなく**工学的判断**（安定性優先）[Tier3]。
        既存の `infant_body.apply_neck_tone` と同じ式にしてある。
        """
        head_bid = int(self.model.body("head").id)
        inertia = float(self.model.body_inertia[head_bid][0])
        mass = float(self.model.body_mass[head_bid])
        parent = int(self.model.body_parentid[head_bid])
        arm = float(np.linalg.norm(np.array(self.data.xpos[head_bid])
                                   - np.array(self.data.xpos[parent])))
        I = inertia + mass * arm ** 2
        return 2.0 * float(np.sqrt(max(k, 1e-12) * max(I, 1e-12)))

    # ------------------------------------------------------------------
    def _stiffness_for_age(self):
        """月齢に応じてスケーリングした支える強さ [N·m/rad]。

        4ヶ月以下、または頭の3関節（HOLD_JOINTS）以外を支える構成のときは
        `self.stiffness` をそのまま返す（挙動を変えない）。
        それ以外は、支えている関節の中で最大の重力等バイアス力
        （`data.qfrc_bias`。今の姿勢で spring 以外がその関節にかけている力で、
        ほぼ重力）を、4ヶ月時の基準値 `HOLD_REF_GRAV_MOMENT_4MO` と比べた比を
        `self.stiffness` に掛ける。**mj_forward 済みの `data` が前提**
        （呼び出し側は reset 直後に呼ぶこと。既存の `hold()` の使い方どおり）。
        """
        if (self.age is None or float(self.age) <= HOLD_AGE_SCALE_FROM_MO
                or self.joints != HOLD_JOINTS):
            return self.stiffness
        tau = max((abs(float(self.data.qfrc_bias[u["dof"]])) for u in self._ids),
                  default=0.0)
        ratio = tau / HOLD_REF_GRAV_MOMENT_4MO if HOLD_REF_GRAV_MOMENT_4MO > 0 else 1.0
        return float(self.stiffness * max(1.0, ratio))

    # ------------------------------------------------------------------
    def hold(self, target=None, stiffness=None, verbose=False):
        """頭を支え始める。

        Args:
            target: 目標角の辞書 {関節名: 度}。None なら**今の角度**で支える
                （＝静止中は力がゼロ＝触れているだけの手）。
            stiffness: 強さの上書き。None なら初期化時の値
                （`age` が指定されていれば月齢に応じてスケーリングした値）。
        Returns:
            支えた関節の数
        """
        if stiffness is not None:
            k = float(stiffness)
        elif self._stiffness_explicit:
            k = float(self.stiffness)     # 実験ファイルで明示された値は age より優先
        else:
            k = float(self._stiffness_for_age())
        c = float(self._critical_damping(k) if self._damping is None else self._damping)
        n = 0
        for u in self._ids:
            if u["name"] not in self._saved:      # 最初の1回だけ元の値を控える
                self._saved[u["name"]] = (
                    float(self.model.jnt_stiffness[u["jid"]]),
                    float(self.model.qpos_spring[u["qadr"]]),
                    float(self.model.dof_damping[u["dof"]]),
                )
            if target is not None and u["name"] in target:
                tgt = float(np.radians(target[u["name"]]))
            else:
                tgt = float(self.data.qpos[u["qadr"]])      # 今の角度
            self.model.jnt_stiffness[u["jid"]] = k
            self.model.qpos_spring[u["qadr"]] = tgt
            # 注意：減衰は元の値と臨界減衰の**大きい方**（元より弱くはしない）
            self.model.dof_damping[u["dof"]] = max(
                self._saved[u["name"]][2], c)
            n += 1
        self._holding = True
        if verbose and n:
            angs = " ".join(
                f"{u['name']}{np.degrees(self.model.qpos_spring[u['qadr']]):+.1f}"
                for u in self._ids)
            print(f"[caregiver] 親が頭を支える: {n}関節 強さ={k:.1f}N·m/rad "
                  f"減衰={c:.3f}(臨界) 目標角({angs})度 "
                  f"[人間の実験でも実験者が頭を支えている＝Hunter & Richards 2003]")
        return n

    # ------------------------------------------------------------------
    def release(self, verbose=False):
        """支えるのをやめて、元のバネ（首の筋緊張）に戻す。"""
        for u in self._ids:
            s = self._saved.get(u["name"])
            if s is None:
                continue
            self.model.jnt_stiffness[u["jid"]] = s[0]
            self.model.qpos_spring[u["qadr"]] = s[1]
            self.model.dof_damping[u["dof"]] = s[2]
        self._holding = False
        if verbose:
            print("[caregiver] 親が手を離した（首の筋緊張だけに戻る）")

    # ------------------------------------------------------------------
    @property
    def holding(self):
        return self._holding

    def head_angles(self):
        """今の頭の角度 [度] を辞書で返す（支えられているかの確認用）。"""
        return {u["name"]: float(np.degrees(self.data.qpos[u["qadr"]]))
                for u in self._ids}

    def offsets(self):
        """目標角からのずれ [度]。支えが効いているかはこれで判定する。"""
        return {u["name"]: float(np.degrees(self.data.qpos[u["qadr"]]
                                            - self.model.qpos_spring[u["qadr"]]))
                for u in self._ids}
