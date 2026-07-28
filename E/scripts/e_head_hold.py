"""★実験の測定条件 — 実験者が乳児の頭を抑える（視線誘導反射を測るため）。

⚠️【置き場所について、2026-07-28】最初 `taro_core/src/caregiver/hands.py` に置いたが、
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
  ③★関節のバネ（`jnt_stiffness` + `qpos_spring`）… MuJoCo が**物理ステップごとに解く**ので
    行き過ぎない。決めるパラメータは「支える強さ」1つだけ。← 採用

  ★目標角を**支え始めた時点の角度**にするので、静止しているあいだは力がゼロになる。
    ＝「触れているが押していない」という手の振る舞いに一致する。

【B-1 と B-2】ユーザーとの合意（2026-07-28）で、まず B-1 から始めて後で B-2 へ進む。
    B-1  ほぼ完全に固定する（HOLD_STIFFNESS_FIRM）
         まず「頭が固定されていれば視線誘導反射が測れるのか」を確かめる。
         柔らかさの調整を後回しにするのは、うまくいかなかったときに
         「支えが甘いのか反射が悪いのか」を切り分けられるようにするため。
    B-2  柔らかく支える（HOLD_STIFFNESS_SOFT）
         力を入れれば動くが放っておけば戻る＝枕で支えられた乳児に近い。

⚠️**これは太郎の身体ではなく実験の設定**。太郎自身は「頭が支えられている」ことを知らない。

【日常の世話は別に作る】やることリストの「世話をする他者が環境に居ない」
（うつ伏せから戻す・泣いたら抱く）は太郎の環境そのものなので core 側に作る。
このファイルはあくまで**測定のために頭を抑える**もので、両者を混ぜないこと。
"""
import numpy as np

# 支える強さ [N·m/rad]。
#
# ⚠️[Tier3・ARBITRARY] 親の手の剛性を測った文献は無い。次の物理から決めた目安：
#   新生児の頭の重力モーメントは約 0.38 N·m（仰向け・中立）。
#   バネが k のとき、ずれ θ で釣り合うので θ = 0.38 / k。
#       k =  21.7 → 1.0度ずれて釣り合う
#       k = 217   → 0.1度
#       k =  10   → 2.2度
#       k =   2   → 11度
# ★人間の実験で許される頭のずれの実測値は無い（Hunter & Richards 2003 は
#   「中心を見ていない試行は除外」とだけ書いてあり、角度の閾値が無い）。
HOLD_STIFFNESS_FIRM = 200.0   # B-1：ほぼ完全固定（0.1度程度のずれ）
HOLD_STIFFNESS_SOFT = 10.0    # B-2：柔らかく支える（2度程度のずれ）

# 支える首の関節（":" の後ろの名前）。頭の3軸すべて。
HOLD_JOINTS = ("head_swivel", "head_tilt", "head_tilt_side")


class CaregiverHands:
    """親が手で頭を支える。関節のバネとして表現する。

    使い方::

        hands = CaregiverHands(model, data)
        hands.hold()          # リセット直後に呼ぶ（今の角度で支え始める）
        ...                   # 実験を回す
        hands.release()       # 支えるのをやめる（元のバネに戻る）

    ⚠️`hold()` は**モデルの値を書き換えるだけ**なので毎ステップ呼ぶ必要はない。
      毎ステップ呼んでも害はないが、目標角が更新されるので支えの意味が変わる
      （＝「今いる場所で支え続ける」＝頭がゆっくり流れていってしまう）。
    """

    def __init__(self, model, data, joints=None, stiffness=None, damping=None):
        """
        Args:
            stiffness: 支える強さ [N·m/rad]。None なら B-1（ほぼ完全固定）。
            damping: 減衰。None なら臨界減衰（ちょうど振動しない値）を計算する。
        """
        self.model = model
        self.data = data
        self.joints = tuple(HOLD_JOINTS if joints is None else joints)
        self.stiffness = float(HOLD_STIFFNESS_FIRM if stiffness is None else stiffness)
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

        c = 2√(k·I)。⚠️文献値ではなく**工学的判断**（安定性優先）[Tier3]。
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
    def hold(self, target=None, stiffness=None, verbose=False):
        """頭を支え始める。

        Args:
            target: 目標角の辞書 {関節名: 度}。None なら**今の角度**で支える
                （＝静止中は力がゼロ＝触れているだけの手）。
            stiffness: 強さの上書き。None なら初期化時の値。
        Returns:
            支えた関節の数
        """
        k = float(self.stiffness if stiffness is None else stiffness)
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
            # ⚠️減衰は元の値と臨界減衰の**大きい方**（元より弱くはしない）
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
        """目標角からのずれ [度]。★支えが効いているかはこれで判定する。"""
        return {u["name"]: float(np.degrees(self.data.qpos[u["qadr"]]
                                            - self.model.qpos_spring[u["qadr"]]))
                for u in self._ids}
