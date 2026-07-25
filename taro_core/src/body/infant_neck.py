"""新生児の「首がすわっていない」を再現する補正。⚠️**恣意的な逸脱**（下記ラベリング参照）。

【なぜ要るか＝実測で判明したMIMoの限界】
首の「持ち上げ能力比」＝(首のトルク) ÷ (頭の重力モーメント) を実測したところ：
    age=0（新生児） : 4.21倍   ← 新生児のほうが相対的に**強い**
    age=18ヶ月      : 3.00倍
＝**発達の向きが逆転している**。原因は MIMo の `mimoGrowth/physics.py` 冒頭に明記のとおり
「乳児の詳細な筋力測定データが無いので、gear値(筋力)を最も近いgeomの**体積**から計算した」
ため。新生児は頭が相対的に大きい(体の24.7%)＝頭geomの体積が大きい→首の筋力も大きく計算され、
かつ首関節〜頭重心の腕が短い→重力モーメントは小さい。この2つが重なって比が跳ね上がる。
実際の新生児は head lag（引き起こすと頭が遅れる）が標準所見で、首がすわるのは3〜4ヶ月。

【文献で確かめられたこと / 確かめられなかったこと】
  ✅ 頸部ROM：回転110度・側屈70度（健常乳児38人の実測）→ MIMoの±111/±70は**正しい**＝触らない
  ✅ 首がすわるのは3〜4ヶ月＝それ以前は相対的に弱い（発達の"向き"は確立した事実）
  ❌ 新生児の頸部筋力の**定量値(N·m)は存在しない**（臨床は質的スコアで評価。2ヶ月児の
     筋機能スコア中央値=1、10ヶ月で3〜4）
  ❌ head lag の角度カットオフも「文献に無い」と明記されている
  ❌ 正常な新生児も「引き上げ時に頭は遅れるが**完全には倒れない**」＝比<1.0が正しいとも限らない
  → **どれだけ弱いべきかは文献から決められない**。以下は恣意的な設定。

【⚠️恣意的な部分（逸脱リストに記録すること）】
  ①**目標比を1.0**に置いた。物理的に一意な点（頭の重さとちょうど釣り合う境界）を選んで
    恣意性を最小化したが、文献の裏づけはない。
  ②**4ヶ月で補正解除する線形の回復曲線**。首がすわる時期(3〜4ヶ月)は文献どおりだが、
    曲線の形（線形）は恣意的。
  ③**tiltのみ**補正し swivel/tilt_side は触らない。仰向けでは回転・側屈は重力に抗さない
    ので head lag を根拠に下げられないため（解剖学的にも屈曲は別の筋群）。
    ＝「振り回し」が残る場合、それは首の筋力でなく**方策**の問題（E1の学習で解く対象）。
"""
import numpy as np

GRAVITY = 9.81
TILT_JOINT = "robot:head_tilt"
TILT_ACTUATOR = "act:head_tilt"
TARGET_RATIO_AT_BIRTH = 1.0    # ⚠️恣意的（上記①）
HEAD_CONTROL_AGE = 4.0         # 首がすわる月齢。ここで補正解除 ⚠️曲線の形は恣意的（上記②）


def _head_bodies(model):
    """head とその子孫（目など）のbody idを集める。頭として一緒に持ち上がる質量。"""
    ids = []
    for b in range(model.nbody):
        p = b
        while p != 0:
            if model.body(p).name == "head":
                ids.append(b)
                break
            p = model.body_parentid[p]
    return ids


def head_gravity_torque(model, data):
    """今の姿勢での「頭の重力モーメント」[N·m]＝首が支えるべき負荷。

    首関節(head_tilt)のアンカーから頭群の重心までの**水平距離**を腕として計算する
    （重力は鉛直なので、モーメントの腕は水平成分）。
    """
    jid = [j for j in range(model.njnt) if model.joint(j).name == TILT_JOINT][0]
    anchor = data.xanchor[jid]
    ids = _head_bodies(model)
    mass = sum(model.body(i).mass[0] for i in ids)
    com = sum(model.body(i).mass[0] * data.xipos[i] for i in ids) / mass
    arm = float(np.linalg.norm((com - anchor)[:2]))
    return float(mass * GRAVITY * arm), float(mass), arm


def lift_ratio(model, data):
    """持ち上げ能力比＝首のトルク ÷ 頭の重力モーメント。<1 なら頭を持ち上げられない。"""
    tau, _, _ = head_gravity_torque(model, data)
    aid = [i for i in range(model.nu) if model.actuator(i).name == TILT_ACTUATOR][0]
    gear = abs(float(model.actuator_gear[aid, 0]))
    return gear / max(tau, 1e-9), gear, tau


def target_ratio_for_age(age, birth_ratio=TARGET_RATIO_AT_BIRTH,
                         settle_age=HEAD_CONTROL_AGE, natural_ratio=None):
    """月齢に応じた目標比。0ヶ月=birth_ratio、settle_age以降は補正なし（＝元の比）。

    ⚠️線形補間は恣意的（逸脱リスト）。ただし「新生児が最も弱く成長で強くなる」という
    **向き**は文献（首がすわるのは3〜4ヶ月）に一致させている。
    """
    if natural_ratio is None or age >= settle_age:
        return None if age >= settle_age else birth_ratio
    w = max(0.0, min(1.0, age / settle_age))
    return (1.0 - w) * birth_ratio + w * natural_ratio


def apply_newborn_neck(model, data, age, birth_ratio=TARGET_RATIO_AT_BIRTH,
                       settle_age=HEAD_CONTROL_AGE, verbose=True):
    """首(tilt)のgearを、月齢に応じた目標比になるよう補正する。

    ⚠️**恣意的な逸脱**。ROM・swivel・tilt_side・他の関節は一切変更しない。
    Returns: (before_ratio, after_ratio) 補正しない場合は after=before。
    """
    before, gear, tau = lift_ratio(model, data)
    if age >= settle_age:
        if verbose:
            print(f"[neck] age={age}mo >= {settle_age}mo: no correction (ratio {before:.2f})")
        return before, before
    target = target_ratio_for_age(age, birth_ratio, settle_age, natural_ratio=before)
    aid = [i for i in range(model.nu) if model.actuator(i).name == TILT_ACTUATOR][0]
    new_gear = target * tau
    sign = np.sign(model.actuator_gear[aid, 0]) or 1.0
    model.actuator_gear[aid, 0] = sign * new_gear
    after, _, _ = lift_ratio(model, data)
    if verbose:
        print(f"[neck] age={age}mo: lift ratio {before:.2f} -> {after:.2f} "
              f"(gear {gear:.3f} -> {new_gear:.3f} N.m, head torque {tau:.3f} N.m) "
              f"[ARBITRARY: see deviation list]")
    return before, after
