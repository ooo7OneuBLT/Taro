"""四肢の「発達の向きの逆転」を解消する補正。⚠️恣意性は小さいが逸脱リスト対象。

【実測で判明した問題】筋力/自重比（＝gear ÷ その関節から先の重力モーメント）を月齢間で比較：
    90アクチュエータ中 **82個(91%)** で age=0 のほうが age=18 より大きい（中央値 ×1.49）
      四肢 : 肘36.3vs29.5 / 膝25.3vs18.8 / 肩9.6vs8.0 / 股6.9vs4.3 … **全て逆転**
      体幹 : chest_lean 0.75vs2.28 / chest_twist 0.38vs1.14 … **逆転なし**
＝**新生児のほうが相対的に強い**＝発達の向きが逆。新生児が最も弱いはず（首がすわるのは3-4ヶ月、
リーチは4-5ヶ月、寝返りは5ヶ月＝新生児は何もできない）という発達の順序と矛盾する。

【原因は解析的に説明できる】
    gear ∝ geom体積（MIMo `mimoGrowth/physics.py` の近似）
    重力モーメント = 質量 × g × 腕 = (体積 × 密度) × g × 腕
    ⇒ 比 = gear / (質量·g·腕) ∝ 1 / 腕
＝**四肢が短いほど比が大きくなる**。新生児は四肢が短いので比が跳ね上がる。
裏づけ：長さスケール比 (2.888/11.267)^(1/3)=0.635 と、逆転比中央値の逆数 1/1.49=0.671 が一致。

【補正の性格】
 ✅ 根拠がある部分：「新生児が最も弱い」という発達の順序（文献で確立）に反する**内部矛盾**を、
    測定値に基づいて打ち消す。倍率は恣意的な決め打ちでなく「逆転量そのもの」。
 ⚠️ 恣意的な部分：目標を「reference_age(既定18ヶ月)と同じ比」に置いたこと。本当は新生児は
    18ヶ月より**さらに弱い**はずだが、どれだけ弱いかの定量文献が無い（首と同じ事情）。
    ＝これは**最小限の補正**（逆転を消すだけ）であり、新生児の弱さの完全な再現ではない。
 ❌ 対象外：**体幹(chest_*)は逆転していないので触らない**。**首(head_*)は別ロジック**
    （head lagという臨床所見があるので比1.0を目標にできる＝`e_infant_neck.py`）。
"""
import numpy as np

G = 9.81
REFERENCE_AGE = 18.0        # 比の目標にする月齢（＝この月齢と同じ相対強度まで下げる）
_CACHE = {}                 # 参照月齢の比。環境構築が重いので月齢ごとに1回だけ測る


def _descendants(model, bid):
    ids = []
    for b in range(model.nbody):
        p = b
        while p != 0:
            if p == bid:
                ids.append(b)
                break
            p = model.body_parentid[p]
    return ids


def actuator_ratios(model, data):
    """各アクチュエータの「筋力 ÷ その関節から先の重力モーメント」を返す。"""
    out = {}
    for i in range(model.nu):
        jid = int(model.actuator_trnid[i, 0])
        if jid < 0:
            continue
        ids = _descendants(model, int(model.jnt_bodyid[jid]))
        mass = sum(model.body(k).mass[0] for k in ids)
        if mass <= 0:
            continue
        com = sum(model.body(k).mass[0] * data.xipos[k] for k in ids) / mass
        arm = float(np.linalg.norm((com - data.xanchor[jid])[:2]))
        tau = mass * G * arm
        out[model.actuator(i).name] = (abs(float(model.actuator_gear[i, 0])),
                                       tau, abs(float(model.actuator_gear[i, 0])) / max(tau, 1e-9))
    return out


def _reference_ratios(age_ref):
    """参照月齢のモデルを1度だけ作り、各アクチュエータの比を測ってキャッシュする。"""
    if age_ref in _CACHE:
        return _CACHE[age_ref]
    from d_supine_env import SupineMimoEnv     # 循環importを避けるため関数内で
    env = SupineMimoEnv(vision_params=None, age=age_ref)
    env.reset(seed=0)
    _CACHE[age_ref] = {k: v[2] for k, v in actuator_ratios(env.model, env.data).items()}
    env.close()
    return _CACHE[age_ref]


def apply_limb_inversion_fix(model, data, age, reference_age=REFERENCE_AGE,
                             verbose=True):
    """四肢のgearを下げ、age の比が reference_age の比を超えないようにする。

    首(head_*)・体幹(chest_*)は対象外（前者は別ロジック、後者は逆転していない）。
    比が既に reference 以下のアクチュエータは**触らない**（下げすぎないため）。

    Returns: dict(n_fixed, median_scale, before_median, after_median)
    """
    if age >= reference_age:
        if verbose:
            print(f"[limbs] age={age}mo >= {reference_age}mo: no correction")
        return dict(n_fixed=0, median_scale=1.0)

    ref = _reference_ratios(reference_age)
    cur = actuator_ratios(model, data)
    scales, before, after = [], [], []
    n = 0
    for i in range(model.nu):
        name = model.actuator(i).name
        if name not in cur or name not in ref:
            continue
        if name.startswith("act:head") or name.startswith("act:chest"):
            continue                      # 対象外（上のドキュメント参照）
        r_now, r_ref = cur[name][2], ref[name]
        before.append(r_now)
        if r_now <= r_ref * 1.0:          # 逆転していない＝触らない
            after.append(r_now)
            continue
        s = r_ref / r_now                 # 逆転量そのもの＝これで割る
        model.actuator_gear[i, 0] *= s
        scales.append(s)
        after.append(r_now * s)
        n += 1
    if verbose:
        ms = float(np.median(scales)) if scales else 1.0
        print(f"[limbs] age={age}mo: fixed {n} actuators, median gear scale x{ms:.3f} "
              f"(ratio median {np.median(before):.2f} -> {np.median(after):.2f}) "
              f"[target = same relative strength as age {reference_age:.0f}mo]")
    return dict(n_fixed=n, median_scale=float(np.median(scales)) if scales else 1.0,
                before_median=float(np.median(before)), after_median=float(np.median(after)))
