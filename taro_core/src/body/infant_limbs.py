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
 ⚠️ **この補正の目標値には根拠が無い**（下の★2026-07-25 参照）。目標を
    「reference_age(既定18ヶ月)と同じ比」に置いたのは**恣意的**[Tier3]。
 ❌ 対象外：**体幹(chest_*)は逆転していないので触らない**。**首(head_*)は別ロジック**
    （head lagという臨床所見があるので比1.0を目標にできる＝`infant_neck.py`）。

【★2026-07-25 文献調査で前提が揺らいだ（重要・格下げ）】
当初この補正は「新生児が最も弱いという発達の順序に反する**内部矛盾**を打ち消す」ものとして
根拠があると考えていた。しかし文献調査（文献調査4本）で**前提そのものが疑わしく**なった。

★**「体重あたりの相対筋力」は月齢とともに増えるのではなく、むしろ微減する**方向を
2つの独立した根拠が示す：
  1. **実測**：出生→24ヶ月で体重は約3.6倍、除脂肪量は約3.1倍
     （MIBCRS/PMC11537963・PMC7070317）＝**体重あたりでは微減**
  2. **理論**：筋力∝断面積(L²)、重力モーメント∝L⁴ ⇒ 比∝L⁻²
     ＝**大きくなるほど不利**（Jaric 2003 の 0.67乗則。アリと象の関係と同じ）
＝**「新生児のほうが相対的に強い」のはスケーリング則としては正しい**可能性がある。
＝この補正は**逆転を直しているのではなく、正しい姿を壊しているかもしれない**。

一方で反対方向の要因もある：**筋の「質」（比張力＝単位断面積あたりの力）が未成熟**
（ヒト胎児筋は成人の約7%、Lindqvist et al. PMC3832119）。ただしこれは妊娠12〜15週の
データで満期産新生児ではなく、「小児と成人で比張力に差なし」（O'Brien）という
**反対の報告もある**。⇒ **どちらが優勢か判定できるデータは存在しない**。

★**新生児の関節トルクの直接測定は文献に存在しない**（新生児に最大随意努力を求める測定は
原理的に不可能＝「随意最大努力」という概念が成立しない）。MIMo開発チーム自身が
「乳児の筋力測定は稀で、その大半は複数筋を使うタスク遂行能力の測定」と明記している
（arXiv:2509.09805）。國吉研の胎児モデル(390筋)も筋力パラメータの根拠が非公開。
＝**この空白は「調べれば分かる」ものではなく、分野全体が抱える欠落**。

【★だからどうするか（2026-07-25 の判断）】
1. 補正は**維持する**。変える根拠も無いため（外す根拠＝上記1,2 と、維持する根拠＝筋の質、
   のどちらも決定的でない）。ただし**根拠が無いことを隠さない**＝このラベル。
2. `scale` 引数で**感度分析**する。値は決められないが「値の不確実性が結論を左右するか」は
   決められる。⚠️差が出たら「太郎の結論は根拠のない筋力の仮定に乗っている」という
   **弱点の発見**になる（通常の実験と目的が逆＝差が出ないことを確かめたい）。
3. ★**「新生児が寝返りできてしまう」問題はこの補正では解かない**。文献の結論は
   「寝返り不能の主因は筋力ではなく**協調（体幹の分節的回旋）と神経成熟（皮質脊髄路の
   髄鞘化）**」。決め手＝**脳性麻痺（痙直型）は筋緊張がむしろ過剰なのに寝返りが遅れる**。
   ⇒ 対処は**制御側**（体幹の分節化を新生児期は制限する）。
   詳細は `doc/人間模倣からの逸脱リスト.md` 2026-07-25続き14。
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


def actuator_ratios(model, data, actuation_model=None):
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
        # ⚠️★【2026-07-25 修正】筋力を actuator_gear から読んでいたが、MuscleModel は
        #   毎ステップ gear を上書きする（muscle.py:331）ので筋力の指標にならない。
        #   筋肉モデルでは actuation_model.fmax を読む（infant_body.actuator_strength）。
        from infant_body import actuator_strength
        _s = actuator_strength(model, i, actuation_model)
        out[model.actuator(i).name] = (_s, tau, _s / max(tau, 1e-9))
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
                             verbose=True, scale=1.0, actuation_model=None):
    """四肢のgearを下げ、age の比が reference_age の比を超えないようにする。

    首(head_*)・体幹(chest_*)は対象外（前者は別ロジック、後者は逆転していない）。
    比が既に reference 以下のアクチュエータは**触らない**（下げすぎないため）。

    Args:
        scale: ★補正後の gear にさらに掛ける係数（感度分析用。1.0＝補正そのまま）。
            **この補正の目標値（reference_age＝18ヶ月と同じ相対強度）には根拠が無い**ため
            （下の【★2026-07-25 文献調査で前提が揺らいだ】参照）、結論がこの仮定に
            依存していないかを確かめるために振る。>1 で強く、<1 で弱くなる。

    Returns: dict(n_fixed, median_scale, before_median, after_median)
    """
    if age >= reference_age:
        if verbose:
            print(f"[limbs] age={age}mo >= {reference_age}mo: no correction")
        return dict(n_fixed=0, median_scale=1.0)

    ref = _reference_ratios(reference_age)
    cur = actuator_ratios(model, data, actuation_model)
    scales, before, after = [], [], []
    n = 0
    _unapplied = 0
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
        s = (r_ref / r_now) * float(scale)   # 逆転量そのもの＝これで割る（×感度分析の係数）
        # ★筋力の書き換えは infant_body 経由（筋肉モデルなら fmax、それ以外は gear）。
        from infant_body import scale_actuator_strength
        if not scale_actuator_strength(model, i, s, actuation_model):
            _unapplied += 1
        scales.append(s)
        after.append(r_now * s)
        n += 1
    if verbose:
        ms = float(np.median(scales)) if scales else 1.0
        _tag = "" if abs(float(scale) - 1.0) < 1e-9 else f" [SENSITIVITY scale=x{scale}]"
        if _unapplied:
            _tag += f" ⚠️[{_unapplied}件未適用＝fmaxがスカラーで関節ごとに変えられない]"
        print(f"[limbs] age={age}mo: fixed {n} actuators, median strength scale x{ms:.3f} "
              f"(ratio median {np.median(before):.2f} -> {np.median(after):.2f}) "
              f"[target = same relative strength as age {reference_age:.0f}mo]{_tag}")
    return dict(n_fixed=n, median_scale=float(np.median(scales)) if scales else 1.0,
                before_median=float(np.median(before)), after_median=float(np.median(after)))
