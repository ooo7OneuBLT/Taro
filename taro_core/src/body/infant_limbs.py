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
import json
import os

import numpy as np

G = 9.81
REFERENCE_AGE = 18.0        # 比の目標にする月齢（＝この月齢と同じ相対強度まで下げる）
_CACHE = {}                 # 参照月齢の比。プロセス内で1回だけ読む

# ★★【2026-07-29 重大なバグの修正】基準を「その場で測る」のをやめ、保存した値を読む。
#
# 【何が起きていたか】この基準は「素の18ヶ月の体」の比でなければならないのに、
#   以前は apply_limb_inversion_fix の中で **裏に18ヶ月の体を1体作って測って**いた。
#   ところがそれが呼ばれるのは「太郎の体を作っている最中」＝新生児の体型補正
#   （手0.70倍・脚0.80倍…）が既に効いた状態で、その補正が裏の18ヶ月の体にも漏れていた。
#   実測（2026-07-29）：
#       何も作らずに測った基準            median = 64.00   ← ★正しい
#       age=4.0 の体を作る途中で測った基準  median = 64.00   （体型補正が無いので無事）
#       age=0.0 の体を作る途中で測った基準  median = 135.93  ← ★2.1倍ずれ
#   結果、**0〜3ヶ月の太郎は79個中31個しか筋力を下げられていなかった**
#   （＝自分の体に対して2倍強い腕を持っていた＝この補正が直そうとした
#     「発達の向きの逆転」が、新生児でだけ直っていなかった）。
#
# 【なぜ定数（保存値）にするか】
#   ・測る時点の状態に依存しない＝**汚染されない**
#   ・体を1体作るたびに裏でもう1体作らずに済む（段階的成長で36回作り直すと36体分の無駄）
#   ・値がファイルに見える＝後から検証・議論できる
# ⚠️体の定義（体型・質量・関節）を変えたら**基準も測り直す**必要がある。
#   照合と再生成： `.venv/Scripts/python.exe taro_core/tools/measure_limb_reference.py`
#   （--update で更新。何も付けなければ保存値と実測を比べるだけ）
_REF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "limb_reference_18mo.json")


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


def measure_reference_ratios(age_ref=REFERENCE_AGE):
    """参照月齢の素の体を作って比を測る。

    ⚠️★**他の体を1体も作っていない状態で呼ぶこと**。新生児の体型補正が効いた後に
      呼ぶと、その補正がここで作る体にも漏れて基準が2倍ずれる（上の説明を参照）。
      通常の実行から呼んではいけない。基準を作り直す tools からだけ呼ぶ。
    """
    from d_supine_env import SupineMimoEnv     # 循環importを避けるため関数内で
    env = SupineMimoEnv(vision_params=None, age=age_ref)
    env.reset(seed=0)
    out = {k: v[2] for k, v in actuator_ratios(env.model, env.data).items()}
    env.close()
    return out


def _reference_ratios(age_ref):
    """保存してある基準（素の18ヶ月の体の比）を読む。★測り直さない。"""
    if age_ref in _CACHE:
        return _CACHE[age_ref]
    if not os.path.exists(_REF_PATH):
        raise FileNotFoundError(
            f"四肢の筋力補正の基準がない: {_REF_PATH}\n"
            "  作る: .venv/Scripts/python.exe taro_core/tools/measure_limb_reference.py --update\n"
            "  ⚠️その場で測る実装に戻してはいけない（新生児の体型補正が漏れて基準が2.1倍ずれる）")
    blob = json.load(open(_REF_PATH, encoding="utf-8"))
    if abs(float(blob["age"]) - float(age_ref)) > 1e-9:
        raise ValueError(f"基準の月齢が違う: 保存={blob['age']} 要求={age_ref}。"
                         "measure_limb_reference.py --update で作り直すこと")
    _CACHE[age_ref] = {k: float(v) for k, v in blob["ratios"].items()}
    _vals = list(_CACHE[age_ref].values())
    print("[limbs-ref] age=%.1f の基準比 median=%.2f n=%d （保存値を使用: %s）" %
          (age_ref, float(np.median(_vals)), len(_vals),
           os.path.basename(_REF_PATH)), flush=True)
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


# ============================================================================
# ★四肢の筋緊張（2026-07-29 新設）— 脱力しても腕が体の前に保たれる
# ----------------------------------------------------------------------------
# 【なぜ要るか】4ヶ月の太郎を脱力させると、腕が重力で真横に伸びきる。
# 実測：手からおもちゃまで 22.7cm。人間の4ヶ月児は脱力しても肘が曲がり、
# 手が体の前にあるので、対象までの距離がはるかに短い。この差のせいで
# リーチングの実験が成立していなかった（30秒の自発運動で接触0回）。
#
# 【人間ではどうか（2026-07-29 の文献調査）】
#   ★**3〜5ヶ月児の安静時の関節角度を度数で測った研究は存在しない**。
#     乳児の筋緊張評価は Amiel-Tison法・HINE など**すべて段階評価（0/1/2）**で、
#     連続値の角度やトルクを報告していない
#     （Goo, Tucker & Johnston 2018, Dev Med Child Neurol の系統的レビューで確認。
#      97研究21手法を調べ、妥当性が中程度以上の4手法はいずれも序数尺度）。
#   ⇒ 「値が無い」ことが確定した壁。1つに決めず**振って確かめる**（感度分析）。
#
#   間接的な根拠は一致している：
#     Dubowitz et al. 1970  満期新生児の安静姿勢は「腕も脚も完全屈曲」（Score 4）。
#                           腕を伸ばして離すと **90度以上**まで自分で戻る（arm recoil）
#     Shah et al. 2017      新生児の肘は他動でも **19.2度** 伸びきらない（構造的制限）
#     Watanabe et al. 1979  肘の伸展制限 出生時 -14度 → 2〜4週 -6度 → **4〜8ヶ月で0度**
#     Solopova et al. 2019  他動運動への反射的な筋反応は **6ヶ月頃まで残る**
#                           （Front Physiol 10:1158、健常児54名）
#     定性的な記述          3〜7ヶ月の典型的安静姿勢は「肩外転・肘屈曲」。
#                           伸展位という報告は**皆無**
#
#   ★重要な区別：**関節の構造的な制限（4〜8ヶ月で解消）**と
#     **筋緊張による屈曲（6ヶ月頃まで残る）**は別物。太郎に欠けているのは後者。
#     4ヶ月は「構造的な壁はほぼ消えたが、筋緊張はまだ効いている」時期にあたる。
#
# 【首との違い】首は `infant_body.apply_neck_tone`。剛性 0.40 N·m/rad は
# 死後標本の実測上限（Luck 2008）という根拠があるが、四肢には対応する実測が無い。
# ⇒ 四肢は「腕が体の前に保たれる」という**機能から逆算**する（下記 tone_from_gravity）。
#
# ⚠️これは太郎の**身体の性質**であって脳ではない。意思とも別。
#   バネなので、筋の指令がバネより強ければ腕は動く（力を抜けば戻る）。
LIMB_TONE_UNTIL_MO = 6.0     # [Tier2] Solopova 2019：反射的な筋反応は6ヶ月頃まで
# ★部位を細かく分ける（2026-07-29）。
#   【なぜ】ひじだけ筋緊張がきつすぎて可動域の15%しか使えず、自発運動で
#   おもちゃに近づけなかった（実測：ひじ151度の可動域に対し28度しか動かない）。
#   一方で肩は緩めると腕が垂れる。⇒ **部位ごとに強さを変えられる必要がある**。
LIMB_TONE_GROUPS = {
    "shoulder": ("shoulder_horizontal", "shoulder_ad_ab", "shoulder_rotation"),
    "elbow": ("elbow",),
    "wrist": ("hand1", "hand2", "hand3"),
    "hip": ("hip1", "hip2", "hip3"),
    "knee": ("knee",),
    "ankle": ("foot1", "foot2", "foot3"),
}
# まとめて指定するための別名（従来の "arm" / "leg" もそのまま使える）
LIMB_TONE_ALIASES = {
    "arm": ("shoulder", "elbow", "wrist"),
    "leg": ("hip", "knee", "ankle"),
}
GROUP_JP_LIMB = {"shoulder": "肩", "elbow": "ひじ", "wrist": "手首",
                 "hip": "股", "knee": "ひざ", "ankle": "足首"}


def _expand_groups(groups):
    """"arm" のような別名を細かい部位に展開する。"""
    out = []
    for g in (groups or ()):
        out.extend(LIMB_TONE_ALIASES.get(g, (g,)))
    return tuple(dict.fromkeys(out))     # 重複を消して順序は保つ


def limb_tone_joints(model, groups=("arm", "leg")):
    """筋緊張を効かせる関節の名前（":" の後ろ）を返す。"""
    return [nm for nm, _g in limb_tone_joints_by_group(model, groups)]


def limb_tone_joints_by_group(model, groups=("arm", "leg")):
    """(関節名, 部位名) の一覧を返す。部位ごとに強さを変えるために使う。"""
    import mujoco as _mj
    want = {}
    for g in _expand_groups(groups):
        for base in LIMB_TONE_GROUPS.get(g, ()):
            want[base] = g
    out = []
    for j in range(model.njnt):
        if int(model.jnt_type[j]) != int(_mj.mjtJoint.mjJNT_HINGE):
            continue
        nm = (model.joint(j).name or "").split(":")[-1]
        base = nm
        for pre in ("right_", "left_"):
            if base.startswith(pre):
                base = base[len(pre):]
                break
        if base in want:
            out.append((nm, want[base]))
    return out


def gravity_moment(model, data, joint_name, worst_case=True):
    """その関節から先（末端側）にぶら下がる重力モーメント [N·m] を測る。

    ＝「バネがどれだけの力に抗う必要があるか」。剛性を機能から逆算するために使う。

    Args:
        worst_case: True（既定）なら**姿勢によらない上界**を返す。
            False なら「今この瞬間の」モーメント。

    ⚠️★【2026-07-29 に踏んだ】最初は「今の姿勢での軸まわりのモーメント」を
      使っていた。ところが腕が関節軸と平行に近い姿勢だと重力が効かず、
      モーメントがほぼゼロと計算される。実測：右肩の水平方向で **0.016**
      （左は0.073）＝ほぼゼロの剛性が入り、**右手だけ17cm落ちた**。
      姿勢が少し変われば重力は効き始めるので、その瞬間の値で決めてはいけない。

      ⇒ 既定を「軸に垂直な距離 × 重さ」の合計に変えた。
        関節をどう回してもその先の相対配置は変わらないので、この値は
        **姿勢によらず、重力がいちばん効く向きでのモーメント**にあたる。
    """
    import numpy as _np
    jid = None
    for j in range(model.njnt):
        if (model.joint(j).name or "").split(":")[-1] == joint_name:
            jid = j
            break
    if jid is None:
        return None
    axis_world = _np.array(data.xaxis[jid], dtype=float)
    na = float(_np.linalg.norm(axis_world))
    if na > 1e-12:
        axis_world = axis_world / na
    anchor = _np.array(data.xanchor[jid], dtype=float)
    tau = 0.0
    for bid in _descendants(model, int(model.jnt_bodyid[jid])):
        mass = float(model.body_mass[bid])
        if mass <= 0.0:
            continue
        com = _np.array(data.xipos[bid], dtype=float)
        r = com - anchor
        if worst_case:
            # 軸に垂直な距離＝てこの最大の腕。重力の向きに関係なく決まる
            perp = r - _np.dot(r, axis_world) * axis_world
            tau += mass * G * float(_np.linalg.norm(perp))
        else:
            f = _np.array([0.0, 0.0, -mass * G], dtype=float)
            tau += float(_np.dot(axis_world, _np.cross(r, f)))
    return abs(tau)


def tone_from_gravity(model, data, joint_name, hold_deg=20.0):
    """「重力に対して hold_deg 度のずれで釣り合う」剛性 [N·m/rad] を返す。

    【なぜこの決め方か】四肢の筋緊張の実測値は文献に存在しない（上記）。
    値を勘で決める代わりに、**機能から逆算**する：
        バネ剛性 k のとき、重力モーメント τ に対するずれは θ = τ / k
        ⇒ 「θ度のずれで止まってほしい」なら k = τ / radians(θ)
    ⚠️hold_deg 自体は文献値ではない[Tier3]。ただし「腕が保たれる／落ちる」という
      観察可能な条件に対応しているので、勘で剛性を決めるより検証しやすい。
    """
    import numpy as _np
    tau = gravity_moment(model, data, joint_name)
    if tau is None:
        return None
    return float(tau / max(_np.radians(float(hold_deg)), 1e-9))


def apply_limb_tone(model, data, age=0.0, target=None, stiffness=None,
                    hold_deg=20.0, groups=("arm", "leg"),
                    until_mo=LIMB_TONE_UNTIL_MO, verbose=True):
    """四肢の筋緊張＝脱力しても腕が体の前に保たれる弱いバネ。

    Args:
        target: 戻る目標角の辞書 {関節名: 度}。**None なら「今の姿勢」を目標にする**
            ＝Viewer で作った姿勢がそのまま筋緊張の落ち着き先になる
        stiffness: 剛性 [N·m/rad]。None なら重力から逆算（`tone_from_gravity`）
        hold_deg: 逆算に使う「許すずれ」[度]。小さいほど硬い。
            ★**辞書 {部位名: 度} を渡すと部位ごとに変えられる**
            （例 {"shoulder":10, "elbow":60}）。数値なら全部位に同じ値。
        groups: "arm" / "leg"、または細かい部位（"shoulder" / "elbow" / "wrist" /
            "hip" / "knee" / "ankle"）
        until_mo: この月齢を超えたら何もしない

    ⚠️部位ごとに変えられるようにした理由（2026-07-29 の実測）：
      ひじだけ筋緊張がきつすぎて可動域151度のうち**28度しか使えず**、
      自発運動でおもちゃに近づけなかった。緩めると近づくが、肩まで緩めると
      腕が垂れて出発点が遠くなる。⇒ 部位ごとに別の値が要る。

    Returns:
        dict(n=効かせた関節数, k=関節名→剛性)
    """
    import numpy as _np
    import mujoco as _mj
    if float(age) >= float(until_mo):
        if verbose:
            print(f"[limb-tone] age={age}mo >= {until_mo}mo: 何もしない")
        return dict(n=0, k={})
    _mj.mj_forward(model, data)      # ⚠️重力モーメントを測る前に姿勢を確定させる
    pairs = dict(limb_tone_joints_by_group(model, groups))
    names = set(pairs)
    n, ks = 0, {}
    for j in range(model.njnt):
        nm = (model.joint(j).name or "").split(":")[-1]
        if nm not in names:
            continue
        # 部位ごとの「許すずれ」。辞書でなければ全部位に同じ値
        _hd = (float(hold_deg.get(pairs[nm], hold_deg.get("default", 10.0)))
               if isinstance(hold_deg, dict) else float(hold_deg))
        # 部位ごとの剛性指定にも対応
        _st = (stiffness.get(pairs[nm]) if isinstance(stiffness, dict) else stiffness)
        adr = int(model.jnt_qposadr[j])
        dof = int(model.jnt_dofadr[j])
        k = (float(_st) if _st is not None
             else tone_from_gravity(model, data, nm, _hd))
        if k is None or not _np.isfinite(k):
            continue
        tgt = (_np.radians(float(target[nm])) if (target and nm in target)
               else float(data.qpos[adr]))       # None なら今の角度
        model.jnt_stiffness[j] = k
        model.qpos_spring[adr] = tgt
        # 減衰は臨界減衰（ちょうど振動しない値）。⚠️工学的判断で実測ではない[Tier3]
        #   首（apply_neck_tone）と同じ式にそろえてある
        bid = int(model.jnt_bodyid[j])
        inertia = float(model.body_inertia[bid][0])
        mass = float(model.body_mass[bid])
        arm = float(_np.linalg.norm(_np.array(data.xipos[bid])
                                    - _np.array(data.xanchor[j])))
        I = max(inertia + mass * arm ** 2, 1e-12)
        c_crit = 2.0 * float(_np.sqrt(max(k, 1e-12) * I))
        model.dof_damping[dof] = max(float(model.dof_damping[dof]), c_crit)
        data.qvel[dof] = 0.0
        ks[nm] = k
        n += 1
    if verbose and n:
        vals = list(ks.values())
        if isinstance(hold_deg, dict):
            how = "重力から逆算（部位ごと " + " ".join(
                f"{GROUP_JP_LIMB.get(g, g)}{v:g}度" for g, v in hold_deg.items()) + "）"
        else:
            how = f"重力から逆算（{float(hold_deg):g}度のずれで釣り合う）"
        print(f"[limb-tone] age={age}mo: {n}関節 剛性{how} "
              f"中央値{_np.median(vals):.3f}N·m/rad "
              f"目標{'今の姿勢' if not target else '指定'} "
              f"[⚠️Tier3: 乳児の四肢の筋緊張の実測値は文献に存在しない"
              f"（Goo et al. 2018 の系統的レビューで確認）。"
              f"間接根拠＝Dubowitz 1970・Solopova 2019 は屈曲位を支持]")
    return dict(n=n, k=ks)
