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
  ①~~目標比を1.0に置いた~~ → ★**2026-07-28 に 0.5 へ改訂し、根拠が付いた**。
    Öhman & Beckung (2008) の2ヶ月児の実測（頭を体幹と水平に5秒保持できる＝比1.0）と
    4ヶ月の目標を結んだ直線の外挿値。詳細は TARGET_RATIO_AT_BIRTH のコメント。
    ⚠️ただし姿勢が違う（文献は腹臥位懸垂／太郎は仰向け）ので測り直しが要る。
  ②**線形の回復曲線**。首がすわる時期(3〜4ヶ月)は文献どおりだが、曲線の形（線形）は恣意的。
    ★2026-07-28 に「4ヶ月で打ち切る」のをやめた。打ち切ると比が 24.4 → 47.5 と
    崖のように跳ぶ（＝成人男性より強い首になる）ため、18ヶ月まで繋いで連続にした。
  ③**tiltのみ**補正し swivel/tilt_side は触らない。仰向けでは回転・側屈は重力に抗さない
    ので head lag を根拠に下げられないため（解剖学的にも屈曲は別の筋群）。
    ＝「振り回し」が残る場合、それは首の筋力でなく**方策**の問題（E1の学習で解く対象）。
  ④★**曲げる筋と伸ばす筋を区別していない**（2026-07-28 に判明）。MIMoの筋肉モデルは
    1関節に2本の筋があり、力が別々に設定されている（head_tilt は負方向31.39／正方向16.86
    ＝1.86倍の差。人間でも首の後ろが前より1.4〜1.9倍強く、この比自体は妥当）。
    太郎の補正は**両方に同じ倍率**を掛けており、方向ごとの調整はしていない[簡略化]。

【★2026-07-28 の文献調査＝判断の足場（重要）】
  成人の持ち上げ能力比（比の意味は lift_ratio 参照）を一次文献から計算した：
    Vasavada, Li & Delp (2001) Spine 26(17):1904-1909（男11・女5の実測、全文入手）
    ＋頭部質量（Suderman & Vasavada 2019, PMC6986765）＋モーメントアーム 4.3〜7.4cm
      成人男性  伸展 17.1〜29.3倍 ／ 屈曲  9.8〜16.9倍
      成人女性  伸展  8.0〜13.8倍 ／ 屈曲  5.7〜 9.8倍
  → 太郎の18ヶ月（伸展27.7／屈曲14.9）は**成人男性の範囲に収まる**。
    ＝**MIMoの素の首の筋力そのものは妥当**で、問題は「その筋力を小さい体に載せると
      相対的に強くなりすぎる」ことだけだった（4ヶ月で47.5倍＝成人を超える）。

  ★Lavallee, Ching & Nuckley (2013) J Biomech 46:527-534（6〜23歳 n=91）の成熟曲線
    （%MVC = -0.0879·年齢² + 6.018·年齢 + 8.120）を0歳へ**外挿する案は破綻した**。
    外挿すると0ヶ月でも比 8.8〜15倍になり、head lag と矛盾する（データ下限が6歳なので
    切片8.12%がそのまま残るため）。この論文から使えるのは向きだけ
    ＝「8歳児は頭囲が成人の91%なのに首の強さは成人の50%」＝頭が先に育ち、首は後から追う。
"""
import numpy as np

GRAVITY = 9.81
TILT_JOINT = "robot:head_tilt"
TILT_ACTUATOR = "act:head_tilt"
# ★【2026-07-28 改訂：1.0 → 0.5】0ヶ月の目標比。
#
# 【なぜ変えたか】1.0 は「頭の重さとちょうど釣り合う境界」＝物理的に一意な点なので
# 恣意性が無い、という理由で選ばれていた。しかしそれだと**生まれた瞬間から頭を
# 持ち上げられる**ことになり、head lag（新生児は引き起こすと頭が遅れる。標準所見）を
# 再現できない。首がすわるのは3〜4ヶ月なので、0ヶ月は比<1.0 でなければおかしい。
#
# 【0.5 はどこから来たか】★勘で決めた数字ではなく、**実測点からの外挿**。
#
#   根拠のある点  2ヶ月で比 ≈ 1.0
#     Öhman & Beckung (2008) Pediatr Phys Ther 20(1):53-58
#     DOI:10.1097/PEP.0b013e31815ebb27（正期産・健常児 n=38、2/4/6/10ヶ月）[Tier1]
#     体幹を水平に保持して頭部の支持を外し、頭の位置を5段階で採点する検査
#     （0=水平より下に垂れる／1=体幹と水平／2=やや上／3=かなり上／4=非常に高く上）。
#     ★**各段階は5秒以上保持できて初めて認定**される＝静的な保持能力の測定。
#     月齢別の平均：2ヶ月 1.0（範囲0〜2）／4ヶ月 2.6／6ヶ月 3.0／10ヶ月 3.4
#     → 段階1「頭が体幹と水平」は**物理的に意味が一意**（頭の重力モーメントが最大に
#       なる姿勢を静的に保てる）＝**比 1.0 に対応する**。
#       2ヶ月児の平均がこの段階なので、2ヶ月 ≈ 比1.0 と読める。
#
#   仮に置いた点  4ヶ月で比 1.5（HEAD_CONTROL_RATIO。根拠は下の定数のコメント）
#
#   → この2点を結んだ直線を0ヶ月へ伸ばすと **0.5**。
#     （0.5 + 0.25×月齢 という直線。2ヶ月で1.00、4ヶ月で1.50）
#
# ⚠️【この値の弱いところ・必ず読むこと】
#   1. ★**姿勢が違う**。Öhman の検査は体幹を水平に持ち上げた姿勢（腹臥位懸垂）で、
#      太郎がこの比を測っているのは**仰向け**。姿勢が変われば頭の重力モーメントも
#      変わるので、厳密には**太郎で同じ姿勢を作って測り直す必要がある**。
#      → やることリストに登録（首座りの測定器を作るときに一緒にやる）。
#   2. 4ヶ月の 1.5 に直接の根拠が無い（下の HEAD_CONTROL_RATIO 参照）ので、
#      0.5 はその不確かさを引き継いでいる。
#   3. Öhman のスコアは個体差が大きい（2ヶ月で範囲0〜2）。平均を1点として使っている。
#   4. スコア2以上（「やや上」「かなり上」）には**角度の定義が無い**ので、
#      4ヶ月の実測点としては使えない（＝4ヶ月を実測で固定できていない理由）。
#   → 感度分析の対象：0.3 / 0.5 / 0.8 を振って首座りの結論が変わらないか確かめる。
#
# 【★教訓】Öhman のスコアは**このファイルに2026-07-20時点で既に書かれていた**
#   （下の「❌ 新生児の頸部筋力の定量値は存在しない（臨床は質的スコアで評価。
#   2ヶ月児の筋機能スコア中央値=1…）」）。当時は「質的スコアだから使えない」と判断して
#   捨てていたが、**段階1が物理的に一意である**ことに気づいていなかった。
#   ＝持っていたデータの使い道に気づいていなかった。→ 落とし穴チェックリストへ。
TARGET_RATIO_AT_BIRTH = 0.5    # [Tier2] Öhman 2008 の2ヶ月実測点からの外挿
HEAD_CONTROL_AGE = 4.0         # 首がすわる月齢 ⚠️曲線の形は恣意的（上記②）

# ★【2026-07-28 新設】首がすわった時点（4ヶ月）の目標比。
#
# 【なぜ要るか】従来は HEAD_CONTROL_AGE で**補正を打ち切っていた**ため、4ヶ月ちょうどで
# MIMoの素の値が露出し、持ち上げ能力比が **24.4 → 47.5** と崖のように跳んでいた
# （実測 `E/scripts/e_age_body_audit.py`）。しかも47.5倍は**成人男性より強い**。
#
# 【成人の比＝判断の足場、2026-07-28 の文献調査】
#   Vasavada, Li & Delp (2001) Spine 26(17):1904-1909（男11・女5の実測、全文入手）
#   ＋ 頭部質量（Suderman & Vasavada 2019, PMC6986765）＋ モーメントアーム4.3〜7.4cm から
#     成人男性  伸展 17.1〜29.3倍 ／ 屈曲  9.8〜16.9倍
#     成人女性  伸展  8.0〜13.8倍 ／ 屈曲  5.7〜 9.8倍
#   ＝太郎の18ヶ月（伸展27.7／屈曲14.9）は**成人男性の範囲に収まる＝MIMoの素の値は妥当**。
#   問題は「その筋力を4ヶ月の小さい体に載せると成人を追い越す」ことだけだった。
#
# 【なぜ 1.5 か】⚠️[Tier3・ARBITRARY]
#   ★**乳児の頸部筋力の実測は文献に存在しない**（最大随意努力を指示できないため原理的に
#     測れない。2026-07-28 の調査で再確認）。
#   ★Lavallee, Ching & Nuckley (2013) J Biomech 46:527-534 の成熟曲線
#     （%MVC = -0.0879·年齢² + 6.018·年齢 + 8.120）を0歳へ外挿する案は**破綻した**。
#     外挿すると0ヶ月でも比 8.8〜15倍になり、「新生児は頭を持ち上げられない」という
#     臨床所見と矛盾する（データの下限が6歳なので、切片8.12%がそのまま残るため）。
#     この論文から使えるのは「頭が先に育ち、首の力が後から追いつく」という**向き**だけ
#     （同論文：8歳児は頭囲が成人の91%なのに首の強さは成人の50%）。
#   → 代わりに **head lag が消えるのは生後4ヶ月前後**（Bradshaw et al. 2023 J Pediatr
#     253:225-231、100名を毎月追った縦断研究）から置いた。「4ヶ月＝頭を持ち上げられる
#     ようになったばかり」なので、比が1.0（頭の重さと釣り合う境界＝物理的に一意な点）を
#     わずかに超えたところ、として 1.5 を採る。
#   ⚠️**1.5 の 1.5 に根拠は無い**。根拠があるのは「4ヶ月で1.0を横切る」ことだけ。
#     → 感度分析の対象（1.2 / 1.5 / 3.0 を振る）。pull-to-sit の測定器を作って
#       実測で決め直すのが本筋（やることリスト）。
HEAD_CONTROL_RATIO = 1.5

# ここまで来たら補正しない＝MIMoの素の値を使う月齢。
# 太郎の18ヶ月の比（伸展27.7／屈曲14.9）は成人男性の範囲に収まるので、
# 「18ヶ月＝素のMIMoで妥当」と判断した（上記の文献調査）。四肢の補正の基準月齢と同じ。
MATURE_AGE = 18.0


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

    ⚠️★**呼ぶ前に mj_forward が済んでいること**。data.xanchor / data.xipos は
      物理計算を1度も走らせていないと初期値のままで、姿勢が反映されない
      （→ apply_newborn_neck の説明を参照。実際にこれで補正が6倍ずれていた）。
    """
    jid = [j for j in range(model.njnt) if model.joint(j).name == TILT_JOINT][0]
    anchor = data.xanchor[jid]
    ids = _head_bodies(model)
    mass = sum(model.body(i).mass[0] for i in ids)
    com = sum(model.body(i).mass[0] * data.xipos[i] for i in ids) / mass
    arm = float(np.linalg.norm((com - anchor)[:2]))
    return float(mass * GRAVITY * arm), float(mass), arm


def lift_ratio(model, data, actuation_model=None):
    """持ち上げ能力比＝首の筋力 ÷ 頭の重力モーメント。<1 なら頭を持ち上げられない。

    ⚠️★【2026-07-25 修正】筋力を `actuator_gear` から読んでいたが、**MuscleModel は
    毎ステップ gear を上書きする**（muscle.py:331）ので、gear を読むと
    「今まさに出力しているトルク」を読んでしまい、筋力の指標にならない。
    筋肉モデルでは `actuation_model.fmax` を読む（→ infant_body.actuator_strength）。
    """
    tau, _, _ = head_gravity_torque(model, data)
    aid = [i for i in range(model.nu) if model.actuator(i).name == TILT_ACTUATOR][0]
    from infant_body import actuator_strength
    gear = actuator_strength(model, aid, actuation_model)
    return gear / max(tau, 1e-9), gear, tau


def target_ratio_for_age(age, birth_ratio=TARGET_RATIO_AT_BIRTH,
                         settle_age=HEAD_CONTROL_AGE, natural_ratio=None,
                         control_ratio=HEAD_CONTROL_RATIO, mature_age=MATURE_AGE):
    """月齢に応じた目標比。**崖を作らずに**MIMoの素の値まで繋ぐ。

    ★【2026-07-28 全面改訂】旧実装は settle_age(4ヶ月)で補正を打ち切っていたため、
    そこで比が 24.4 → 47.5 と跳んでいた（実測）。3区間に分けて連続にする：

        0 〜 4ヶ月    birth_ratio(1.0) → control_ratio(1.5)      首がすわるまで
        4 〜 18ヶ月   control_ratio(1.5) → natural_ratio        素のMIMoへ戻る
        18ヶ月〜      None（補正しない）                          素のMIMoそのまま

    ⚠️線形補間は恣意的[Tier3]。ただし「新生児が最も弱く成長で強くなる」という**向き**は
    文献に一致させてある（head lag が消えるのは4ヶ月前後＝Bradshaw et al. 2023）。

    Args:
        natural_ratio: 補正しなかった場合の比（＝その月齢のMIMoの素の値）。
            None なら4ヶ月以降の補間ができないので control_ratio で止める。
    """
    if age >= mature_age:
        return None                      # 補正しない
    if age < settle_age:                 # 0 〜 4ヶ月
        w = max(0.0, min(1.0, age / settle_age))
        return (1.0 - w) * birth_ratio + w * control_ratio
    if natural_ratio is None:            # 素の比が分からない＝繋ぎようがない
        return control_ratio
    w = (age - settle_age) / max(mature_age - settle_age, 1e-9)   # 4 〜 18ヶ月
    w = max(0.0, min(1.0, w))
    return (1.0 - w) * control_ratio + w * float(natural_ratio)


def apply_newborn_neck(model, data, age, birth_ratio=TARGET_RATIO_AT_BIRTH,
                       settle_age=HEAD_CONTROL_AGE, verbose=True, actuation_model=None):
    """首(tilt)のgearを、月齢に応じた目標比になるよう補正する。

    ★【2026-07-28 修正】冒頭で `mj_forward` を呼ぶようにした。

    【何が起きていたか】この関数は環境の `__init__` の中（＝物理計算を1度も走らせる前）に
    呼ばれる。`data.xanchor` / `data.xipos` はまだ初期値のままなので、
    **仰向けにした姿勢が反映されていない状態**で頭の重力モーメントを測っていた。
    その結果、補正時に見ていた負荷は 0.066 N·m（腕 0.76cm）だったのに、
    reset で体が落ち着くと 0.384 N·m（腕 4.40cm）になり、**比が約6倍ずれた**：
        設計した目標比 1.0 → 実測 0.17（0ヶ月）
        設計した目標比 1.5 → 実測 0.25（4ヶ月）
    ＝目標値をいくら丁寧に決めても、その値が体に反映されていなかった。

    ⚠️mj_forward は qpos0（リセット時の姿勢）での物理量を計算するだけで、
      settle（体が床に落ち着くまでの100ステップ）は走らない。完全一致はしないが、
      「腕が0.76cm」のような桁違いのズレは消える。

    ⚠️**恣意的な逸脱**。ROM・swivel・tilt_side・他の関節は一切変更しない。
    Returns: (before_ratio, after_ratio) 補正しない場合は after=before。
    """
    import mujoco
    mujoco.mj_forward(model, data)   # ★姿勢を反映させてから測る（上の説明を参照）
    before, gear, tau = lift_ratio(model, data, actuation_model)
    # ★【2026-07-28】打ち切りの月齢を settle_age(4ヶ月)から MATURE_AGE(18ヶ月)へ。
    #   4ヶ月で切ると比が 24.4 → 47.5 と跳んでいた（＝成人男性より強い首になる）。
    if age >= MATURE_AGE:
        if verbose:
            print(f"[neck] age={age}mo >= {MATURE_AGE}mo: no correction (ratio {before:.2f})")
        return before, before
    target = target_ratio_for_age(age, birth_ratio, settle_age, natural_ratio=before)
    aid = [i for i in range(model.nu) if model.actuator(i).name == TILT_ACTUATOR][0]
    new_gear = target * tau
    # ★筋力の書き換えは infant_body 経由（筋肉モデルなら fmax、それ以外は gear）。
    #   ⚠️直接 actuator_gear を書くと MuscleModel では次のステップで消える。
    from infant_body import scale_actuator_strength
    factor = new_gear / max(gear, 1e-12)
    ok = scale_actuator_strength(model, aid, factor, actuation_model)
    after, _, _ = lift_ratio(model, data, actuation_model)
    if verbose:
        note = "" if ok else " ⚠️[fmaxがスカラー＝関節ごとに変えられないため未適用]"
        print(f"[neck] age={age}mo: lift ratio {before:.2f} -> {after:.2f} "
              f"(strength {gear:.3f} -> {new_gear:.3f} N.m, head torque {tau:.3f} N.m)"
              f"{note} [ARBITRARY: see deviation list]")
    return before, after
