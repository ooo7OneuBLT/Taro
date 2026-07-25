"""新生児の体型（プロポーション）— 太郎の身体そのものの定義。

【なぜ core にあるか、2026-07-25】この体型補正はもともと目標Eのおもちゃ環境
（`E/scripts/e_toy_env.py`）の中に書かれていた。その結果**環境を変えると太郎の体型が
変わる**状態になり、実際に「おもちゃ環境では補正あり／学習に使う仰向け環境では補正なし」
という食い違いが起きていた（＝太郎は成人プロポーションのまま学習していた）。
体型は**太郎そのもの**であって環境の性質ではないので core へ移した。
方針：[[feedback-core-vs-experiment-placement]]（本実装は core、実験は目標フォルダ）。

【経緯：v2（目視で決定）→ v3（実測で決定）】
mimoGrowth の age=0 は「大きさは新生児だが四肢の比率が成人」だった（人間模倣からの逸脱リスト
2026-07-20 その4）。加えて頭が球なので、頭囲34cmでも真上から見た頭が15%小さい。
→ ①頭を体軸方向に楕円化（頭囲を保ったまま見かけを人間に）②四肢・手足を縮小。

- **v2（2026-07-21）**：Viewerでの目視だけで係数を決めた。質量も絶対サイズも一度も
  測っていなかった。
- **★v3（2026-07-25）**：測定器（`E/scripts/e_body_measure.py`）を新設して初めて実測。
  v2の実態は**身長35.6cm（人間の71%）・体重1.44kg（41%）・2.8頭身**（人間の新生児は
  4.2頭身）で、**アニメの赤ちゃんの比率**だった。`E/scripts/e_body_fit.py` で
  人間の実測値に合わせて探索し直した。

**v3の寸法**（`e_body_measure.py` で再現できる）：
| | 太郎v3 | 人間の新生児 | v2（旧） |
|---|---|---|---|
| 身長 | 49.3cm | 49.9cm | 35.6cm |
| 体重 | 3.54kg | 3.50kg | 1.44kg |
| 頭身 | 3.9 | 4.2 | **2.8** |
| 頭の質量比 | **25.0%** | **25.0%** | **49.5%** |
| 下肢/身長 | 38.9% | 39.2% | 29% |

★**縮めるのは脚だけ**で、腕・胴・太さはむしろ伸ばす／太くするのが正しかった。
素のmimoGrowthの脚長19.2cmは人間19.6cmとほぼ一致しており、v2の0.45倍は過剰だった。
★頭の質量は**密度で**合わせる（`apply_head_mass`）。大きさ（頭囲34cm）は既に合っているので、
サイズを変えると頭身が壊れる＝**見た目を変えずに質量だけ人間に一致させる**。

⚠️文献で裏取りできたのは足長7.58cmのみ。下肢19.6cm・上肢20.96cmは測定定義が
成人的で新生児に不整合＝**当てにしない**（実際、上肢をこの値に合わせると腕が伸びすぎて
手が腰を越え、目視で新生児に見えなくなる）。詳細は `doc/人間模倣からの逸脱リスト.md`。

【使い方】
    from infant_body import body_scale_custom, NEWBORN_SHAPE_DEFAULTS
    custom = body_scale_custom(age=0.0, scales=NEWBORN_SHAPE_DEFAULTS)
    env = SomeMimoEnv(age=0.0, custom_measurements=custom)

⚠️環境変数（`E_LEG_SCALE` 等）の読み取りは**目標フォルダ側の責務**。core は係数の既定値と
変換ロジックだけを持つ（[[feedback-core-target-neutral-naming]]＝core に目標プレフィックスを
書かない）。
"""

# 部位グループ -> 既定の係数。1.0 は「触らない」。
#
# 【★2026-07-25 改訂・v3】旧v2（leg 0.45 / arm 0.62 / trunk_len 0.74 / 太さ 0.60）は
# **目視だけで決めて数値を一度も測っていなかった**。測定器（`E/scripts/e_body_measure.py`）
# を作って初めて実測したところ、身長35.6cm（人間の71%）・体重1.44kg（41%）・
# **2.8頭身**（人間の新生児は4.2頭身）という壊れた体だった。
# ＝「赤ちゃんらしく見える」ようにデフォルメされていただけで、実物とは別物。
#
# v3は `E/scripts/e_body_fit.py` で人間の実測値に合わせて探索した値：
#   身長49.3cm（人間49.9）／体重3.54kg（3.50）／頭の質量25.0%（25.0、密度で補正）／**3.9頭身**（4.2）／下肢/身長 38.9%（39.2）
# ★**縮めるのは脚だけ**で、腕・胴・太さはむしろ伸ばす／太くするのが正しかった
# （素のmimoGrowthの脚長19.2cmは人間19.6cmとほぼ一致しており、0.45倍は過剰だった）。
#
# ⚠️上肢/下肢比だけは人間（1.07）に対して0.86と外れる。太郎の「上肢」は
# 肩bodyの中心→手bodyの中心の距離で、人間の計測値（肩峰〜手首）と**測る場所が違う**。
# 比に合わせると腕が伸びすぎて手が腰を越える（目視で確認）ため、見た目を優先した。
# ⚠️[Tier2] 個々の係数そのものは実測に合わせた結果であって文献値ではない。
NEWBORN_SHAPE_DEFAULTS = {
    "leg":        0.801,
    "leg_thick":  1.19,
    "arm":        1.00,
    "arm_thick":  1.19,
    "trunk_len":  1.03,
    "foot":       0.72,
    "hand":       0.70,
    "trunk_width": 1.19,
    "foot_width":  1.19,
}

# 頭を体軸方向に 12.5/10.8 倍＝球→楕円（頭囲は不変）。
HEAD_ELONGATION = 1.16

# 頭が全体重に占める割合。人間の新生児は約25%（成人は約8%）＝**発達の主役**。
# ⚠️[Tier1] 出典は新生児の体節質量比の標準値。首がすわらない（head lag）ことも、
# 仰向けから転がりやすいことも、この「頭の重さ」が物理的な原因になっている。
HEAD_MASS_FRACTION = 0.25
# 何ヶ月まで頭の質量を補正するか。首の筋力補正（4ヶ月で解除）と揃えてある[Tier3]。
HEAD_MASS_UNTIL_MO = 4.0

# 足＝相似縮小するgeom群（左側だけ指定すれば右側は自動でミラーされる）
_FOOT_GEOMS = [
    "geom:left_foot1", "geom:left_foot2", "geom:left_foot3",
    "geom:left_toes1", "geom:left_toes2",
    "geom:left_big_toe1", "geom:left_big_toe2",
]

# 手＝掌ブロック＋全指のgeom。相似縮小（全indexを一律）。
# ⚠️指のknuckle/middle/distalは連鎖するので、掌だけでなく指も含めないと
#   「掌は小さいが指は元のまま」というちぐはぐになる。
_HAND_GEOMS = [
    "geom:left_hand1", "geom:left_hand2", "geom:left_hand3", "geom:left_hand4",
    "geom:left_ffknuckle1", "geom:left_ffmiddle1", "geom:left_ffdistal1",
    "geom:left_mfknuckle1", "geom:left_mfmiddle1", "geom:left_mfdistal1",
    "geom:left_rfknuckle1", "geom:left_rfmiddle1", "geom:left_rfdistal1",
    "geom:left_lfmetacarpal1", "geom:left_lfmetacarpal2",
    "geom:left_lfknuckle1", "geom:left_lfmiddle1", "geom:left_lfdistal1",
    "geom:left_thbase1", "geom:left_thhub1", "geom:left_thdistal1",
]

# (グループ名) -> ([geom名...], 変えるindex)。idx=None は「全indexを一律」＝相似縮小。
#
# ⚠️【重要・2026-07-21 に踏んだ罠】**どの index が「長さ」かは部位で違う**。
# MIMoのcapsuleは `size=[radius, half_length]` だが、**bodyの位置がスキーマ上どちらを
# 参照するかが部位ごとに異なる**（`SCHEMA_V2["bodies"]`）：
#     lower_body.pos = (lb.size[0] + cb.size[0]) × 0.752      ← radius
#     upper_body.pos = (cb.size[0] + ub1.size[0]) × 0.867     ← radius
#     chest.pos      = ub3.size[0] × 2.195                    ← radius
#     lower_leg.pos  = −(upper_leg1.size[0]×2 + size[1])      ← 両方
# ＝**胴体は「横向きのcapsuleを体軸方向に積んだ」構造**で、体軸方向の厚みは radius
# （index 0）、左右の幅が half_length（index 1）。当初 index 1 だけを変えていたため
# **体幹長がまったく変わらなかった**（19.8cmのまま。「効かない」と誤って結論しかけた）。
_GROUPS = {
    "leg":         (["geom:left_upper_leg1", "geom:left_lower_leg1",
                     "geom:left_lower_leg2"], 1),
    "leg_thick":   (["geom:left_upper_leg1", "geom:left_lower_leg1",
                     "geom:left_lower_leg2"], 0),      # ⚠️脚長にも効く（bodyのposが参照）
    "arm":         (["left_uarm1", "left_larm"], 1),
    "arm_thick":   (["left_uarm1", "left_larm"], 0),   # ⚠️前腕→手の距離にも効く
    "trunk_len":   (["lb", "cb", "ub1", "ub2", "ub3"], 0),   # ← 体軸方向
    "trunk_width": (["lb", "cb", "ub1", "ub2", "ub3"], 1),   # ← 左右の幅
    # 足＝全indexを一律に縮める（相似）。実測 8.36cm に対し人間の新生児は
    # 7.58±0.44cm（n=500, foot length は在胎週数の推定に使われる標準指標）
    "foot":        (_FOOT_GEOMS, None),
    "foot_width":  (_FOOT_GEOMS, 1),   # 甲の広さ（どのindexが幅かは足の向きで変わる＝要目視）
    "hand":        (_HAND_GEOMS, None),   # 手全体を相似縮小（掌＋指）
}


def elongate_head(spec, ratio):
    """頭を体軸方向に伸ばして球→楕円にする（頭囲は変えない）。

    【なぜ要るか】MIMoの頭は**球**で、直径は頭囲(34cm)から計算される＝10.8cm。
    しかし**人間の頭は楕円**で、頭囲34cmでも頭頂〜顎は約12.5cmある。
    ＝MIMoは頭囲が正しいのに、**真上から見た頭が15%小さい**。
    この差を四肢を縮めて埋めようとすると身長が犠牲になり、身長を優先すると頭囲が壊れる。
    **球のままでは頭囲・身長・見かけの3つを同時に満たす解が無い**。

    【何をするか】左右方向（＝頭囲を決める軸）は変えず、MIMoローカルのz（頭頂方向＝
    仰向けでは体軸方向）だけを ratio 倍する。
      球   [r, 0, 0]        頭囲 2πr        真上から見た長さ 2r
      楕円 [r, r, r*ratio]  頭囲 2πr（不変） 真上から見た長さ 2r*ratio
    ratio = 12.5/10.8 ≒ 1.16 で人間の新生児に一致する。

    【なぜ core にあるか、2026-07-25】この処理は `E/scripts/e_toy_env.py`（おもちゃ環境）
    にしかなく、**学習に使う仰向け環境では頭が球のまま**だった。実測で
    頭の長さ10.8cm（人間12.0cm）・頭/身長0.224（人間0.240）とズレ、体型補正で身長を
    合わせると「新生児に見えない」原因になっていた。体型補正と**同じ構造の問題**
    （身体の設定が環境に散らばっている）。方針＝[[feedback-core-vs-experiment-placement]]。

    ⚠️目のカメラ位置は head の geom size から計算されるが、その計算は成長モジュール
    （このフックより前）で終わっている。z方向にだけ伸ばすので目が頭に埋もれることは
    無いはずだが、**視界の画像で必ず確認すること**。

    Args:
        spec: MuJoCo の MjSpec（compile 前）。
        ratio: 体軸方向の伸長率。1.0 なら何もしない。
    """
    import mujoco
    import numpy as np
    if abs(float(ratio) - 1.0) < 1e-9:
        return False
    for g in spec.geoms:
        if g.name != "head":
            continue
        r = float(np.asarray(g.size).ravel()[0])
        g.type = mujoco.mjtGeom.mjGEOM_ELLIPSOID
        g.size = [r, r, r * ratio]
        print(f"[body] 頭を楕円化: 半径{r*100:.2f}cm → "
              f"[{r*100:.2f}, {r*100:.2f}, {r*ratio*100:.2f}]cm "
              f"（頭囲は不変、真上から見た長さ {2*r*ratio*100:.2f}cm）")
        return True
    print("[body] ⚠️頭のgeomが見つからず、楕円化をスキップした")
    return False


_SCHEMA_BACKUP = None


def _restore_growth_schema():
    """★MIMo側のバグ対策：`mimoGrowth` のスキーマ（グローバル辞書）を元に戻す。

    【バグの中身、2026-07-25 に発見】`MIMo/mimoGrowth/growth.py:167` は
        schema = SCHEMA_V2 if mimo_version == "v2" else SCHEMA
        if custom:
            schema["geoms"][geom_name]["size"][index] = custom_size
    と書かれており、`schema` は**コピーでなくグローバル辞書への参照**。つまり
    `custom_measurements` を渡すたびに**グローバルの SCHEMA_V2 が破壊的に書き換わる**。
    結果、同じプロセスで環境を作り直すと補正が**累積**して体が縮み続ける：
        1回目 元の値×0.45 → 2回目 (0.45倍された値)×0.45 → 3回目 0.091倍 …
    実測：同じ設定で身長 35.6cm → 26.8cm → 26.8cm、体重 1.442 → 1.009 → 0.850kg。

    【なぜ MIMo を直さないか】MIMo は `.gitignore` 済み（Git管理外）で、書き換えると
    再現性が失われる方針。首の筋力補正などと同様に**太郎側で実行時に防御**する。

    毎回スキーマを初期状態へ戻してから custom を適用すれば累積しない。
    """
    global _SCHEMA_BACKUP
    import copy
    from mimoGrowth.schema.schema import SCHEMA_V2
    if _SCHEMA_BACKUP is None:
        _SCHEMA_BACKUP = copy.deepcopy(SCHEMA_V2)   # 初回＝まだ汚れていない状態を保存
    else:
        SCHEMA_V2.clear()
        SCHEMA_V2.update(copy.deepcopy(_SCHEMA_BACKUP))


def body_scale_custom(age, scales=None, verbose=True):
    """体の部位ごとに geom の寸法を係数倍する custom 辞書を作る。

    mimoGrowth の `custom` は `{(geom名, index): 値}` でスキーマを直接上書きする。
    左側だけ指定すれば右側は自動でミラーされる（右を渡すと警告されて無視される）。

    Args:
        age: 体年齢（月）。
        scales: 部位グループ -> 係数。None なら NEWBORN_SHAPE_DEFAULTS を使う。
        verbose: 適用した補正を1行表示する。

    Returns:
        {(geom名, index): 値} の辞書。空なら補正なし。
    """
    from mimoGrowth.growth import get_growth_params

    if scales is None:
        scales = NEWBORN_SHAPE_DEFAULTS
    _restore_growth_schema()   # ★累積バグ対策（上の説明を参照）
    base = get_growth_params(age, "v2")["geoms"]
    custom, note = {}, []
    for grp, (names, idx) in _GROUPS.items():
        k = float(scales.get(grp, 1.0))
        if abs(k - 1.0) < 1e-9:
            continue
        n = 0
        for nm in names:
            size = base.get(nm, {}).get("size")
            if size is None:
                continue
            idxs = range(len(size)) if idx is None else ([idx] if len(size) > idx else [])
            for j in idxs:
                if float(size[j]) <= 0:
                    continue          # 0 は「使っていない次元」なので触らない
                # 同じ (geom, index) を複数グループが指す場合は掛け合わせる
                cur = custom.get((nm, j), float(size[j]))
                custom[(nm, j)] = cur * k
                n += 1
        note.append(f"{grp}x{k:.2f}(×{n})")
    if note and verbose:
        print("[body] 体型補正: " + " ".join(note) + " [逸脱リスト参照]")
    return custom


# ============================================================================
# ★統一の窓口 — 太郎の身体はここ1箇所で決まる（2026-07-25）
# ----------------------------------------------------------------------------
# 【なぜ作ったか】体型・頭の楕円化・首の筋力・四肢の筋力が**環境クラスごとにバラバラに
# 適用**されており、実測で**3種類以上の体**が存在していた：
#     LeanMimoEnv      : 補正なし
#     SupineMimoEnv    : 体型は呼び出し側次第・頭の楕円化のみ
#     ToySupineEnv     : 体型・頭・首・四肢の全部
# ＝**環境を変えると太郎の体が変わる**状態。ユーザー指摘「正直一個でいいんだけど」。
# 身体は環境の性質ではなく**太郎そのもの**なので、ここに集約する。
# 方針：[[feedback-core-vs-experiment-placement]]
#
# 【2段階ある理由】MIMoの身体は
#   ①モデル構築前（geomの寸法・スキーマ）… build_body_kwargs / elongate_head
#   ②モデル構築後（アクチュエータのgear＝筋力）… apply_runtime_corrections
# の2段階でしか変えられない。①は環境のコンストラクタ引数、②は構築後の上書き。
# ============================================================================

def build_body_kwargs(age, scales=None, head_elongation=None):
    """環境のコンストラクタに渡す身体の設定を作る（モデル構築前の分）。

    Args:
        age: 体年齢（月）。
        scales: 部位グループ->係数。None なら NEWBORN_SHAPE_DEFAULTS。
        head_elongation: 頭の楕円化率。None なら HEAD_ELONGATION。

    Returns:
        {"custom_measurements": ..., "head_elongation": ...}（不要な項目は入れない）
    """
    kwargs = {}
    custom = body_scale_custom(age, scales)
    if custom:
        kwargs["custom_measurements"] = custom
    he = HEAD_ELONGATION if head_elongation is None else float(head_elongation)
    if abs(he - 1.0) > 1e-9:
        kwargs["head_elongation"] = he
    return kwargs


def apply_head_mass(model, age=0.0, fraction=None, verbose=True):
    """頭の**密度**を上げて、全体重に占める頭の割合を人間の新生児に合わせる。

    【なぜ必要か、2026-07-25】頭の**大きさ**（頭囲34cm）は人間と合っているのに、
    質量は全体重の20.5%しかなかった（人間25%）＝MIMoの頭は人間より密度が低い。
    大きさが合っているので**サイズを変えて合わせてはいけない**（頭身が崩れる）。
    密度＝質量だけを上げる＝**見た目を1ミリも変えずに質量を人間に一致させる**。

    ⚠️慣性テンソルも同じ倍率でスケールする（形は変わらないので質量に比例する）。
    これを忘れると「重いのに回りやすい」物理的にありえない頭になる。

    ⚠️必ず首・四肢の筋力補正**より先に**呼ぶこと。首の補正は「頭を持ち上げられるか」を
    頭の質量から計算するので、後から頭を重くすると首の補正が古い前提のままになる。

    ⚠️★**新生児期（age < HEAD_MASS_UNTIL_MO）だけに適用する**。四肢の筋力補正は
    「18ヶ月児と比べて発達の向きが逆転していないか」を見るために**18ヶ月の基準モデルを
    別に構築する**ので、月齢で止めないと**基準モデルの頭まで25%に書き換えてしまい、
    基準そのものが歪む**（2026-07-25、実測ログで発覚：18ヶ月モデルの頭 1.846kg が
    0.865kg に書き換えられていた）。首の補正が4ヶ月で解除されるのと同じ形にしてある。
    ⚠️人間の頭の質量比は新生児25%→成人8%と月齢で下がるが、その中間の文献値を
    持っていないので**新生児期だけ合わせて以降は触らない**[Tier3・簡略化]。

    Args:
        age: 月齢。これが HEAD_MASS_UNTIL_MO 以上なら何もしない。
        fraction: 目標の割合。None なら HEAD_MASS_FRACTION（0.25）。
    """
    import numpy as np
    if float(age) >= HEAD_MASS_UNTIL_MO:
        return
    frac = HEAD_MASS_FRACTION if fraction is None else float(fraction)
    hid = int(model.body("head").id)
    # 太郎の身体だけを数える（床・柵・おもちゃを含めない）
    skip = ("floor", "wall", "fence", "object", "toy", "target", "world")
    ids = [i for i in range(1, model.nbody)
           if not any(k in model.body(i).name for k in skip)]
    total = float(sum(model.body_mass[i] for i in ids))
    head = float(model.body_mass[hid])
    others = total - head
    if others <= 0 or not (0.0 < frac < 1.0):
        return
    new_head = others * frac / (1.0 - frac)
    k = new_head / head if head > 0 else 1.0
    model.body_mass[hid] = new_head
    model.body_inertia[hid] = np.asarray(model.body_inertia[hid]) * k
    if verbose:
        print(f"[head] 頭の質量 {head:.3f}kg -> {new_head:.3f}kg "
              f"(全体の {head/total*100:.1f}% -> {frac*100:.1f}%, 密度 x{k:.2f}) "
              f"[Tier1: 人間の新生児の体節質量比]")


def apply_runtime_corrections(model, data, age, neck=True, limbs=True, head_mass=True):
    """モデル構築後に上書きする補正（筋力＝アクチュエータのgear）。

    - **首の筋力**（`infant_neck`）：MIMoは gear を geom の体積から計算するため、
      頭が大きい新生児ほど首も強くなり**発達の向きが逆転**する（age=0で持ち上げ能力比
      4.21倍 > age=18ヶ月の3.00倍）。首がすわっていない（head lag）を再現するため
      月齢に応じて下げる。⚠️目標比1.0・4ヶ月で解除は恣意的[Tier3]。
    - **四肢の筋力**（`infant_limbs`）：同じ理由で四肢も発達の向きが逆転しているのを解消。

    ⚠️MIMo本体は書き換えない（Git管理外で再現性が失われるため）＝実行時に上書きする。
    """
    # ★頭の質量は首・四肢より先。首の補正が頭の質量を前提に計算するため。
    if head_mass:
        apply_head_mass(model, float(age))
    if neck:
        from infant_neck import apply_newborn_neck
        apply_newborn_neck(model, data, float(age))
    if limbs:
        from infant_limbs import apply_limb_inversion_fix
        apply_limb_inversion_fix(model, data, float(age))
