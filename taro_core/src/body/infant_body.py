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
# 首がすわらない（head lag）ことも、仰向けから転がりやすいことも、
# この「頭の重さ」が物理的な原因になっている。
#
# ⚠️★【2026-07-25 ラベルを [Tier1] から [Tier2〜3] に格下げ】
#   従来は「[Tier1] 出典は新生児の体節質量比の標準値」と書いていたが、
#   文献調査の結果 ★**「新生児の頭部は体重の25%」は一次文献の裏が取れなかった**
#   （調べた範囲ではブログ等の二次情報どまり。使用は推奨されないと報告された）。
#   乳児の体節慣性パラメータの一次研究（Jensen 1986 は4-15歳／Sun & Jensen 1994 は
#   2-9ヶ月児27名）は存在するが、**新生児の頭部質量比の具体値には到達できていない**。
#   → チェックリスト項16（貼った根拠ラベル自体を一次文献で検証する）の**再発**。
#   ⚠️値そのものは変えない（体型v3は身長・体重・頭身とセットで人間に一致させてあり、
#     ここだけ動かすと全体の整合が崩れる）。**ラベルだけを正直にする**。
#   ★やること：Sun & Jensen 1994 の本文を入手して新生児の頭部質量比を確認する
#     （やることリストに登録）。それまでは感度分析の対象（型B）として扱う。
HEAD_MASS_FRACTION = 0.25   # [Tier2〜3・一次文献の裏取り未了]
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


def apply_distal_mass(model, scale=1.0, verbose=True):
    """★手・足（末端）の**質量だけ**を倍率で変える（サイズは変えない）＝感度分析用。

    【なぜ要るか、2026-07-25】体型v3は身長・体重・頭身・頭の質量比を人間に合わせたが、
    **四肢の内訳**を測ると末端が成人比で極端に軽かった：
        手 0.09%（成人0.60% ＝ **1/7**）／足 0.43%（成人1.45% ＝ **1/3.4**）
    原因は判明している——MIMoは「**各部位の密度は一定と仮定し、幾何形状の体積から質量を
    決める**」設計（MIMo論文 arXiv:2509.09805 III-B に明記）で、手を0.70倍に縮小すると
    質量は 0.70³=0.34倍になる。★MIMoの元データ AnthroKids には**体節ごとの質量が
    そもそも無い**（全87項目のうち質量は「全身の体重」1つだけ、残りは外形寸法）。

    ⚠️**正しい値は誰も知らない**：新生児の体節質量比は**実測が存在しない**
    （乳児のBSP研究 Jensen 1986／Schneider & Zernicke 1992／Sun & Jensen 1994 は
    すべて「幾何モデル＋密度の仮定」による推定。乳児の死体解剖データは無い。
    しかも PMC2667919 は「**手足は小さすぎて正確な測定ができない**」として除外している）。
    ⇒ 値は決められないので、**振って結論が変わらないことを確かめる**（筋力と同じ扱い）。

    ★**サイズは変えない**のが重要。`hand` 係数を振るとサイズも変わり、
    「質量が効いたのか見た目（視野に映る面積）が効いたのか」が**交絡する**。
    hand regard の実験に効く可能性があるので、そこを分離しておく。

    ⚠️慣性テンソルも同じ倍率でスケールする（`apply_head_mass` と同じ理由）。
    ⚠️手の指・足の指も含める（`left_hand` から先の子孫すべて）。

    Args:
        scale: 質量の倍率。1.0 で何もしない。成人の比率に合わせるなら 手≈6.7 / 足≈3.4。
    """
    import numpy as np
    if abs(float(scale) - 1.0) < 1e-9:
        return
    roots = ("left_hand", "right_hand", "left_foot", "right_foot")
    before = after = 0.0
    n = 0
    for rname in roots:
        try:
            rid = int(model.body(rname).id)
        except Exception:
            continue
        for b in range(model.nbody):          # rid の子孫（自分を含む）
            p = b
            while p != 0:
                if p == rid:
                    before += float(model.body_mass[b])
                    model.body_mass[b] = float(model.body_mass[b]) * float(scale)
                    model.body_inertia[b] = np.asarray(model.body_inertia[b]) * float(scale)
                    after += float(model.body_mass[b])
                    n += 1
                    break
                p = int(model.body_parentid[p])
    if verbose:
        print(f"[distal] 手足の質量 x{scale}: {before*1000:.1f}g -> {after*1000:.1f}g "
              f"({n}body) [SENSITIVITY: 新生児の体節質量比は実測が存在しない]")


# ★★【筋力の読み書き・2026-07-25】★これを間違えていて、首と四肢の筋力補正が
# **筋肉モデルでは一度も効いていなかった**。
#
# 【何が起きていたか】太郎の補正は `model.actuator_gear[aid, 0]` を書き換えていた。
# ところが MuscleModel は**毎ステップ gear を上書きする**：
#     muscle.py:331  self.env.model.actuator_gear[self.actuators, 0] = self.joint_torque.copy()
#     （「gear は単なるスカラー倍率なので、計算したトルクをそこに書いて
#       action=1 を出す」という設計。XMLとgymインタフェースを共通化するため）
# ＝★書き換えた値は次のステップで消える。ログには出るが実効ゼロ。
#
# 筋力の実体は **actuator_user[:, 1] / [:, 2]（fmax_neg / fmax_pos）** から読まれ、
# `actuation_model.fmax`（180次元 = 90関節 × neg/pos）に保持される。
# ⚠️`actuator_user` を後から変えても反映されない（初期化時に一度読むだけ）ので、
#   `actuation_model.fmax` を直接書き換える。
#
# 【症状（これが出たら疑う）】
#   ・筋力の係数を16倍振っても、学習後のモデルの重みが**完全一致**する
#   ・`[limbs] median gear scale x0.044` のようなログは出るのに挙動が変わらない
# → 検証は `E/scripts/e_condition_check.py`（チェックリスト項43）。

def actuator_strength(model, aid, actuation_model=None):
    """アクチュエータの「筋力」を返す。筋肉モデルなら fmax、それ以外は gear。

    ★筋肉モデルでは neg/pos の2本があるので**大きい方**を返す（関節を動かせる強さの上限）。
    """
    import numpy as np
    f = getattr(actuation_model, "fmax", None)
    if f is None:
        return abs(float(model.actuator_gear[aid, 0]))
    f = np.asarray(f, dtype=float)
    if f.ndim == 0:                      # キャリブレーションファイルが無いとスカラーになる
        return float(f)
    n = int(getattr(actuation_model, "n_actuators", len(f) // 2))
    return max(float(f[aid]), float(f[aid + n]))


def scale_actuator_strength(model, aid, factor, actuation_model=None):
    """アクチュエータの筋力を factor 倍する。筋肉モデルなら fmax、それ以外は gear。

    ⚠️スカラーの fmax は**全身に一律**なので、その場合は何もしない（関節ごとに
      調整できないため。黙って全身を変えると別の逸脱になる）。
    """
    import numpy as np
    f = getattr(actuation_model, "fmax", None)
    if f is None:
        model.actuator_gear[aid, 0] *= float(factor)
        return True
    f = np.asarray(f)
    if f.ndim == 0:
        return False                     # 関節ごとに変えられない
    n = int(getattr(actuation_model, "n_actuators", len(f) // 2))
    f[aid] *= float(factor)
    f[aid + n] *= float(factor)
    return True


# 【生理的屈曲（physiological flexion）・2026-07-25】
# 新生児は放っておいても股・膝・肘が曲がっている（屈曲拘縮）。一次文献の実測値：
#   股関節 -32度  Ishida et al. 1997, Rev Bras Ortop 32(1):37-45
#                 （n=80、満期産・生後0-4日、受動ROM、オープンアクセスで全文取得）[Tier1]
#   膝     -21度  Broughton, Wright & Menelaus 1993, J Pediatr Orthop 13(2):263-4
#                 （n=57、縦断。出生時21.4度→3ヶ月10.7度→6ヶ月3.3度）[Tier1]
#   肘     -14度  Watanabe et al. 1979（n=62。Norkin & White 教科書 Table 16-2 経由）[二次]
#
# 実測（補正なし・MuscleModel）：股 +1.6度 / 膝 -1.5度 / 肘 -14.4度
# ＝股と膝がほぼ伸びた姿勢で落ち着いており、人間の新生児と大きく違う。
#
# 【実装機構の判断】生理的屈曲は**受動的な組織特性**（屈筋トーン・関節包の張力）であって、
# 筋の能動的な姿勢保持ではない。後者は 2026-07-25 に2つの[Tier1]で否定されている
# （仰臥位は姿勢筋の需要が最小／固める方向は人間でも病的サイン＝cramped-synchronised）。
# → jnt_stiffness（関節のバネ）＋ qpos_spring（バネの中立位置）で表現する。
#   Ishida 1997 自身が「新生児の受動ROMの主因は屈筋トーンと子宮内姿勢による
#   **軟部組織の抵抗**」と書いており、この機構に対応する。
# ⚠️MuscleModel は jnt_stiffness=0 で、筋の受動力 fp(lce) で代替している。
#   fp を強める案（fpmax）は**スカラーで全身の筋に一律に効く**ので関節ごとに調整できない。
#   だから jnt_stiffness を 0 から立てる。fp は触らない。
FLEXION_TARGETS = {"hip1": -32.0, "knee": -21.0, "elbow": -14.0}
# バネの強さ。⚠️新生児の関節の受動剛性(N·m/rad)の文献値は**存在しない**（調査済み）。
# この値は「目標角度に落ち着く点を実測から逆算し、達成できる中で可動範囲が最大になる点」
# として選んだ [Tier2・実測から逆算]。実測（e_flexion_probe.py）：
#   stiffness  股eq/ROM      膝eq/ROM      肘eq/ROM
#   0          +1.6/96.7     -1.5/99.8     -14.4/148.8
#   1         -26.0/84.8    -19.9/72.9     -13.8/77.4
#   2 ←採用   -30.0/71.2    -19.2/59.1     -13.8/63.5   ＝3関節すべて目標±3度
#   4         -30.4/52.0    -20.5/46.6     -14.0/45.0   ＝可動範囲が落ちる
#   32        -31.8/14.5    -21.0/14.7     -14.0/7.9    ＝ほぼ動かない（異常の方向）
# 参考：人間の股関節の excursion は 75.5±13.0度（Chen et al. 2021, n=9, 3-4ヶ月児）で、
#       補正なしの96.7度より stiffness=2 の71.2度のほうが近い。
#       ⚠️ただし対象月齢も測り方も同じとは保証できないので「オーダーが合う」までの主張。
FLEXION_STIFFNESS = 2.0
# 生理的屈曲は発達で解消する（膝：出生21.4度→3ヶ月10.7度→6ヶ月3.3度）。
# ⚠️この月齢まで一定に保ち、それ以降は触らない[Tier3・簡略化]。
# 間の推移を線形補間する案は、3点しか実測が無いので今は入れない（恣意性を増やさない）。
FLEXION_UNTIL_MO = 3.0
# ★実装機構。2026-07-26 に "spring" から "range" へ変更した（理由は
# apply_physiological_flexion の docstring 参照。バネは曲がった関節を逆に伸ばしていた）。
#   "range"  ： jnt_range の伸展側に壁を立てる＝「これ以上は伸ばせない」（屈曲拘縮の意味）
#   "spring" ： 旧実装。qpos_spring + jnt_stiffness で目標角へ引き寄せる（比較用に残す）
FLEXION_MODE = "range"

# 【屈筋トーン（flexor tone）・2026-07-26】
# 新生児は脱力しても手足が曲がったまま＝**曲げる筋肉が常にうっすら働いている**。
# 大人が脱力すると重力で手足が伸びるのと逆。上の「壁」（拘縮＝伸展の限界）だけでは
# 再現できない（ユーザーの目視「膝が伸びきってる」で発覚）。
#
# 【機構】伸張反射（筋紡錘 → 脊髄 → 筋収縮）を jnt_stiffness + qpos_spring で近似する。
#   Solopova et al. (2019) Front Physiol [PMC6769424]（乳児54名、他動屈伸中の筋電）
#     ・他動運動に対する筋電反応が高頻度に観察され、潜時 460〜630ms の**トニックな収縮**
#     ・著者らは「脊髄・脊髄上ネットワークの興奮性水準の反映」＝**純粋な機械的特性ではない**と明記
#   Dolinskaya et al. (2023) Biology 12(5):724 [PMC10215963] も同様の反応を報告
# ⚠️[Tier2・近似] バネは伸張反射と**機能は似るが機構は違う**：
#     人間 … 潜時 460〜630ms／γループで反射ゲインが状況依存に変わる
#     バネ … 遅れゼロ／強さ固定
#   → 逸脱リストに登録。遅れ付きの反射として実装するのは将来の候補。
#
# 【なぜ持続的な筋活性化にしないか】Schloon, O'Brien, Scholten & Prechtl (1976)
#   Neuropädiatrie 7(4):384-415 [PMID 1036764] が安静時の多点筋電で
#   **屈筋優位の持続的活動を見出さなかった**＝随意的な姿勢保持ではない。
#
# 【目標角（バネの中立位置）の根拠】
#   肘 -105度  ★Farmania et al. (2017) J Neurosci Rural Pract 8(3) [PMC5602260]
#              満期産 n=74 の arm recoil（腕を伸ばして離すと戻る角度）105.3 ± 14.2度
#              ⚠️[Tier2] 反跳は「離した直後に戻る角度」で、静止時の角度とは厳密には別。
#                ユーザーの目視は -134度 で約2SD分ずれる。文献値を優先して採用し、
#                目視と食い違えば再検討する。
#   ★肘の実装値は -111度（下記）。文献の 105.3 ± 14.2度 の 1SD 内なので矛盾しない。
#   股・膝・肩  ⚠️[Tier3・ARBITRARY]
#              ★安静時の関節角度の一次文献は**存在しない**（2026-07-26 調査で確認）。
#              Cioni, Ferrari & Prechtl (1989) Early Hum Dev 18(4):247-262 は
#              無刺激ビデオ観察をしたが「完全屈曲への選好は確認できず、個人差が大きい」
#              と報告するのみで度数化していない。
#              → ユーザーが Viewer で新生児の見た目に合わせて作った値を採用する
#                （`E/docs/pose_editor_saved.json`、2026-07-26）。
#              ⚠️この値は **qpos0 の誤りを修正した後**に作り直したもの。
#                修正前は「数値は曲がっているのに実際は真っ直ぐ」だった（項49）。
TONE_TARGETS = {
    "hip1": -71.0,                 # 股（前後）
    "hip2": -36.0,                 # 股（開き）＝新生児の蛙のような開き
    "knee": -90.0,
    "shoulder_horizontal": 58.0,   # 肩（前後）
    "shoulder_ad_ab": 3.0,         # 肩（開き）
    "shoulder_rotation": -65.0,    # 肩（ひねり）
    "elbow": -111.0,               # [Tier2] 文献 105.3±14.2度（Farmania 2017）の範囲内
}
# バネの強さ [N·m/rad]。★肘の arm recoil で較正した実測値（2026-07-26）。
#   `E/scripts/e_arm_recoil_test.py` で 0.005〜0.2 を振り、肘を伸展位から離したときの
#   落ち着く角度を測定：
#       0.005〜0.02 → 14度台（ほぼ戻らない）／0.05 → 23度／0.1 → 91.9度／0.2 → 103.2度
#   目標 105.3 ± 14.2度（Farmania 2017）に対し **0.2 が差 2.1度** で最も近い。
# ★このプロジェクトで珍しく、目視でなく文献の実測から決まった値 [Tier2]。
#   ⚠️暫定で置いていた 0.02 は10倍小さく、較正しなければ気づけなかった。
TONE_STIFFNESS = 0.2
# 屈筋トーンが解消する月齢。⚠️[Tier3] 一次典拠が見つかっていない（2026-07-20/26 の調査）。
# 拘縮の解消（膝 21.4→10.7(3ヶ月)→3.3度(6ヶ月)、Broughton 1993）を目安に置く。
TONE_UNTIL_MO = 3.0


def apply_physiological_flexion(model, age=0.0, stiffness=None, verbose=True,
                                mode=None, data=None):
    """新生児の生理的屈曲を実装する。

    ★【2026-07-26 機構の訂正】既定を "spring" から "range" に変えた。

    【なぜ変えたか】ユーザーの目視「膝もひじも曲がってない」から発覚。
    実測すると、バネ方式は**曲がっている関節を逆に伸ばしていた**：

        関節      屈曲OFF     屈曲ON(バネ)   目標
        右ひざ     -3.7度     -19.1度       -21度   ✓ 正しく曲がった
        右ひじ   -114.9度     -15.3度       -14度   ✗ ★曲がっていたのが伸びた
        右股       -7.4度     -29.6度       -32度   ✓

    原因は**文献の意味の取り違え**。FLEXION_TARGETS の値は「屈曲拘縮」＝
    **これ以上は伸展できないという限界**であって、「ここに戻る」という目標角ではない。
      ・拘縮の意味   ： 膝は21度より真っ直ぐに【伸ばせない】＝伸展側の壁
      ・バネの意味   ： 膝を21度に【引き寄せる】＝両方向から目標へ
    バネは両方向に効くので、目標より深く曲がった関節を無理に伸ばしてしまう。

    ★正しい実装は逸脱リストに既に書かれていた（「次の候補（未実装）：関節可動域の制約…
      実装が明確（jnt_range を狭める）」）。実装時にそれを参照していなかった。

    【range 方式が一度却下された理由と、その解決】
    旧docstring：「range を狭めると初期姿勢が範囲外になってリセット時に関節が跳ねる」
    → 範囲を狭めるのと同時に **qpos0（リセット時の姿勢）も範囲内へ入れる**ことで解決する。

    Args:
        mode: "range"（既定・伸展側の壁）／"spring"（旧実装・比較用に残す）
        data: 渡すと現在の qpos も範囲内に入れる（跳ね防止）
    """
    import numpy as np   # このファイルの他の関数と同じくローカルにimportする
    if float(age) >= FLEXION_UNTIL_MO:
        if verbose:
            print(f"[flexion] age={age}mo >= {FLEXION_UNTIL_MO}mo: no correction")
        return dict(n=0)

    mode = FLEXION_MODE if mode is None else str(mode)

    if mode == "spring":
        # ⚠️旧実装。曲がった関節を伸ばしてしまうので既定では使わない（上記参照）。
        k = FLEXION_STIFFNESS if stiffness is None else float(stiffness)
        n, applied = 0, []
        for base, target in FLEXION_TARGETS.items():
            for side in ("right_", "left_"):
                try:
                    j = model.joint("robot:" + side + base)
                except Exception:
                    continue
                model.qpos_spring[int(model.jnt_qposadr[j.id])] = np.radians(target)
                model.jnt_stiffness[j.id] = k
                n += 1
            applied.append(f"{base}{target:+.0f}")
        if verbose:
            print(f"[flexion/spring] age={age}mo: {n} joints, stiffness={k} "
                  f"({' '.join(applied)}) ⚠️旧方式（曲がった関節を伸ばす）")
        return dict(n=n, stiffness=k, mode="spring")

    # --- range 方式（既定）：伸展側に壁を立てる ---
    n, applied, clamped = 0, [], 0
    for base, target in FLEXION_TARGETS.items():
        tgt = np.radians(target)
        for side in ("right_", "left_"):
            try:
                j = model.joint("robot:" + side + base)
            except Exception:
                continue
            jid = int(j.id)
            qadr = int(model.jnt_qposadr[jid])
            lo, hi = float(model.jnt_range[jid, 0]), float(model.jnt_range[jid, 1])
            # 目標が可動域の外なら触らない（体型やモデルが変わった場合の保険）
            if not (lo < tgt < hi):
                continue
            model.jnt_range[jid, 1] = tgt      # 伸展側（正の側）の限界を target に
            model.jnt_limited[jid] = 1
            # ⚠️★【2026-07-26 修正】ここで model.qpos0 を書き換えてはいけない。
            #   MuJoCo の関節の回転は **qpos - qpos0** で計算されるので、
            #   qpos0 を動かすと**角度のゼロ点ごとずれる**。
            #   実際 qpos0 と qpos を同じ値にしていたため回転がゼロになり、
            #   qpos は -105度なのに実際の屈曲は 5度、という状態になっていた
            #   （ユーザーの目視「ひじ膝伸びきってる」で発覚。→ チェックリスト項49）
            #   跳ね防止は data.qpos 側だけで行う。
            if data is not None and float(data.qpos[qadr]) > tgt:
                data.qpos[qadr] = tgt
                clamped += 1
            n += 1
        applied.append(f"{base}{target:+.0f}")
    if verbose:
        print(f"[flexion/range] age={age}mo: {n} joints, 伸展側の限界を設定 "
              f"({' '.join(applied)}) qpos0を{clamped}件クランプ "
              f"[Tier1: 限界角は実測（屈曲拘縮）]")
    return dict(n=n, mode="range", clamped=clamped)


def apply_flexor_tone(model, age=0.0, stiffness=None, targets=None,
                      verbose=True, data=None):
    """屈筋トーン＝安静姿勢へ戻ろうとする弱いバネ（伸張反射の近似）。

    上の `apply_physiological_flexion`（壁＝伸展の限界）と**併用する**。
    壁は「これ以上は伸ばせない」を決めるだけで、屈曲側の深さを決めない。
    新生児が脱力しても手足が曲がったままなのは、曲げる筋肉が常にうっすら
    働いているから（＝屈筋トーン）。それをバネで近似する。

    ⚠️2026-07-25 の "spring" 実装との違い：あれはバネの中立位置に
      **拘縮の限界**（膝-21度）を使っていたので、深く曲がった肘を逆に伸ばしていた。
      ここでは中立位置を**安静姿勢**（膝-107度）にする。根拠は上の TONE_TARGETS 参照。

    Args:
        data: 渡すと現在の qpos も目標角へ揃える（初期姿勢を安静姿勢にする）
    """
    import numpy as np
    if float(age) >= TONE_UNTIL_MO:
        if verbose:
            print(f"[tone] age={age}mo >= {TONE_UNTIL_MO}mo: no correction")
        return dict(n=0)
    k = TONE_STIFFNESS if stiffness is None else float(stiffness)
    tgts = TONE_TARGETS if targets is None else dict(targets)
    n, applied = 0, []
    for base, target in tgts.items():
        tgt = np.radians(target)
        for side in ("right_", "left_"):
            try:
                j = model.joint("robot:" + side + base)
            except Exception:
                continue
            jid = int(j.id)
            qadr = int(model.jnt_qposadr[jid])
            lo, hi = float(model.jnt_range[jid, 0]), float(model.jnt_range[jid, 1])
            tgt_c = float(np.clip(tgt, lo, hi))     # 壁の内側に収める
            model.qpos_spring[qadr] = tgt_c
            model.jnt_stiffness[jid] = k
            # ⚠️★【2026-07-26 修正】model.qpos0 は書き換えない。
            #   MuJoCo の関節の回転は qpos - qpos0 なので、qpos0 を動かすと
            #   角度のゼロ点ごとずれる。ここで qpos0 = qpos = -105度 にしていたため
            #   回転が 0 になり、**数値は曲がっているのに実際は真っ直ぐ**だった
            #   （→ チェックリスト項49）。初期姿勢は data.qpos だけで設定する。
            if data is not None:
                data.qpos[qadr] = tgt_c
            n += 1
        applied.append(f"{base}{target:+.0f}")
    if verbose:
        print(f"[tone] age={age}mo: {n} joints, stiffness={k} ({' '.join(applied)}) "
              f"[Tier2: 肘はarm recoil実測／Tier3: 股・膝は目視（安静時の文献なし）]")
    return dict(n=n, stiffness=k)


def apply_runtime_corrections(model, data, age, neck=True, limbs=True, head_mass=True,
                              limb_scale=1.0, distal_mass=1.0, flexion=False,
                              flexion_stiffness=None, actuation_model=None,
                              tone=True, tone_stiffness=None):
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
    # ★手足の質量（感度分析用）も筋力補正より先。四肢の補正は「筋力÷その関節から先の
    #   重力モーメント」を見るので、質量を後から変えると補正が古い前提のままになる。
    apply_distal_mass(model, float(distal_mass))
    if neck:
        from infant_neck import apply_newborn_neck
        apply_newborn_neck(model, data, float(age), actuation_model=actuation_model)
    if limbs:
        from infant_limbs import apply_limb_inversion_fix
        # limb_scale＝四肢の筋力補正の感度分析用の係数（1.0＝補正そのまま）。
        # この補正の目標値に根拠が無いことが2026-07-25の文献調査で判明したため、
        # 「結論がこの仮定に依存していないか」を振って確かめられるようにしてある。
        apply_limb_inversion_fix(model, data, float(age), scale=float(limb_scale),
                                 actuation_model=actuation_model)
    # 生理的屈曲は筋力補正の後（gear と独立なので順序は結果に影響しないが、
    # 「筋力→姿勢」の順で読めるようにここに置く）。既定OFF＝目視で確認してから既定ONにする。
    if flexion:
        # ★data を渡す：range 方式では現在の qpos も新しい範囲内へ入れて跳ねを防ぐ
        apply_physiological_flexion(model, float(age), stiffness=flexion_stiffness,
                                    data=data)
        # ★屈筋トーン（バネ）は壁の後。壁の内側に目標角を収めるため順序が必要。
        if tone:
            apply_flexor_tone(model, float(age), stiffness=tone_stiffness, data=data)
