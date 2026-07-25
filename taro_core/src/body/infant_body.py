"""新生児の体型（プロポーション）— 太郎の身体そのものの定義。

【なぜ core にあるか、2026-07-25】この体型補正はもともと目標Eのおもちゃ環境
（`E/scripts/e_toy_env.py`）の中に書かれていた。その結果**環境を変えると太郎の体型が
変わる**状態になり、実際に「おもちゃ環境では補正あり／学習に使う仰向け環境では補正なし」
という食い違いが起きていた（＝太郎は成人プロポーションのまま学習していた）。
体型は**太郎そのもの**であって環境の性質ではないので core へ移した。
方針：[[feedback-core-vs-experiment-placement]]（本実装は core、実験は目標フォルダ）。

【この体型が確定した経緯（2026-07-21「新生児体型v2」）】
mimoGrowth の age=0 は「大きさは新生児だが四肢の比率が成人」だった（人間模倣からの逸脱リスト
2026-07-20 その4）。加えて頭が球なので、頭囲34cmでも真上から見た頭が15%小さい。
→ ①頭を体軸方向に楕円化（頭囲を保ったまま見かけを人間に）②四肢・手足を縮小。

**確定の根拠**：主にViewerでの目視（ユーザー＋育児経験者）。四肢を縮める向きと大きさは、
新生児は頭でっかちで四肢が短いという観察に一致。
⚠️**各係数そのものは目視で決めた恣意値**[Tier3・ARBITRARY]（文献で裏取りしたのは
足長7.58cmのみ。下肢19.6cm・上肢20.96cmは測定定義が成人的で新生児に不整合と判明＝
当てにしない）。詳細は `doc/人間模倣からの逸脱リスト.md` 2026-07-21。

**確定寸法**：身長37.0cm 頭囲34.0cm（不変） 頭12.55 胴14.65 腕12.16 脚13.02 足7.5 手2.4cm
　頭/身長 0.339（人間0.25より頭でっかち＝見た目重視の選択）。
⚠️身長37cmは新生児49.9cmの74%＝**絶対サイズは小さい**。比率（見た目）を優先した結果。

【使い方】
    from infant_body import body_scale_custom, NEWBORN_SHAPE_DEFAULTS
    custom = body_scale_custom(age=0.0, scales=NEWBORN_SHAPE_DEFAULTS)
    env = SomeMimoEnv(age=0.0, custom_measurements=custom)

⚠️環境変数（`E_LEG_SCALE` 等）の読み取りは**目標フォルダ側の責務**。core は係数の既定値と
変換ロジックだけを持つ（[[feedback-core-target-neutral-naming]]＝core に目標プレフィックスを
書かない）。
"""

# 部位グループ -> 既定の係数。1.0 は「触らない」。
# ⚠️[Tier3・ARBITRARY] 目視で決めた値。文献で裏取りできたのは足長のみ。
NEWBORN_SHAPE_DEFAULTS = {
    "leg":        0.45,
    "leg_thick":  0.60,
    "arm":        0.62,
    "arm_thick":  0.62,
    "trunk_len":  0.74,
    "foot":       0.72,
    "hand":       0.70,
    # 既定で使わない微調整用（1.0）。目視で必要になったら呼び出し側で振る。
    "trunk_width": 1.0,
    "foot_width":  1.0,
}

# 頭を体軸方向に 12.5/10.8 倍＝球→楕円（頭囲は不変）。
HEAD_ELONGATION = 1.16

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
