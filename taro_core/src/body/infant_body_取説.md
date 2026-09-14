# infant_body.py 取扱説明書（AI用）

本体: taro_core\src\body\infant_body.py（1,030行）
最終同期: 2026-08-14 / 本体の行数: 1030

## このファイルは何をするか
乳児（新生児〜18ヶ月）の身体モデル（MIMoの体型・頭部質量・関節の筋力・生理的屈曲・
筋緊張・首のバネ・眼球の正中位）を人間の新生児に近づけるための補正ロジック一式。
モデル構築前（geom寸法）と構築後（アクチュエータのgear／バネ）の両方に対応する。

## 主要なクラス・関数
このファイルにクラスは無い（すべてモジュール関数）。

| 名前 | 行 | 何をするか |
|---|---|---|
| `taper_weight` | 112-116 | 新生児向け補正を月齢で線形に薄める重み（0ヶ月=1.0→until_mo=0.0） |
| `elongate_head` | 182-226 | 頭を体軸方向に楕円化する（球のままだと真上から見て15%小さいため） |
| `_restore_growth_schema` | 232-257 | MIMo側のグローバル辞書バグ対策。custom適用前にスキーマを毎回元に戻す |
| `body_scale_custom` | 260-309 | 部位ごとのgeom寸法を係数倍する custom 辞書を作る（体型補正の本体） |
| `build_body_kwargs` | 330-348 | 環境コンストラクタに渡す身体設定（構築前の分）をまとめて作る |
| `apply_head_mass` | 351-407 | 頭の密度を上げ、全体重に占める頭の質量比を人間の新生児(25%)に合わせる |
| `apply_distal_mass` | 410-461 | 手足の質量だけを倍率で変える（サイズは変えない、感度分析用） |
| `actuator_strength` | 484-497 | アクチュエータの筋力を返す（筋肉モデルならfmax、それ以外はgear） |
| `scale_actuator_strength` | 500-517 | アクチュエータの筋力をfactor倍する（同上の書き込み版） |
| `apply_physiological_flexion` | 698-789 | 新生児の生理的屈曲（股・膝・肘が伸展できない限界）を実装する |
| `apply_runtime_corrections` | 804-884 | モデル構築後の補正をまとめて呼ぶ統一入口（頭質量→四肢質量→首→四肢筋力→屈曲→筋緊張→首の緊張→関節可動域の滑らかさ、の順） |
| `center_eyes` | 920-953 | 眼球の関節角を基準位置（正中位）に戻す。リセットのたびに呼ぶ想定 |
| `apply_neck_tone` | 956-1030 | 首の筋緊張（頭が重力で倒れきらないよう支える弱いバネ）を効かせる |

## 主要な定数（ファイル冒頭〜中盤に多数の根拠付き数値がある。変更時は必ずコメントの
Tierラベルと根拠実験を読むこと）
- `NEWBORN_SHAPE_DEFAULTS`（67-77行目）: 部位ごとの体型補正係数
- `HEAD_ELONGATION`（80行目）、`HEAD_MASS_FRACTION`（97行目）、`HEAD_MASS_UNTIL_MO`（99行目）
- `SHAPE_UNTIL_MO`（131行目）: 体型補正を何ヶ月まで効かせるか
- `_FOOT_GEOMS` / `_HAND_GEOMS` / `_GROUPS`（134-179行目）: 体型補正が触るgeom名の対応表
- `FLEXION_TARGETS` / `FLEXION_STIFFNESS` / `FLEXION_UNTIL_MO` / `FLEXION_MODE`（540-562行目）
- `TONE_TARGETS` / `TONE_STIFFNESS` / `TONE_UNTIL_MO`（599-625行目、2026-08-08時点で
  TONE_STIFFNESSは直接参照されないが経緯の記録として残っている）
- `NECK_TONE_STIFFNESS` / `NECK_TONE_TARGET` / `NECK_TONE_JOINTS` / `NECK_TONE_UNTIL_MO`
  （654-695行目）
- `EYE_REST_VERTICAL_DEG`（917行目）: 環境変数 `E_EYE_REST_V` から読む眼球の上下基準角

## 触るときの注意
- **coreに目標プレフィックス（E_/目標E固有の名前）を書かない方針**（7-8, 45-47行目）。
  環境変数の読み取りは目標フォルダ側（`run/scene_tools/e_scene.py`等）の責務で、
  このファイルは係数の既定値と変換ロジックだけを持つ。
- `_restore_growth_schema`（232-257行目）は必須の防御。これを外すと同一プロセス内で
  環境を作り直すたびに体が縮み続けるバグ（MIMo側のグローバル辞書破壊）が再発する。
- `apply_head_mass` は**必ず首・四肢の筋力補正より先に**呼ぶこと（362行目）。
  順序を変えると首の補正が古い頭質量を前提に計算される。`apply_distal_mass` も
  四肢の筋力補正より先（844-845行目）。
- `apply_physiological_flexion` は "spring" 方式が**曲がった関節を逆に伸ばす**バグを
  2026-07-26に踏んでいる（702-720行目）。既定は "range" 方式。`mode="spring"` は
  比較用に残しているだけで、通常は使わない。
- 筋力補正（gear）は**筋肉モデル使用時は毎ステップ上書きされる**ため
  `model.actuator_gear` を書いても効かない（464-482行目）。実体は
  `actuation_model.fmax` を直接書き換える必要がある。この罠は過去に「筋力を16倍振っても
  学習結果が完全一致する」という形で発覚した。
- `apply_runtime_corrections` 内の呼び出し順序に依存関係が多い（頭質量→四肢質量→首→
  四肢筋力→生理的屈曲→筋緊張→首の緊張→関節可動域の滑らかさ）。804-839行目のdocstringに
  理由が書かれているので、順序を変える前に必読。
- `flexion=True` 経由での筋緊張起動（`apply_runtime_corrections`内、866-869行目）と、
  `run/scene_tools/e_scene.py`の`_apply_limb_tone`（`setup.limb_tone`を読む）は
  **完全に別の入口**。GUIとの整合は後者側でのみ保証されている（822-831行目）。
- 数値のほとんどに Tier1/2/3 の根拠ラベルが付いている。値を変更する場合は
  ラベルと出典コメントを一緒に更新すること（このファイルの慣例）。

## よくある変更の場所
- 体型比率を調整したいなら `NEWBORN_SHAPE_DEFAULTS`（67-77行目）と `_GROUPS`（165-179行目）。
- 頭の質量比・楕円化率を変えたいなら `HEAD_MASS_FRACTION`（97行目）・
  `HEAD_ELONGATION`（80行目）、適用関数は `apply_head_mass`（351行目）・
  `elongate_head`（182行目）。
- 新生児補正が消える月齢（崖を作らないための線形の薄め方）を変えたいなら
  `SHAPE_UNTIL_MO`（131行目）・`HEAD_MASS_UNTIL_MO`（99行目）・`FLEXION_UNTIL_MO`
  （557行目）・`TONE_UNTIL_MO`（625行目）・`NECK_TONE_UNTIL_MO`（695行目）と、
  それぞれが使う `taper_weight`（112行目）。
- 生理的屈曲の目標角・強さを変えたいなら `FLEXION_TARGETS`／`FLEXION_STIFFNESS`
  （540-553行目）、適用は `apply_physiological_flexion`（698行目）。
- 首のバネの強さ・目標角を変えたいなら `NECK_TONE_STIFFNESS`／`NECK_TONE_TARGET`
  （654-673行目）、適用は `apply_neck_tone`（956行目）。
- モデル構築後の補正を丸ごと呼ぶ入口を変えたいなら `apply_runtime_corrections`
  （804行目）。呼び出し元は主に `run/scene_tools/e_scene.py` の `build()`。
