# infant_limbs.py 取扱説明書（AI用）

本体: `taro_core\src\body\infant_limbs.py`（703行）
最終同期: 2026-08-14 / 本体の行数: 703

## このファイルは何をするか
乳児の四肢（腕・脚。首・体幹は対象外）について、(1) 月齢間で筋力/自重比が逆転する
問題を補正する`apply_limb_inversion_fix`系と、(2) 脱力しても腕が体の前に保たれる
「筋緊張（バネ）」を実装する`apply_limb_tone`系の、2つの独立した機構を持つ。
MuJoCoの`model`/`data`を受け取り、gear・剛性(jnt_stiffness)・初期姿勢を書き換える。

## 主要なクラス・関数
| 名前 | 行 | 何をするか |
|---|---|---|
| `G` / `REFERENCE_AGE` / `_CACHE`（定数） | 117-119 | 重力加速度・基準月齢(18ヶ月)・基準比のプロセス内キャッシュ |
| `_REF_PATH` | 142-143 | 基準比の保存ファイル`limb_reference_18mo.json`のパス |
| `_descendants` | 146-155 | あるbodyから末端側にぶら下がる全body idを列挙 |
| `actuator_ratios` | 158-178 | 各アクチュエータの「筋力÷その先の重力モーメント」比を算出 |
| `measure_reference_ratios` | 181-193 | 参照月齢(既定18ヶ月)の素の体を新規に作って比を測る（基準ファイル生成専用、通常経路から呼ばない） |
| `_reference_ratios` | 196-214 | 保存済み基準比ファイルを読む（その場で測り直さない） |
| `apply_limb_inversion_fix` | 217-279 | 四肢のgear(筋力)を下げ、age時点の比がreference_age以下になるよう補正。`scale`引数で感度分析可 |
| `LIMB_TONE_UNTIL_MO`（定数） | 318 | 筋緊張バネを効かせる月齢の上限(6.0mo, Tier2) |
| `LIMB_TONE_GROUPS`（辞書） | 323-335 | 部位名(shoulder/elbow/wrist/hip/knee/ankle/hip_sagittal)→関節名タプル |
| `LIMB_TONE_ALIASES`（辞書） | 337-340 | "arm"/"leg"のような大まかな別名→細かい部位への展開表 |
| `GROUP_JP_LIMB`（辞書） | 341-343 | 部位名の日本語表示用マップ |
| `_expand_groups` | 346-351 | "arm"のような別名を細かい部位タプルへ展開 |
| `limb_tone_joints` | 354-356 | 筋緊張の対象になる関節名（側の接頭辞なし部分）の一覧を返す |
| `limb_tone_joints_by_group` | 359-378 | (関節フル名, 部位名) のペア一覧を返す（部位ごとの強さ変更用） |
| `gravity_moment` | 381-427 | ある関節から先にぶら下がる重力モーメントを測る（worst_case=姿勢によらない上界） |
| `tone_from_gravity` | 430-444 | 「hold_deg度のずれで釣り合う」剛性[N·m/rad]を重力モーメントから逆算 |
| `TONE_PROFILES`（辞書） | 469-522 | `apply_limb_tone`の宣言的プリセット。"newborn_flexor"(新生児・固定目標角)と"reach_limb"(現在の姿勢を保持)の2種 |
| `disable_limb_tone_spring` | 525-562 | `apply_limb_tone`が入れた継続的バネ(jnt_stiffness)を0に戻し無効化する（新駆動モジュール用） |
| `apply_limb_tone` | 565-703 | 四肢の筋緊張バネ本体。剛性・目標角・減衰(臨界減衰)を関節ごとに設定する |

## 触るときの注意
- **この補正・筋緊張の目標値には両方とも根拠が無い**ことがファイル冒頭(1-110行)に
  詳しく書かれている。数値を変えるときは必ず[Tier]ラベルと根拠（無ければ「根拠無し」）
  を明記する慣習を踏襲する。
- `apply_limb_inversion_fix`の基準比は**その場で測ってはいけない**（197-204行の
  `_reference_ratios`のdocstring・121-141行のコメント参照）。新生児の体型補正が
  裏の基準体にも漏れて2.1倍ずれるバグを2026-07-29に踏んだ実例あり。基準を作り直す
  ときは`taro_core/tools/measure_limb_reference.py --update`を使う。
- 首(`head_*`)・体幹(`chest_*`)は`apply_limb_inversion_fix`の対象外（246行で
  明示的にスキップ）。体幹は逆転していない・首は別ロジック(`infant_body.py`の
  `apply_neck_tone`)のため。
- `apply_limb_tone`は月齢ごとに剛性・目標角を計算するだけでなく、`force_qpos=True`
  のとき初期姿勢(data.qpos)も書き換える（672-676行）。`disable_limb_tone_spring`は
  stiffnessだけを0に戻し、この初期姿勢そのものは戻さない（543-545行に明記）。
  両者を混同すると「継続的なバネを切ったのに姿勢が変わっている」と誤認しうる。
- `infant_body`モジュールへの参照は循環import回避のため、関数の**呼び出し時に
  遅延import**する設計（175行・255行・611行・626行）。モジュール先頭に
  `from infant_body import ...`を書かないこと。
- `apply_limb_tone`を実際に呼んでいるのは`run/scene_tools/scene_io.py`の
  `_apply_limb_tone()`（本ファイルの外）。呼び出し側の引数の渡し方を変える必要が
  あるときはそちらを触る必要があるが、本取説の対象外（536-541行の「想定外」記録も参照）。
- hold_deg=20.0（newborn_flexorプロファイルの既定、507行）はFarmania 2017の肘反跳実測
  との整合（Tier2）を手放し、能動運動の自由を優先した2026-08-10の判断。数値を戻す/
  変えるときは492-506行の経緯を読むこと。

## よくある変更の場所
- 筋力逆転補正の強さを感度分析したいなら`apply_limb_inversion_fix`の`scale`引数
  （217-235行）。シーンJSON側の`body.limb_scale`から渡す設計（既定1.0、0ヶ月
  自発運動シーンは10.0）。
- 対象の部位（肩/肘/手首/股/膝/足首）を追加・変更したいなら`LIMB_TONE_GROUPS`
  （323-335行）と`LIMB_TONE_ALIASES`（337-340行）。
- 筋緊張の強さ(hold_deg)や対象月齢を変えたいなら`TONE_PROFILES`（469-522行）の
  該当プロファイル、または`apply_limb_tone`呼び出し時の`hold_deg`/`until_mo`引数
  （565-567行の引数一覧）。
- 新しいプリセット（プロファイル）を追加したいなら`TONE_PROFILES`辞書に
  キーを1つ足す（groups/target/stiffness/hold_deg/until_mo/clamp_to_range/
  force_qposの7項目、511-521行が最小のテンプレになる）。
- 重力モーメントの測り方（worst_case/今の姿勢）を変えたいなら`gravity_moment`
  （381-427行）の`worst_case`引数まわり。
