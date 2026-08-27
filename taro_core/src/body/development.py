"""月齢(age_months) → 発達パラメータ の対応表を一本化する（2026-08-23新設）。

【なぜ要るか】
太郎の発達パラメータの一部が「月齢に連動すべきなのに固定値」になっていて、
実験のたびに手で辻褄を合わせる事故が起きていた：
  - 視力の設定は6ヶ月（5.64 cycles/度）なのに、レンダリング解像度が128px固定
    （視野60度）で 1.07 cycles/度しか出ていなかった（F1-7で発覚）
  - 声道が既定 stage=0 では「わ」が出せず、run/taro_setup.py が stage=2 を
    **手で固定値として**与えていた（doc/人間模倣からの逸脱リスト.md 逸脱その25。
    月齢連動は未配線とコード内に明記されていた）

このモジュールは、月齢を受け取り各パラメータを返す関数を1箇所にまとめる。
目標横断（E/F/どの目標からも）で使うため、目標固有の表記（F_ 等）は置かない。

【呼び出し側】
  - `E/scripts/e_toy_env.py`（視力・解像度）：既存の `_acuity_cpd`/`required_px`
    をここへ委譲する形に変更（二重実装の解消）。
  - `run/taro_setup.py`（声道の成熟ステージ）：`produce.develop_from_age=true`
    のときだけ `vocal_tract_stage_for_age()` を使う（既定は従来どおり固定値）。
"""

import math


# ============================================================================
# 視力（acuity） — Mayer, Fkalitzka & Stager (1995) の実測テーブル
# ============================================================================
# 【出典・二重実装についての注記】
#   一次実装は `MIMo/mimoVision/vision.py:192-219`（`SimpleVision._get_acuity_function`
#   内）。あの関数は「acuity値をテーブルから引く」処理と「そのacuityからMTF配列
#   (height×width)を作る」処理が1つの関数に混ざっており、スカラーのacuity値だけを
#   取り出すモジュール関数が存在しないため import できない（この判断は
#   `E/scripts/e_toy_env.py` の `_acuity_cpd` docstring・2026-08-22 で既出）。
#   このテーブルの数値自体は e_toy_env.py 側の複製（当時の行336-342）と
#   1文字も変えていない。
#
#   今回、taro_core（目標横断で共有する場所）とE（目標固有）の2箇所に同じ表が
#   存在する二重実装を解消するため、**canonicalな置き場所をここに移し**、
#   e_toy_env.py 側は本モジュールをimportして使う形に変更した
#   （依存の向き：E → taro_core。逆方向にしない。ノウハウ「e_scene.pyの値を
#   taro_core側のクラスへ渡す」と同じ理由＝taro_coreは目標フォルダの存在を
#   知らない状態を保つ）。
_ACUITY_AGES_MONTHS = [
    1., 1.169, 1.366, 1.596, 1.865, 2.179, 2.547, 2.976,
    3.478, 4.064, 4.75, 5.55, 6.486, 7.58, 8.858, 10.351,
    12.096, 14.135, 16.519, 19.304, 22.558, 26.362, 30.806, 36.,
]
_ACUITY_CPD = [
    0.852, 1.0005, 1.1745, 1.3795, 1.621, 1.9065, 2.2445,
    2.645, 3.121, 3.6885, 4.367, 5.179, 6.1425, 7.1985,
    8.218, 9.008, 9.405, 9.544, 9.6775, 10.079, 11.013,
    12.54, 14.6785, 17.4195,
]


def vision_acuity_cpd(age_months):
    """月齢 → 視力[cycles/度]（Mayer et al. 1995・実測テーブルの線形補間）。

    テーブル範囲(1.0〜36.0ヶ月)の外は端の値にクランプする
    （vision.py:192-219 と同じ挙動。1ヶ月未満は「1ヶ月児の視力で代用」という
    既知の逸脱がそのまま残る＝新生児(0ヶ月)の実測値はこのテーブルに存在しない）。
    """
    age_months = float(age_months)
    ages, acuities = _ACUITY_AGES_MONTHS, _ACUITY_CPD
    if age_months in ages:
        return acuities[ages.index(age_months)]
    if age_months < ages[0]:
        return acuities[0]
    if age_months > ages[-1]:
        return acuities[-1]
    for i in range(len(ages)):
        if ages[i] > age_months:
            weight = (age_months - ages[i - 1]) / (ages[i] - ages[i - 1])
            return acuities[i - 1] + (acuities[i] - acuities[i - 1]) * weight
    return acuities[-1]  # 到達しないはずの保険


def vision_required_px(age_months, fovy_deg):
    """視力フィルタが効く上限まで画素を用意するのに必要な解像度[px]（F1-7と同じ式）。

        px = 2 * acuity[cpd] * fovy_deg * 1.15   （ナイキスト×余裕15%）
        戻り値 = max(64, ceil(px/16)*16)          （16の倍数へ切り上げ、下限64）

    検算：6.0ヶ月・15度 → acuity=5.642cpd → px=194.6 → 208px
    （F1-7仕様書・e_toy_env.py の検算コメントと一致）。
    """
    acuity = vision_acuity_cpd(age_months)
    px = 2.0 * acuity * float(fovy_deg) * 1.15
    return max(64, int(math.ceil(px / 16.0)) * 16)


# ============================================================================
# 声道(vocal tract)の成熟ステージ
# ============================================================================
# 【taro_core/src/brain/vocal_tract.py 118-149行を確認した結果】
#   STAGE_ALLOWED_PLACE/MANNER/VOICING/VOWEL の4つのテーブル自体には、
#   stage0→1→2→3のしきい値となる月齢は**一切書かれていない**。
#   唯一の月齢言及は、143行の一文だけ：
#     「人間の音節性鼻音は規準喃語（6ヶ月〜）と同時に現れる」
#   （vowel index5=「ん」がstage1から解禁される理由付け）。
#   ＝ **stage1→2の境界だけ、コード内に月齢の根拠がある**。
#   stage0→1・stage2→3 の境界には、このコードベース内に根拠となる記述が
#   見当たらなかった。
#
# 【今回の割り当て（根拠の強さは境界ごとに違う）】
#   0〜4ヶ月   : stage0（母音のみ）
#   4〜6ヶ月   : stage1（+両唇・鼻音・破裂音＝「ばばば」「まままま」）
#   6〜12ヶ月  : stage2（+歯茎・歯茎硬口蓋・硬口蓋＝「だだだ」「ななな」）
#   12ヶ月〜   : stage3（+声帯の無声化＝「ぱ」と「ば」の使い分け、全解放）
#
#   [Tier1・コード内根拠あり] 6ヶ月＝stage1→2（規準喃語の開始。
#     vocal_tract.py:143の記述をそのまま採用）。
#   [Tier2・コード外の一般的知見、暫定] 4ヶ月＝stage0→1。周辺喃語
#     （marginal babbling、子音様の音が混じり始める）の開始目安として
#     一般に引用される月齢だが、**このコードベース内には根拠記述が無い**。
#     「根拠なし」の代わりに置いた暫定値。
#   [Tier3・ARBITRARY・根拠なし、暫定] 12ヶ月＝stage2→3。初語が出始める
#     目安月齢を代理指標として流用した。このコードベース・一般文献のどちらにも
#     「無声音の分化がこの月齢で起きる」という直接の根拠は無い。
#
#   実際の使われ方：run/taro_setup.py が `produce.develop_from_age=true` の
#   ときだけこの関数を使う（既定False＝従来どおり固定値 `produce.vocal_tract_stage`
#   （既定2）を使う。1ビットも変わらない）。
def vocal_tract_stage_for_age(age_months):
    """月齢 → 声道の成熟ステージ(0〜3)。境界の根拠の強さは上のコメント参照。"""
    age_months = float(age_months)
    if age_months >= 12.0:
        return 3
    if age_months >= 6.0:
        return 2
    if age_months >= 4.0:
        return 1
    return 0


# ============================================================================
# まとめて見るための便利関数（判断材料の表を作るためだけに使う）
# ============================================================================
def develop(age_months, vision_fovy_deg=60.0, fovea_fovy_deg=15.0):
    """月齢1つから、今回配線した各パラメータをまとめて返す（表作成・検算用）。"""
    return {
        "age_months": float(age_months),
        "vision_acuity_cpd": vision_acuity_cpd(age_months),
        "vision_periphery_px": vision_required_px(age_months, vision_fovy_deg),
        "vision_fovea_px": vision_required_px(age_months, fovea_fovy_deg),
        "vocal_tract_stage": vocal_tract_stage_for_age(age_months),
    }
