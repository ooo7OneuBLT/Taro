"""既存のプリセットからシーンファイルを作る（シーン方式への引っ越し）。

【なぜ要るか、2026-07-29】これまでの環境の条件は
コードの定数・環境変数・Viewerの `PRESETS`・`viewer_saved.json` の4か所に
散らばっていた。それを1つのシーンファイルへ集める。

姿勢（qpos）は「作ってリセットし、少し落ち着かせた状態」を保存する。
そこから先はViewerで人が調整して保存し直す（それが本来の使い方）。

使い方:
    .venv/Scripts/python.exe run/scene_tools/e_scene_make.py            # 全部作る
    .venv/Scripts/python.exe run/scene_tools/e_scene_make.py 名前       # 1つだけ
"""

# 注意：古い方式（2026-07-30 に整理）。新しい実験は `run/main.py` を通す。
#   【経緯】目標Eの実験スクリプトが118本あり、うち66本が**独立に環境を組み立てていた**。
#     そのため「学習は関節モード（90関節を独立に駆動＝逸脱リスト 逸脱5）、
#     測定とViewerは筋肉モード（拮抗筋2本/関節）」という**別の体で動く**事故が起きた
#     （ユーザーの目視「視線誘導反射の実験の時とは動きが全然違う」で発覚。
#      実測で動きが人間の新生児の約3.3倍速かった）。
#   【設計と移行計画】`E/docs/実行基盤_設計.md`
#   注意：このファイルは**記録として残す**（削除しない方針）。
#     中の測り方は再利用できるので、プラグインへ移すときの元にする。
import os
import sys
import warnings

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np      # noqa: E402
import e_scene          # noqa: E402


def _scene(name, note, body=None, world=None, setup=None):
    sc = e_scene.default_scene(name)
    sc["note"] = note
    for key, over in (("body", body), ("world", world), ("setup", setup)):
        if over:
            sc[key] = e_scene._merge(sc[key], over)
    return sc


# ============================================================================
# 引っ越し元＝`e_viewer.py` の PRESETS と、2026-07-28 までの起動コマンド
# ============================================================================
SCENES = [
    _scene(
        "視線誘導反射_仰向け_頭を支える",
        "4ヶ月・仰向け・実験者が頭を支える。3シードで反射の機能を確認済み"
        "（反射ON 3.2度 vs OFF 10.2度）",
        body={"age_months": 4.0, "eye_rest_vertical_deg": 0.0},
        world={"recline_deg": 0.0, "fence": False,
               "toy": {"shape": "sphere", "radius": 0.0056, "mode": "hold",
                       "dist": 0.086, "delay_sec": 0.0}},
        setup={"head_hold": {"stiffness": 200.0, "target_deg": None}},
    ),
    _scene(
        "リーチング_リクライニング60度",
        "4ヶ月・リクライニング60度・実験者が頭を60度で支える（顎を引く）・眼球-15度。"
        "注意おもちゃの位置は未確定（遮蔽で候補が体に隠れる）",
        body={"age_months": 4.0, "eye_rest_vertical_deg": -15.0},
        world={"recline_deg": 60.0, "seat_friction": 2.0, "fence": False,
               "toy": {"shape": "sphere", "radius": 0.0056, "mode": "hold",
                       "dist": 0.18, "delay_sec": 0.0}},
        setup={"head_hold": {"stiffness": 200.0,
                             "target_deg": {"head_tilt": 60.0}},
               # 肩から手先と指と眼球だけを自由にし、それ以外は椅子とベルトが支える
               #   （ユーザーの提案 2026-07-29）。リーチングは腕と目でやる動作なので、
               #   体幹や脚が動いて姿勢が崩れるのを実験条件として止める。
               #   注意：脚の固定は人間からの逸脱（現実の乳児は椅子でも脚は自由）。
               "body_support": {"stiffness": 200.0, "free": ["arm", "finger"]}},
    ),
    _scene(
        "リーチング_リクライニング60度_肘屈曲強化",
        "4ヶ月・リクライニング60度・実験者が頭を60度で支える（顎を引く）・眼球-15度。"
        "「リーチング_リクライニング60度」を土台に、limb_toneの肘の目標角度だけを"
        "-130度へ変更（2026-08-03、頭ダブルタッチ報酬の代替案＝候補2・姿勢バイアス版）。"
        "狙い：新生児の生理的屈曲姿勢（膝窩角90度未満がDubowitz et al. 2005、"
        "肘の屈曲優勢がZoia et al. 2007）を参考に腕を体に近い角度で保つと自己接触が"
        "増えるか確かめる実験。注意：屈曲姿勢が自己接触を増やすという因果自体は"
        "一次資料で未検証（`E/docs/リーチング/文献調査/屈曲姿勢と自己接触頻度_2026-08-03.md`）。"
        "肘-130度は[Tier3・工学的判断]、どれだけ曲げれば十分かの基準は文献に無い"
        "（`doc/人間模倣からの逸脱リスト.md` ⑧参照）。他の設定は土台と同じ",
        body={"age_months": 4.0, "eye_rest_vertical_deg": -15.0},
        world={"recline_deg": 60.0, "seat_friction": 2.0, "fence": False,
               "toy": {"shape": "sphere", "radius": 0.0056, "mode": "hold",
                       "dist": 0.18, "delay_sec": 0.0}},
        setup={"head_hold": {"stiffness": 200.0,
                             "target_deg": {"head_tilt": 60.0}},
               "body_support": {"stiffness": 200.0, "free": ["arm", "finger"]},
               # 肘だけ目標角を明示（-130度＝屈曲側）。他の部位（肩・手首・脚）は
               #   target_deg を書かないので土台と同じく「今の姿勢」が落ち着き先になる。
               "limb_tone": {"hold_deg": 10.0, "groups": ["arm", "leg"],
                            "target_deg": {"right_elbow": -130.0,
                                          "left_elbow": -130.0}}},
    ),
    _scene(
        "リーチング_リクライニング60度_自己接触版",
        "4ヶ月・リクライニング60度・実験者が頭を60度で支える（顎を引く）・眼球-15度。"
        "自己接触（reach_self、頭・胸・反対の手への接触）の検証専用シーン"
        "（2026-08-03、仕様:作業記録（非公開）"
        "2026-08-03_自己接触専用シーンの新設.md）。"
        "「リーチング_リクライニング60度」を土台に、①おもちゃを無効化（world.toy.enabled=false）"
        "②body_support.freeへ体幹(trunk)を追加し胴体を自由にした。"
        "狙い：土台シーンはもともとおもちゃへのリーチング課題用に作られ、おもちゃが目から"
        "6.28cmの至近距離にあり体幹・脚が完全固定（工学的な逸脱）。自己接触の検証には"
        "どちらも不要で、屈曲姿勢を試すとおもちゃと屈曲した腕・胴体が干渉して姿勢が壊れる"
        "不具合が実測で判明した（報告:2026-08-03_おもちゃ配置の衝突回避.md）。"
        "注意[Tier3・工学的判断]：胴体（体幹）を自由にするのは初めての試みで、恣意的な判断。"
        "文献根拠なし（脚は引き続き固定のまま＝土台と同じ）。座面からずり落ちる懸念があるため"
        "settle・写真の両方で確認すること。`doc/人間模倣からの逸脱リスト.md`参照。"
        "limb_toneは土台の実際の保存値（hold_deg=10, groups=[arm,leg], target_degなし＝"
        "今の姿勢を保つ）をそのまま複製した（e_scene_make.pyの土台エントリ自体には"
        "limb_toneの記載が無いという既知の食い違いがあるため、実際に保存されている"
        "土台JSONの値を基準にした。詳細は上記おもちゃ配置の衝突回避報告 2-1'節）。"
        "屈曲版との唯一の違いを肘の角度だけにするため、target_degは指定しない",
        body={"age_months": 4.0, "eye_rest_vertical_deg": -15.0},
        world={"recline_deg": 60.0, "seat_friction": 2.0, "fence": False,
               "toy": {"enabled": False}},
        setup={"head_hold": {"stiffness": 200.0,
                             "target_deg": {"head_tilt": 60.0}},
               "body_support": {"stiffness": 200.0,
                                "free": ["arm", "finger", "trunk"]},
               "limb_tone": {"hold_deg": 10.0, "groups": ["arm", "leg"]}},
    ),
    _scene(
        "リーチング_リクライニング60度_自己接触版_肘屈曲強化",
        "4ヶ月・リクライニング60度・実験者が頭を60度で支える（顎を引く）・眼球-15度。"
        "「リーチング_リクライニング60度_自己接触版」を土台に、limb_tone.target_degの"
        "肘だけ-130度へ変更（2026-08-03、頭ダブルタッチ報酬の代替案＝候補2・姿勢バイアス版の"
        "おもちゃ無し・胴体自由版）。それ以外（おもちゃ無効化・体幹free・limb_toneの"
        "hold_deg/groups）は自己接触版と完全に同じ。"
        "狙い：新生児の生理的屈曲姿勢（膝窩角90度未満がDubowitz et al. 2005、"
        "肘の屈曲優勢がZoia et al. 2007）を参考に腕を体に近い角度で保つと自己接触が"
        "増えるか確かめる実験。注意：屈曲姿勢が自己接触を増やすという因果自体は"
        "一次資料で未検証（`E/docs/リーチング/文献調査/屈曲姿勢と自己接触頻度_2026-08-03.md`）。"
        "肘-130度は[Tier3・工学的判断]、どれだけ曲げれば十分かの基準は文献に無い"
        "（`doc/人間模倣からの逸脱リスト.md`参照）。おもちゃが無いため、旧版"
        "（`リーチング_リクライニング60度_肘屈曲強化`）で起きたおもちゃとの接触による"
        "弾き飛ばしは起きない見込み（要検証）",
        body={"age_months": 4.0, "eye_rest_vertical_deg": -15.0},
        world={"recline_deg": 60.0, "seat_friction": 2.0, "fence": False,
               "toy": {"enabled": False}},
        setup={"head_hold": {"stiffness": 200.0,
                             "target_deg": {"head_tilt": 60.0}},
               "body_support": {"stiffness": 200.0,
                                "free": ["arm", "finger", "trunk"]},
               "limb_tone": {"hold_deg": 10.0, "groups": ["arm", "leg"],
                            "target_deg": {"right_elbow": -130.0,
                                          "left_elbow": -130.0}}},
    ),
    _scene(
        "リーチング_リクライニング60度_自己接触版_体幹固定",
        "4ヶ月・リクライニング60度・実験者が頭を60度で支える（顎を引く）・眼球-15度。"
        "「リーチング_リクライニング60度_自己接触版」（おもちゃ無し・体幹自由）と対になる、"
        "比較用の2本目のセット（2026-08-03、仕様:作業記録（非公開）"
        "指示\\2026-08-03_体幹固定版の自己接触シーン.md）。"
        "自己接触版から body_support だけを、元の「リーチング_リクライニング60度」"
        "（体幹固定・おもちゃあり）と同じ設定に戻した（free=[arm,finger]、trunkを外す。"
        "pin_jointsも元シーンに合わせ明示でTrue＝既定値と同じだが、"
        "「元と完全に同じ設定に戻す」という指示の意図を明確にするため明示した）。"
        "おもちゃはこちらも無効化したまま（自己接触の検証に不要という土台版の判断を継続）。"
        "ユーザーは体幹固定を『正しい』とは考えていない（工学的な逸脱）が、"
        "体幹自由版と両方のデータを比較するために作る。"
        "limb_toneは自己接触版と同じ（hold_deg=10, groups=[arm,leg], target_degなし）。"
        "体幹が椅子とベルトで固定されたままなので、体幹自由版で必要だった presettle 35秒"
        "（PRESETTLE_SEC辞書）は不要と判断し、既定の2.0秒のまま（元シーンと同じ支え方の"
        "はずなので、元シーンと同様に短時間で落ち着く見込み。3節の settle 確認で検証する）",
        body={"age_months": 4.0, "eye_rest_vertical_deg": -15.0},
        world={"recline_deg": 60.0, "seat_friction": 2.0, "fence": False,
               "toy": {"enabled": False}},
        setup={"head_hold": {"stiffness": 200.0,
                             "target_deg": {"head_tilt": 60.0}},
               "body_support": {"stiffness": 200.0,
                                "free": ["arm", "finger"],
                                "pin_joints": True},
               "limb_tone": {"hold_deg": 10.0, "groups": ["arm", "leg"]}},
    ),
    _scene(
        "リーチング_リクライニング60度_自己接触版_体幹固定_肘屈曲強化",
        "4ヶ月・リクライニング60度・実験者が頭を60度で支える（顎を引く）・眼球-15度。"
        "「リーチング_リクライニング60度_自己接触版_体幹固定」を土台に、limb_tone.target_degの"
        "肘だけ-130度へ変更（2026-08-03、体幹自由版の屈曲版と同じ肘の値。"
        "仕様:作業記録（非公開）"
        "2026-08-03_体幹固定版の自己接触シーン.md）。"
        "それ以外（おもちゃ無効化・体幹固定・limb_toneのhold_deg/groups）は"
        "体幹固定版と完全に同じ。"
        "肘-130度は[Tier3・工学的判断]、根拠は⑧・⑩と同じものを流用（新しい根拠は無い）",
        body={"age_months": 4.0, "eye_rest_vertical_deg": -15.0},
        world={"recline_deg": 60.0, "seat_friction": 2.0, "fence": False,
               "toy": {"enabled": False}},
        setup={"head_hold": {"stiffness": 200.0,
                             "target_deg": {"head_tilt": 60.0}},
               "body_support": {"stiffness": 200.0,
                                "free": ["arm", "finger"],
                                "pin_joints": True},
               "limb_tone": {"hold_deg": 10.0, "groups": ["arm", "leg"],
                            "target_deg": {"right_elbow": -130.0,
                                          "left_elbow": -130.0}}},
    ),
    _scene(
        "リーチング_リクライニング0度_自己接触版",
        "4ヶ月・リクライニング0度（仰向け・フラット）・自己接触（reach_self、頭・胸・"
        "反対の手への接触）の検証専用シーン（2026-08-04）。"
        "「リーチング_リクライニング60度_自己接触版」のリクライニング角度だけを0度に変えた"
        "比較用シーン。それ以外（おもちゃ無効化・body_support.free・limb_tone・"
        "age_months・eye_rest_vertical_deg）は一切変えていない。"
        "狙い：土台の60度自己接触版で頭への自己接触が全ログ0回・胸も0.07〜0.45%と稀"
        "だったことから、リクライニング角度自体が可動域を制限しているのではという疑問を"
        "確かめる（ユーザー提起）。"
        "注意[Tier3・工学的判断・実装時に変更]：設計時点ではhead_hold.target_deg.head_tilt=60度"
        "（土台と同じ値）をそのまま持ち込む案だったが、実装時の姿勢確認（2026-08-04）で"
        "崩れが見つかったため、実装の判断でtarget_deg.head_tilt=0.0へ切り替えた。"
        "理由：head_tiltは体幹に対する相対角（関節ローカル角度）であり、"
        "リクライニング60度では体幹の後傾60度を打ち消して正面視を作る補正値だったが、"
        "リクライニング0度（体幹が後傾していない）にそのまま60度を持ち込むと、"
        "頭が世界座標系で約59度前屈し、顔が座面（床）に接触した"
        "（実測：MuJoCoのcontactリストに`floor`-`head`のペアが出現。"
        "真横からの写真でも顔が座面に埋まっているのが確認できた）。"
        "生理的な可動域を明らかに超えて折れ曲がる姿勢と判断し、target_deg.head_tilt=0.0"
        "（体幹相対角ゼロ＝頭が体幹の延長線上を向く）へ切り替えたところ、顔の床接触は"
        "解消した（詳細はE/docs/研究日誌.md 2026-08-04参照）。"
        "この切り替えにより『リクライニング角度だけを変数にする』という当初の単一変数比較"
        "の前提からは外れる（頭の目標角度も変わっている）。比較の解釈時にはこの点を"
        "踏まえること。`doc/人間模倣からの逸脱リスト.md`⑬参照。"
        "体幹を自由にする判断自体は⑩と同じで新しい判断ではないが、リクライニング0度での"
        "姿勢確認は初めてのため改めて実施する",
        body={"age_months": 4.0, "eye_rest_vertical_deg": -15.0},
        world={"recline_deg": 0.0, "seat_friction": 2.0, "fence": False,
               "toy": {"enabled": False}},
        setup={"head_hold": {"stiffness": 200.0,
                             "target_deg": {"head_tilt": 0.0}},
               "body_support": {"stiffness": 200.0,
                                "free": ["arm", "finger", "trunk"]},
               "limb_tone": {"hold_deg": 10.0, "groups": ["arm", "leg"]}},
    ),
    _scene(
        "新生児_仰向け_柵あり",
        "0ヶ月・仰向け・柵あり。運動性喃語（自発運動）の学習で使ってきた環境",
        body={"age_months": 0.0, "flexion": True},
        world={"recline_deg": 0.0, "fence": True,
               "toy": {"shape": "box", "radius": 0.020, "mode": "tether",
                       "dist": 0.086, "delay_sec": 1.0}},
        setup={"neck_tone": {"target_deg": None, "stiffness": None}},
    ),
    _scene(
        "新生児_仰向け_柵なし",
        "0ヶ月・仰向け・柵なし・おもちゃなし。2026-07-31 新設。"
        "柵を外した理由：柵の寸法（±31cm×±16cm）は新生児に合わせたもので、"
        "体を育てる実験で見直されていなかった。実測すると4ヶ月の体は柵の左右幅の97%を占め"
        "（余裕8mm）、頭と目が柱に当たったまま学習していた。しかも0ヶ月でも"
        "腕を広げた幅34.0cm > 柵の内寸29.6cm。"
        "柵の目的はコードのコメントによれば『長時間の学習で遠くへ行かない保険』であって"
        "学習を助けるものではない。半径スイープの実測でも柵なしが最もズレが小さかった"
        "（なし0.088m / 0.23m柵0.163m）。"
        "おもちゃを外した理由：tetherで吊るすので揺れて動く＝自分の動き以外の変化が入る。"
        "自己モデル（自分の行動の結果を予測する）の学習には不要。"
        "注意柵なしで長時間動かすと遠くへ行く可能性がある。Viewerで目視して確かめること。",
        body={"age_months": 0.0, "flexion": True},
        world={"recline_deg": 0.0, "fence": False,
               "toy": {"enabled": False}},
        setup={"neck_tone": {"target_deg": None, "stiffness": None}},
    ),
]


# 【2026-08-03 追加】保存前に落ち着かせる秒数（既定2.0秒）を、シーンごとに変えられるようにした。
#
# 【なぜ要るか】自己接触専用シーン2本は body_support.free に "trunk"（体幹）を初めて
# 加えた。体幹の関節は limb_tone の対象外（LIMB_TONE_GROUPSに無い）なので、支え
# （CaregiverHands）から外すと**モデル本来の受動バネ（native jnt_stiffness、リクライニング
# の目標角とは無関係な"まっすぐ"に近い姿勢が中立点）へ引かれて新しい釣り合い位置へ動く**。
# 実測（`shot_scene.py`によるスイープ、E/scenes配下の実行ログ参照）：
#   体幹の傾き 71.9度(2秒後)→81.4度(2.5秒後)→以後30秒間 81.3〜81.6度で完全に静止
# ＝**「崩れ続ける」のではなく「別の釣り合い位置へ0.5秒程度で移り、その後は極めて安定」**。
# 既定の2.0秒だと、ちょうどこの移行の途中でスナップショットを撮ってしまい、
# settle（3秒）の判定窓の中に移行そのものが入って ok=false（目の移動1.25cm）になった。
# → 移行が終わったあとの秒数まで presettle を延ばせば、settle窓は「移行後の静止」だけを
#   見るようになり、正しく安定を判定できる。既存シーンは辞書に無ければ2.0秒のまま＝不変。
# 【2026-08-03 実測で20秒に修正】6秒ではまだ収束の途中で、settleの3秒窓の中に
#   わずかな残り移動が入ってしまい ok=False（目の移動1.2〜1.3cm、上限1cm）になった。
#   presettleを10秒→20秒と伸ばして実測したところ：
#     10秒: 目の移動1.22cm ok=False　20秒: 目の移動0.99cm ok=True
#   角度そのもの（首・体幹）は6秒の時点で既に0.19度しかずれておらず安定していた
#   （上限3.0度に対し十分小さい）ので、「崩れ続けている」のではなく
#   「座面との接触が落ち着くまでの、ミリ単位のごく遅い尾の収束」と判断した。
PRESETTLE_SEC = {
    "リーチング_リクライニング60度_自己接触版": 35.0,
    "リーチング_リクライニング60度_自己接触版_肘屈曲強化": 35.0,
    "リーチング_リクライニング0度_自己接触版": 35.0,
}


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    made = []
    for sc in SCENES:
        if only and sc["name"] != only:
            continue
        print("=" * 74)
        print(f" {sc['name']}")
        print("=" * 74)
        print(f"  {sc['note']}")
        env, hands = e_scene.build(sc, orient=False, vor=True, seed=0, verbose=False)
        # 物理を少し進めて落ち着かせてから保存する
        #   ＝作った直後は力が釣り合っておらず、そのまま保存すると
        #     測定を始めた瞬間に動いてしまう
        u = env.unwrapped
        dt = float(u.model.opt.timestep) * int(u.frame_skip)
        a = np.zeros(env.action_space.shape[0], dtype=np.float32)
        presettle = PRESETTLE_SEC.get(sc["name"], 2.0)
        for _ in range(int(presettle / dt)):
            env.step(a)
        # 【2026-07-29】落ち着いた**後**に、おもちゃを視線の正面へ置き直す。
        #
        # 【なぜ要るか】これを入れないと、おもちゃは「リセット直後の視線の正面」に
        # 置かれ、そのあと2秒ぶん頭が動いた分だけ視野の端へ寄る。
        # 実測：視線誘導反射の測定で初期ずれが **+0.44 ぶん右へ偏り**、
        #   視野の右端（+4.0cm）が見えなくなった（移行前は見えていた）。
        # 移行前の環境は「1秒待ってから親が運んでくる」ので、
        # **運び終わった時点＝落ち着いた姿勢での正面**に置かれていた。それに揃える。
        # 注意：体を留める設定があるシーンでは、ここで留め直してはいけない。
        #   `build` が**落ち着かせる前の姿勢**で既に留めている。それが正しい。
        #   落ち着かせてから留め直すと「転がり落ちた位置」で固定されてしまう
        #   （2026-07-29 の実測：リクライニング60度で頭が中心から25cm 横へずれ、
        #     視線が真後ろを向いた状態のまま保存された）。
        if u._toy and hasattr(u, "_set_anchor"):
            u._set_anchor()
            e_scene.place_toy(env, u._rest_pos)
            for _ in range(int(0.3 / dt)):
                env.step(a)
        saved, drift = e_scene.save(sc, env=env, hands=hands,
                                    settle_seconds=3.0, verbose=True)
        fp = saved["fingerprint"]
        print(f"  首 {fp['head_angles_deg']}")
        print(f"  体幹の傾き {fp['trunk_tilt_deg']}度")
        if fp.get("toy_pos"):
            print(f"  おもちゃ {fp['toy_pos']}  "
                  f"目から{fp['toy_dist_cm']}cm  視線から{fp['toy_angle_deg']}度  "
                  f"{'見える' if fp.get('toy_visible_left') else '見えない'}  "
                  f"腕の{(fp.get('toy_reach_ratio') or 0)*100:.0f}%")
        env.close()
        made.append(sc["name"])
        print()
    print("=" * 74)
    print(f" {len(made)}件のシーンを作った → run/scenes/")
    print("=" * 74)
    for n in made:
        print(f"  {n}")


if __name__ == "__main__":
    main()
