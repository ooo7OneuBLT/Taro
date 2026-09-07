"""太郎一式（脳・学習器・神経調節・小脳）を組み立てる。

【なぜ切り出したか、2026-07-30】これは `E/scripts/e_growth_train.py` の
562〜700行を**そのまま写した**もの。元は930行の関数の中に埋まっていて、
53個のモジュール変数（`_LR` `_MUSCLE` …）を直接読んでいたため
実験ファイルから設定を渡せなかった。→ `run/config.py` の Config を受け取る形にする。

注意：写すときに守ったこと：**乱数を消費する順序を変えない**。
  順序が変わると同じシードでも違う初期値になり、過去の実験と比較できなくなる
  （落とし穴チェックリスト 項3「reset を1回足すだけで乱数列がずれる」）。
  順序は：env作成 → fusion → target_fusion → env.reset(seed) → 予測対象の次元を測る
        → brain → 脊髄CPG → learner → 神経調節 → 小脳 → モデル読み込み

【この中に「測り方」は入れない】測るのはプラグインの仕事（`run/plugins/`）。
【この中に「環境の作り方」も入れない】環境は `run/plugins/common/scene.py` だけが作る。
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
_CORE = os.path.join(_ROOT, "taro_core")
for _sub in ("wrapper", "senses", "brain", "body"):
    _p = os.path.join(_CORE, "src", _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
for _p in (os.path.join(_CORE, "tests"), os.path.join(_ROOT, "E", "scripts"),
           os.path.join(_ROOT, "D", "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch                                                    # noqa: E402
import numpy as np                                              # noqa: E402

from fusion import MinimalFusion                                # noqa: E402
from somatosensory_cortex import build_touch_map_from_env       # noqa: E402
from touch_adaptation import TouchAdaptation                    # noqa: E402
from taro_brain_motor import TaroBrainWithMotor                 # noqa: E402
from basal_ganglia import TaroLearner                           # noqa: E402
from dopamine import Dopamine                                   # noqa: E402
from locus_coeruleus import LocusCoeruleus                      # noqa: E402
from developmental_clock import DevelopmentalClock              # noqa: E402
from cerebellum_motor import MotorCerebellum                    # noqa: E402
# 発話小脳（口の動き→音の対応の帳面）。手足用のMotorCerebellumとは別クラス
#   （taro_core/src/brain/cerebellum.py冒頭コメント参照）。既定OFF（cfg.produce=None）
#   では_setup_produce内で一度もCerebellum()を作らない＝乱数消費なし。
from cerebellum import Cerebellum as SpeechCerebellum           # noqa: E402
from learning_progress import LearningProgress                  # noqa: E402
from homeostatic_scaling import HomeostaticScaling              # noqa: E402
from test_phase8_motor_learning import CombinedParams, rescale_action, to_tensor  # noqa: E402
# 手先位置の目標表現（案C）。既定（goal_space != "reach_self"）では一度も使われない
#   （Taro.__init__ 内で条件付きに構築する。設計の統合判断「決定1」）。
from proprioceptive_map import build_arm_proprio_map_from_env, _opposite_side  # noqa: E402
from goal_babbling.reach_goal_head import ReachGoalHead          # noqa: E402
# 頭へのダブルタッチを報酬に直結する（2026-08-03）。既定（reach_space無効）では
#   一度も使われない（Taro.__init__ 内で条件付きに構築する。決定1と同じパターン）。
# 【2026-08-12追記】口元への自己接触報酬。mouth_point_mask は
#   cfg.mouth_touch_bonus != 0.0 のときだけ呼ばれる（仕様3節）。
from double_touch import DoubleTouchDetector, mouth_point_mask   # noqa: E402

mse = torch.nn.functional.mse_loss

# 【なぜ複製するか、2026-08-13】run/trainer.py の DT（=0.01秒=MuJoCoの1物理ステップ）
#   と同じ値。taro_setup.py は trainer.py からimportされる側で、trainer.py を
#   importすると circular import になるため、値を複製する
#   （run/tools/record_self_touch_clip.py 等、既存の複数ファイルが同じ理由で
#   同じ値を複製している前例に倣う。仕様
#   作業記録（非公開） 3節）。
_PHYS_DT = 0.01

# シナジーの関節index（MIMo身体の配列。d_c5_motor_quality.py と同一）
# 注意：【逸脱・Tier3・2026-07-30】「まとめてしか動かせない」を**配線で作っている**。
#   人間（および國吉研の胎児モデル）は、振動子を互いに繋がず、
#   身体・床・羊水を介した物理的な力の伝達で位相が揃う（引き込み現象）。
#   ＝「まとめて動く」は**結果**であって入力ではない。
#   国吉 & Sangawa 2006（Biological Cybernetics 95(6)）は
#   「複数筋を協調させる回路を一切事前に組み込まずに」全身協調運動を創発させている。
#   ⇒ 人間模倣からの逸脱リスト「2026-07-30 シナジーを配線で作っている」を参照。
#     注意：太郎の環境には羊水も子宮壁もないので、外すと単に消える可能性がある（要検証）。
#
# 【2026-08-11・符号バグの修正】各要素は (index, sign) のタプル。sign は
# 「生のqpos正方向が屈曲なら+1、伸展なら-1」（`spinal_cord/cpg.py` の `_blend` が使う）。
# MuJoCoの関節は正方向の定義が関節ごとにバラバラで統一されていないため、
# グループへ同符号の共通信号をそのまま配ると、屈曲側で符号が揃わず
# 「肩が曲がると肘は伸びる」のように関節同士が打ち消し合っていた
# （引用元の実測：肩-肘の屈曲量相関 修正前 −0.253±0.098、膝-足首 修正前 0.462±0.385。
#   この2つの値は監査報告・依頼文が引用したもの。今回の修正実装（本ファイルの
#   sign値）を実際にsyn_w=1.0・3シード・6000stepで検算した結果は
#   肩水平-肘 −0.254±0.098 → +0.204±0.164、膝-足首 0.462±0.385 → +0.744±0.200
#   （syn_w=0.6の標準テンプレートではより穏やかな改善：肩水平-肘 +0.013→+0.177、
#   膝-足首 +0.818→+0.897）。詳細・数値の出所は実装の作業記録を参照）。
# 根拠（静的mj_kinematicsスイープ7点＋動力学つき複数シード検算。詳細は
# 作業記録（非公開））：
#   shoulder_horizontal(14/43)  sign=+1（+方向=屈曲。動力学つき・複数シードで確定）
#   elbow(17/46)                sign=-1（+方向=伸展。静的スイープで単調・確定）
#   hip1(72/81)                 sign=-1（+方向=伸展。静的スイープで単調・確定）
#   knee(75/84)                 sign=-1（+方向=伸展。静的スイープで単調・確定）
#   foot1/ankle(76/85)          sign=+1（+方向=屈曲。静的スイープで単調・確定）
# 次の関節は符号を推測せず、シナジーグループから除外した（推測で割り当てない）：
#   shoulder_ad_ab(15/44)：静的スイープで非単調・確定した符号の記載なし
#   hand2/wrist_flexion(19/48)：neutral付近が最伸展という構造で二値の符号判定が
#     原理的に成り立たない
#   hip2/hip_abduction(73/82)：静的スイープでは単調だが、Dominici 2011の
#     kicking synergy（股矢状面・膝・足首の協調）に外転内転は含まれない
LEG_R = [(72, -1), (75, -1), (76, 1)]
LEG_L = [(81, -1), (84, -1), (85, 1)]
ARM_R = [(14, 1), (17, -1)]
ARM_L = [(43, 1), (46, -1)]


# 【2026-08-11・新しい駆動モジュール：伸張反射＋揺らぐ振動子の共通駆動】
# 設計：作業記録（非公開）
#
# 【なぜCPGのLEG_R等（(index, sign)タプル、一部関節のみ）を再利用しないか】
# moment_1/moment_2経由の変換（taro_core側、common_drive.py・stretch_reflex.py）は
# 「生のqpos正方向が屈曲か伸展か」を人力で判定する必要が構造的に無い（設計4節）。
# したがって cpg.py が符号確定できず除外した関節（shoulder_ad_ab・hand2など）も
# 含めて、四肢すべての関節を対象にできる（ユーザー確定Q2・Q3）。
# 生の配列インデックスをここで決め打ちせず、**関節の基底名**
# （`infant_limbs.LIMB_TONE_ALIASES`・`LIMB_TONE_GROUPS`と同じ命名規則）で
# actuator名から引き直す（ノウハウ2026-08-11「グループへ共通信号を一括で配るとき、
# 各要素の符号が揃っている保証はない」の教訓＝生のインデックスを直接書かない）。
def _reflex_common_joint_indices(env):
    """4肢（両腕・両脚）すべての関節を、行動配列における関節index(0..n_joint-1、
    neg側のインデックス。moment_1/moment_2と同じ並び）で返す。

    Returns:
        {"right_arm": [index, ...], "left_arm": [...],
         "right_leg": [...], "left_leg": [...]}

    【2026-08-11・バグ修正】以前はMIMoの**アクチュエータ名**
    （`u.model.actuator(int(aid)).name`、"shoulder_abduction"のような名前）を
    `LIMB_TONE_GROUPS`が持つ**関節名**（"shoulder_ad_ab"のような名前）と
    直接照合していた。MIMoではこの2つが別の名前空間で、一部（肩水平・肘・膝）を
    除き綴りが一致せず、大半の関節が漏れていた（実装担当仕様
    作業記録（非公開）担当A節に
    詳細）。修正：アクチュエータをループしつつ、`am.mimo_actuated_joints`
    （`MIMo/mimoActuation/muscle.py` 189行、`self.actuators`・`moment_1`・
    `moment_2`と同じ並び・同じ長さn_actuatorsの「各アクチュエータ番目に対応する
    関節id」の配列）から関節idを引き、その関節idの**関節名**で照合するように変えた。
    enumerateする対象・返すindexの意味（moment_1/moment_2配列でのposition）は
    変えていない。
    """
    from infant_limbs import LIMB_TONE_ALIASES, LIMB_TONE_GROUPS
    u = env.unwrapped
    am = u.actuation_model
    base_names = {}
    for limb, groups in LIMB_TONE_ALIASES.items():        # "arm"->(shoulder,elbow,wrist) 等
        s = set()
        for g in groups:
            s.update(LIMB_TONE_GROUPS.get(g, ()))
        base_names[limb] = s
    out = {"right_arm": [], "left_arm": [], "right_leg": [], "left_leg": []}
    joint_ids = am.mimo_actuated_joints   # 長さ n_actuators。moment_1/2と同じ並び
    for i, jid in enumerate(joint_ids):
        nm = (u.model.joint(int(jid)).name or "").split(":")[-1]
        for side in ("right_", "left_"):
            if not nm.startswith(side):
                continue
            base = nm[len(side):]
            side_key = side.rstrip("_")
            for limb, names in base_names.items():
                if base in names:
                    out[f"{side_key}_{limb}"].append(i)
            break
    return out


def _reflex_common_groups(grouping, idx_by_limb):
    """common_drive_grouping設定から {group名: [関節index, ...]} を作る（設計7-4節）。

    "none"     ：各関節を1要素だけの個別グループにする（RhythmicCommonDriveGroupの
                性質上、1要素グループでは「共通」成分も他の関節と共有されないため、
                関節間の相関は生まれない＝設計の「相関なし」と数値的に等価。
                検証2で確認する）。
    "per_limb" ：右腕・左腕・右脚・左脚をそれぞれ1グループ
    "whole_body"：四肢全体を1グループ
    辞書        ：{group名: [limb名 または 関節index, ...]}（limb名は
                "right_arm"等をそのまま展開、数値はindexとして扱う）
    """
    if grouping in (None, "none"):
        out = {}
        for limb, idxs in idx_by_limb.items():
            for i in idxs:
                out[f"{limb}_{i}"] = [i]
        return out
    if grouping == "per_limb":
        return {limb: list(idxs) for limb, idxs in idx_by_limb.items() if idxs}
    if grouping == "whole_body":
        allidx = [i for idxs in idx_by_limb.values() for i in idxs]
        return {"whole_body": allidx} if allidx else {}
    if isinstance(grouping, dict):
        out = {}
        for gname, members in grouping.items():
            idxs = []
            for m in members:
                if isinstance(m, str) and m in idx_by_limb:
                    idxs.extend(idx_by_limb[m])
                else:
                    idxs.append(int(m))
            out[gname] = idxs
        return out
    raise ValueError(f"common_drive_grouping が不明: {grouping!r}")


def _setup_reflex_common(taro, cfg, env, *, verbose=True):
    """Taro.__init__・on_body_change の両方から呼ぶ、reflex_commonの配線本体。

    【なぜ両方から呼ぶ必要があるか】moment_1/moment_2・関節indexは env（MuJoCoモデル）
    ごとに固定される値。体を作り直す実験（cfg.grows）では env が作り直されるたびに
    これらが変わりうるため、初回構築（__init__）だけでなく、体が変わるたび
    （on_body_change、trainer.py._regrowから呼ばれる）にも組み立て直す必要がある。
    注意：組み立て直すと振動子（WanderingOscillator）の内部状態は初期値へリセットされる
    （継続はしない）。今回のスコープの主眼（検証1〜4・既定挙動不変）には影響しない
    軽微な簡略化のため、作業記録に明記する。
    """
    if str(cfg.spinal_drive_mode) != "reflex_common":
        return False
    am = env.unwrapped.actuation_model
    idx_by_limb = _reflex_common_joint_indices(env)
    groups = _reflex_common_groups(cfg.common_drive_grouping, idx_by_limb)
    osc_params = dict(cfg.common_drive_osc_params or {})
    # "amp"（基準長オフセットへの変換スケールA、案C4-2節）はWanderingOscillator自身の
    #   引数ではないので、ここで取り出してから残りをoscillator_kwargsとして渡す。
    amp = float(osc_params.pop("amp", 1.0))
    taro.brain.enable_reflex_common(
        n_joint=int(am.n_actuators), moment_1=am.moment_1, moment_2=am.moment_2,
        groups=groups, rho=float(cfg.common_drive_rho), osc_params=osc_params,
        amp=amp, seed=taro.seed)
    taro.brain.set_reflex_common_baseline(am.muscle_lengths)
    # apply_limb_tone（run/scene_tools/e_scene.py、触ってよいファイルの一覧に無い）が
    #   四肢に効かせた「継続的なバネ」役割を事後的に無効化する（infant_limbs.py参照）。
    from infant_limbs import disable_limb_tone_spring
    n_disabled = disable_limb_tone_spring(env.unwrapped.model, groups=("arm", "leg"))
    if verbose:
        n_joints_total = sum(len(v) for v in groups.values())
        print(f"[反射+共通駆動] reflex_common ON: grouping={cfg.common_drive_grouping} "
              f"rho={cfg.common_drive_rho} amp={amp} 対象関節数={n_joints_total} "
              f"（apply_limb_toneの継続的バネ役割を{n_disabled}関節ぶん無効化。"
              f"初期姿勢の設定＝役割Aは維持）", flush=True)
    return True


def _setup_postural_gate(taro, cfg, env, *, verbose=True):
    """姿勢制御反射（層1）の配線。Taro.__init__・on_body_change の両方から呼ぶ。

    【なぜ両方から呼ぶ必要があるか】_setup_reflex_common と同じ理由：
    アクチュエータ・関節のindexはenv（MuJoCoモデル）ごとに固定される値。
    体を作り直す実験（cfg.grows）ではenvが作り直されるたびにこれらが変わりうるため、
    初回構築（__init__）だけでなく、体が変わるたび（on_body_change）にも
    組み立て直す必要がある。

    仕様：作業記録（非公開）
    設計：作業記録（非公開）
    """
    if not cfg.posture_reflex:
        # 既定OFF。属性は必ず持たせておく（e_toy_env.py側がNoneかどうかで
        #   ON/OFFを判定できるように、仕様の必須要件）。
        taro.postural_gate = None
        return
    if not cfg.is_muscle:
        raise ValueError(
            "posture_reflex=True には actuation=muscle が要る。\n"
            "  postural_gate.py はMuscleModel前提（行動配列が2*n_actuator次元、"
            "neg側/pos側の拮抗筋構造）でゲートを組み立てる。")
    from spinal_cord.postural_gate import PosturalGate
    taro.postural_gate = PosturalGate(env.unwrapped.model)
    # 【なぜ、2026-08-15・想定外】E/scripts/e_toy_env.py の step() は
    #   VOR・orienting_reflexと同じ並びでreflexを適用する設計（仕様C節）だが、
    #   VOR/orienting_reflexはenvが自前で構築するのに対し、postural_gate/
    #   righting_damperはTaro側（taro_setup.py）で構築される。envがstep()内で
    #   これへ到達するには env.taro の参照が要る。trainer.pyの変更はreward switch
    #   文への追加のみに制限されているため（触ってよいファイルの一覧）、
    #   ここ（_setup_postural_gate/_setup_righting_damper、どちらも新設が
    #   許可されている関数）でenv.unwrapped.taroを設定する。この配線が
    #   仕様の想定と一致しているかは作業記録「想定外」に明記する。
    env.unwrapped.taro = taro
    if verbose:
        print(f"[姿勢制御反射] postural_gate ON: 対象アクチュエータ数="
              f"{len(taro.postural_gate.actuator_idx)}"
              f" tilt_deadzone_deg={taro.postural_gate.tilt_deadzone_deg}", flush=True)


def _setup_righting_damper(taro, cfg, env, *, verbose=True):
    """立ち直り反射（層2）の配線。Taro.__init__・on_body_change の両方から呼ぶ。

    頭・首を動かす筋（act:head_tilt、前後・Y軸）のneg/pos indexを、
    体が変わるたびに引き直す（_setup_postural_gateと同じ理由）。
    """
    if not cfg.righting_reflex:
        taro.righting_damper = None
        taro.righting_neg_indices = None
        taro.righting_pos_indices = None
        return
    if not cfg.is_muscle:
        raise ValueError(
            "righting_reflex=True には actuation=muscle が要る。\n"
            "  righting_damper.py はMuscleModel前提（neg_indices/pos_indicesの"
            "拮抗筋構造）でバイアスを配る。")
    from spinal_cord.righting_damper import RightingDamper
    taro.righting_damper = RightingDamper(
        k_d0=float(cfg.righting_reflex_gain),
        deadzone_dps=float(cfg.righting_reflex_deadzone_dps))
    model = env.unwrapped.model
    n = int(model.nu)
    # 頭の前後(pitch)を動かす筋は act:head_tilt の1本のみ（infant_neck.py 参照。
    #   act:head_tilt_side は左右(roll)用でここでは対象外）。
    aid = int(model.actuator("act:head_tilt").id)
    taro.righting_neg_indices = [aid]
    taro.righting_pos_indices = [aid + n]
    # env.taroの配線理由は_setup_postural_gate冒頭のコメント参照。
    env.unwrapped.taro = taro
    if verbose:
        print(f"[立ち直り反射] righting_damper ON: k_d0={cfg.righting_reflex_gain} "
              f"deadzone_dps={cfg.righting_reflex_deadzone_dps} "
              f"対象アクチュエータ=act:head_tilt(neg={aid}, pos={aid + n})", flush=True)


def _setup_hearing(taro, cfg, *, verbose=True):
    """耳＋連合器の配線（2026-08-18新設・F1-3）。Taro.__init__ から呼ぶ。

    F1-2「耳の移植」（F/docs/仕様_F1-2_耳の移植.md）で taro_core 側に置いた部品
    （taro_core/src/senses/hearing.py の Hearing、taro_core/src/brain/lexicon.py の
    Lexicon）をそのまま装着するだけ。env（体）に依存しないので、_setup_postural_gate
    等と違い on_body_change での再構築は不要。

    既定OFF（cfg.hearing=False）：taro.hearing/taro.lexicon/taro.vision_backendは
    Noneのまま＝run/trainer.py の step_k 内の配線コードが1行も実行されない
    （既存実験の挙動は1ビットも変わらない）。

    【2026-08-19追記・F1-3b】Lexiconへ渡す視覚表現の作り方を差し替え可能にした
    （taro_core/src/senses/vision_backends.py の登録式レジストリ）。cfg.lexicon_vision
    が既定None（未指定）なら、taro.fusion.vision（従来の未訓練CNN、64次元）を
    そのまま包む"custom"バックエンドが選ばれ、Lexiconのstate_dimも従来どおり64になる
    （既定挙動は1ビットも変わらない）。設計：
    F/docs/仕様_F1-3b_視覚バックエンドの差し替え機構.md
    """
    if not cfg.hearing:
        taro.hearing = None
        taro.lexicon = None
        taro.vision_backend = None
        return
    from hearing import Hearing
    from lexicon import Lexicon
    from vision_backends import get_backend
    taro.hearing = Hearing()
    # 手打ち数字の根絶：state_dimはバックエンドのdimプロパティから取る
    #   （既定null時はcustomバックエンド経由でfusion.visionの出力次元＝従来どおり64）。
    taro.vision_backend = get_backend(cfg.lexicon_vision, vision_encoder=taro.fusion.vision)
    # 【F1-5・2026-08-21】lexicon_mode="sum"（既定）なら旧挙動と完全一致。
    #   "contrast"のときだけ対照学習（引き寄せ＋引き離し）が有効になる。
    #   設計：F/docs/設計_F1-5_連合器の対照学習化.md
    taro.lexicon = Lexicon(state_dim=taro.vision_backend.dim, mode=cfg.lexicon_mode,
                          eta_pull=cfg.lexicon_eta_pull, eta_push=cfg.lexicon_eta_push)
    if verbose:
        print(f"[耳] taro.hearing/taro.lexicon ON（視覚バックエンド="
              f"{taro.vision_backend.name} state_dim={taro.vision_backend.dim}）",
              flush=True)


def _setup_produce(taro, cfg, env, *, verbose=True):
    """見た物の名前を言う（初語）の配線（2026-08-22新設・F2）。

    設計：F/docs/設計_F2_初語（見た物の名前を言う）.md 第2部「実装作業」より：
      ① 逆引き：taro.lexicon.reverse_lookup()（既存の連合器に追加したメソッド、
         taro_core/src/brain/lexicon.py）
      ② 産出：taro.brain.generate()（既存の調音ヘッド＋VocalTract、新規機構なし）
      ③ 報酬：taro_core/src/brain/imitation_reward.py（B原本 instincts/imitation.py
         からの移植）
      ④ 学習：taro.produce_learner（**運動学習(taro.learner)とは別インスタンス**。
         設計「⚠新たに判明したリスク：学習器の共有」案a）

    既定OFF（cfg.produce=None）：taro.produce_vocab/produce_vocal_tract/
    produce_learner/produce_dopはNoneのまま＝run/trainer.pyの
    _apply_word_productionが最初のifで即returnし、既存実験の挙動は
    1ビットも変わらない。

    【なぜ Taro.__init__ の _load() の"後"に呼ぶか】taro.brainのembedding/
    perception_headはvocab_size=3（PAD/BOS/EOS）で構築される（これまで
    generate()を呼ぶ経路が無く、発声に使う文字を持たせる必要が無かったため。
    設計「調査で確定した事実」表：Fのループにgenerate()の呼び出しが
    そもそも無い）。_load()より前にresize_embedding()すると、_load内の
    _match()（形が合う層だけコピー）が「保存時のembedding(3語)」と
    「resize後のembedding(74語)」の形の違いで弾かれ、生成用に伸ばした分
    どころか元の3語分の重み（PAD/BOS/EOS）までロードされない。先に元の形で
    読み込み、あとから伸ばす方が安全（PAD/BOS/EOSの3行は保存値のまま、
    伸ばした分だけ新規初期化になる。resize_embedding()自体が
    「既存の重みを先頭にコピーし、増えた分だけ新規初期化」する実装なので
    これで両立する）。
    """
    if cfg.produce is None:
        taro.produce_vocab = None
        taro.produce_vocal_tract = None
        taro.produce_learner = None
        taro.produce_dop = None
        taro._last_produce_sec = None
        # 【F2-1・2026-08-23】発話小脳・喃語用の持ち越しhidden。既定OFFでは
        #   どちらもNoneのまま＝run/trainer.pyの_apply_word_productionが
        #   最初のifで即returnし、既存実験の挙動は1ビットも変わらない。
        taro.produce_cerebellum = None
        taro._produce_hidden = None
        # 【作業A・2026-08-24・修正】以前はここで無条件に
        #   `taro._pending_produce_cerebellum = None` としていたため、_load()が
        #   保存モデルから退避しておいた帳面（forward_map/inverse_map/
        #   experience_count）が、produceキー無し（OFF）の実験を1回挟んだだけで
        #   静かに捨てられていた（実測：帳面339行・68種を持つモデルをproduce無し
        #   実験で保存すると、保存後のblobから"produce_cerebellum"キーが消滅）。
        #   cfg.produce=Noneではproduce_cerebellumインスタンス自体は作らない
        #   （既存の既定OFF挙動は変えない）が、_load()が退避した帳面データその
        #   ものは保持したまま素通しし、save()（下方）でblobへ書き戻す。
        #   cfg.modelを指定していない（=_load()が一度も呼ばれていない）新規
        #   モデルでは self._pending_produce_cerebellum 属性自体がまだ無いので
        #   getattrで安全にNone扱いする。
        taro._pending_produce_cerebellum = getattr(taro, "_pending_produce_cerebellum", None)
        # 【F2-2・実装作業②・2026-08-23】ブローカ野（発話計画）。既定OFF
        #   （cfg.produce=None）ではNoneのまま＝run/trainer.pyの
        #   _apply_word_productionが最初のifで即returnし、既存実験の挙動は
        #   1ビットも変わらない。設計：F/docs/設計_F2-2_見た物の名前を言う.md 第4部②。
        taro.produce_broca = None
        # 【文脈・2026-08-30】既定OFFでは文脈も無し（設計_文脈（コンテキスト）.md）。
        taro._context_enabled = False
        taro._context_hidden = None
        taro._context_last_target = None
        taro._context_speaker_ids = None
        taro._speech_gate = None
        taro._social_cfg = None
        taro._social_pending = None
        taro._visual_projection = None
        taro.language_hippocampus = None
        # 【M4・2026-09-06・仕様_M4_消えた物について「○○ないね」と言う】
        #   既定OFF（cfg.produce=None）ではGONEトークンも登録されず、
        #   run/trainer.pyのvanish_input分岐は getattr(t, "_gone_id", None) が
        #   Noneのまま＝一切通らない。
        taro._gone_id = None
        # 【M4d・2026-09-06・仕様_M4d_あるの印】既定OFF（cfg.produce=None）では
        #   HEREトークンも登録されない＝run/trainer.pyのhere_input分岐は
        #   getattr(t, "_here_id", None) がNoneのまま＝一切通らない。
        taro._here_id = None
        # 【M4e・2026-09-07・仕様_M4e_状態の線と驚きの書き込み】既定False。
        #   cfg.produce=Noneではtrainer.pyの各forward_hidden呼び出しは
        #   self.cfg.produce（None扱い→ {}）から state_channel を読み、常にFalse
        #   ＝state_idは常にNone＝1ビットも変わらない。
        taro._state_channel = False
        taro._gone_strength = 1.0
        # 【塊レベル層・2026-09-07・仕様_M5_塊レベル層.md §2】既定OFF
        #   （cfg.produce=None）ではchunk_vocab/chunk_brain/_chunk_optimizer/
        #   _chunk_context_hiddenはNoneのまま＝run/trainer.pyの新しい分岐は
        #   一度も実行されない＝既存実験の挙動は1ビットも変わらない。
        #   _pending_chunk_brain/_pending_chunk_vocabは_load()が退避した値を
        #   （あれば）そのまま素通しする（visual_projection/language_hippocampus
        #   と同じ流儀。cfg.model未指定の新規モデルではまだ属性自体が無いので
        #   getattrで安全にNone扱いする）。
        taro.chunk_level = False
        taro.chunk_vocab = None
        taro.chunk_brain = None
        taro._chunk_optimizer = None
        taro._chunk_context_hidden = None
        taro._pending_chunk_brain = getattr(taro, "_pending_chunk_brain", None)
        taro._pending_chunk_vocab = getattr(taro, "_pending_chunk_vocab", None)
        return
    from vocal_tract import VocalTract
    from hearing import Vocabulary
    pd = cfg.produce
    vt = VocalTract()
    # 【Tier3・工学的判断・2026-08-22】「わんわん」の「わ」(両唇+半母音)は
    #   coupledモード（調音点→調音法が自動決定）では出せない（両唇は鼻音に
    #   自動的に決まるため「ま」にしかならない、vocal_tract.py:165-172の
    #   COUPLED_PLACE_TO_MANNER）。半母音(manner index6)はstage2から解禁
    #   （vocal_tract.py:130-135のSTAGE_ALLOWED_MANNER）。
    # 【2026-08-23・月齢連動を配線】`develop_from_age`（既定False）で切替。
    #   False（既定・キー無し実験も常にこちら）のときは**従来と1ビットも変わらない**
    #   （固定値 `vocal_tract_stage`、既定2）。True のときだけ
    #   `taro_core/src/body/development.vocal_tract_stage_for_age()` で
    #   env.age（月齢）から自動計算する（境界の根拠の強さは development.py 側の
    #   コメント参照。6ヶ月の境界だけコード内に根拠があり、他は暫定値）。
    #   run/config.py（共通ファイル、変更範囲外）に新しいtaroキーを追加できない
    #   ため、既存の自由辞書 `produce` の中にキーを足す形にした
    #   （`produce.develop_from_age`。`body.develop_from_age`＝視覚側とは別キー。
    #   声道と視覚を必ず連動させる設計ではないため、あえて分けている）。
    if bool(pd.get("develop_from_age", False)):
        from development import vocal_tract_stage_for_age
        # 【2026-08-23・バグ修正】`env` はラッパーで、月齢は `env.unwrapped.age`
        #   にある（同じファイルの `env.unwrapped.model` と同じ事情）。
        #   直前の実装は `getattr(env, "age", 0.0)` だったため常に既定の 0.0 を
        #   拾い、12ヶ月と指定しても **stage0（母音のみ）** で走っていた。
        #   実測：12ヶ月・300ステップで出た音が「あいうえお」の5種だけ
        #   （6ヶ月・stage2では15種で子音も出ていた）。
        #   ログには `age=12.00mo` と出るのに実効が無い型のバグ＝
        #   `E/scripts/e_body_wiring_check.py` 冒頭に記録された2026-07-25の
        #   3件と同型。**静かに既定値へ落ちないよう、取れなければ止める。**
        _env_age = getattr(getattr(env, "unwrapped", env), "age", None)
        if _env_age is None:
            raise ValueError(
                "produce.develop_from_age=true だが env から月齢(age)が取れない。"
                "シーンに age_months が入っているか確認すること"
                "（静かに stage0＝母音のみで走るのを防ぐため、ここで止める）。")
        vt.stage = vocal_tract_stage_for_age(float(_env_age))
    else:
        vt.stage = int(pd.get("vocal_tract_stage", 2))
    if pd.get("vocal_tract_decoupled", True):
        vt.force_decouple()
    taro.produce_vocal_tract = vt
    pv = Vocabulary()
    # 【⓪・2026-08-31・設計_二語文へ】保存済みの脳の名簿があれば**同じ番号で**復元
    #   する（embedding の行と番号の対応がずれると学習済みの重みが壊れるため）。
    #   その上で口の音を足す（復元済みなら全部 no-op）。聞いた音は今後、実行中に
    #   動的に足される（trainer の _ensure_brain_capacity）。
    _pbv = getattr(taro, "_pending_brain_vocab", None)
    if _pbv is not None:
        pv.char2idx = dict(_pbv["char2idx"])
        pv.idx2char = {int(i): c for c, i in pv.char2idx.items()}
        pv.size = max(pv.idx2char) + 1
    for ch in vt.get_all_chars():
        pv.encode(ch)
    # 【文脈・2026-08-30・設計_文脈（コンテキスト）.md 決定3】話者の印。
    #   produce.context=true のときだけ語彙に2トークン足す。既定falseでは
    #   語彙サイズが変わらない＝embeddingも知覚ヘッドも従来と同じ形＝
    #   generate()のsoftmaxの分母も変わらず、既存実験は1ビットも変わらない。
    #   （常に足すと、使わなくてもlogitが2本増えて全産出の確率が微妙に動く）
    taro._context_enabled = bool(pd.get("context", False))
    taro._context_hidden = None
    taro._context_last_target = None
    taro._context_speaker_ids = None
    if taro._context_enabled:
        taro._context_speaker_ids = {"parent": pv.add_special("<PARENT>"),
                                     "self": pv.add_special("<SELF>")}
    # 【M4・2026-09-06・仕様_M4_消えた物について「○○ないね」と言う】
    #   「消えた」を、話者の印(<PARENT>/<SELF>)と同じ仕組みの離散トークンとして
    #   1つ足す（逸脱・Tier3：内側の状態を離散の印にする。仕様書「『だね』と
    #   『ないね』の区別」節・doc\人間模倣からの逸脱リスト.md に登録済み）。
    #   既定False（produce.vanish_input無し）ではtaro._gone_idはNoneのまま＝
    #   語彙サイズも変わらず1ビットも変わらない。話者トークンと同じ手順
    #   （Vocabulary.add_special＋resize_embeddingでの埋め込み拡張）を踏むことで、
    #   既存の .pt を読んでも壊れない（resize_embeddingは旧重みを先頭へコピーし、
    #   増えた行だけ新規初期化する。taro_core/src/brain/cerebral_cortex/
    #   recurrent_core.py resize_embedding参照）。
    taro._gone_id = None
    if bool(pd.get("vanish_input", False)):
        if not taro._context_enabled:
            raise ValueError(
                "produce.vanish_input=true には produce.context=true が必要"
                "（<GONE>はGRUの文脈入力へ流す仕組みで、文脈が無いと意味を持たない）。")
        taro._gone_id = pv.add_special("<GONE>")
    # 【M4d・2026-09-06・仕様_M4d_あるの印】「あるの印」を「消えたの印」と同じ
    #   仕組みで足す。注意中の物体ファイルが見えている（消えていない）とき、
    #   話者の印の直後に入れる特殊トークン（口には出さない・生成列から除いて
    #   復号する。逸脱・Tier3：ある/ないを離散2値で言語に渡す。提案2で連続値へ
    #   移行予定。doc\人間模倣からの逸脱リスト.md その46に登録済み）。
    #   here_input=true は vanish_input=true を前提にする（あるの印は消えたの印
    #   と対で初めて意味を持つ＝gru_hippoとλの共存禁止と同じ流儀）。
    taro._here_id = None
    if bool(pd.get("here_input", False)):
        if not bool(pd.get("vanish_input", False)):
            raise ValueError(
                "produce.here_input=true には produce.vanish_input=true が必要"
                "（<HERE>は<GONE>と対で初めて意味を持つ）。")
        taro._here_id = pv.add_special("<HERE>")
    # 【M4e・2026-09-07・仕様_M4e_状態の線と驚きの書き込み】状態の線（GRU入口へ
    #   1音ごとに毎回同じ値を足す仕組み）を使うかどうかの設定読み取り。
    #   taro.brain.state_embedding は TaroBrain.__init__ で常に作られる
    #   （taro_core/src/brain/cerebral_cortex/recurrent_core.py）ため、ここでは
    #   「使うかどうか」だけを読む。run/trainer.pyの各forward_hidden呼び出しは
    #   self.cfg.produce.get("state_channel", False) を直接読むため、この属性
    #   taro._state_channel 自体はtrainer.pyの分岐には使われない（診断・将来の
    #   参照用に一貫した場所へ置いておく）。既定False（設定なし）では従来と
    #   1ビットも変わらない。<GONE>/<HERE>と同じく、意味を持たせるには
    #   produce.context=true が必要（GRUの文脈が無いと状態の線を1音ごとに
    #   足す先＝GRU入口が文脈を持たないため）。
    taro._state_channel = bool(pd.get("state_channel", False))
    if taro._state_channel and not taro._context_enabled:
        raise ValueError(
            "produce.state_channel=true には produce.context=true が必要"
            "（状態の線はGRUの文脈入力へ流す仕組みで、文脈が無いと意味を持たない）。")
    taro.brain.resize_embedding(pv.size)
    taro.brain.set_vocab_mapping(pv.char2idx)
    # 【聞く学習・2026-08-31・設計_文脈（コンテキスト）.md 追補】聞いた発話の
    #   次トークン予測でGRUを学習する。学習率・クリップ・損失の形は目標Bの実績値
    #   （B/src/taro/brain/basal_ganglia.py:23-49, lr既定0.005）をそのまま借りる。
    #   更新するのは embedding・GRU・知覚ヘッドだけ（調音ヘッドには触れない＝
    #   喃語で作った発話回路を直接は動かさない。共有GRU経由の間接影響は
    #   産出テストの完全一致率で監視する）。
    #   既定OFF（produce.listen_learn 無し）では optimizer も作らない＝挙動不変。
    #   optimizer は resize_embedding の**後**に作ること（resizeはモジュールを
    #   丸ごと差し替えるので、先に作ると古いパラメータを掴んで学習が空振りする）。
    # 【V1・2026-08-31・設計_視覚とGRUの統合.md】視覚トークンの変換層。
    #   produce.vision_context=true のときだけ作る（既定false＝1ビットも変わらない）。
    #   listen optimizer より先に作ること（学習対象に入れるため）。
    taro._visual_projection = None
    if bool(pd.get("vision_context", False)):
        from cerebral_cortex.visual_projection import VisualProjection
        taro._visual_projection = VisualProjection(
            in_dim=384, out_dim=taro.brain.embedding.embedding_dim)
        _pvp = getattr(taro, "_pending_visual_projection", None)
        if _pvp is not None:
            taro._visual_projection.load_state_dict(_pvp)
    # 【2026-09-04・混乱防止】word_choice="gru_hippo"では、λブレンド（決定4）の
    #   結果は段階2ブロックに丸ごと上書きされ、context_lambdaは一切効かない
    #   （run/trainer.py _apply_word_production参照）。効かない値を設定したまま
    #   気づかない「設定が静かに無視される」事故（過去5件目相当）を防ぐため、
    #   起動時に組み合わせを検証して止める。
    if (str(pd.get("word_choice", "lexicon")) == "gru_hippo"
            and float(pd.get("context_lambda", 0.0)) != 0.0):
        raise ValueError(
            "produce.word_choice='gru_hippo' のとき produce.context_lambda は"
            "効きません（段階2の選択がλブレンドの結果を上書きするため）。"
            "context_lambdaを0にするか、word_choiceを'lexicon'にすること。")
    taro._listen_learn = bool(pd.get("listen_learn", False))
    # 【分節第2案・2026-09-03】設計_分節（語の切れ目の発見）.md 第2案 第2部
    #   「listen_eos」節。既定False＝聞く学習の列にEOSを足さない＝1ビットも変わらない。
    taro._listen_eos = bool(pd.get("listen_eos", False))
    # 【分節第2案・2026-09-03】同上「segment_mode」節。既定"valley"＝従来の
    #   real_confidence の谷のまま（"end_prob"にすると第2案の切り出しに変わる）。
    taro._segment_mode = str(pd.get("segment_mode", "valley"))
    taro._listen_optimizer = None
    taro._listen_grad_clip = float(pd.get("listen_grad_clip", 1.0))
    taro._listen_self = bool(pd.get("listen_self", True))   # 既定True＝従来どおり自己発話でも学ぶ
    if taro._listen_learn:
        if not taro._context_enabled:
            raise ValueError(
                "produce.listen_learn=true には produce.context=true が必要"
                "（聞く学習は文脈のhiddenから予測を始める設計。設計追補参照）。")
        import itertools
        _extra = (list(taro._visual_projection.parameters())
                  if taro._visual_projection is not None else [])
        _params = itertools.chain(taro.brain.embedding.parameters(),
                                  taro.brain.gru.parameters(),
                                  taro.brain.perception_head.parameters(),
                                  _extra)
        taro._listen_optimizer = torch.optim.Adam(
            _params, lr=float(pd.get("listen_lr", 0.005)))
    # 【言語海馬・段階1・2026-09-01・設計_言語海馬と睡眠リプレイ.md】
    #   produce.hippocampus が無ければ None＝run/trainer.pyの新しい分岐は
    #   一度も実行されない＝既存実験の挙動は1ビットも変わらない。
    taro.language_hippocampus = None
    if pd.get("hippocampus") is not None:
        # 【なぜこのパス、2026-09-01】brain/hippocampus.py（MotorHippocampus、
        #   recurrent_core_motor.pyがfrom hippocampus importで参照）と同名の
        #   パッケージ brain/hippocampus/ を作ると、sys.pathでbrain直下がフラット
        #   import対象になっているため既存importを覆い隠して壊す（実測で発覚）。
        #   衝突を避け brain/language_hippocampus/ に置いた。
        from language_hippocampus.language_hippocampus import LanguageHippocampus
        taro.language_hippocampus = LanguageHippocampus(pd.get("hippocampus"))
        _plh = getattr(taro, "_pending_language_hippocampus", None)
        if _plh is not None:
            taro.language_hippocampus.load_state_dict(_plh)
    # 【塊レベル層・2026-09-07・仕様_M5_塊レベル層.md §2】塊レベルのGRU。
    #   既定False（produce.chunk_level無し）ではchunk_vocab/chunk_brain/
    #   _chunk_optimizer/_chunk_context_hiddenはNoneのまま＝run/trainer.pyの
    #   新しい分岐は一度も実行されない＝既存実験の挙動は1ビットも変わらない。
    taro.chunk_level = bool(pd.get("chunk_level", False))
    taro.chunk_vocab = None
    taro.chunk_brain = None
    taro._chunk_optimizer = None
    taro._chunk_context_hidden = None
    if taro.chunk_level:
        if not taro._context_enabled or str(pd.get("word_choice", "lexicon")) != "gru_hippo":
            raise ValueError(
                "produce.chunk_level=true には produce.context=true と "
                "produce.word_choice='gru_hippo' が必要"
                "（塊レベルGRUは文脈GRU＋海馬の経路にだけ差し込む設計。"
                "共存禁止と同じ流儀＝仕様_M5_塊レベル層.md §2）。")
        from cerebral_cortex.chunk_vocab import ChunkVocab
        from cerebral_cortex.recurrent_core import TaroBrain as _ChunkBrainCls
        cv = ChunkVocab()
        _pcv = getattr(taro, "_pending_chunk_vocab", None)
        if _pcv is not None:
            cv.load_state_dict(_pcv)
        # 【仕様書§1・§2】<PARENT>/<SELF>/<GONE>/<HERE>を音の名簿と同じ4種
        #   （id空間は別）。specialsのキーはtrainer.pyの_chunk_context_feedが
        #   speaker（"parent"/"self"）とgone/hereの印を直接引く形に合わせた
        #   （taro._context_speaker_ids={"parent":..,"self":..}と同じ発想。
        #   仕様書§1は「音の名簿と同名」とのみ指定しキーの正確な文字列までは
        #   決めていなかったため、trainer.py側の呼び出し規約に合わせてここで
        #   判断した＝作業記録「仕様に無かった判断」に記載）。
        cv.add_special("parent")
        cv.add_special("self")
        cv.add_special("gone")
        cv.add_special("here")
        taro.chunk_vocab = cv
        # 【重要・既定不変の落とし穴・2026-09-07】nn.Embedding/nn.GRUの初期化は
        #   torchのグローバル乱数生成器を消費する。fork_rngでこの構築だけを
        #   支流へ逃がし、chunk_level無しの既存実験の乱数列を1つもずらさない
        #   （recurrent_core.py state_embeddingの同種コメント・落とし穴と同じ）。
        with torch.random.fork_rng(devices=[]):
            taro.chunk_brain = _ChunkBrainCls(
                vocab_size=cv.size, embedding_dim=64, hidden_dim=128,
                body_state_dim=0)
        _pcb = getattr(taro, "_pending_chunk_brain", None)
        if _pcb is not None:
            _own = taro.chunk_brain.state_dict()
            _matched = {k: v for k, v in _pcb.items()
                        if k in _own and _own[k].shape == v.shape}
            taro.chunk_brain.load_state_dict(_matched, strict=False)
            if verbose:
                print(f"  [塊GRU] ロード{len(_matched)}/{len(_own)}層", flush=True)
        taro._chunk_optimizer = torch.optim.Adam(
            taro.chunk_brain.parameters(), lr=float(pd.get("listen_lr", 0.005)))
        taro._chunk_context_hidden = None
        # 【仕様書§2】海馬：chunk_level真ならlanguage_hippocampus.unit="chunk"。
        #   保存済み海馬のunitが"mora"（または無し）なら中身を捨てて空から
        #   （print で明示）。
        if taro.language_hippocampus is not None:
            if taro.language_hippocampus.unit != "chunk":
                if taro.language_hippocampus.episodes and verbose:
                    print("注意[塊レベル] 保存済み海馬はunit='mora'（モーラの列）でした。"
                          "塊レベル層では海馬は塊の列で書き直すため、中身を空にします"
                          f"（捨てたエピソード数={len(taro.language_hippocampus.episodes)}）。",
                          flush=True)
                taro.language_hippocampus.episodes = []
                taro.language_hippocampus.unit = "chunk"
    else:
        taro._pending_chunk_brain = getattr(taro, "_pending_chunk_brain", None)
        taro._pending_chunk_vocab = getattr(taro, "_pending_chunk_vocab", None)
    # 【M4e・2026-09-07・仕様_M4e_状態の線と驚きの書き込み】驚きの書き込み。
    #   消えたの印が立っている間に聞いた発話は、海馬に書く強さをgone_strength倍
    #   にする（既定1.0＝従来と1ビットも変わらない）。run/trainer.pyの_context_feed
    #   が self.cfg.produce.get("hippocampus", {}).get("gone_strength", 1.0) を
    #   直接読むため、この属性自体はtrainer.pyの分岐には使われない
    #   （taro._state_channel と同じく、診断・将来の参照用に一貫した場所へ置く）。
    taro._gone_strength = float((pd.get("hippocampus") or {}).get("gone_strength", 1.0))
    taro.produce_vocab = pv
    # 【設計「⚠学習器の共有」案a】運動学習(taro.learner)とは別インスタンス。
    #   generate()が使うtaro.brainのパラメータ（embedding/gru＝音声用でmotor_gruとは
    #   別物/head_place等の4ヘッド）は、運動系（motor_gru/motor_cortex/
    #   forward_model_head等、taro_brain_motor.py）とは重みを共有しない別モジュール
    #   （taro_brain_motor.py冒頭コメント「音声用GRUとは別に運動専用GRU」参照）。
    #   したがってtaro.produce_learnerとtaro.learnerが同じtaro.brainを指していても、
    #   それぞれのbackward()が実際に勾配を作る先は重ならない
    #   （run/trainer.py _apply_word_production参照）。
    taro.produce_learner = TaroLearner(taro.brain, lr=float(pd.get("lr", cfg.lr)))
    taro.produce_dop = Dopamine()
    # 【発話の動機・2026-08-31・設計_発話の動機.md】言う確率πの門。
    #   produce.social が無ければ None＝trainer の新経路は1行も実行されない。
    #   保存済みモデルに門があれば（_pending_speech_gate）復元する。
    taro._social_cfg = pd.get("social")
    taro._speech_gate = None
    taro._social_pending = None
    if taro._social_cfg is not None:
        from subcortical_nuclei.speech_gate import SpeechGate
        sc = taro._social_cfg
        taro._speech_gate = SpeechGate(
            init_prob=float(sc.get("init_prob", 0.9)),
            lr=float(sc.get("gate_lr", 0.5)),
            dopamine=Dopamine(),
            speak_cost=float(sc.get("speak_cost", 0.1)))
        _pending_sg = getattr(taro, "_pending_speech_gate", None)
        if _pending_sg is not None:
            taro._speech_gate.load_state(_pending_sg)
    taro._last_produce_sec = None
    # 【F2-1・実装作業③・2026-08-23】発話小脳（口の動き→音の帳面）。
    #   generate()は既に cerebellum= を受け取れる（taro_brain.py:177-179、
    #   コード改変ゼロ）。mode に関わらず cfg.produce が有効なら常に構築する
    #   （＝word/babbleどちらのモードでも学習経験を帳面に貯める）。
    #   【判断を仰ぐ点・作業記録参照】forward_map/inverse_map/experience_countは
    #   素のdictでありTaro.save()/_load()のblobには現時点で載らない
    #   （state_dict()を持たない）。持ち越し学習（cfg.model指定）をすると
    #   帳面は毎回空から再スタートする。保存方法は勝手に決めず報告する。
    taro.produce_cerebellum = SpeechCerebellum()
    # 【F2-1・作業A・2026-08-23】_load()が退避した帳面があれば、作った直後に
    #   流し込む（_load()は_setup_produceより前に呼ばれるため、_load()の時点では
    #   まだCerebellum()が存在せず直接復元できない。_load()のコメント参照）。
    pending = getattr(taro, "_pending_produce_cerebellum", None)
    if pending is not None:
        taro.produce_cerebellum.forward_map = pending["forward_map"]
        taro.produce_cerebellum.inverse_map = pending["inverse_map"]
        taro.produce_cerebellum.experience_count = pending["experience_count"]
        if verbose:
            print(f"  [発話小脳] 復元完了：forward_map={len(pending['forward_map'])}件"
                  f" inverse_map={len(pending['inverse_map'])}件"
                  f" experience_count={len(pending['experience_count'])}件", flush=True)
    taro._pending_produce_cerebellum = None
    # 【F2-1・実装作業④⑦・2026-08-23】喃語モード用に持ち越すGRU隠れ状態。
    #   B原本 core_b.py self_babble() の「自分の声を聞く」ループ
    #   （610-615行）と同じく、発声のたびに自分の出力を聞かせて更新する。
    #   word モードでは使わない（既存どおり毎回 hidden=None）。
    taro._produce_hidden = None
    # 【F2-2・実装作業②・2026-08-23】ブローカ野（発話計画の中枢）。
    #   taro_core/src/brain/left_frontal_lobe/brocas_area.py（F2-1で移植済み・
    #   plan()を呼ぶ配線は未実装だったもの）。mode に関わらず cfg.produce が
    #   有効なら常に構築する（produce_cerebellum と同じ扱い）。plan()自体は
    #   trainer.py _apply_word_production（wordモード）でしか呼ばない
    #   （babbleモードは計画を立てずに探索的に発声する＝従来どおり）。
    #   設計：F/docs/設計_F2-2_見た物の名前を言う.md 第4部「②」。
    from left_frontal_lobe.brocas_area import BrocasArea
    taro.produce_broca = BrocasArea()
    mode = pd.get("mode", "word")
    if verbose:
        print(f"[産出] taro.produce ON（mode={mode}）：語彙={pv.size}文字"
              f"（vocal_tract stage={vt.stage} decoupled={not vt.is_coupled()}）"
              f" threshold={pd.get('threshold', 0.80)}"
              f" cooldown_sec={pd.get('cooldown_sec', 2.0)}", flush=True)


class _DoubleTouchBonusContributor:
    """taro.reward_contributors の1要素（2026-08-05・全身一般化の設計1-4節）。

    「compute(ctx)->floatを持つオブジェクト」という緩い規約に従うだけの、
    抽象基底クラスを持たない最小の実装。ctx.last_double_touch['hit'] が
    True なら bonus を返し、そうでなければ0を返す（run/trainer.py 749行付近が
    以前「hitならcfg.double_touch_bonusを足す」と1行で書いていたのを、
    このオブジェクトへ切り出しただけで中身の判定は変えていない）。
    """

    def __init__(self, bonus):
        self.bonus = float(bonus)

    def compute(self, ctx):
        ld = getattr(ctx, "last_double_touch", None)
        if ld is not None and ld.get("hit"):
            return self.bonus
        return 0.0


class _MouthTouchBonusContributor:
    """taro.reward_contributors の1要素（2026-08-12・口元自己接触報酬の実装仕様3節）。

    設計：作業記録（非公開） 5-1節・5-2節（決定日2026-08-12）。
    「手のひらが口元に触れた瞬間」に報酬を1回だけ与える（立ち上がり検出＝前tickで
    口元に触れておらず今tickで触れている、という遷移。押し付け続けても加点しない）。

    立ち上がり検出の状態（`_prev_hit`）はこのオブジェクトが持つ。エピソードが
    リセットされたら、`reset()`（trainer.py の `reset_state()` の汎用ループから
    呼ばれる）で必ず初期化する必要がある（そうしないと、エピソード境界をまたいで
    「前回の最後のtick」を引きずり、次のエピソード最初のtickで誤って立ち上がりを
    見逃す/誤検出する恐れがある）。
    """

    def __init__(self, double_touch_detector, bonus, mouth_threshold=0.5,
                 mouth_gain=1.0, toucher_threshold=0.5):
        # double_touch_detector＝t.double_touch（口元マスクを保持するインスタンス）
        #   への参照。構築順序上、Taro.__init__ 内で t.double_touch を先に作ってから
        #   このコンストラクタへ渡す（仕様3節「実装時に確認すること」への回答。
        #   ctx.taro.double_touch 経由でも到達できるが、コンストラクタで直接参照を
        #   持たせる方が、この報酬寄与クラスが「何に依存しているか」が読んで分かる）。
        self._dt = double_touch_detector
        self.bonus = float(bonus)
        self.mouth_threshold = float(mouth_threshold)
        self.mouth_gain = float(mouth_gain)
        # toucher_threshold＝t.double_touch.threshold（cfg.double_touch_threshold）を
        #   そのまま使い回す（仕様3節。新しい設定キーを増やさない）。
        self.toucher_threshold = float(toucher_threshold)
        self._prev_hit = False

    def reset(self):
        """エピソード境界で呼ぶ。前回tickの状態を引き継がない。"""
        self._prev_hit = False

    def compute(self, ctx):
        ld = getattr(ctx, "last_double_touch", None)
        last = getattr(ctx, "last", None)
        if ld is None or last is None:
            return 0.0
        toucher_presence = float(ld.get("toucher_presence", 0.0))
        # touch_flat は ctx.last["obs_out"]["touch"]（trainer.py が
        #   double_touch.detect() に渡しているのと同じ値）。
        touch_flat = to_tensor(last["obs_out"]["touch"])
        mouth_presence = self._dt.mouth_presence(touch_flat, gain=self.mouth_gain)
        hit_now = (mouth_presence > self.mouth_threshold) and \
                  (toucher_presence > self.toucher_threshold)
        rising = hit_now and not self._prev_hit
        self._prev_hit = hit_now
        return self.bonus if rising else 0.0


class Taro:
    """太郎そのもの。脳・学習器・神経調節・小脳・海馬を持つ。

    注意：ここは「太郎の中身」だけ。環境（env）は組み立てに必要なので受け取るが、
      **保持しない**（体を作り直しても Taro は作り直さないため）。
      env が要る操作（step / reset）は呼び出し側（run/trainer.py）が持つ。

    属性:
        brain, fusion, target_fusion, nat_head, emb_proj
        learner, dop, ne, homeo, dev_clock, cereb, cere_opt, hippo, lp
        n_act      脳が出す行動の次元（拮抗筋モードでは環境の半分）
        out_dim    予測対象の次元
        blocks     予測対象のブロック境界 [(start, end, 名前), ...]
    """

    def __init__(self, cfg, env, *, seed, verbose=True):
        self.cfg = cfg
        self.seed = int(seed)
        self.blocks = None            # ln_prop が最初の呼び出しで作る

        # ---- 触覚の次元（reset せずに読める。reset を足すと乱数列がずれる）----
        # 注意：get_sensor_count() は「センサ点の数」で、観測は1点あたり力の3成分。
        touch_dim = int(env.observation_space["touch"].shape[0]) if cfg.touch else 0
        # 固有感覚の次元は駆動モードで変わる（関節モード=621 / 筋肉モード=801）
        prop_dim_space = int(env.observation_space["observation"].shape[0])

        # ---- 視覚の解像度（fusion に渡す）------------------------------------
        # 注意：2026-08-13より、taro.vision は「脳が視覚を使うか」（ここでのvres）と、
        #   「環境が視覚センサ自体を持つか」（e_scene.build()へ渡るvision_params。
        #   run/plugins/common/scene.py参照）の**両方**を連動して切り替える。
        #   vision=False は環境の視覚センサごと無効化するアブレーションになり、
        #   LeanMimoEnv.strip_texturesによりメモリも節約される。
        vres = 0
        if cfg.vision:
            from e_toy_env import VISION_RES
            vres = VISION_RES
            # 【2026-09-02・F2-44】シーンの body.vision_px で目の解像度を変えたとき、
            #   体側の視覚エンコーダ（VisionEncoder）は画像サイズ依存なので、定数でなく
            #   実際の眼球カメラの一辺から組む（従来128なら同値＝挙動不変）。
            _vp = getattr(getattr(env, 'unwrapped', env), 'vision_params', None) or {}
            if isinstance(_vp.get('eye_left'), dict) and _vp['eye_left'].get('width'):
                vres = int(_vp['eye_left']['width'])
                if verbose and vres != VISION_RES:
                    print(f'[vision] 眼球カメラ {vres}px に合わせて視覚エンコーダを構築（既定{VISION_RES}）', flush=True)

        # ---- 体性感覚系（触覚ONのときだけ）----------------------------------
        touch_map = None
        if cfg.somatosensory and cfg.touch:
            touch_map = build_touch_map_from_env(env)
            assert touch_map.total_dim == touch_dim, \
                f"触覚の地図{touch_map.total_dim} != 観測{touch_dim}"
            if verbose:
                print(f"[体性感覚系] SomatosensoryCortex 有効：部位数="
                      f"{len(touch_map.group_names)} 触覚総次元={touch_dim}", flush=True)

        # ---- 触覚の順応（同じ場所を押され続けると感じ方が弱まる、2026-08-13）--
        # 設計：作業記録（非公開）
        # 指示：作業記録（非公開）
        #   既定OFF（cfg.touch_adaptation=False）＝self.touch_adaptationはNoneのまま
        #   ＝apply_touch_adaptation()は何もせずobsをそのまま返す＝既存実験は不変。
        self.touch_adaptation = None
        if cfg.touch_adaptation:
            if not (cfg.somatosensory and cfg.touch):
                raise ValueError(
                    "touch_adaptation=True には touch=true, somatosensory=true が要る。\n"
                    "  順応は触覚の生の力ベクトル(obs['touch'])に適用し、"
                    "SomatosensoryCortexの手前(fusion.py)へ差し込むため。")
            self.touch_adaptation = TouchAdaptation(
                touch_map.n_points, dt=_PHYS_DT * cfg.K,
                fa_enabled=bool(cfg.touch_adapt_fa),
                sa_enabled=bool(cfg.touch_adapt_sa),
                include_cortical=bool(cfg.touch_adapt_include_cortical),
                tau_peripheral_s=float(cfg.touch_adapt_tau_peripheral_s),
                tau_cortical_s=float(cfg.touch_adapt_tau_cortical_s),
                sa_floor=float(cfg.touch_adapt_sa_floor),
                tau_recover_s=float(cfg.touch_adapt_tau_recover_s),
                fa_gain=float(cfg.touch_adapt_fa_gain))
            if verbose:
                layer = "末梢+脳" if cfg.touch_adapt_include_cortical else "末梢のみ"
                cortical_part = (f" τ脳={cfg.touch_adapt_tau_cortical_s}秒"
                                 if cfg.touch_adapt_include_cortical else "")
                print(f"[触覚の順応] ON：粒度=point 階層={layer}"
                      f" τ末梢={cfg.touch_adapt_tau_peripheral_s}秒{cortical_part}"
                      f" 残存率={cfg.touch_adapt_sa_floor} 速順応倍率={cfg.touch_adapt_fa_gain}"
                      f" 回復τ={cfg.touch_adapt_tau_recover_s}秒 対象点数={touch_map.n_points}"
                      f" エピソード境界でリセット={bool(cfg.touch_adapt_reset_on_episode)}",
                      flush=True)
        elif verbose:
            # OFF時にも必ず表示する（静かに壊れるタイプの変更であるため。仕様8節）。
            print("[触覚の順応] OFF", flush=True)

        # ---- ① 融合層（感覚をまとめる）--------------------------------------
        # target_fusion は**凍結した別インスタンス**（RND式）。予測側と正解側が
        #   同じ学習中の層だと「出力を平坦にすれば当たる」抜け道で崩壊する（目標Cで実際に踏んだ）。
        self.fusion = MinimalFusion(touch_dim, vision_res=vres,
                                    proprio_dim=prop_dim_space,
                                    touch_map=touch_map)
        self.target_fusion = MinimalFusion(touch_dim, vision_res=vres,
                                           proprio_dim=prop_dim_space,
                                           touch_map=touch_map).freeze()
        if cfg.somatosensory and cfg.touch and self.fusion.touch is not None and verbose:
            print(self.fusion.touch.summary(), flush=True)

        # 【2026-08-18新設・F1-3】耳＋連合器（既定OFF）。fusion（視覚エンコーダ含む）を
        #   作った直後＝taro.fusion.visionが以後いつでも呼べる状態になってから。
        _setup_hearing(self, cfg, verbose=verbose)

        # ---- ② 行動の次元 ---------------------------------------------------
        n_env_act = env.action_space.shape[0]
        # 拮抗筋モード：脳・CPG・prev_a はすべて n_joint 次元、環境には to_env_action で
        # 2*n_joint に写像して渡す。既定は n_act == n_env_act。
        self.n_act = n_env_act // 2 if (cfg.is_muscle and cfg.antagonist) else n_env_act

        # ---- ③ 最初の reset（必ず seed を渡す）------------------------------
        # 環境の乱数（env.unwrapped.np_random）は gym が別に管理しており、
        # torch.manual_seed も np.random.seed も効かない。シードなし reset だと
        # 毎回ちがう姿勢から始まり再現できない（2026-07-15 のバグ）。
        obs, _ = env.reset(seed=self.seed)
        self.sdim = self.fusion.encode(obs).shape[0]
        self.prop_dim = to_tensor(obs["observation"]).shape[0]

        # ---- ④ 予測対象の次元を**測る**（手計算だとズレる）--------------------
        self.out_dim = int(self.encode_target(obs).shape[0])
        if cfg.target_has_vision and verbose:
            bd = "／".join(f"{nm}:{e - s}" for (s, e, nm) in (self.blocks or []))
            print(f"[予測対象] {cfg.target_kind} → 全{self.out_dim}次元  内訳 {bd}")
            print(f"     誤差はブロックごとに平均してから足す（次元数の影響を除く。"
                  f"λ_v={cfg.lam_v}）")

        # ---- ⑤ 脳 -----------------------------------------------------------
        self.brain = TaroBrainWithMotor(vocab_size=3, sensory_dim=self.sdim,
                                        n_actuators=self.n_act, proprio_dim=self.out_dim,
                                        latent_deterministic=bool(cfg.latent_deterministic))
        # 【運動性喃語（脊髄CPG）】noise=colored のとき太郎の中で色付き探索を有効化する。
        # 既定 white では呼ばれない＝spinal_cpg=None＝白色ガウス＝従来と数値完全一致。
        if str(cfg.noise) == "colored":
            # 注意：シナジーの index は90-actuator前提。筋肉モードでは pair_offset を入れて
            #   対になる筋（伸ばす側）に符号反転で同じシナジーを混ぜる（spinal_cord/cpg.py）。
            pair_offset = (self.n_act // 2) if (cfg.is_muscle and not cfg.antagonist) else 0
            self.brain.enable_spinal_babble(
                self.n_act, leg_r=LEG_R, leg_l=LEG_L, arm_r=ARM_R, arm_l=ARM_L,
                beta=cfg.beta, synergy=bool(cfg.synergy), syn_w=cfg.syn_w, seed=self.seed,
                antagonist=(cfg.is_muscle and cfg.antagonist),
                co_activation=cfg.coactivation, pair_offset=pair_offset)
            if verbose:
                print(f"[脊髄CPG] 色付き探索ON: β={cfg.beta} synergy={bool(cfg.synergy)}"
                      f" syn_w={cfg.syn_w}"
                      f"{f' pair_offset={pair_offset}' if pair_offset else ''}"
                      f"{' 【拮抗筋モードON】coactivation=' + str(cfg.coactivation) if cfg.is_muscle and cfg.antagonist else ''}",
                      flush=True)
        # 【2026-08-11・新しい駆動モジュール】伸張反射＋揺らぐ振動子の共通駆動。
        # 既定 spinal_drive_mode="cpg" では _setup_reflex_common が即 False を返して
        # 何もしない＝self.brain.reflex_common は None のまま＝既存の経路（explore()・
        # spinal_cpg）は1バイトも変わらない。cfg._check() で actuation=muscle・
        # noise!=colored であることは既に保証済み（config.pyのバリデーション）。
        self.reflex_common_active = _setup_reflex_common(self, cfg, env, verbose=verbose)
        # 【2026-08-15・座位保持の学習】層1（姿勢制御反射・ゲート）・層2（立ち直り反射・
        #   角速度ダンパー）。既定OFF＝taro.postural_gate/taro.righting_damperはNoneの
        #   まま＝E/scripts/e_toy_env.py側のstep()は何も呼ばず、既存実験の挙動は不変。
        _setup_postural_gate(self, cfg, env, verbose=verbose)
        _setup_righting_damper(self, cfg, env, verbose=verbose)

        # 【2026-07-25】D-a/D-b の層は**太郎の中（core）のもの**を使う。
        # 旧実装はここで別に作っており、core の層は作られるだけで使われていなかった
        # （＝脳が二重に存在していた）。旧名を別名として残す（参照箇所が多いため）。
        self.emb_proj = self.brain.motor_input_proj
        self.nat_head = self.brain.forward_model_head

        # ---- ⑥ 学習器と神経調節 ----------------------------------------------
        self.learner = TaroLearner(CombinedParams(self.brain, self.fusion), lr=cfg.lr)
        self.dop = Dopamine()
        self.ne = LocusCoeruleus(relative=bool(cfg.ne_relative))
        self.homeo = HomeostaticScaling(dim=self.sdim)
        self.dev_clock = DevelopmentalClock()   # ③発達年齢（累積学習回数）
        # 運動小脳。ON/OFFで乱数列を揃えるため cfg.cerebellum に関わらず**常に構築**する
        # （使う/学習するのは ON のときだけ。gate/imitation は乱数を消費しない）。
        self.cereb = MotorCerebellum(self.brain.latent_dim, self.n_act)
        self.cere_opt = torch.optim.Adam(self.cereb.parameters(), lr=cfg.lr)
        self.hippo = self.brain.hippocampus     # 睡眠リプレイのバッファ（core に一元化済み）
        self.lp = LearningProgress(surprise_bonus=cfg.progress_surprise_bonus,
                                    surprise_decay=cfg.progress_surprise_decay,
                                    surprise_var_tau=cfg.progress_surprise_var_tau,
                                    surprise_threshold=cfg.progress_surprise_threshold)
        # 予測誤差の速い/遅い走行平均（surprise_bonus既定0.0なら現状と同じ計算のみ）

        # ---- ⑦ 続きから学習する（形が合う層だけ）------------------------------
        if cfg.model:
            self._load(cfg.model, verbose=verbose)

        # 【2026-08-22新設・F2】見た物の名前を言う（初語）。既定OFF（cfg.produce=None）：
        #   taro.produce_vocab/produce_vocal_tract/produce_learner/produce_dopは
        #   Noneのまま＝run/trainer.pyのstep_k内の配線が一度も実行されない＝既存実験の
        #   挙動は1ビットも変わらない。有効時は_load()の"後"（脳の重みを保存値から
        #   復元した"後"）に呼ぶ：埋め込み層を伸ばす前に読み込む方が安全（下記
        #   _setup_produceのdocstring参照）。有効時のみtorchの乱数を追加消費する
        #   （nn.Embedding/nn.Linearの初期化）＝既存実験（produce未使用）の乱数列は
        #   1つもずれない。
        _setup_produce(self, cfg, env, verbose=verbose)

        # ---- ⑧ 努力コストの重み ---------------------------------------------
        # 筋力（最大トルク）が大きい筋ほど動かすとコストが高い（代謝の標準：活性化²×筋サイズ）。
        # 注意：体を作り直しても**更新していない**（元の実装もそうだった）。
        #   体を育てる実験で努力コストを使うときは、ここが古い体の値であることに注意。
        gear = np.abs(env.unwrapped.model.actuator_gear[:self.n_act, 0]).astype(np.float32)
        self.eff_w = torch.tensor(gear / (gear.sum() + 1e-8))
        self.first_obs = obs

        # ---- ⑨ 手先位置の目標表現（案C・Goal Babbling段階1）------------------
        # 【なぜ①〜⑧の"あと"に置くか、2026-08-02】乱数を消費する順序を守るため
        #   （このファイル冒頭の注記と同じ理由）。既定（goal_space!="reach_self"）
        #   ではこのブロックは一度も実行されない＝乱数消費もパラメータ数も
        #   現状と完全に不変（spinal_cpgの「条件付き構築」パターンを踏襲。
        #   設計の統合判断「決定1」。taro_brain_motor.py は1行も変更しない）。
        self.arm_map = None
        self.reach_touch_groups = None
        self.reach_blocks = None
        self.reach_dim = 0
        self.reach_head = None
        self.reach_opt = None
        self.home_reach_goal = None
        # 頭へのダブルタッチを報酬に直結する（2026-08-03、2026-08-05に全身一般化）。
        #   既定は None のまま＝run/trainer.py の `t.double_touch is not None`
        #   ガードで一切実行されない。
        self.double_touch = None
        # 【2026-08-05追記：全身一般化】報酬に足す寄与を集めるリスト。
        #   「compute(ctx)->floatを持つオブジェクト」という緩い規約
        #   （設計1-4節。抽象基底クラスは作らない）。空のままなら
        #   run/trainer.py のforループは0回まわり既存挙動を1ビットも変えない。
        self.reward_contributors = []
        reach_space = cfg.goal_babbling and cfg.goal_space == "reach_self"
        if reach_space:
            self.arm_map = build_arm_proprio_map_from_env(env, side=cfg.reach_arm_side)
            self.reach_touch_groups = ["head", "chest",
                                       f"{_opposite_side(cfg.reach_arm_side)}_palm"]
            g0 = self.encode_reach_goal(obs)
            self.reach_dim = int(g0.shape[0])
            self.reach_blocks = [(0, 7, "arm"), (7, self.reach_dim, "touch")]
            self.reach_head = ReachGoalHead(self.brain.latent_dim, self.n_act, self.reach_dim)
            # 独立optimizer（設計の統合判断「決定2」）。既存の自己モデル学習
            #   （nat_head・pe）の loss には混ぜない＝既存の学習曲線を汚さない。
            self.reach_opt = torch.optim.Adam(self.reach_head.parameters(), lr=cfg.lr)
            # ホーム姿勢＝reset直後の初期姿勢（成長のたびに on_body_change 側で
            #   再計算する。可動域自体が成長で変わりうるため）
            self.home_reach_goal = g0.detach()
            if verbose:
                print(f"[goal_babbling] reach_self 有効：目標{self.reach_dim}次元"
                      f"（腕7＋自己接触{self.reach_dim - 7}） 腕={cfg.reach_arm_side}"
                      f" 対象部位={self.reach_touch_groups}（2026-08-02時点で未検証・診断段階）",
                      flush=True)
        # 【2026-08-05：全身一般化・reach_self依存の除去】
        #   構築条件を「reach_space（既存、後方互換）または
        #   cfg.double_touch_bonus != 0.0」のORへ拡張した（仕様2節）。
        #   自前でSomatosensoryCortexを構築する（taro_setup.py冒頭の
        #   build_touch_map_from_env を使い、taro自身の target_fusion.touch には
        #   一切依存しない＝touch=false のシーンでも動く。double_touch.py
        #   冒頭docstring(2)参照）。
        # 【2026-08-12追記：口元への自己接触報酬】構築条件を
        #   「reach_space（既存）または double_touch_bonus!=0.0（既存）または
        #   mouth_touch_bonus!=0.0（新規）」のORへ拡張した（仕様3節）。
        #   既存2条件がFalseかつmouth_touch_bonusも0.0なら、このブロックは
        #   今まで通り一度も実行されない＝既存実験の挙動は1ビットも変わらない。
        if reach_space or cfg.double_touch_bonus != 0.0 or cfg.mouth_touch_bonus != 0.0:
            touch_map = build_touch_map_from_env(env)
            # mouth_touch_bonus!=0.0 のときだけ口元マスクを計算する（仕様3節。
            #   double_touch_bonus単体の既存挙動を1ビットも変えないため、
            #   mouth_touch_bonus=0.0（既定）ならこの計算を一切行わない）。
            mouth_mask = None
            if cfg.mouth_touch_bonus != 0.0:
                mouth_mask = mouth_point_mask(
                    touch_map, env.unwrapped.model, env.unwrapped.touch,
                    x_frac=cfg.mouth_touch_x_frac, z_frac=cfg.mouth_touch_z_frac)
            self.double_touch = DoubleTouchDetector(
                touch_map=touch_map, threshold=cfg.double_touch_threshold,
                touched_names=cfg.double_touch_touched_groups,
                mouth_mask=mouth_mask, mouth_x_frac=cfg.mouth_touch_x_frac,
                mouth_z_frac=cfg.mouth_touch_z_frac)
            self.reward_contributors.append(
                _DoubleTouchBonusContributor(cfg.double_touch_bonus))
            if cfg.mouth_touch_bonus != 0.0:
                # 既存の努力コスト・CAPS減算の"後"に足される、という既存の順序が
                #   そのまま適用される（reward_contributorsの同じリストに積むため）。
                self.reward_contributors.append(
                    _MouthTouchBonusContributor(
                        self.double_touch, cfg.mouth_touch_bonus,
                        mouth_threshold=cfg.mouth_touch_threshold,
                        toucher_threshold=cfg.double_touch_threshold))
            if verbose:
                mouth_info = (f" 口元presenceしきい値={cfg.mouth_touch_threshold}"
                              f" 口元ボーナス={cfg.mouth_touch_bonus}"
                              f" (x_frac={cfg.mouth_touch_x_frac} z_frac={cfg.mouth_touch_z_frac}"
                              f" 口元点数={int(mouth_mask.sum())}/{touch_map.n_points})"
                              if cfg.mouth_touch_bonus != 0.0 else "")
                print(f"[double_touch] 有効：対象部位={cfg.double_touch_touched_groups}"
                      f" しきい値={cfg.double_touch_threshold} ボーナス={cfg.double_touch_bonus}"
                      f"（2026-08-05・全身一般化。既定headのみでは既存実験の挙動は不変）"
                      f"{mouth_info}",
                      flush=True)

    # ------------------------------------------------------------ 読み込み
    def _load(self, path, *, verbose=True):
        def _match(module, sd, tag):
            own = module.state_dict()
            matched = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
            module.load_state_dict(matched, strict=False)
            if verbose:
                print(f"  [{tag}] ロード{len(matched)}/{len(own)}層", flush=True)
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if verbose:
            print(f"続きから学習：{os.path.basename(path)} を読み込み", flush=True)
        # 【⓪・2026-08-31】聞く学習で育った embedding（78〜80行以上）は、構築時の
        #   3行と形が合わず _match で**静かにスキップ**されていた（触覚の読み戻し
        #   忘れ 2026-07-30 と同型。F2-21の鎖30本で毎回白紙に戻っていたのを実測で
        #   発見）。保存時のサイズへ先に伸ばしてから読む。
        _bemb = blob["brain"].get("embedding.weight")
        if _bemb is not None and _bemb.shape[0] > self.brain.embedding.num_embeddings:
            self.brain.resize_embedding(int(_bemb.shape[0]))
        _match(self.brain, blob["brain"], "脳")
        # 【⓪】脳の名簿（トークン↔番号）も退避（_setup_produce が同じ番号で再現する）
        self._pending_brain_vocab = blob.get("brain_vocab")
        self.fusion.insula.load_state_dict(blob["fusion_insula"])
        self.fusion.proprio.load_state_dict(blob["fusion_proprio"])
        self.fusion.vestibular.load_state_dict(blob["fusion_vestibular"])
        # 触覚のエンコーダ。save は保存していたのに**読み戻していなかった**
        #   （2026-07-30 の点検で発覚）。黙って白紙に戻るので、
        #   「続きから学習できている」ように見えて触覚だけ学習しなおしになる。
        if "fusion_touch" in blob:
            if self.fusion.touch is not None:
                _match(self.fusion.touch, blob["fusion_touch"], "触覚")
            else:
                print("注意[load] 保存されたモデルは触覚あり、いまの設定は触覚なしです。"
                      "触覚のエンコーダは読み込みません（taro.touch を確認）", flush=True)
        elif self.fusion.touch is not None:
            print("注意[load] いまの設定は触覚ありですが、保存されたモデルに触覚が"
                  "ありません。触覚のエンコーダは白紙から学習します", flush=True)
        if "emb_proj" in blob:
            print("  [注意] 旧形式のチェックポイント（emb_proj/nat_head が別層）です。"
                  "層の構成が変わったため、その2層は読み込まれません＝学習しなおしになります。",
                  flush=True)
        if self.cfg.cerebellum and "cereb" in blob:
            _match(self.cereb, blob["cereb"], "小脳")
        # progress報酬の内部状態（pe_fast/pe_slow）。2026-08-05に発覚：これを
        #   読み戻していなかったため、続きから学習するたびにpe_fast/pe_slowが
        #   既定init=1.0からゼロスタートし、起動直後のprogressが「自己接触とは
        #   無関係の助走アーティファクト」として見かけ上大きくプラスに振れていた
        #   （progress_breakdown_selfmodel_seed0系実験、累計step500と3500でほぼ
        #   同じ値が再現したことで発覚）。fusion_touch読み戻し漏れ（2026-07-30）
        #   と同じ種類の見落とし。
        if "lp_pe_fast" in blob and "lp_pe_slow" in blob:
            self.lp.pe_fast = blob["lp_pe_fast"]
            self.lp.pe_slow = blob["lp_pe_slow"]
            if verbose:
                print(f"  [progress] pe_fast/pe_slowを保存値から復元"
                      f"（pe_fast={self.lp.pe_fast:.4f}, pe_slow={self.lp.pe_slow:.4f}）",
                      flush=True)
        else:
            print("注意[load] 保存されたモデルにprogress報酬の内部状態(pe_fast/pe_slow)が"
                  "ありません（2026-08-05より前の旧形式のチェックポイント）。"
                  f"既定init={self.lp.pe_fast:.1f}からの助走が入り、続きから学習した直後の"
                  "progressがしばらく大きくプラスに振れます（自己接触とは無関係の"
                  "アーティファクト）。", flush=True)
        # 【2026-08-19・F1-4a】語彙(hearing.vocab / lexicon)の復元。
        #   _load は __init__ から呼ばれるが、_setup_hearing はそれより前
        #   （taro_setup.py内の呼び出し順）に実行済みなので、この時点で
        #   self.hearing/self.lexicon は（hearing有効なら）生成済み。
        if self.hearing is not None:
            if "hearing_vocab" in blob and "lexicon" in blob:
                hv = blob["hearing_vocab"]
                self.hearing.vocab.char2idx = hv["char2idx"]
                self.hearing.vocab.idx2char = hv["idx2char"]
                self.hearing.vocab.size = hv["size"]
                lx = blob["lexicon"]
                self.lexicon.counts = lx["counts"]
                self.lexicon.state_sum = lx["state_sum"]
                self.lexicon.min_len = lx["min_len"]
                # 【分節第2案・2026-09-03】設計_分節（語の切れ目の発見）.md 第2案
                #   第2部「発話まるごとの記録」節。旧保存にはキーが無いので空辞書。
                self.lexicon.utterance_counts = lx.get("utterance_counts", {})
                self.lexicon.end_prob_sum = float(lx.get("end_prob_sum", 0.0))   # 【分節第2案】走行平均も引き継ぐ
                self.lexicon.end_prob_n = int(lx.get("end_prob_n", 0))
                # 【F1-5・2026-08-21】旧blob（mode/protoキー無し）はそのまま
                #   sumモード・proto空辞書として復元される（従来どおり）。
                if "mode" in lx:
                    self.lexicon.mode = lx["mode"]
                # 【F2-12・2026-08-27】設計：F/docs/設計_F2-12_意味を感覚ごとに
                #   分けて持つ.md。channels優先で読み、無ければ旧キー
                #   (state_dim/proto/view_sum/view_n)からvisionチャンネルを
                #   組み立てる（後方互換。旧形式のモデルにはchannelsキーが無い）。
                if "channels" in lx:
                    self.lexicon.channels = lx["channels"]
                else:
                    self.lexicon.channels = {
                        "vision": {
                            "dim": lx["state_dim"],
                            "proto": lx.get("proto", {}),
                            "view_sum": lx.get("view_sum"),
                            "view_n": lx.get("view_n", 0),
                        }
                    }
                if verbose:
                    print(f"  [語彙] 復元：耳={self.hearing.vocab.size}文字"
                          f" 語彙={len(self.lexicon.counts)}語", flush=True)
            else:
                print("注意[load] いまの設定は耳(hearing)ありですが、保存されたモデルに"
                      "語彙(hearing_vocab/lexicon)がありません"
                      "（2026-08-19以前の保存）。語彙は空から開始します。", flush=True)
        # 【F2-1・作業A・2026-08-23】発話小脳（口の動き→音の帳面）の復元。
        #   _load() は Taro.__init__ の"⑦続きから学習する"（_setup_produceより前）
        #   で呼ばれるため、この時点では self.produce_cerebellum はまだ存在しない
        #   （_setup_produce は __init__ の"後"、_load()の"後"に呼ばれる。
        #   taro_setup.py の _setup_produce docstring参照）。ここでは blob の中身を
        #   一時属性へ退避するだけにして、_setup_produce が Cerebellum() を
        #   作った直後にそこへ流し込む。
        # 【発話の動機・2026-08-31】門の退避（_setup_produce が作った直後に流し込む）
        self._pending_speech_gate = blob.get("speech_gate")
        self._pending_visual_projection = blob.get("visual_projection")
        # 【言語海馬・段階1・2026-09-01】visual_projectionと同じ流儀
        #   （_setup_produceがLanguageHippocampusを作った直後に流し込む）。
        self._pending_language_hippocampus = blob.get("language_hippocampus")
        # 【塊レベル層・2026-09-07・仕様_M5_塊レベル層.md §2】visual_projectionと
        #   同じ流儀（_setup_produceがChunkVocab/TaroBrainを作った直後に流し込む）。
        self._pending_chunk_brain = blob.get("chunk_brain")
        self._pending_chunk_vocab = blob.get("chunk_vocab")
        if "produce_cerebellum" in blob:
            self._pending_produce_cerebellum = blob["produce_cerebellum"]
            if verbose:
                pc = blob["produce_cerebellum"]
                print(f"  [発話小脳] 復元予定：forward_map={len(pc['forward_map'])}件"
                      f" inverse_map={len(pc['inverse_map'])}件"
                      f" experience_count={len(pc['experience_count'])}件", flush=True)
        else:
            self._pending_produce_cerebellum = None

    # -------------------------------------------------------- 体が変わったとき
    def on_body_change(self, env):
        """体を作り直したら、触覚の地図を差し替える。**学習した重みは保つ。**

        【なぜ要るか、2026-07-31】触覚センサの点は成長で増える（0ヶ月4,824 →
        4ヶ月9,804次元）。SomatosensoryCortex は「点→部位」の対応表と
        各点の位置を持っているので、これを新しい体のものに入れ替える必要がある。

        注意：入れ替えを忘れても**例外は出ない**。配列が長くなる方向の変化では
          古いインデックスが範囲内に収まり、静かに別の部位を読む
          （落とし穴チェックリスト 項86）。ここを通す設計にしてあるのはそのため。
        """
        # 【2026-08-05追記：全身一般化】self.double_touch は taro自身の
        #   fusion.touch/target_fusion.touch とは別の、自前のSomatosensoryCortex
        #   インスタンスを持つ（double_touch.py冒頭docstring(2)参照）。
        #   下の早期return（fusion.touchがNoneなら戻る）は既存の2インスタンス
        #   専用のガードであり、これより"後"にrebuild呼び出しを置くと
        #   cfg.touch=False のシーンでは一度も実行されない＝成長後も古い触覚地図を
        #   参照し続け、しかも例外が出ない（落とし穴チェックリスト項86と同型）。
        #   したがって、この早期returnより**前**に置く（設計1-7節・実装
        #   ノウハウ2026-08-05項の必須要件）。
        if self.double_touch is not None:
            tm_dt = build_touch_map_from_env(env)
            # 【2026-08-12追記】model/touch を常に渡す。口元マスクを使っていない
            #   （mouth_touch_bonus=0.0）場合はDoubleTouchDetector.rebuild内部で
            #   使われず無害。口元マスクを使っている場合はこれが無いと
            #   古い点数・古い並びのマスクを参照し続け、例外を出さずに別の点を読む
            #   （落とし穴チェックリスト項86）。
            self.double_touch.rebuild(
                tm_dt, model=env.unwrapped.model, touch=env.unwrapped.touch)
        # 【2026-08-11・新しい駆動モジュール】体を作り直す実験（cfg.grows）では
        #   moment_1/moment_2・関節indexがenvごとに変わりうるため、体が変わるたびに
        #   組み立て直す（_setup_reflex_common のdocstring参照。既定"cpg"では
        #   self.cfg.spinal_drive_mode!="reflex_common" のため即Falseで戻り、
        #   何も実行しない＝既存の成長実験の挙動は1ビットも変わらない）。
        self.reflex_common_active = _setup_reflex_common(self, self.cfg, env, verbose=False)
        # 【2026-08-15・座位保持の学習】既定OFF（cfg.posture_reflex/righting_reflex=False）
        #   では即returnして何もしない＝既存の成長実験の挙動は1ビットも変わらない。
        _setup_postural_gate(self, self.cfg, env, verbose=False)
        _setup_righting_damper(self, self.cfg, env, verbose=False)
        if self.fusion.touch is None or not hasattr(self.fusion.touch, "rebuild"):
            return None
        tm = build_touch_map_from_env(env)
        self.fusion.touch.rebuild(tm)
        self.target_fusion.touch.rebuild(tm)
        # 【2026-08-13】触覚の順応も体が変わったら全点リセット（3-6節・設計
        #   9節「成長時に順応の状態をリセットする」）。touch_adaptationが有効なら
        #   必ずfusion.touchも有効（同じ条件cfg.somatosensory and cfg.touchで
        #   ゲートされている）ので、このifは早期returnの後でも安全に届く。
        if self.touch_adaptation is not None:
            self.touch_adaptation.rebuild(tm.n_points)
        # ---- 手先位置の目標表現（案C）：腕の索引を引き直す ------------------
        # 【なぜ、2026-08-02】腕の関節構成は月齢で不変と実装確認済みだが、
        #   「不変のはず」を仮定にせず、体を作り直すたびに必ず引き直す
        #   （落とし穴チェックリスト 項86。q_touch側は名前で毎回引くのでキャッシュしない）。
        if self.cfg.goal_babbling and self.cfg.goal_space == "reach_self":
            old_idx = list(self.arm_map.idx) if self.arm_map is not None else None
            self.arm_map = build_arm_proprio_map_from_env(env, side=self.cfg.reach_arm_side)
            if old_idx is not None and self.arm_map.idx != old_idx:
                print(f"注意[goal_babbling] 体を作り直したら腕の固有感覚indexが変わった"
                      f"（{old_idx} → {self.arm_map.idx}）。想定外なので確認すること。",
                      flush=True)
            missing = [nm for nm in self.reach_touch_groups if nm not in tm.group_names]
            if missing:
                raise AssertionError(
                    f"体を作り直したら reach_goal の対象部位が消えた: {missing}\n"
                    f"  いまの部位: {tm.group_names}")
        return tm

    # -------------------------------------------------------- 触覚の順応
    def apply_touch_adaptation(self, obs, is_reset=False):
        """触覚の順応を1回だけ進め、obs["touch_percept"]を追加した新しいdictを返す。

        【なぜ、2026-08-13】advance()は「新しい物理観測が生まれた瞬間」にだけ
          呼ぶ必要がある。呼び出し元は run/trainer.py の reset_state()・step_k()
          （K回のenv.stepループを終えた最後のoにだけ）の2箇所に厳密に限定する。
          fusion.py・encode_target 側では読むだけで、advance()は絶対に呼ばない
          （設計2節で確認済みの罠：同一物理観測に対しtarget_fusion.touchが
          最大3回呼ばれるが、そこでadvance()を呼ぶと順応が最大3倍の速さで進む）。

        Args:
            obs: 環境から返された観測（dict）。obs["touch"]は書き換えない
                （元のdictも破壊的に書き換えない。他のコードが元のobsへの
                参照を保持している可能性があるため、新しいdictを作って返す）。
            is_reset: env.reset()直後の呼び出しならTrue。
                cfg.touch_adapt_reset_on_episode=True のときだけ、
                advance()の前に順応の状態（ゲイン）を1.0へ戻す
                （既定Falseでは何もしない＝エピソードをまたいで持続させる。
                2026-08-13、ユーザー決定）。
        """
        if self.touch_adaptation is None:
            return obs
        if is_reset and self.cfg.touch_adapt_reset_on_episode:
            self.touch_adaptation.reset_episode()
        self.touch_adaptation.advance(obs["touch"])
        new_obs = dict(obs)
        new_obs["touch_percept"] = self.touch_adaptation.adapted()
        return new_obs

    # -------------------------------------------------------- 予測の対象
    def encode_target(self, obs):
        """予測する対象を作る（元 `ln_prop`）。既定は固有感覚のみ。

        2つの設計判断（詳細は E/scripts/e_target.py）：
          (1) **凍結した別インスタンス**のエンコーダを使う（RND式）。学習中のエンコーダを
              正解側に使うと「出力を平坦にすれば当たる」抜け道で崩壊する。
          (2) **固有感覚と視覚を別々に layer_norm** してから連結する。全体を一度に
              正規化すると621次元が平均・分散を支配して64次元が埋もれる（3回踏んだ罠）。
        """
        ln = torch.nn.functional.layer_norm
        cfg = self.cfg
        v = to_tensor(obs["observation"])
        # 【B案】somatosensory=True のときは生の触覚を混ぜず、S1相当のembedをブロックで足す
        # 【2026-08-13】touch_adaptationが有効なら順応後の値(obs["touch_percept"])を
        #   使う。無効（既定）ならキー自体が無いのでobs["touch"]にフォールバック
        #   ＝1ビットも変わらない後方互換（仕様5節、片方だけ直すと非対称になるので
        #   下のtouch_embedブロックとも揃える）。
        if cfg.touch and cfg.touch_mode == "target" and not cfg.somatosensory:
            v = torch.cat([v, to_tensor(obs.get("touch_percept", obs["touch"]))])
        parts = [ln(v, v.shape).detach()]
        names = ["prop"]
        f = self.target_fusion
        if cfg.somatosensory and cfg.touch and cfg.touch_mode == "target":
            if getattr(f, "touch", None) is not None and "touch" in obs:
                with torch.no_grad():
                    e = f.touch(to_tensor(obs.get("touch_percept", obs["touch"])))
                parts.append(ln(e, e.shape)); names.append("touch_embed")
        if cfg.target_has_vision:
            with torch.no_grad():             # 正解側は勾配を流さない（RND式）
                # 内受容は入れない（2026-07-20 の文献調査による判断）：人間は内受容の
                #   予測誤差を自律反射（心拍・血管）で解消するが、太郎にはその出力が無い
                #   ＝**誤差を減らす手段が構造的に存在しない**ので progress報酬が生まれない。
                if cfg.target_has_all:
                    for nm, enc, key in (("vest", getattr(f, "vestibular", None), "vestibular"),
                                         ("touch", getattr(f, "touch", None), "touch")):
                        if enc is None or key not in obs:
                            continue
                        # 【なぜ、2026-08-03バグ修正】somatosensory かつ touch かつ
                        #   touch_mode=="target" のとき、上のブロック（326-330行付近）で
                        #   既に同じ f.touch エンコーダを同じ obs["touch"] に適用し
                        #   "touch_embed" として parts/names に追加済み。ここでまた
                        #   "touch" として追加すると、数値的に同一の埋め込みが2つの
                        #   別ブロックとして block_pe に渡り、触覚の予測誤差だけ構造的に
                        #   2倍の重みで progress の計算に効いてしまう（調査報告
                        #   2026-08-03 1-4節）。既に追加済みならここでは足さない。
                        #   既定設定（touch=False）ではこの分岐に到達しないため、
                        #   既存実験の数値には影響しない。
                        if nm == "touch" and "touch_embed" in names:
                            continue
                        e = enc(to_tensor(obs[key]))
                        parts.append(ln(e, e.shape)); names.append(nm)
                if getattr(f, "vision", None) is not None and "eye_left" in obs:
                    e = f.vision(obs["eye_left"], obs["eye_right"])
                    parts.append(ln(e, e.shape)); names.append("vision")
        if self.blocks is None or len(self.blocks) != len(parts):
            self.blocks, o = [], 0
            for nm, p in zip(names, parts):
                self.blocks.append((o, o + int(p.shape[-1]), nm))
                o += int(p.shape[-1])
        return torch.cat(parts, dim=-1).detach() if len(parts) > 1 else parts[0]

    # -------------------------------------------------- 手先位置の目標表現（案C）
    def encode_reach_goal(self, obs):
        """目標ベクトル g（22次元）＝腕の固有感覚(7)＋自己接触(15)。座標は作らない。

        【なぜ encode_target と別の関数か】`encode_target`（予測誤差の"正解"を作る
        既存関数）は一切変更しない、という仕様の要求を厳密に守るため
        （設計の統合判断「決定3」）。中身の設計は同じ思想（凍結した別インスタンス
        から取る＝RND式、勾配は流さない）だが、対象・次元・用途が別物。

        q_arm（7次元）：腕（cfg.reach_arm_side）の肩3・肘1・手首3の関節角度を、
          可動域(jnt_range、固定の身体定数)で[-1,1]に線形正規化したもの。
          座標変換は一切挟まない（ユーザー確認済み・2026-08-02仕様2節 論点1）。
        q_touch（15次元＝3部位×5）：head（顔+口相当）・chest（胸）・
          opposite_palm（反対の手）の3部位。SomatosensoryCortex.part_features()の
          [有無,強さ,重心x,y,z]を、**凍結した target_fusion 側**から取る
          （予測側と正解側が同じ学習中の層だと崩壊する、目標Cで実際に踏んだ罠）。
        部位のインデックスは**毎回「名前」で検索する**（整数indexをキャッシュしない。
        rebuild後にグループの並びが変わったとき、静かに別の部位を読まないため）。

        呼ぶ前提：self.arm_map / self.reach_touch_groups が構築済みであること
          （cfg.goal_babbling and cfg.goal_space=="reach_self" のときだけ Taro.__init__
          が構築する。それ以外のときにこのメソッドを呼ぶのは呼び出し側の誤り）。
        """
        v = to_tensor(obs["observation"])
        lo = torch.as_tensor(self.arm_map.lo, dtype=v.dtype)
        hi = torch.as_tensor(self.arm_map.hi, dtype=v.dtype)
        raw = v[self.arm_map.idx]
        q_arm = torch.clamp(2.0 * (raw - lo) / (hi - lo) - 1.0, -1.05, 1.05)
        f = self.target_fusion
        with torch.no_grad():
            feat = f.touch.part_features(to_tensor(obs["touch"]))     # (G, 5)
        names = f.touch.group_names
        parts = [q_arm]
        for nm in self.reach_touch_groups:
            gi = names.index(nm)
            parts.append(feat[gi])
        return torch.cat(parts, dim=-1).detach()

    def dummy_reach_goal(self, like=None):
        """陰性対照：実在の目標と似た値域だが、意味的な相関の無いランダム目標を作る。

        仕様の確定3（陰性対照を必ず入れる）の実装。太郎の外から与える測定器で
        あり、太郎の機能ではない（検証の落とし穴チェックリスト 項28の分類）。
        q_arm は可動域内(=[-1,1])の一様乱数、q_touch は部位ごとに「有無」を確率0.3で
        決めてから、有無=1のときだけ強さ・重心を一様乱数でサンプルする（完全な
        無構造乱数だと「可動域外テスト」と見分けがつかなくなるため、"もっともらしい
        値域だが意味的な相関はない"という設計にする。設計Q6を踏襲）。
        現在地（gclp、実測値）は一切汚さない。軌道の**終点だけ**をこれに差し替える。
        """
        q_arm = torch.rand(7) * 2.0 - 1.0
        parts = [q_arm]
        for _ in self.reach_touch_groups:
            presence = (torch.rand(1) < 0.3).float()
            if presence.item() > 0.5:
                strength = torch.rand(1) * 2.0 - 1.0
                centroid = torch.rand(3) * 2.0 - 1.0
            else:
                strength = torch.zeros(1)
                centroid = torch.zeros(3)
            parts.append(torch.cat([presence, strength, centroid]))
        return torch.cat(parts, dim=-1)

    def block_pe(self, pred, target, blocks=None):
        """予測誤差＝**ブロックごとに平均してから足す**（次元数の影響を除く）。

        【なぜ】従来は連結したベクトル全体を1回で平均していたので、寄与が次元数比で
        決まっていた（固有感覚621 + 視覚64 → 視覚の寄与は 9.3%）。これは「視覚が
        重要でない」という判断ではなく、**次元数という無関係な量**が重みを決めている状態。
        同じ罠を目標D0とE1で計3回踏んだ。
        根拠＝Ohata & Tani 2020（`1/(2Rp)`）、Idei et al. 2025（1,150倍差を正規化のみで処理）、
        Ichiwara & Ogata 2022（`1/(H·W·C)`）。［参考文献リスト §目標E-17］
        注意：(1+λ_v) で割るのは全体のスケールを保つため（割らないと「視覚を足した効果」と
          「学習率が実質変わった効果」が混ざる＝交絡）。

        引数 blocks（既定 None）：省略時は self.blocks（既存の予測対象のブロック）を
          使う＝既存呼び出し `t.block_pe(pred, nlp)` は無変更で従来どおり動く。
          2026-08-02、手先位置の目標表現（案C）で q_arm(7次元)とq_touch(15次元)を
          同じ理由（次元数の希釈、落とし穴 項11）で分けて合成するために追加した。
        """
        use_blocks = self.blocks if blocks is None else blocks
        if not use_blocks or len(use_blocks) <= 1:
            return mse(pred, target)          # 固有感覚のみ＝従来と完全に同一
        tot, wsum = 0.0, 0.0
        for (s, e, nm) in use_blocks:
            lam = self.cfg.lam_v if nm == "vision" else 1.0
            tot = tot + lam * mse(pred[..., s:e], target[..., s:e])
            wsum += lam
        return tot / max(wsum, 1e-9)

    # ------------------------------------------------------------ 脳の操作
    def infer_latent(self, sv, prev_a, cf, h):
        """[感覚, 前回の行動] → 内部表現 z（元 `zc`）。太郎の infer_latent を呼ぶだけ。"""
        return self.brain.infer_latent(sv, prev_a, cf, h)

    def act_mean(self, z):
        """決定的な行動平均（評価・agency用）。太郎の act_deterministic を呼ぶだけ。"""
        return self.brain.act_deterministic(
            z, cerebellum=(self.cereb if self.cfg.cerebellum else None))

    def motor_drive(self, z):
        """行動の平均と揺らぎ。運動野の精密制御＋小脳の自動化ブレンド。"""
        return self.brain.motor_drive(
            z, self.ne.get_ne_level(),
            cerebellum=(self.cereb if self.cfg.cerebellum else None))

    def infer_goal_action(self, z, clp, init_mean, g, n_steps=15, lr_inf=0.1):
        """目標指向：凍結した順モデルを反転し、望む感覚 g に届く行動を推論する。

        注意：2026-07-30 の実測でこの機構は**有害**と判明（config.py の注記参照）。
          原典（Rolf, Steil & Gienger 2010）は目標を**手先位置の低次元**に取るが、
          ここは固有感覚621次元まるごとを目標にしている＝別物。
        """
        target = (g - clp).detach()
        raw = torch.atanh(torch.clamp(init_mean, -0.999, 0.999)).detach().requires_grad_(True)
        opt = torch.optim.Adam([raw], lr=lr_inf)
        for _ in range(n_steps):
            opt.zero_grad()
            ((self.nat_head(torch.cat([z, torch.tanh(raw)], dim=-1)) - target) ** 2).mean().backward()
            opt.step()
        return torch.tanh(raw).detach()

    def infer_reach_goal_action(self, z, gclp, init_mean, g, n_steps=15, lr_inf=0.1):
        """目標指向（手先位置の目標表現・案C版）：凍結した reach_head を反転し、
        望む reach_goal g に届く行動を推論する。

        `infer_goal_action` とは別の新設メソッド（既存は1文字も変更しない、
        設計の統合判断「決定3」）。解き方（distal teacher、凍結ヘッドをAdamで
        n_steps回逆算）は全く同じで、対象を nat_head→reach_head に差し替えただけ。

        注意：段階1ではこの解き方（distal teacher）を一時的に残す。Rolfが名指しで
          批判する方式だが、`doc/やることリスト.md` で段階1の診断目的として
          既に合意済み。段階2で局所線形マップに置き換える前提。
        """
        target = (g - gclp).detach()
        raw = torch.atanh(torch.clamp(init_mean, -0.999, 0.999)).detach().requires_grad_(True)
        opt = torch.optim.Adam([raw], lr=lr_inf)
        for _ in range(n_steps):
            opt.zero_grad()
            ((self.reach_head(z, torch.tanh(raw)) - target) ** 2).mean().backward()
            opt.step()
        return torch.tanh(raw).detach()

    def init_state(self, obs):
        """学習ループが持つ「いまの状態」の初期値。"""
        return {"obs": obs, "hidden": self.brain.init_motor_hidden(),
                "prev_a": torch.zeros(self.n_act)}

    def save(self, path, *, extra=None):
        """確立した自己モデルを保存する。条件も一緒に入れる。

        注意：条件を入れないと、あとから「どの設定で保存されたか」が分からない
          （2026-07-25 に感度分析のモデルが同一だと分かっても原因を追えなかった）。
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # 【2026-08-21・上書き事故の再発防止】保存先に既存ファイルがあれば、
        #   上書きの前に <元名>.prev.pt へ退避する（1世代のみ・毎回入れ替え）。
        #   経緯：実験JSONのsave先を残したまま再走した際、過去の合格を出した
        #   モデル2本を上書きし、git管理外のため復元不能になった（F日誌
        #   2026-08-21追記12）。モデル保存系の事故は累計4件目のため、
        #   文書でなく機械で防ぐ（CLAUDE.md自浄ルール）。
        if os.path.exists(path):
            backup = path + ".prev.pt"
            try:
                if os.path.exists(backup):
                    os.remove(backup)
                os.replace(path, backup)
                print(f"注意[save] 既存のモデルを退避しました: {backup}")
            except OSError as e:
                print(f"注意[save] 既存モデルの退避に失敗（続行します）: {e}")
        blob = {"brain": self.brain.state_dict(),
                "fusion_insula": self.fusion.insula.state_dict(),
                "fusion_proprio": self.fusion.proprio.state_dict(),
                "fusion_vestibular": self.fusion.vestibular.state_dict(),
                "cereb": self.cereb.state_dict(),
                "lp_pe_fast": self.lp.pe_fast,
                "lp_pe_slow": self.lp.pe_slow,
                "config": dict(self.cfg.as_dict(),
                               sdim=self.sdim, prop_dim=self.prop_dim,
                               out_dim=self.out_dim, n_act=self.n_act,
                               **(extra or {}))}
        if self.fusion.touch is not None:
            blob["fusion_touch"] = self.fusion.touch.state_dict()
        # 【2026-08-19・F1-4a】語彙の保存。耳(hearing)が有効なときだけ足す
        #   （既定hearing=Falseでは何も足さない＝既存モデルとバイト互換）。
        #   視覚未保存事件(2026-08-19発覚)・語彙未保存(同日発覚)に続く同型バグ
        #   （「センサ/学習器を足したのに保存を足し忘れる」）を、下のassertで
        #   機械的に止める。
        if self.hearing is not None:
            vocab = self.hearing.vocab
            blob["hearing_vocab"] = {"char2idx": vocab.char2idx,
                                      "idx2char": vocab.idx2char,
                                      "size": vocab.size}
            blob["lexicon"] = {"counts": self.lexicon.counts,
                                "state_sum": self.lexicon.state_sum,
                                "state_dim": self.lexicon.state_dim,
                                "min_len": self.lexicon.min_len,
                                # 【分節第2案・2026-09-03】設計_分節（語の切れ目の
                                #   発見）.md 第2案 第2部「発話まるごとの記録」節。
                                "utterance_counts": self.lexicon.utterance_counts,
                                "end_prob_sum": float(getattr(self.lexicon, "end_prob_sum", 0.0)),
                                "end_prob_n": int(getattr(self.lexicon, "end_prob_n", 0))}
            # 【F1-5・2026-08-21】mode/protoは既定sumモードでは追加しない
            #   （旧blobとバイト互換を維持する。落とし穴メモリ「新キーは条件付きで」
            #   と同じ流儀。設計：F/docs/設計_F1-5_連合器の対照学習化.md）。
            if self.lexicon.mode == "contrast":
                blob["lexicon"]["mode"] = self.lexicon.mode
                blob["lexicon"]["proto"] = self.lexicon.proto
                # 【F2-8・2026-08-25】「見慣れた景色」の平均も保存する。
                #   これを保存しないと、続きから学習したとき逆引きの引き算が
                #   ゼロからやり直しになる（＝最初の数十歩は引き算が効かない）。
                #   視覚未保存・語彙未保存・帳面未保存に続く同型事故を作らない。
                blob["lexicon"]["view_sum"] = self.lexicon.view_sum
                blob["lexicon"]["view_n"] = self.lexicon.view_n
                # 【F2-12・2026-08-27】設計：F/docs/設計_F2-12_意味を感覚ごとに
                #   分けて持つ.md。旧キー(proto/view_sum/view_n)と新キー
                #   (channels)を両方書く。旧キーは上の3行と同じ中身の別名
                #   （channels["vision"]の中身と一致）なので、旧コードで読んでも
                #   新コードで読んでも同じ結果になる。
                blob["lexicon"]["channels"] = self.lexicon.channels
        assert self.hearing is None or ("hearing_vocab" in blob and "lexicon" in blob), (
            "耳(hearing)が有効なのに語彙(vocab/lexicon)がblobに入っていない。"
            "視覚未保存事件(2026-08-19発覚)・語彙未保存(同日発覚)に続く同型バグを"
            "機械で止める（このassertを消さないこと）。")
        # 【F2-1・作業A・2026-08-23】発話小脳（口の動き→音の帳面）の保存。
        #   produce_cerebellumがNone（既定・cfg.produce未設定）のときは1キーも
        #   足さない＝既存モデルとバイト互換（hearing_vocab/lexiconと同じ
        #   「既定OFFでは足さない」流儀、上のブロック参照）。
        #   forward_map/inverse_map/experience_countは素のdict（state_dict()は
        #   持たない）なのでそのまま入れる。
        # 【作業A・2026-08-24】cfg.produce=OFFの実験でも、_load()で読み込んだ
        #   モデルが帳面を持っていれば（_pending_produce_cerebellum）、それを
        #   素通しして保存する。produce_cerebellumインスタンス自体が無い（OFF）
        #   からといって帳面まで捨てない（作業A本体、上のOFF分岐のコメント参照）。
        # 【発話の動機・2026-08-31】門の保存。無ければ1キーも足さない
        #   （発話小脳と同じ流儀）。読み込んだモデルが門を持ち、今回OFFでも素通し。
        # 【⓪・2026-08-31】脳の名簿の保存（produce有効時のみ。無ければキーを足さない）
        if getattr(self, "produce_vocab", None) is not None:
            blob["brain_vocab"] = {"char2idx": dict(self.produce_vocab.char2idx)}
        # 【V1】視覚投射の保存（無ければキーを足さない流儀）
        _pending_vp = getattr(self, "_pending_visual_projection", None)
        if getattr(self, "_visual_projection", None) is not None:
            blob["visual_projection"] = self._visual_projection.state_dict()
        elif _pending_vp is not None:
            blob["visual_projection"] = _pending_vp
        # 【言語海馬・段階1・2026-09-01】無ければキーを足さない流儀
        #   （visual_projectionと同じ。既定OFFではblobはバイト互換のまま）。
        _pending_lh = getattr(self, "_pending_language_hippocampus", None)
        if getattr(self, "language_hippocampus", None) is not None:
            blob["language_hippocampus"] = self.language_hippocampus.state_dict()
        elif _pending_lh is not None:
            blob["language_hippocampus"] = _pending_lh
        # 【塊レベル層・2026-09-07・仕様_M5_塊レベル層.md §2】chunk_level真の
        #   ときだけキーを足す（無ければキーを足さない流儀。visual_projection・
        #   language_hippocampusと同じ）。
        _pending_cb = getattr(self, "_pending_chunk_brain", None)
        if getattr(self, "chunk_brain", None) is not None:
            blob["chunk_brain"] = self.chunk_brain.state_dict()
        elif _pending_cb is not None:
            blob["chunk_brain"] = _pending_cb
        _pending_cv = getattr(self, "_pending_chunk_vocab", None)
        if getattr(self, "chunk_vocab", None) is not None:
            blob["chunk_vocab"] = self.chunk_vocab.state_dict()
        elif _pending_cv is not None:
            blob["chunk_vocab"] = _pending_cv
        _pending_sg = getattr(self, "_pending_speech_gate", None)
        if getattr(self, "_speech_gate", None) is not None:
            blob["speech_gate"] = self._speech_gate.state()
        elif _pending_sg is not None:
            blob["speech_gate"] = _pending_sg
        _pending_pc = getattr(self, "_pending_produce_cerebellum", None)
        if self.produce_cerebellum is not None:
            pc = self.produce_cerebellum
            blob["produce_cerebellum"] = {
                "forward_map": pc.forward_map,
                "inverse_map": pc.inverse_map,
                "experience_count": pc.experience_count,
            }
        elif _pending_pc is not None:
            blob["produce_cerebellum"] = {
                "forward_map": _pending_pc["forward_map"],
                "inverse_map": _pending_pc["inverse_map"],
                "experience_count": _pending_pc["experience_count"],
            }
        assert (self.produce_cerebellum is None and _pending_pc is None) or (
            "produce_cerebellum" in blob), (
            "発話小脳(produce_cerebellum)を持っている（有効化中、または読み込んだ"
            "モデルが帳面を保持中）のに帳面がblobに入っていない。"
            "視覚未保存事件(2026-08-19発覚)・語彙未保存(同日発覚)に続く同型バグ"
            "（今回で3回目）を機械で止める（このassertを消さないこと）。")
        torch.save(blob, path)
        print(f"SAVED MODEL {path}", flush=True)
