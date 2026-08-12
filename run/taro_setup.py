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
from taro_brain_motor import TaroBrainWithMotor                 # noqa: E402
from basal_ganglia import TaroLearner                           # noqa: E402
from dopamine import Dopamine                                   # noqa: E402
from locus_coeruleus import LocusCoeruleus                      # noqa: E402
from developmental_clock import DevelopmentalClock              # noqa: E402
from cerebellum_motor import MotorCerebellum                    # noqa: E402
from learning_progress import LearningProgress                  # noqa: E402
from homeostatic_scaling import HomeostaticScaling              # noqa: E402
from test_phase8_motor_learning import CombinedParams, rescale_action, to_tensor  # noqa: E402
# 手先位置の目標表現（案C）。既定（goal_space != "reach_self"）では一度も使われない
#   （Taro.__init__ 内で条件付きに構築する。設計の統合判断「決定1」）。
from proprioceptive_map import build_arm_proprio_map_from_env, _opposite_side  # noqa: E402
from goal_babbling.reach_goal_head import ReachGoalHead          # noqa: E402
# 頭へのダブルタッチを報酬に直結する（2026-08-03）。既定（reach_space無効）では
#   一度も使われない（Taro.__init__ 内で条件付きに構築する。決定1と同じパターン）。
from double_touch import DoubleTouchDetector                     # noqa: E402

mse = torch.nn.functional.mse_loss

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
        # 注意：シーンを使う場合、環境は**常に**視覚を持つ（e_scene が vision_params を渡す）。
        #   vision=False は「脳が視覚を無視する」アブレーションになる（環境は変わらない）。
        vres = 0
        if cfg.vision:
            from e_toy_env import VISION_RES
            vres = VISION_RES

        # ---- 体性感覚系（触覚ONのときだけ）----------------------------------
        touch_map = None
        if cfg.somatosensory and cfg.touch:
            touch_map = build_touch_map_from_env(env)
            assert touch_map.total_dim == touch_dim, \
                f"触覚の地図{touch_map.total_dim} != 観測{touch_dim}"
            if verbose:
                print(f"[体性感覚系] SomatosensoryCortex 有効：部位数="
                      f"{len(touch_map.group_names)} 触覚総次元={touch_dim}", flush=True)

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
                                        n_actuators=self.n_act, proprio_dim=self.out_dim)
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
        if reach_space or cfg.double_touch_bonus != 0.0:
            touch_map = build_touch_map_from_env(env)
            self.double_touch = DoubleTouchDetector(
                touch_map=touch_map, threshold=cfg.double_touch_threshold,
                touched_names=cfg.double_touch_touched_groups)
            self.reward_contributors.append(
                _DoubleTouchBonusContributor(cfg.double_touch_bonus))
            if verbose:
                print(f"[double_touch] 有効：対象部位={cfg.double_touch_touched_groups}"
                      f" しきい値={cfg.double_touch_threshold} ボーナス={cfg.double_touch_bonus}"
                      f"（2026-08-05・全身一般化。既定headのみでは既存実験の挙動は不変）",
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
        _match(self.brain, blob["brain"], "脳")
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
            self.double_touch.rebuild(tm_dt)
        # 【2026-08-11・新しい駆動モジュール】体を作り直す実験（cfg.grows）では
        #   moment_1/moment_2・関節indexがenvごとに変わりうるため、体が変わるたびに
        #   組み立て直す（_setup_reflex_common のdocstring参照。既定"cpg"では
        #   self.cfg.spinal_drive_mode!="reflex_common" のため即Falseで戻り、
        #   何も実行しない＝既存の成長実験の挙動は1ビットも変わらない）。
        self.reflex_common_active = _setup_reflex_common(self, self.cfg, env, verbose=False)
        if self.fusion.touch is None or not hasattr(self.fusion.touch, "rebuild"):
            return None
        tm = build_touch_map_from_env(env)
        self.fusion.touch.rebuild(tm)
        self.target_fusion.touch.rebuild(tm)
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
        if cfg.touch and cfg.touch_mode == "target" and not cfg.somatosensory:
            v = torch.cat([v, to_tensor(obs["touch"])])
        parts = [ln(v, v.shape).detach()]
        names = ["prop"]
        f = self.target_fusion
        if cfg.somatosensory and cfg.touch and cfg.touch_mode == "target":
            if getattr(f, "touch", None) is not None and "touch" in obs:
                with torch.no_grad():
                    e = f.touch(to_tensor(obs["touch"]))
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
        torch.save(blob, path)
        print(f"SAVED MODEL {path}", flush=True)
