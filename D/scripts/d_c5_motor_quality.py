"""taro-C5：運動の質(なめらかさ)を、クリーンな仰向け環境で測る／見るハーネス。

【なぜ】太郎の運動が非人間的（Prechtl's GMAの基準ではカクつき過大＝異常寄り）と判明した。
最初の実験①（活性化ダイナミクス）はD視覚環境で試したが、交絡（倒れた姿勢・ベータ・視覚
レンダリング）が多すぎ、指標（眼球コマ間差分）も100ms間隔で35msの効果を検出できず判定不能
だった。ここでは**視覚なし・ベータなし・仰向け**のクリーンな環境で、GMAの"jerkiness"に直接
対応する**ジャーク（＝加速度の時間変化率）**を物理ステップ解像度で測る。これなら①の効果も
原理的に見える。

【測るもの】
  ・mean|jerk|：関節角加速度(qacc)の時間微分の絶対値平均。小さいほどなめらか（Flash&Hogan
    1985の最小ジャークの発想＝ヒトの滑らかな運動はジャークを最小化する）。
  ・境界ジャーク vs 内部ジャーク：1ティック(K=100)の「切り替わり目」と「保持中」でジャークを
    分けて集計。スナップ＆ホールドなら境界で跳ねるはず。①はこの境界の跳ねを抑えるのが狙い。
  ・per-tick 行動変化量：毎ティック行動がどれだけ"ジャンプ"するか（スナップの大きさ）。

【変えないもの】脳・方策・1秒に1回の判断。ACTION_SCALEのようなD側の後付けは使わない
（＝Cで実際に学習した方策そのものの運動を、素で測る）。

使い方:
  python d_c5_motor_quality.py view off          # 見る（従来トルク SpringDamperModel）
  python d_c5_motor_quality.py view on           # 見る（活性化ダイナミクス SmoothTorqueModel）
  python d_c5_motor_quality.py measure off [n]   # 測る（ヘッドレス, n=60ティック既定, 従来）
  python d_c5_motor_quality.py measure on  [n]   # 測る（活性化ダイナミクス）
  末尾に babble を付けると探索ノイズ(運動性喃語)込みで動かす（既定は決定的な方策平均）。

  【taro-C6 Step1】環境変数 C5_CTRL_M で制御の刻みを変える（既定=100=従来の1秒保持）：
    C5_CTRL_M=10 python d_c5_motor_quality.py view off      # 0.1秒ごとに感覚を見て出し直す＝速い連続制御
    C5_CTRL_M=10 python d_c5_motor_quality.py measure off 200
  ＝再学習なしで"実行だけ"速い制御に変え、jerkが下がるか（＝速い制御はなめらかにするか）を
  学習の交絡なしで測る（C4未実施の角度）。C5_CKPTで測るモデルを差替（推奨=c5_progress_seed0.pt）。
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import mujoco

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
torch.set_num_threads(4)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "taro_core"))
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "C", "scripts"))  # fusion再エクスポート等
import paths
paths.setup_brain_path()
sys.path.insert(0, os.path.join(paths.SRC, "body"))   # smooth_actuation
sys.path.insert(0, paths.MIMO_DIR)

import mujoco.viewer
from hybrid_env import HybridEnv
from fusion import MinimalFusion, to_tensor
from taro_brain_motor import TaroBrainWithMotor
from cerebellum_motor import MotorCerebellum
from test_phase8_motor_learning import rescale_action
from d_supine_env import SupineMimoEnv
from mimoActuation.actuation import SpringDamperModel
from smooth_actuation import SmoothTorqueModel
from spinal_cord.cpg import CPG, ColoredNoiseGenerator
from spinal_cord.grasp_reflex import GraspReflex
from brainstem.atnr import ATNR
from corticospinal import project as corticospinal_project

# 既定は目標Cの学習済みモデル。C5_CKPT環境変数で別モデル（例：新生児＋努力コストで再学習した版）に差替可。
CKPT = os.environ.get("C5_CKPT",
                      os.path.join(_HERE, os.pardir, os.pardir, "C", "models", "c_pred_abs_seed0.pt"))
K = 100
# 【taro-C6 Step1】制御の刻み。C5_CTRL_M=10 で「毎10tick(0.1秒)ごとに感覚を見て行動を出し直す」
# ＝速い連続制御（閉ループ）。既定=K=100＝従来の1秒保持と完全に同一。再学習せず"実行だけ"変える
# ＝「速い制御は運動をなめらかにするか」を学習の交絡なしで測る（C4未実施の角度）。
CTRL_M = int(os.environ.get("C5_CTRL_M", str(K)))
# 体の月齢（成長モジュール）。C5_AGE=0 で新生児。空=18ヶ月児（従来）。
AGE = float(os.environ["C5_AGE"]) if os.environ.get("C5_AGE") else None
# 【目標E1】おもちゃ（押すと動く随伴対象）のある仰向け環境で見る/測る。E_TOY=1でopt-in。
# 既定OFF＝従来のD/C5と1バイトも変わらない。おもちゃは観測にも行動にも入らない（予測対象は
# 固有感覚のまま）ので、**C5の学習済みモデルをそのまま実行できる**＝再学習なしで目視できる。
E_TOY = os.environ.get("E_TOY", "0") == "1"

# 【目標E ①②④】運動性喃語の生成方式。既定は従来と完全同一（白色・w_mean=1.0・1秒ホールド）。
# 詳細は E/docs/研究日誌.md 続き10（4原因の仕分けと実装雛形）。
E_NOISE = os.environ.get("E_NOISE", "white")               # "white"(従来)|"colored"(②色付き1/f^β)
E_BETA = float(os.environ.get("E_BETA", "0.8"))            # ②色付き度。0.7(8週相当)〜0.9(30週相当)
E_WMEAN = float(os.environ.get("E_WMEAN", "1.0"))          # ①meanの重み。1.0=従来／小=精密制御器を退ける
E4_CONTINUOUS = os.environ.get("E4_CONTINUOUS", "0") == "1"  # ④物理step刻みでノイズ更新（1秒ホールド解消）
# 【目標E ① 本丸への着手】mean（関節指令の中心値）の作り方を変える。研究日誌 続き11で
# 「境界jerk(ガクガク)の89%はmean（精密制御器）の1秒ごとの段差」と実測確定したことへの対処。
# 予備A（E_INTERP）：精密制御器のmeanを境界間で線形補間して段差を消す＝境界jerkの"理論下限"を測る
#   物差し（精密制御器は残すので①の本質は解かない、下限測定用の使い捨て。continuous時のみ有効）。
# 本命B-min（E_WMEAN<1）：mean =(1-w)×低周波生成器 + w×精密制御器。w=0で精密制御器を完全に外し
#   素朴な自発運動だけにする。1本のツマミwで①をアブレーションでき、そのまま本番の骨格になる。
E_INTERP = os.environ.get("E_INTERP", "0") == "1"          # 予備A：境界間でmeanを線形補間（下限測定）
E_GENBETA = float(os.environ.get("E_GENBETA", "0.95"))    # B-min生成器の色付き度（高=低周波=ゆっくり）
E_GENAMP = float(os.environ.get("E_GENAMP", "0.3"))       # B-min生成器の振幅（関節指令[-1,1]に対する）
# 【目標E ③シナジー（粗い多関節協調）】既定OFF、E_SYNERGY=1で有効。研究日誌続き15。
# 文献の範囲に忠実な2部位のみ実装：
#   脚（hip_flex/abduction, knee, foot_flexion）：Dominici et al. 2011 [Tier1]が
#     新生児は生得の粗いシナジー2個を持つと報告（stepping時の脚筋）。原著の実測負荷係数は
#     未取得なため、「左右が逆位相で粗く協調する」という定性的性質のみ反映[ARBITRARY・簡略化]。
#   腕（shoulder_horizontal/abduction, elbow, wrist_flexion）：Physiopedia他の臨床知見[Tier2]
#     「肩+肘+手首の伸展が同時に起きる」を反映。左右は独立（アーム間の協調は文献未確認）。
# 体幹・手指は対象外（文献なし、今まで通り独立）。実装は「グループの先頭関節の生成値に、
# 残りの関節が追従する」簡易版（新規の生成器を増やさず既存gen_npを流用）。
E_SYNERGY = os.environ.get("E_SYNERGY", "0") == "1"
E_SYN_W = float(os.environ.get("E_SYN_W", "0.6"))          # 追従の強さ。[ARBITRARY]
# 【writhing→fidgety移行スケジュール、既定OFF】E_DEV_SCHEDULE=1で有効。研究日誌続き16、
# developmental_schedule.py参照。C5_AGE(体の月齢)に応じてE_WMEAN/E_SYN_Wを上書きする
# （皮質脊髄路の成熟+CPGの多筋協調洗練、Frontiers論文の同一"感受性の窓"に基づく）。
E_DEV_SCHEDULE = os.environ.get("E_DEV_SCHEDULE", "0") == "1"
if E_DEV_SCHEDULE:
    from developmental_schedule import schedule_w_mean, schedule_syn_w
    E_WMEAN = schedule_w_mean(AGE)
    E_SYN_W = schedule_syn_w(AGE, baseline=E_SYN_W)
    print(f"[dev_schedule] age={AGE}mo → E_WMEAN={E_WMEAN:.3f} E_SYN_W={E_SYN_W:.3f}")
_LEG_R = [72, 73, 75, 76]   # right: hip_flex, hip_abduction, knee, foot_flexion
_LEG_L = [81, 82, 84, 85]   # left: 同上
_ARM_R = [14, 15, 17, 19]   # right: shoulder_horizontal, shoulder_abduction, elbow, wrist_flexion
_ARM_L = [43, 44, 46, 48]   # left: 同上
# 【把握反射】既定OFF、E_GRASP_REFLEX=1で有効。研究日誌続き17、spinal_cord/grasp_reflex.py参照。
# 強さは 1-E_WMEAN に連動（新しい独立スケジュールを作らず、皮質脊髄路の成熟による抑制を再利用）。
E_GRASP_REFLEX = os.environ.get("E_GRASP_REFLEX", "0") == "1"
# 【ATNR】既定OFF、E_ATNR_REFLEX=1で有効。研究日誌続き18、brainstem/atnr.py参照。
# 強さは把握反射と同じく 1-E_WMEAN に連動。
E_ATNR_REFLEX = os.environ.get("E_ATNR_REFLEX", "0") == "1"
# 【E_GEN_UPDATE_M】B-min生成器を M物理stepごとに新サンプル・間は線形補間で滑らかに接続する。
# 既定=1（毎step更新＝100Hz）。10なら10Hz更新＝人間の運動指令に近い低頻度化。連続時のみ有効。
# [仮説Y：経験的テスト、文献根拠なし＝逸脱リストに記録]
E_GEN_UPDATE_M = int(os.environ.get("E_GEN_UPDATE_M", "1"))
# 【E_TRACE】run_measureで関節指令の時系列を .npz に保存する（グラフ用）。パスを指定すると保存。
E_TRACE = os.environ.get("E_TRACE", "")                    # 例: "/path/to/trace.npz"


# 【運動野リファクタ、2026-07-23】旧チェックポイント（motor_head/pc_latent/motor_gruが
# TaroBrainWithMotorの直属だった頃に保存）のキーを、MotorCortexへの移動後の新キーに
# 読み替える対応表。これが無いと精密制御器の学習済み重みが「作り直し」扱いになり、
# 訓練前のランダム状態に初期化されてしまう。
_OLD_TO_NEW_KEY_PREFIX = {
    "motor_gru.": "motor_cortex.motor_gru.",
    "pc_latent.": "motor_cortex.pc_latent.",
    "motor_head.": "motor_cortex.motor_head.",
}


def _migrate_old_keys(sd):
    out = {}
    for k, v in sd.items():
        for old, new in _OLD_TO_NEW_KEY_PREFIX.items():
            if k.startswith(old):
                k = new + k[len(old):]
                break
        out[k] = v
    return out


def load_matching(module, sd, tag):
    sd = _migrate_old_keys(sd)
    own = module.state_dict()
    matched = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    skipped = [k for k in own if k not in matched]
    module.load_state_dict(matched, strict=False)
    note = f"（作り直し: {sorted(set(k.split('.')[0] for k in skipped))}）" if skipped else "（全層一致）"
    print(f"  [{tag}] ロード{len(matched)}層/作り直し{len(skipped)}層 {note}")


def build(mode_actuation, age=None):
    """age（月齢0〜24）を渡すと、MIMoの成長モジュールが体をその月齢に自動調整する
    （env内部で adjust_mimo_to_age を呼ぶ＝mimo_env.py）。既定 None＝18ヶ月児（従来）。"""
    seed = 0
    torch.manual_seed(seed); np.random.seed(seed)
    act_model = SmoothTorqueModel if mode_actuation == "on" else SpringDamperModel
    _kw = {"age": age} if age is not None else {}
    # 【2026-07-20 修正・重要】E1では**視覚と触覚を実際に脳へ繋ぐ**。
    # それまでは vision_params=None / touch_dim=0 で、太郎は内受容+固有感覚+前庭覚の
    # 3つだけ（sdim=192）で動いていた＝「視覚を入力に足した」という以前の記述は**誤り**で、
    # 環境側にパラメータを用意しただけで build から渡していなかった。
    # ⚠️sdimが192→320に変わるので、C5のチェックポイントは**一部の層が作り直しになる**
    #   （load_matching が形の合う層だけ読む）。E1は「視覚を使って自分の手を学ぶ」段階なので、
    #   視覚なしで学んだ重みをそのまま持ち越すほうが不自然、という判断で許容する。
    #   E_VISION=0 / E_TOUCH=0 で個別に切れる（アブレーション用）。E_TOY=0 なら従来と完全同一。
    _use_vision = E_TOY and os.environ.get("E_VISION", "1") == "1"
    _use_touch = E_TOY and os.environ.get("E_TOUCH", "1") == "1"
    if E_TOY:   # 目標E1：おもちゃ入りの仰向け環境（E/scripts/e_toy_env.py）
        sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "E", "scripts"))
        from e_toy_env import ToySupineEnv, infant_vision_params, VISION_RES
        _vp = infant_vision_params() if _use_vision else None
        env = HybridEnv(ToySupineEnv(vision_params=_vp, actuation_model=act_model, **_kw))
        print("[E1] おもちゃ環境（ToySupineEnv）で実行")
    else:
        env = HybridEnv(SupineMimoEnv(vision_params=None, actuation_model=act_model, **_kw))
    obs, _ = env.reset(seed=seed)
    n_act = env.action_space.shape[0]
    _touch_dim = int(np.asarray(obs["touch"]).shape[0]) if (_use_touch and "touch" in obs) else 0
    _vres = VISION_RES if _use_vision else 0
    fusion = MinimalFusion(touch_dim=_touch_dim, vision_res=_vres)
    if E_TOY:
        print(f"[E1] 感覚: 視覚={'ON(' + str(_vres) + 'px)' if _vres else 'OFF'}"
              f"／触覚={'ON(' + str(_touch_dim) + '次元)' if _touch_dim else 'OFF'}")
    sdim = fusion.encode(obs).shape[0]
    prop_dim = to_tensor(obs["observation"]).shape[0]
    print(f"融合次元 sdim={sdim}／固有感覚 prop_dim={prop_dim}／行動 n_act={n_act}")
    brain = TaroBrainWithMotor(vocab_size=3, sensory_dim=sdim, n_actuators=n_act)
    emb_proj = nn.Linear(sdim + n_act, brain.sensory_proj.out_features)
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    print(f"チェックポイント読込: {os.path.basename(CKPT)}")
    load_matching(brain, blob["brain"], "脳")
    fusion.insula.load_state_dict(blob["fusion_insula"])
    fusion.proprio.load_state_dict(blob["fusion_proprio"])
    fusion.vestibular.load_state_dict(blob["fusion_vestibular"])
    load_matching(emb_proj, blob["emb_proj"], "emb_proj")
    cereb = MotorCerebellum(brain.latent_dim, n_act)
    load_matching(cereb, blob["cereb"], "小脳")
    return env, brain, fusion, emb_proj, cereb, n_act


def actuated_dofs(model):
    """アクチュエータが駆動する関節のDOFアドレス（＝脳が動かす関節の集合）。"""
    dofs = []
    for i in range(model.nu):
        if model.actuator(i).name.startswith("beta_"):
            continue
        jid = model.actuator_trnid[i, 0]
        if jid >= 0:
            dofs.append(int(model.jnt_dofadr[jid]))
    return np.array(sorted(set(dofs)), dtype=int)


def make_policy(brain, fusion, emb_proj, cereb, n_act, babble,
                 noise_mode="white", beta=0.8, noise_seed=None, w_mean=1.0,
                 continuous=False):
    """1ティックぶんの決定的（または喃語込み）行動を返すクロージャ。

    noise_mode="white"（既定）＝従来通り毎tickの独立ガウス（自己相関≈0、実測0.034＝
      e_efference_check.pyで測定、E/docs/研究日誌.md 2026-07-21）。呼び出し側の挙動は
      1バイトも変えない。
    noise_mode="colored"＝1/f^βの色付きノイズ（colored_noise.py、baby noise由来）。
      分散(std)の式は従来と同一＝探索の"激しさ"は変えず、時間的な"粘り"だけ加える。
      beta: 0.7(生後8週相当)〜0.9(生後30週相当)。既定0.8はその中間の暫定値
      （発達に応じたβスケジュールは、学習進捗を追う仕組みができてから対応する）。

    w_mean（既定1.0＝従来と完全同一）：babble時のmean（方策+小脳の決定的出力）の重み。
      [ARBITRARY・工学的対処 — 文献根拠なし、doc/人間模倣からの逸脱リスト.md参照]
      背景：meanは目標Cで「感覚フィードバックに正確に反応する」よう学習した閉ループ制御の
      出力で、実測で強い負の自己相関(-0.180、過修正)を持つ。一方、新生児期は皮質脊髄路が
      未髄鞘化で、そうした閉ループ精密制御の行動的証拠がない（Pediatric Neurology Briefs、
      corticospinal tract in newborns）。このズレを、meanの重みを下げてノイズを主役にする
      ことで緩和する狙いだが、「新生児のGeneral Movementsを生成する具体的アルゴリズム」は
      2026年時点の文献でも未解明（"in our ongoing work", Infants' spontaneous movements
      explore arm dynamics, Commun Biol 2026）のため、w_meanの値そのものに文献根拠はない。
    """
    ne_level = 0.095  # 学習後期のNE水準（ログ実測値）。babble時のノイズ幅に使う。
    cn = ColoredNoiseGenerator(n_act, seed=noise_seed) if noise_mode == "colored" else None
    # 【目標E ①本命B-min＝脊髄CPG相当】自発運動の生成器（spinal_cord/cpg.py::CPG）。
    # w_mean<1 のとき皮質脊髄路(corticospinal.py)経由でmeanに混ざる。E_GENAMPが0なら
    # 生成しない（軽量化）。探索ノイズ(cn)と系列が被らないよう seed をずらす。
    cpg = (CPG(n_act, leg_r=_LEG_R, leg_l=_LEG_L, arm_r=_ARM_R, arm_l=_ARM_L,
               seed=(noise_seed + 1 if noise_seed is not None else 12345))
           if E_GENAMP > 0.0 and w_mean < 1.0 else None)
    cache = {}  # 【目標E ④連続化】mean/std/hiddenをCTRL_Mごとにキャッシュする箱

    def policy(obs, prev_a, hidden, recompute=True, frac=0.0):
        """recompute=True（既定＝従来と完全同一）で毎回meanを計算。continuous時は呼び出し側が
        CTRL_Mの境界だけrecompute=Trueにし、間の物理stepはrecompute=Falseで**meanは再計算せず
        ノイズだけ毎step更新**する（④＝1秒ホールドの離散をなくす。脳の判断頻度は変えない）。
        frac：境界からの相対位置[0,1]。E_INTERP時のみ使用（前のmeanと今のmeanを線形補間）。"""
        if recompute or "mean" not in cache:
            if "mean" in cache:
                cache["prev_mean"] = cache["mean"]   # 予備A補間用：前ブロックのmeanを保存
            sv = fusion.encode(obs); cf = sv.detach()
            emb = emb_proj(torch.cat([sv, prev_a], dim=-1)).unsqueeze(0).unsqueeze(0)
            out, nh = brain.motor_gru(emb, hidden)
            z, _, _ = brain.pc_latent.infer(hidden[-1, 0], out[0, -1], cf)
            z = z.detach()
            # 【2026-07-25】太郎の motor_drive を呼ぶだけに変更（core へ一元化）。
            # 旧実装は同じ式を手書きしていた＝**数値は完全に同一**。
            # ★ACTION_SCALEなし＝Cで学習した素の方策、という性質も変わらない。
            _mean, _std, _, _ = brain.motor_drive(z, ne_level, cerebellum=cereb)
            cache["mean"] = _mean.detach()
            cache["std"] = _std.detach()
            cache["hidden"] = nh.detach()
            if "prev_mean" not in cache:
                cache["prev_mean"] = cache["mean"]   # 最初のブロックは補間不能→現在値で埋める
        mean, std, nh = cache["mean"], cache["std"], cache["hidden"]
        # 【予備A】E_INTERP時は前meanと今meanを frac で線形補間して段差を消す。
        # E_INTERP=0の既定では frac がどんな値でも mean は使わない＝従来と1バイト差なし。
        if E_INTERP:
            mean_used = (1.0 - frac) * cache["prev_mean"] + frac * mean
        else:
            mean_used = mean
        if babble:
            noise = (torch.as_tensor(cn.sample(beta), dtype=mean_used.dtype) if cn is not None
                     else torch.randn_like(mean_used))
            # 【B-min＝脊髄CPG】cpg>0 & w_mean<1 のときのみ動く。既定 w_mean=1.0 では cpg=None＝
            # composed = mean_used ＝ 従来と1バイト差なし。
            if cpg is not None:
                # 【仮説Y】E_GEN_UPDATE_M > 1 ならCPGを低頻度化＋線形補間で滑らかに繋ぐ。
                # 100Hzのままだと「ぴくぴく」に見える(90関節独立)。10なら10Hz更新。
                def _sample_lowfreq(fn, key, beta):
                    if E_GEN_UPDATE_M <= 1:
                        return fn(beta)
                    if key not in cache:
                        cache[key] = {"step": 0, "prev": None, "curr": None}
                    c = cache[key]
                    if c["step"] % E_GEN_UPDATE_M == 0:
                        c["prev"] = c["curr"] if c["curr"] is not None else fn(beta)
                        c["curr"] = fn(beta)
                    gfrac = (c["step"] % E_GEN_UPDATE_M) / E_GEN_UPDATE_M
                    out = (1.0 - gfrac) * c["prev"] + gfrac * c["curr"]
                    c["step"] += 1
                    return out

                gen_np = _sample_lowfreq(
                    lambda b: cpg.sample(b, synergy=E_SYNERGY, syn_w=E_SYN_W), "gen", E_GENBETA)
                gen_out = torch.as_tensor(gen_np, dtype=mean_used.dtype) * E_GENAMP
                # 【皮質脊髄路】運動野(mean)と脊髄CPG(gen_out)をw_meanで混合（corticospinal.py）。
                composed = corticospinal_project(mean_used, gen_out, w_mean)
            else:
                composed = w_mean * mean_used
            a = torch.clamp(composed + std * noise, -1, 1)
        else:
            a = torch.clamp(mean_used, -1, 1)
        return a.detach(), nh

    return policy


class JerkMeter:
    """物理ステップごとのqaccを受け取り、ジャーク（qaccの時間微分）を集計する。"""

    def __init__(self, dofs, dt):
        self.dofs = dofs; self.dt = dt
        self.prev_qacc = None
        self.boundary, self.interior = [], []

    def observe(self, qacc, is_boundary):
        a = qacc[self.dofs]
        if self.prev_qacc is not None:
            jerk = np.abs(a - self.prev_qacc) / self.dt
            (self.boundary if is_boundary else self.interior).append(float(jerk.mean()))
        self.prev_qacc = a

    def summary(self):
        b = np.array(self.boundary) if self.boundary else np.array([0.0])
        it = np.array(self.interior) if self.interior else np.array([0.0])
        allj = np.concatenate([b, it])
        return {"mean": float(allj.mean()), "boundary": float(b.mean()),
                "interior": float(it.mean()), "max": float(allj.max())}


def run_view(mode_actuation, babble):
    """ライブビューア。C5_REALTIME=1 で**等倍速**（既定は計算任せ＝早送り）。

    等倍速でないと「人間の赤ちゃんと比べて速すぎ/遅すぎ」が判断できず目視の意味が薄れる。
    一時停止＝スペース、巻き戻し＝左パネルのHistory（MuJoCoビューアの標準機能）。
    """
    import time
    # 再生速度。1.0=等倍速、0=待たない（最速＝学習を早く進めたいとき）。
    # 実行中にキーで変えられる（等速だと学習の進みを見るのに時間がかかりすぎるため）：
    #   .（>） 2倍速く   ,（<） 2倍遅く   0 等倍に戻す   M 最速（待たない）
    speed = [1.0 if os.environ.get("C5_REALTIME", "0") == "1" else 0.0]

    def _key_cb(keycode):
        try:
            ch = chr(keycode)
        except ValueError:
            return
        if ch in ".>":
            speed[0] = min(speed[0] * 2 if speed[0] > 0 else 64.0, 64.0)
        elif ch in ",<":
            speed[0] = max(speed[0] / 2 if speed[0] > 0 else 1.0, 0.0625)
        elif ch == "0":
            speed[0] = 1.0
        elif ch in "mM":
            speed[0] = 0.0
        else:
            return
        print(f"  [speed] x{speed[0]:.4g}" if speed[0] > 0 else "  [speed] MAX (no wait)",
              flush=True)
    env, brain, fusion, emb_proj, cereb, n_act = build(mode_actuation, age=AGE)
    policy = make_policy(brain, fusion, emb_proj, cereb, n_act, babble,
                         noise_mode=E_NOISE, beta=E_BETA, w_mean=E_WMEAN, continuous=E4_CONTINUOUS)
    m, d = env.unwrapped.model, env.unwrapped.data
    dofs = actuated_dofs(m)
    dt_env = m.opt.timestep * env.unwrapped.frame_skip
    meter = JerkMeter(dofs, dt_env)
    # 【目視確認のため run_measure と揃える、2026-07-23】従来はここに反射の配線が無く、
    # E_GRASP_REFLEX/E_ATNR_REFLEXをONにしてもViewerには反映されない欠落があった。
    grasp_reflex = GraspReflex(m) if E_GRASP_REFLEX else None
    atnr = ATNR(m) if E_ATNR_REFLEX else None

    obs, _ = env.reset(seed=0)
    hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)
    _ctrl_note = f"制御刻み={CTRL_M}tick({'速い連続制御' if CTRL_M < K else '1秒保持'})"
    print(f"\nビューア起動。仰向けの太郎が学習済みの脳で動きます（活性化ダイナミクス={mode_actuation.upper()}"
          f"／{'喃語込み' if babble else '決定的'}／{_ctrl_note}）。スペース=一時停止、左History=巻き戻し。")
    print("  キー操作: . = 速く / , = 遅く / 0 = 等倍 / M = 最速（待たない）")

    def _show_speed(viewer, eff=0.0):
        """現在の再生倍率を**画面右上のUI**に出す。

        第1版はシーン内にラベルgeomを浮かせたが、3D空間の物体として描かれるので
        カメラを動かすと位置がずれ、見づらかった。ビューアのオーバーレイ(mjr_overlay)は
        passive viewer から直接触れないため、**ウィンドウのタイトル**に出す方式にする
        （常に画面上部に見え、カメラ操作の影響を受けない）。
        """
        # ★「要求倍率」と「実効倍率」を**両方**出す。第1版は要求だけを出していたが、
        #   実測で x2 以上は計算が追いつかず頭打ち（1物理ステップの計算6.46ms > sim 10ms/倍率）
        #   と判明した＝x64と表示しながら実際は約1.5倍速だった。要求だけの表示は嘘になる。
        req = (f"x{speed[0]:.4g}" if speed[0] > 0 else "MAX")
        txt = f"speed {req} (real x{eff:.1f})"
        if getattr(viewer, "_last_speed_txt", None) == txt:
            return
        viewer._last_speed_txt = txt
        try:
            # set_figures はビューアの右上に固定表示されるオーバーレイ（3D空間ではないので
            # カメラを動かしてもずれない）。mjvFigure のタイトルを速度表示に使う。
            fig = mujoco.MjvFigure()
            mujoco.mjv_defaultFigure(fig)
            fig.title = txt
            fig.flg_legend = 0
            fig.flg_ticklabel[:] = [0, 0]          # 配列フィールドは[:]で代入する
            fig.figurergba[:] = [0.0, 0.0, 0.0, 0.4]
            vp = viewer.viewport
            w = max(int(vp.width * 0.22), 180)
            h = 46
            rect = mujoco.MjrRect(int(vp.width - w - 10), int(vp.height - h - 10), w, h)
            viewer.set_figures([(rect, fig)])
        except Exception as e:
            print(f"  [{txt}] (overlay unavailable: {type(e).__name__})", flush=True)

    tick = 0; ctrl = None
    # 画面の更新は60Hzで十分（人間の目に見える上限）。第1版は**毎物理ステップ**（等倍で100Hz、
    # 倍速時はもっと）描画していて、それ自体が倍速を頭打ちにする原因の一つだった。
    SYNC_DT = 1.0 / 60.0
    with mujoco.viewer.launch_passive(m, d, key_callback=_key_cb) as viewer:
        t_wall = time.perf_counter()   # 次に物理を進めてよい実時刻（sleepの累積誤差を防ぐ）
        t_draw = 0.0
        eff_t0, eff_sim, eff = time.perf_counter(), 0.0, 0.0
        while viewer.is_running():
            for k in range(K):
                boundary = (k % CTRL_M == 0)
                if E4_CONTINUOUS:   # ④/B-min：命令を毎物理stepで更新（連続再生）
                    frac = (k % CTRL_M) / CTRL_M
                    a, hidden = policy(obs, prev_a, hidden, recompute=boundary, frac=frac)
                    if grasp_reflex is not None:
                        a = torch.as_tensor(grasp_reflex.apply(
                            a.numpy(), env.unwrapped.touch.sensor_outputs, max(0.0, 1.0 - E_WMEAN)),
                            dtype=a.dtype)
                    if atnr is not None:
                        a = torch.as_tensor(atnr.apply(
                            a.numpy(), d.qpos, max(0.0, 1.0 - E_WMEAN)),
                            dtype=a.dtype)
                    ctrl = rescale_action(a, env.action_space)
                    if boundary:
                        prev_a = a
                elif boundary:   # 従来：1秒に1回だけ命令を出す
                    a, hidden = policy(obs, prev_a, hidden)
                    if grasp_reflex is not None:
                        a = torch.as_tensor(grasp_reflex.apply(
                            a.numpy(), env.unwrapped.touch.sensor_outputs, max(0.0, 1.0 - E_WMEAN)),
                            dtype=a.dtype)
                    if atnr is not None:
                        a = torch.as_tensor(atnr.apply(
                            a.numpy(), d.qpos, max(0.0, 1.0 - E_WMEAN)),
                            dtype=a.dtype)
                    ctrl = rescale_action(a, env.action_space)
                    prev_a = a
                obs, r, te, tr, info = env.step(ctrl)
                meter.observe(d.qacc.copy(), is_boundary=boundary)
                now = time.perf_counter()
                eff_sim += dt_env
                if now - eff_t0 >= 0.5:       # 実効倍率＝直近0.5秒の「sim時間/実時間」
                    eff = eff_sim / (now - eff_t0)
                    eff_t0, eff_sim = now, 0.0
                if now - t_draw >= SYNC_DT:
                    _show_speed(viewer, eff)
                    viewer.sync()
                    t_draw = now
                if speed[0] > 0:
                    # 目標時刻を積み上げて追従する。第1版の「毎回 sleep(dt/speed)」は
                    # Windowsのsleep分解能(~1-15ms)ぶん必ず遅れ、その誤差が積もっていた。
                    t_wall += dt_env / speed[0]
                    lag = t_wall - time.perf_counter()
                    if lag > 0:
                        time.sleep(lag)
                    elif lag < -0.25:         # 大きく遅れたら追いつくのを諦めて基準を引き直す
                        t_wall = time.perf_counter()
                else:
                    t_wall = now
                if te or tr:
                    break
            tick += 1
            if tick % 20 == 0:
                s = meter.summary()
                print(f"  tick {tick}: mean|jerk|={s['mean']:.1f} 境界={s['boundary']:.1f} "
                      f"内部={s['interior']:.1f} 最大={s['max']:.1f}", flush=True)


def run_measure(mode_actuation, n, babble):
    env, brain, fusion, emb_proj, cereb, n_act = build(mode_actuation, age=AGE)
    policy = make_policy(brain, fusion, emb_proj, cereb, n_act, babble,
                         noise_mode=E_NOISE, beta=E_BETA, w_mean=E_WMEAN, continuous=E4_CONTINUOUS)
    m, d = env.unwrapped.model, env.unwrapped.data
    dofs = actuated_dofs(m)
    dt_env = m.opt.timestep * env.unwrapped.frame_skip
    meter = JerkMeter(dofs, dt_env)
    grasp_reflex = GraspReflex(m) if E_GRASP_REFLEX else None
    atnr = ATNR(m) if E_ATNR_REFLEX else None

    obs, _ = env.reset(seed=0)
    hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)
    action_jumps = []
    trace_actions = [] if E_TRACE else None   # 【グラフ用】毎物理stepの行動指令を保存
    trace_qpos = [] if E_TRACE else None      # 【グラフ用】毎物理stepの関節位置(qpos)も保存
                                              # ＝人間新生児の"観察された動き"と公平に比較するため
    _ctrl_note = f"制御刻み={CTRL_M}tick({'速い連続制御' if CTRL_M < K else '1秒保持'})"
    print(f"\n測定開始（{n}ティック・活性化ダイナミクス={mode_actuation.upper()}"
          f"／{'喃語込み' if babble else '決定的'}／{_ctrl_note}）")
    ctrl = None
    for tick in range(n):
        for k in range(K):
            boundary = (k % CTRL_M == 0)   # 脳が感覚を見て判断を出し直す刻み（＝mean更新の刻み）
            if E4_CONTINUOUS:
                # ④：meanは境界でだけ更新し、探索ノイズは毎物理stepで更新して命令を出し直す
                # ＝1秒ホールドの階段（躍度の不連続）を消す。脳の判断頻度は変えない。
                # frac：予備A（E_INTERP）が境界間で前meanと今meanを線形補間するのに使う。
                frac = (k % CTRL_M) / CTRL_M
                a, hidden = policy(obs, prev_a, hidden, recompute=boundary, frac=frac)
                if grasp_reflex is not None:   # 【把握反射】強さ=1-w_mean（既存スケジュール逆連動）
                    a = torch.as_tensor(grasp_reflex.apply(
                        a.numpy(), env.unwrapped.touch.sensor_outputs, max(0.0, 1.0 - E_WMEAN)),
                        dtype=a.dtype)
                if atnr is not None:   # 【ATNR】強さ=1-w_mean（既存スケジュール逆連動）
                    a = torch.as_tensor(atnr.apply(
                        a.numpy(), d.qpos, max(0.0, 1.0 - E_WMEAN)),
                        dtype=a.dtype)
                ctrl = rescale_action(a, env.action_space)
                if boundary:
                    action_jumps.append(float(np.abs((a - prev_a).numpy()).mean()))
                    prev_a = a
            elif boundary:   # 従来：1秒に1回だけ命令を出し、その間は保持
                a, hidden = policy(obs, prev_a, hidden)
                if grasp_reflex is not None:
                    a = torch.as_tensor(grasp_reflex.apply(
                        a.numpy(), env.unwrapped.touch.sensor_outputs, max(0.0, 1.0 - E_WMEAN)),
                        dtype=a.dtype)
                if atnr is not None:
                    a = torch.as_tensor(atnr.apply(
                        a.numpy(), d.qpos, max(0.0, 1.0 - E_WMEAN)),
                        dtype=a.dtype)
                action_jumps.append(float(np.abs((a - prev_a).numpy()).mean()))
                ctrl = rescale_action(a, env.action_space)
                prev_a = a
            if trace_actions is not None:
                trace_actions.append(a.detach().cpu().numpy().copy())
            obs, r, te, tr, info = env.step(ctrl)
            if trace_qpos is not None:
                trace_qpos.append(d.qpos.copy())
            meter.observe(d.qacc.copy(), is_boundary=boundary)
            if te or tr:
                obs, _ = env.reset()
                hidden = brain.init_motor_hidden()
                break
    if E_TRACE and trace_actions:
        actuator_names = [m.actuator(i).name for i in range(m.nu)]
        joint_names = [m.joint(i).name for i in range(m.njnt)]
        jnt_qposadr = np.array([m.jnt_qposadr[i] for i in range(m.njnt)])
        np.savez(E_TRACE,
                 actions=np.array(trace_actions),
                 qpos=np.array(trace_qpos),
                 actuator_names=actuator_names,
                 joint_names=joint_names,
                 jnt_qposadr=jnt_qposadr)
        print(f"[E_TRACE] 行動指令+関節位置の時系列を保存: {E_TRACE}")
        print(f"          actions.shape={np.array(trace_actions).shape} "
              f"qpos.shape={np.array(trace_qpos).shape}")
    s = meter.summary()
    print(f"\n===== 結果（活性化={mode_actuation.upper()}／{_ctrl_note}）=====")
    print(f"mean|jerk|      = {s['mean']:.2f}   (小さいほどなめらか)")
    print(f"  境界ジャーク  = {s['boundary']:.2f}  (ティック切替の瞬間)")
    print(f"  内部ジャーク  = {s['interior']:.2f}  (行動保持中)")
    print(f"  最大ジャーク  = {s['max']:.2f}")
    print(f"行動ジャンプ = {np.mean(action_jumps):.3f}  (1判断ごとに行動がどれだけ跳ぶか)")
    return s


def run_eyeview(mode_actuation, n, babble):
    """【運動性喃語×egomotion】仰向け・新生児・相手なし・視覚ありで自発運動を動かし、
    第三者視点＋一人称(眼球)視界を録画する。＝C5の運動が"視界を揺らさないか(egomotion)"を、
    姿勢/体/相手の交絡なしで目視するための動画。脳は視覚を使わない（＝眼球カメラを"見るだけ"）。"""
    import cv2
    env, brain, fusion, emb_proj, cereb, n_act = build(mode_actuation, age=AGE)
    policy = make_policy(brain, fusion, emb_proj, cereb, n_act, babble,
                         noise_mode=E_NOISE, beta=E_BETA, w_mean=E_WMEAN, continuous=E4_CONTINUOUS)
    m, d = env.unwrapped.model, env.unwrapped.data
    # オフスクリーン描画のフレームバッファを広げる（既定500pxだと640×480が入らない）
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), 640)
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), 480)

    third_ren = mujoco.Renderer(m, height=480, width=640)
    third_cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(m, third_cam)
    # E1（おもちゃ）は「手とおもちゃの位置関係」を見るのが目的なので寄る＋太郎を追う。
    # 既定の引きだと太郎が豆粒でおもちゃが見えない（実際に一度そうなった）。
    # 【E_EYEVIEW_WIDE=1】新生児体型v2(E_TOY=1)のまま、全身シナジーを見たいときは引きに戻す
    # （2026-07-23：柵で全身が隠れて判断不能と判明、既定の接写と切り替え可能にした）。
    _wide_env = os.environ.get("E_EYEVIEW_WIDE", "0")
    _wide = _wide_env not in ("0", "")
    if _wide_env not in ("0", "1", ""):   # 数値指定なら倍率として使う（例: E_EYEVIEW_WIDE=0.6）
        third_cam.distance *= float(_wide_env)
    else:
        third_cam.distance *= (1.3 if (not E_TOY or _wide) else 0.32)
    # 【E_EYEVIEW_TOP=1】真上から見下ろす（柵は縦の格子なので、真上なら遮られない）。
    # 2026-07-23：距離調整だけでは柵が視界を塞ぐ問題を解消できなかったため追加。
    if os.environ.get("E_EYEVIEW_TOP", "0") == "1":
        third_cam.elevation = -89.0
        third_cam.distance = 1.1
    if E_TOY and not _wide:
        third_cam.elevation = -35.0
    try:
        eye_cid = int(m.camera("eye_left").id)
    except Exception:
        print("⚠️ eye_left カメラが見つからず一人称は録画できません（第三者のみ）"); eye_cid = None
    eye_ren = eye_cam = None
    if eye_cid is not None:
        eye_ren = mujoco.Renderer(m, height=64, width=64)
        eye_cam = mujoco.MjvCamera(); eye_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        eye_cam.fixedcamid = eye_cid

    obs, _ = env.reset(seed=0)
    hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)
    third_frames, eye_frames = [], []
    SUB = 2  # 物理2ステップに1コマ録画（秒内の揺れが見える解像度）
    ctrl = None; step_i = 0
    _age = "新生児(0m)" if AGE == 0 else (f"{int(AGE)}m" if AGE is not None else "18m既定")
    print(f"\n録画開始（{n}ティック・仰向け・体={_age}・視覚あり(眼球)・相手なし・"
          f"{'喃語(運動性喃語)込み' if babble else '決定的'}／制御刻み={CTRL_M}tick）")
    for tick in range(n):
        for k in range(K):
            boundary = (k % CTRL_M == 0)
            if E4_CONTINUOUS:   # ④/B-min：meanは境界のみ、ノイズは毎物理stepで更新して命令を出し直す
                frac = (k % CTRL_M) / CTRL_M   # 予備A補間用
                a, hidden = policy(obs, prev_a, hidden, recompute=boundary, frac=frac)
                ctrl = rescale_action(a, env.action_space)
                if boundary:
                    prev_a = a
            elif boundary:
                a, hidden = policy(obs, prev_a, hidden)
                ctrl = rescale_action(a, env.action_space); prev_a = a
            obs, r, te, tr, info = env.step(ctrl)
            if step_i % SUB == 0:
                if E_TOY:   # 太郎は暴れて移動するのでカメラを体に追従させる
                    third_cam.lookat[:] = d.body("upper_body").xpos
                third_ren.update_scene(d, camera=third_cam)
                third_frames.append(third_ren.render().copy())
                if eye_ren is not None:
                    eye_ren.update_scene(d, camera=eye_cam)
                    eye_frames.append(cv2.resize(eye_ren.render().copy(), (240, 240),
                                                 interpolation=cv2.INTER_NEAREST))
            step_i += 1
            if te or tr:
                obs, _ = env.reset(); hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)
                break

    # 【重要】E1（おもちゃ環境）の動画は**別名・別フォルダ**に出す。
    # 同名で出すと従来のC5動画（おもちゃ無し）を上書きしてしまう（実際に一度やらかした）。
    # ログは消さず分類して保存する、という運用に合わせる。
    _age_tag = int(AGE) if AGE is not None else "def"
    _bab = "babble" if babble else "det"
    if E_TOY:
        out_dir = os.path.join(_HERE, os.pardir, os.pardir, "E", "logs", "video")
        tag = f"e1toy_age{_age_tag}_m{CTRL_M}_{_bab}"
    else:
        out_dir = os.path.join(_HERE, os.pardir, "logs", "video")
        tag = f"c5eye_age{_age_tag}_m{CTRL_M}_{_bab}"
    os.makedirs(out_dir, exist_ok=True)

    def _write(frames, name, size, fps=50):
        p = os.path.join(out_dir, f"{tag}_{name}.mp4")
        vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        for f in frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release(); return p

    p3 = _write(third_frames, "third", (640, 480))
    print("第三者視点mp4:", p3)
    # 静止画シート（等間隔に抜いた6コマ）。動画を再生せずに配置・接触を一目で確認するため。
    try:
        idx = np.linspace(0, len(third_frames) - 1, 6).astype(int)
        rows = [np.hstack([third_frames[i] for i in idx[:3]]),
                np.hstack([third_frames[i] for i in idx[3:]])]
        sheet = np.vstack(rows)
        sp = os.path.join(out_dir, f"{tag}_sheet.png")
        cv2.imwrite(sp, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        print("第三者視点シートpng:", sp)
    except Exception as e:
        print(f"[警告] シート画像の保存に失敗: {type(e).__name__}: {e}")
    if eye_frames:
        pe = _write(eye_frames, "eye", (240, 240))
        print("一人称(眼球)mp4:", pe)
    print("→ 一人称が激しく揺れるほど、運動が視界(egomotion)を汚している＝C6が要る兆候。")
    return p3


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "view"
    actuation = sys.argv[2] if len(sys.argv) > 2 else "off"
    rest = sys.argv[3:]
    babble = "babble" in rest
    nums = [int(x) for x in rest if x.isdigit()]
    n = nums[0] if nums else 60
    print(f"=== taro-C5 運動の質: mode={mode} 活性化ダイナミクス={actuation.upper()} "
          f"{'喃語込み' if babble else '決定的'} ===")
    if mode == "measure":
        run_measure(actuation, n, babble)
    elif mode == "eyeview":
        run_eyeview(actuation, n if nums else 15, babble)
    else:
        run_view(actuation, babble)
