"""
目標E 成長カリキュラム学習ループ【C/scripts/run_c_metrics_ac_lr.py のコピー・改造版】。

【由来・2026-07-23】このファイルは目標C の学習ループ（margin+51 の自己モデルを建てた
実績コード）を**そのまま複製**したもの。オリジナルは無傷で残す（Cの再現性を壊さないため）。
ここでは「白紙の脳＋新生児の体(age=0)から、体を少しずつ育てながら自己モデルと運動を
同時に学習する」成長カリキュラム用に、E側で少しずつ改造していく。

司令塔は e_growth_curriculum.py（段階ごとに E_AGE / E_LOADMODEL を変えて本ループを
サブプロセス起動する）。

【整理方針・2026-07-23】測定器（probe類・CSV・録画）と学習ループ本体は本来別ファイルに
分けるべき（ユーザー指摘）。まずCと挙動一致するコピーを確保し、その後 e_probes.py へ
probe群を切り出す。太郎モデル(taro_core)・環境(d_supine_env/e_toy_env)は既に分離済み。

--- 以下、オリジナルCのdocstring ---
run_c_metrics_seed.py との違いは2点だけ（共有クラスは触らない）：
  D-a: GRU入力を [感覚] → [感覚, 前回の行動] に（遠心性コピーを再帰の中へ）
  D-b: 予測ヘッドを線形 → 非線形MLPに（行動→感覚の線形上限0.22を破る）
根拠：World Model/RSSM・Tani系 MTRNN/PV-RNN の標準形（行動条件づけを再帰に入れて次観測予測）。
狙い：persist>100（過大予測）と自己復元0.22の根＝「行動→感覚の弱い結びつき」を強める。

出力は logs/E/ に ac_metrics_seed{seed}_{日時}.csv。使い方: python e_growth_train.py <seed> [n_train]
"""
import os, sys, csv, time, datetime, warnings

# Windowsのcp932で表示できない文字が1つでもあると print が例外を投げ、**学習が途中で落ちる**。
# 2026-07-25 だけで3回踏んだ（絵文字／上付き2／棒グラフの █）。表示できない文字は「?」に
# 置き換えて**落とさない**。注意これは表示だけの保険で、ログの内容を変える意図はない。
# → 検証の落とし穴チェックリスト 項35。
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)

_BRIDGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # = Taro/E（このコピーはE側。logs出力もE側）
# 脳/感覚/tests は taro_core へ移設済み（doc/移行記録_taro_core化_2026-07-17.md）
_CORE = os.path.join(_BRIDGE, os.pardir, "taro_core")
for sub in ("wrapper", "senses", "brain"):
    sys.path.insert(0, os.path.join(_CORE, "src", sub))
sys.path.insert(0, os.path.join(_CORE, "tests"))
import gymnasium as gym
import mimoEnv  # noqa
from hybrid_env import HybridEnv
from taro_brain_motor import TaroBrainWithMotor
from basal_ganglia import TaroLearner
from dopamine import Dopamine
from locus_coeruleus import LocusCoeruleus
from developmental_clock import DevelopmentalClock
from cerebellum_motor import MotorCerebellum
from learning_progress import LearningProgress, smoothness_cost
from homeostatic_scaling import HomeostaticScaling
from test_phase8_motor_learning import CombinedParams, rescale_action, to_tensor
from sensory_encoders import ProprioceptionEncoder, VestibularEncoder, TouchEncoder
from insula import Insula
import e_probes  # 測定器（probe類・録画）＝太郎の外から測る道具。学習ループ本体からは独立。
# 【なぜ、2026-08-06】infer_goal_actionのデフォルト値（n_steps/lr_inf）を、監査指摘
# （測定器の数値がC/E双方に同一の値でコピーされていた）を受けて一元化した場所から参照する。
# e_probes.py が既に Taro ルートを sys.path へ足しているので、ここでは import するだけでよい。
from run.plugins.common import probe_defaults as _pd

# 仰向け環境（D側で定義）。E_SUPINE=1 のときだけ使う。
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir, "D", "scripts"))
from gymnasium.envs.registration import register
if os.environ.get("E_SUPINE", "0") == "1":
    from d_supine_env import SupineMimoEnv  # noqa
    register(id="TaroSupine-v0", entry_point="d_supine_env:SupineMimoEnv", max_episode_steps=6000)
elif os.environ.get("E_LEAN", "1") == "1":
    from mimo_lean import LeanMimoEnv  # noqa
    # max_episode_steps=6000 は MIMoBenchV2-v0 の登録値と同一（＝太郎の人生の長さを変えない）
    register(id="TaroBenchV2Lean-v0", entry_point="mimo_lean:LeanMimoEnv", max_episode_steps=6000)

mse = torch.nn.functional.mse_loss
DT = 0.01
LOG_DIR = os.environ.get("E_LOGDIR", os.path.join(_BRIDGE, "logs", "E", "growth_curriculum"))
_LR = float(os.environ.get("E_LR", "0.005"))
_MATURE = os.environ.get("E_MATURE", "0") == "1"  # 1=学習進行に合わせて探索を結晶化
_REPLAY = os.environ.get("E_REPLAY", "1") == "1"  # 既定ON＝睡眠中の経験リプレイ（記憶定着）。自己モデル確立の本命機構(taro-C2＝margin+11→+48の頭打ち突破)。E_REPLAY=0で無効化可（アブレーション用）
# 【注意すべき機能】既定ON＝運動小脳（自動化/結晶化）。良性は検証済み（発散・フリーズ・
# agency崩壊なし）だが、現指標では効果ほぼ不変・常時稼働（約26%ブレンド）。将来の壁の
# 切り分け時は「小脳が効いている可能性」を必ず確認する（`注意すべき機能リスト.md` 参照）。
_CEREB = os.environ.get("E_CEREBELLUM", "1") == "1"  # 0で無効化可
# 【taro-C5】努力コスト＝運動の代謝コストを報酬から引く。報酬=予測のうまさ − E_EFFORT×(筋活動)²。
# 既定0＝OFF＝従来と完全に同一。人間は代謝エネルギー最小になるよう動く（Selinger 2015 等）＝
# 大きな力＝損、を入れると自分で力を加減する。注意二乗の形・λの値は恣意的（cost()=Σu²·Tmaxを流用、
# 生理の正確な代謝式ではない）＝感度確認の対象。フリーズ（motor collapse, 逸脱リストB1）と背中
# 合わせなので、|行動|の低下と自己モデル(margin/agency)の生存を必ず併せて確認する。
_EFFORT = float(os.environ.get("E_EFFORT", "0"))
# 【階層化・変更②】action smoothing（CAPS）＝行動の急変にペナルティ。既定OFF(0)。
# 制御頻度を上げる（E_K=10）と探索ノイズが高周波化して力積が打ち消し合うのを防ぐ。
# 実装は core（brain/learning_progress.py::smoothness_cost）。ここは実験の設定だけ。
# 注意：λ は [Tier3・ARBITRARY]＝必ず複数値で振る。強すぎると動きが縮んで固まる。
_CAPS = float(os.environ.get("E_CAPS", "0"))
# 【taro-C5】学習済みモデルから継続学習する（脳をリセットしない方針）。E_LOADMODEL=<pt path>。
# 形が合う層だけロード（転移学習）。既定なし＝従来どおりゼロから学習。
_LOADMODEL = os.environ.get("E_LOADMODEL", "")
# 【taro-C5】体の月齢（成長モジュール）。0〜24。空=デフォルト18ヶ月児。envのage引数に渡すと
# MIMoが adjust_mimo_to_age で体格・体重・筋力を実乳児データに合わせて自動調整（構造=関節数は不変）。
_AGE = os.environ.get("E_AGE", "")
# 【2026-07-29】学習を止めずに体を育てる（＝プロセスを分けない成長）。
#
# 【なぜ要るか】これまで「0ヶ月で学ぶ → 保存 → 別プロセスで4ヶ月として読み込み」で
#   体を切り替えていた。ところが保存されるのは**脳の重みだけ**で、
#   オプティマイザの状態・経験バッファ(goal_buf)・馴化などの内部状態は消える
#   （落とし穴チェックリスト 項75）。段階的成長を「9回保存・読み込み」で作ると、
#   その条件だけ内部状態が8回消える＝**載せ替え回数が交絡になる**。
#   → 同じプロセスの中で env だけ作り直す。脳・オプティマイザ・バッファは
#     Python の変数のまま残るので、何回体を変えても消えるものが無い。
#
# 【使い方】3条件がこの4つの変数だけで書ける（総学習回数 18000 の例）：
#   ずっと4ヶ月    E_AGE=4.0                                   （体の作り直し 0回）
#   急に変える     E_AGE=0.0 E_AGE_TO=4.0 E_AGE_START=12000    （1回）
#   なめらか       E_AGE=0.0 E_AGE_TO=4.0 E_AGE_RAMP=18000     （E_AGE_EVERY ごと）
#
# 注意：E_AGE_TO が空なら月齢は動かない＝**従来と完全に同じ動作**（既存実験は壊れない）。
# 注意：触覚ONでは使えない：センサ点が月齢で変わり(0ヶ月1734→4ヶ月4274)、観測の次元が
#   変わって脳の入力層と合わなくなる。下の assert で止める（既知の構造的限界）。
# 【2026-07-30】シーンから環境を作る。空なら従来どおり環境変数で組み立てる。
#   E_SCENE=リーチング_リクライニング60度 のように名前で指定する（run/scenes/*.json）。
_SCENE_NAME = os.environ.get("E_SCENE", "")
if _SCENE_NAME:
    # シーンは体の月齢（body.age_months）を持っているので、E_AGE を書かずに済ませる。
    # 注意：E_AGE も指定されている場合はそちらを使う（E_AGE_TO で月齢を動かす実験では
    #   開始月齢がシーンと違って当たり前なので、止めずに**食い違いを知らせるだけ**にする）。
    # 【2026-08-05】シーンの置き場所はe_scene.pyのSCENE_DIRを唯一の正とし、
    #   ここで独自にパスを組み立てない（配置整理の設計:
    #   作業記録（非公開） 4-2節）。
    _scene_tools_dir = os.path.join(_BRIDGE, os.pardir, "run", "scene_tools")
    if _scene_tools_dir not in sys.path:
        sys.path.insert(0, _scene_tools_dir)
    import e_scene
    _sp = e_scene.scene_path(_SCENE_NAME)
    if not os.path.exists(_sp):
        raise FileNotFoundError(f"シーンが見つからない: {_sp}\n"
                                f"  使えるシーン: {'、'.join(e_scene.list_scenes())}")
    import json as _json
    _scene_age = float(_json.load(open(_sp, encoding="utf-8"))["body"]["age_months"])
    if not _AGE:
        _AGE = repr(_scene_age)
        print(f"[scene] 月齢をシーンから取得: {_scene_age}ヶ月"
              f"（E_AGE 未指定／シーン={_SCENE_NAME}）", flush=True)
    elif abs(float(_AGE) - _scene_age) > 1e-9:
        print(f"注意[scene] 月齢が食い違う: E_AGE={_AGE} / シーン={_scene_age}"
              f" → E_AGE を使う（体を育てる実験なら正常）", flush=True)
_AGE_TO = os.environ.get("E_AGE_TO", "")            # 終わりの月齢。空=変えない
_AGE_START = int(os.environ.get("E_AGE_START", "0"))    # 何回目から変え始めるか
_AGE_RAMP = int(os.environ.get("E_AGE_RAMP", "0"))      # 何回かけて変えるか。0=一瞬で
_AGE_EVERY = int(os.environ.get("E_AGE_EVERY", "500"))  # 何回ごとに月齢を見直すか


def age_at(step, n_train):
    """学習が step 回進んだ時点の月齢。E_AGE_TO が空なら常に E_AGE のまま。

    注意：[Tier3] 月齢を学習回数の**線形**で進めることに直接の根拠は無い。
      taro_core 側の体の補正（taper_weight / target_ratio_for_age）が線形なのに
      合わせてある。主張できるのは「崖が無い」ことだけ。
    """
    if not _AGE_TO:
        return float(_AGE) if _AGE else None
    a0, a1 = float(_AGE or 0.0), float(_AGE_TO)
    if _AGE_RAMP <= 0:                       # 一瞬で切り替える（＝「急に変える」条件）
        return a1 if step >= _AGE_START else a0
    f = (step - _AGE_START) / float(_AGE_RAMP)
    return a0 + (a1 - a0) * max(0.0, min(1.0, f))


# 【目標E・運動性喃語（脊髄CPG）】既定 white＝白色ガウス探索＝従来と数値完全一致。
# colored で太郎の脊髄CPG（色付きノイズ1/f^β＋粗いシナジー）を有効化し、探索そのものを
# 人間の新生児らしくする。βやシナジーは d_c5_motor_quality.py と同じ意味。探索の性質は
# 太郎の中（brain.enable_spinal_babble → explore）が持つ＝学習ループは配線しない。
_E_NOISE = os.environ.get("E_NOISE", "white")        # white(従来) | colored(色付き＋シナジー)
_E_BETA = float(os.environ.get("E_BETA", "0.8"))     # 色付き度。0.7(8週)〜0.9(30週)
_E_SYNERGY = os.environ.get("E_SYNERGY", "0") == "1"  # 粗いシナジー（脚・腕）
_E_SYN_W = float(os.environ.get("E_SYN_W", "0.6"))   # シナジー追従の強さ[ARBITRARY]
# シナジーの関節index（MIMo身体の配列。d_c5_motor_quality.py と同一）
_LEG_R = [72, 73, 75, 76]; _LEG_L = [81, 82, 84, 85]
_ARM_R = [14, 15, 17, 19]; _ARM_L = [43, 44, 46, 48]
# 【taro-C5】①活性化ダイナミクス（筋の一次遅れ＝力が瞬時に変わらずじわっと立ち上がる）。
# 1でSmoothTorqueModelを使う（境界ジャーク半減）。既定0＝従来の瞬時トルク。
_SMOOTH = os.environ.get("E_SMOOTH", "0") == "1"
# 【筋肉モデル切替】E_MUSCLE=1 で MIMoActuation.MuscleModel を使う（1関節=拮抗筋2本、
# 引くだけ・活性化ダイナミクス・長さ速度依存）。action space が 90次元[-1,1]→180次元[0,1]に
# なり、学習済みモデル（トルクモード）は非互換＝白紙から学習し直しになる。まず「筋肉モデルの
# 太郎が自己モデルを立てられるか」の実現可能性検証。共収縮（拮抗筋の本領）はここでは入れない
# ＝各筋を独立にexplore()の出力で駆動する最小版[Tier3・簡略化]。シナジーは90-actuator index
# 前提なので自動でOFFにする。
# 【2026-07-30 既定を反転】0（関節モード）→ 1（筋肉モード）。
#   【なぜ】測定スクリプトは80本以上が `actuation_model=MuscleModel` を直書きし、
#     シーン（e_scene.build）も筋肉モードが既定だったのに、**学習ループだけが
#     既定で関節モード**だった。＝学習した太郎と、測定・Viewerで見ていた太郎が
#     別の体だった（2026-07-30 にユーザーの目視「視線誘導反射の実験の時とは
#     動きが全然違う／等速でも倍速みたい」で発覚。実測で動きが約3.3倍速い）。
#   【どちらが正しいか】関節モード（90関節を独立に駆動）は
#     **逸脱リストの「逸脱5」に登録済みの明確な人間模倣からの逸脱**
#     （Hadders-Algra 1992［Tier1］＝新生児は拮抗筋を同時に力ませる）。
#     逸脱リスト自身が「拮抗筋なし筋肉モードで先に進む」と判断している。
#     過去の実測でも margin +58.8（筋肉モード）vs 今日の +25〜30（関節モード）。
#   注意：関節モードを使うには **明示的に E_MUSCLE=0** と書く（＝逸脱を選ぶ意思表示）。
#   注意：既定が変わったので、E_MUSCLE を指定していない過去の実験とは条件が違う。
_MUSCLE = os.environ.get("E_MUSCLE", "1") == "1"
if not _MUSCLE:
    print("注意[actuation] 関節モード（90関節を独立に駆動）＝逸脱リスト 逸脱5 の逸脱を"
          "選んでいます。人間の新生児は拮抗筋を同時に力ませる[Tier1]", flush=True)
# 【拮抗筋co-activation】E_ANTAGONIST=1 で拮抗筋モード（要 E_MUSCLE=1）。脳は「関節レベルの
# 指令」(n_joint次元, [-1,1]) を出し、脊髄CPG(cpg.py の antagonist_map)が2本の筋
# (n_muscle=2*n_joint, [0,1])に写像する。共収縮の度合いは E_COACTIVATION（既定0.3、
# [Tier3・ARBITRARY]、Hadders-Algra 1992 は存在確認[Tier1]だが数値は非公開）。
_ANTAGONIST = os.environ.get("E_ANTAGONIST", "0") == "1"
_COACTIVATION = float(os.environ.get("E_COACTIVATION", "0.3"))
_INVPROBE = os.environ.get("E_INVPROBE", "0") == "1"  # 1=逆モデルStage1診断（学習後に1回）
_INVEXEC = os.environ.get("E_INVEXEC", "0") == "1"  # 1=逆モデルStage1.5＝推論a*の実行テスト
_GOALBABBLE = os.environ.get("E_GOALBABBLE", "0") == "1"  # 1=Goal Babbling(目標指向の探索)
_GB_SWITCH = os.environ.get("E_GB_SWITCH", "pe")  # 探索/目標の切替: fixed(i%2) / ne(NE) / pe(予測誤差+NE, 既定)
_CLPROBE = os.environ.get("E_CLPROBE", "0") == "1"  # 1=C4診断＝閉ループ制御 vs 開ループの到達比較
_CLTRAIN = os.environ.get("E_CLTRAIN", "0") == "1"  # 1=閉ループreaching訓練（held goal＋停止条件）
_INVTRAJ = os.environ.get("E_INVTRAJ", "0") == "1"  # 1=各チェックポイントで逆probeも回し、逆モデルの"天井"軌跡(recover_corr/star)を記録
# 内発的動機の切替（既定＝従来の「予測しやすさ」）。Cは触覚なし・margin+51が確立した唯一の
# "うまくいくと分かっている環境"なので、ここで動機だけを差し替えれば**触覚という交絡なしに
# 動機の良し悪しだけ**を判定できる（D0では予測対象の89%が触覚で交絡していた）。
#   progress : pe_slow−pe_fast＝学習進度（Oudeyerの好奇心。誤差が"減っている"ことを求める）既定
#   predict  : 1/(1+誤差)＝予測しやすい状態を求める（旧・注意既知の欠陥あり。下記）
# 【2026-07-25 既定を predict → progress に変更】
#   predictは「大きく動くほど感覚変化がノイズを超えて予測しやすい」ため**大行動バイアス**を持つ
#   （逸脱リストで【最重要・報酬の根本欠陥】と自分で記録済み＝暗い部屋問題の一種）。
#   にもかかわらず既定が predict のままで、margin+58.8 の学習もそれで回っていた。
#   実測（age=0・仰向け・白紙7000step・筋肉モデル・同一seed）：
#     | 報酬 | margin | うつ伏せ% | 体幹回転(最大) | |qvel| | jerk |
#     | predict  | +58.8 | **57.5%** | 105.8°(最大180.0°) | 0.722 | 2501 |
#     | progress | +45.9 | **1.1%**  | 52.6°(最大91.9°)   | 0.256 | 1123 |
#   ＝報酬を1つ変えるだけで**うつ伏せが52分の1**、動きの量1/2.8、jerk 1/2.2。
#   marginが下がるのは想定内（predictは指標を水増しする）。仰向けを保ったままC5当時(+48)と
#   同水準を達成＝**こちらが人間的な条件での本来の姿**。
#   → [検証の落とし穴チェックリスト 項29](../../doc/検証の落とし穴チェックリスト.md)
#      「良い方を既定値にする」に従い既定を変更。predictは対照実験用にのみ残す。
_REWARD = os.environ.get("E_REWARD", "progress")
if _REWARD == "predict":
    print("注意 E_REWARD=predict は【既知の欠陥】があります（大行動バイアス＝"
          "大きく動くほど得なので暴れる。実測でうつ伏せ57.5%・jerk2501）。"
          "対照実験でなければ progress を使ってください。", flush=True)
# NEを相対化するか。2026-07-25：既定を1に変更（progressと対で使う必要があるため）。
# 従来は報酬の絶対値に固定閾値(<0.1で探索/>0.3で活用)を当てるので、値域の違う報酬関数を
# 入れると壊れる（学習進度≒0.04は常に「報酬ゼロ」と誤認される）。
# relative=Trueは「長期基準線と比べていつもより良いか」で決める＝尺度非依存（D0で必要性が判明）。
_NE_REL = os.environ.get("E_NE_RELATIVE", "1") == "1"
# 【2026-07-15】姿勢と触覚の切替。どちらも既定OFF＝従来の立位・触覚なしのCと完全に同じ。
#   E_SUPINE=1 : 仰向けで開始する。録画で判明した通り、既定(立位)のCは**開始3秒で転倒し、
#     以降ずっと床でもがいている**。margin+51はその状態で出た数字。仮説＝「Cが成功したのは
#     転んで偶然"手足が自由に振れる状態"になったから」。仰向けはそれを意図してやる。
#     シーンはCと同一(benchmarkv2)のまま姿勢だけ変える＝比較で変わる要素は姿勢1つだけ。
#   E_TOUCH=1  : 触覚を足す（乳児acuity＝somatotopy比を保ったまま解像度を2倍粗く）。
#     予測対象が固有感覚621→固有感覚+触覚 になるので、評価は必ず分けて見る（合計だけ見ると
#     触覚の次元数に薄められて何も分からない＝D0で踏んだ罠）。
_SUPINE = os.environ.get("E_SUPINE", "0") == "1"
_TOUCH = os.environ.get("E_TOUCH", "0") == "1"
# 触覚を「予測対象」にするか「入力（文脈）」だけにするか。既定は従来どおり予測対象。
#   target : 触覚を fusion の入力にもし、**予測対象にもする**（従来。仰向け3シードで
#            触覚 persist 97〜104%＝学べない）
#   input  : 触覚を fusion の入力にはするが、**予測対象からは外す**（＝予測するのは固有感覚だけ）
# 【根拠・2026-07-15】同じ入力(真の物理状態)・同じデータ・同じ分割・同じ出力次元(55)で
# 予測対象だけを変えた対比：固有感覚 +26.5% / 温度 −53.1% / 接触力 −54.4%（検証R²）。
# ＝接触に由来する信号は、姿勢からの予測が原理的には可能でも**実質的に不可能**（接触は
# 姿勢の不連続な関数）。温度で滑らかにしても解決しなかった（仮説は棄却済み）。
# → **触覚を予測対象にすること自体が誤り**の可能性。触覚は「予測するもの」ではなく
#   「予測に使う手がかり」ではないか、を測るための切替。
_TOUCH_MODE = os.environ.get("E_TOUCH_MODE", "target")
# 【体性感覚系(触覚)の脳内経路化・2026-07-24】E_SOMATOSENSORY=1 で従来の TouchEncoder
# (1枚の巨大変換層)を SomatosensoryCortex(部位別集約=視床VPL相当+統合=S1相当)に差し替える。
# 同時に「B案」(予測は生の触覚でなく触覚embedで行う)も自動有効化：ln_prop が触覚を
# 予測対象に足す場合、target_fusion.touch(obs["touch"])のembed(64次元)を対象にする。
# 根拠は taro_core/src/senses/somatosensory_cortex.py の冒頭docstring参照(視床VPL・S1・
# 予測符号化階層の文献)。既定OFF＝従来と1バイト差なし。
_SOMATOSENSORY = os.environ.get("E_SOMATOSENSORY", "0") == "1"
# 【Viewer】E_VIEW=1 で学習せずリアルタイムViewer再生。E_LOADMODELでチェックポイントを指定する
# 必要がある(白紙脳の再生は無意味)。E_REALTIME=1で等倍速、既定=最速。
# E4_CONTINUOUS=1 で連続制御(1秒ホールドを解消、動きが滑らかになる)。既定OFFで従来の学習と
# 同じ挙動(1秒に1回の判断→100tick同じ命令保持)。連続制御ONにすると訓練と挙動は違うが、
# 動きは人間らしく滑らかに見える。目視の目的次第で使い分ける。
_VIEW = os.environ.get("E_VIEW", "0") == "1"
_REALTIME = os.environ.get("E_REALTIME", "0") == "1"
_E4_CONTINUOUS = os.environ.get("E4_CONTINUOUS", "0") == "1"
_CTRL_M = int(os.environ.get("E_CTRL_M", "100"))  # 連続制御の刻み(既定100=1秒ホールド、10=100Hzで再生成)
# 省メモリ版（絵を落とす。物理は不変・視覚ONなら自動で素に戻る）。詳細は D/scripts/mimo_lean.py。
# 1本 2.64GB→0.28GB＝同時実行 6本→約22本。既定ON（仰向けは常に省メモリ版の上に載る）。
# E_LEAN=0 で従来の素のモデルに戻せる（アブレーション/描画品質が要るとき用）。
_LEAN = os.environ.get("E_LEAN", "1") == "1"
_ENV_ID = "TaroSupine-v0" if _SUPINE else ("TaroBenchV2Lean-v0" if _LEAN else "MIMoBenchV2-v0")
# 【目標E1・2026-07-20】E_E1=1 で E/scripts/e_toy_env.py の ToySupineEnv を使う。
#   仰向け＋柵（＝Ferrari 2007 のnest相当の支持）＋視覚的に貧しい環境（＝White 1966）。
#   おもちゃ・柵・見た目は e_toy_env 側の環境変数（E_TOY_OBJ / E_FENCE / E_PLAIN）で切る。
# 【段階1（いまここ）】視覚は**入力にだけ**入れ、予測対象は固有感覚のまま＝C5と同条件で
#   自己モデルを建て直す。理由＝目標C(自己モデル)→目標E(運動発達)の発達順を飛ばさないため。
#   視覚を足したことで pc_latent など6層が作り直しになったので、まず土台を戻す。
# 【段階2（次）】E_E1_TARGET=1 で視覚を**予測対象**にも入れる（E/scripts/e_target.py）。
#   そこで初めて「自分の手を見る」が報酬を生みうる状態になる。
_E1 = os.environ.get("E_E1", "0") == "1"
_E1_VISION = os.environ.get("E_E1_VISION", "1") == "1"      # E1での視覚入力（0でアブレーション）
# 予測対象に何を入れるか。3条件を比較するための切替：
#   "0"      … 固有感覚のみ（従来＝対照条件A）
#   "vision" … 固有感覚 + 視覚（条件B。"1" も同じ扱い＝後方互換）
#   "all"    … 固有感覚 + 内受容 + 前庭 + 触覚 + 視覚（条件C＝**全感覚**）
# 【条件Cを作る理由】2026-07-20 の実測で「**予測対象に入っていない感覚は内部表現 z から
# 捨てられる**」と判明した（視覚を入力にだけ入れた段階1のモデルは、画面を真っ黒にしても
# z が 0.00% しか変わらなかった）。同じ理屈で触覚・内受容・前庭も捨てられているはずで、
# 人間の脳が一部の感覚だけ予測しているとは考えにくい＝**全感覚を予測するのが人間模倣**。
# BとCを比べると「視覚だけ入れる」という選り好み自体の是非が分かる。
_E1_TGT = os.environ.get("E_E1_TARGET", "0").lower()
_E1_TARGET = _E1_TGT in ("1", "vision", "all")
_E1_TARGET_ALL = _E1_TGT == "all"


def _touch_params():
    if not _TOUCH:
        return None
    from d_supine_env import infant_touch_params
    return infant_touch_params(2.0)
CSV_COLUMNS = ["life_min", "train_step", "classify", "margin", "corr", "persist",
               "agency", "mag_ratio", "real_min", "hand_in_view"]


def hand_in_view_rate(model, data):
    """E1の主指標：手が視野に入っているか（1tickの判定）。

    判定の実体は `E/scripts/e_hand_in_view.py` の `hand_in_view()` に**一本化**してある
    （測定スクリプトと学習ループで基準がズレると比較不能になるため）。
    - **左右どちらかの眼**で視野内なら「見ている」とみなす（2026-07-20 変更。
      切り出し動画の目視で「右目には大きく映っているのに左目には映っていない」場面が
      多いと分かったため。人間側の観察研究も両眼視は問わない）
    - `E_HV_MODE=both` で旧基準（両目とも）に戻せる＝アブレーション用
    注意：人間側に比較できる実測値は存在しないので、判定は**太郎の中での前後比較**で行う。
    """
    # 注意：import失敗を握りつぶさない。黙って0.0を返すと「手が一度も視野に入らなかった」という
    #   結果が静かに出て、E1の結論を誤らせる（＝今日9件出したバグと同じ構造の事故）。
    from e_hand_in_view import hand_in_view
    return float(hand_in_view(model, data))


# MinimalFusion は taro_core/src/senses/fusion.py へ抽出済み（doc/移行記録_taro_core化_2026-07-17.md）。
# ここでは import して従来どおり使う（`from run_c_metrics_ac_lr import MinimalFusion` する
# D側スクリプトの後方互換のため、この名前で再エクスポートする）。視覚を足す拡張(vision_res)も
# 抽出先に入っているので、本番の視覚ONはそちらを使う。
from fusion import MinimalFusion  # noqa: F401  （再エクスポート）


_TGT_FUSION = None      # 正解側の**凍結**融合層（run内で設定）。E_E1_TARGET=1 のとき視覚に使う
_PROP_DIM = None        # 固有感覚の次元数（予測対象の先頭ブロックの長さ）。run内で設定
_BLOCKS = None          # 予測対象のブロック境界 [(start, end, 名前), ...]。ln_prop が構築
# 各ブロックの重み λ。段階1では全部 1.0（＝感覚ごとに平等）から動かさない。
# 段階2で振って比較する予定（doc/やることリスト.md）。E_LAM_V は視覚ブロック専用の指定。
_LAM_V = float(os.environ.get("E_LAM_V", "1.0"))


def block_pe(pred, target):
    """予測誤差＝**ブロックごとに平均してから足す**（次元数の影響を除く）。

    【なぜ（2026-07-20・段階1）】
    従来は連結したベクトル全体を1回で平均していたので、**寄与が次元数比で決まっていた**：
      固有感覚621 + 視覚64 → 視覚の寄与はわずか 9.3%
    これは「視覚が重要でない」という判断ではなく、**次元数という無関係な量**が
    勝手に重みを決めてしまっている状態。同じ罠を目標D0とE1で計3回踏んでいる。

    → **各ブロックの平均を取ってから足す**と、寄与は次元数によらず λ_v で決まる。
      λ_v=1.0 なら 50:50。**これは「重み付け」ではなく無関係な量の影響を除く操作**なので
      恣意的ではない。根拠＝Ohata & Tani 2020（`1/(2Rp)`, `1/(2Rv)` で次元数で割る）、
      Idei et al. 2025（固有感覚28 vs 視覚32,256＝1,150倍差を正規化のみで処理）、
      Ichiwara & Ogata 2022（各項を `1/(H·W·C)` で正規化）、MoPoE-VAE（次元比でスケール）。
      [参考文献リスト §目標E-17](../../doc/参考文献リスト.md)

    注意：(1+λ_v) で割るのは**全体のスケールを保つ**ため。割らないと視覚を足したときだけ
      誤差が約2倍になり、他の損失（KL・恒常性）との比が変わって「視覚を足した効果」と
      「学習率が実質変わった効果」が混ざる（＝交絡）。
    注意：次元を無理に揃える案は**採らない**。どの文献もやっておらず、1,150倍差でも
      正規化だけで扱えている実例がある。
    """
    if not _BLOCKS or len(_BLOCKS) <= 1:
        return mse(pred, target)                     # 従来と完全に同一（固有感覚のみの条件A）
    tot, wsum = 0.0, 0.0
    for (s, e, nm) in _BLOCKS:
        lam = _LAM_V if nm == "vision" else 1.0      # 段階1では視覚も1.0＝全ブロック平等
        tot = tot + lam * mse(pred[..., s:e], target[..., s:e])
        wsum += lam
    return tot / max(wsum, 1e-9)


def ln_prop(obs):
    """予測対象。既定は固有感覚のみ（従来のC）。

    E_TOUCH=1 かつ E_TOUCH_MODE=target のときだけ触覚を予測対象に加える。
    E_TOUCH_MODE=input なら触覚は fusion の入力にだけ入り、予測対象は固有感覚のまま。

    【E_E1_TARGET=1（目標E1・段階2）】視覚エンコーダの出力64次元を予測対象に**足す**。
    これが hand regard（自分の手を見る）の実験の本体：予測対象に入っていないものは
    progress報酬を生まないので、視覚を入れて初めて「手を見ると得をする」状態になる。
    2つの設計判断（詳細は E/scripts/e_target.py）：
      (1) **凍結した別インスタンス**のエンコーダを使う（RND式）。予測側と正解側が同じ
          学習中のエンコーダだと「出力を平坦にすれば当たる」抜け道で崩壊する（目標Cで実際に踏んだ）。
      (2) **固有感覚と視覚を別々に layer_norm** してから連結する。全体を一度に正規化すると
          621次元が平均・分散を支配して64次元が埋もれる（D0・E1で3回踏んだ希釈の罠）。
    注意：それでもMSEへの寄与は次元数比のまま（64/685=9.3%）。重み付けは**恣意的になる**ので
      今はしない。まず等重みで回し、足りなければ精度(precision)の議論として扱う。
    """
    ln = torch.nn.functional.layer_norm
    v = to_tensor(obs["observation"])
    # 【B案：触覚は生でなくembedを予測】E_SOMATOSENSORY=1 のときは、生の触覚を予測対象に
    # 足すのでなく、後で target_fusion.touch(...) の64次元embedをブロックとして足す。
    # ここでは生の触覚を prop に混ぜない（S1相当のembedはfor loop側で追加される）。
    if _TOUCH and _TOUCH_MODE == "target" and not _SOMATOSENSORY:
        v = torch.cat([v, to_tensor(obs["touch"])])
    parts = [ln(v, v.shape).detach()]
    names = ["prop"]
    f = _TGT_FUSION
    # 【B案：触覚embedを予測対象に足す】E_SOMATOSENSORY=1 かつ E_TOUCH_MODE=target のとき、
    # target_fusion(凍結)の SomatosensoryCortex を通して 64次元embedを作り、独立ブロックとして
    # 予測対象に加える。生の触覚を予測対象にしないため、予測ヘッドの出力次元と誤差計算量が
    # 劇的に減る(生1968次元→embed 64次元)。RND式で凍結側を使うので崩壊対策あり。
    if _SOMATOSENSORY and _TOUCH and _TOUCH_MODE == "target" and f is not None:
        if getattr(f, "touch", None) is not None and "touch" in obs:
            with torch.no_grad():
                e = f.touch(to_tensor(obs["touch"]))
            parts.append(ln(e, e.shape)); names.append("touch_embed")
    if _E1_TARGET and f is not None:
        with torch.no_grad():                   # 正解側は勾配を流さない（RND式）
            # 条件C＝固有感覚 + 前庭 + 触覚 + 視覚。
            # 内受容は入れない（2026-07-20 の文献調査による判断）：
            #   人間は内受容の予測誤差を**自律反射**（心拍・血管）で解消するが、太郎には
            #   その出力が無い＝**誤差を減らす手段が構造的に存在しない**ので、予測対象に
            #   入れても progress報酬（＝予測が上達した分）が生まれない。
            #   加えてE1の interoception は定数 [0.3,0,0,0.3] で中身が無い（逸脱リスト参照）。
            #   注意：内受容は既に homeostasis.py で**報酬系**に使われており、予測対象にも
            #   入れると同じ信号が二重に効く。
            if _E1_TARGET_ALL:
                for nm, enc, key in (("vest", getattr(f, "vestibular", None), "vestibular"),
                                     ("touch", getattr(f, "touch", None), "touch")):
                    if enc is None or key not in obs:
                        continue
                    e = enc(to_tensor(obs[key]))
                    parts.append(ln(e, e.shape)); names.append(nm)
            if getattr(f, "vision", None) is not None and "eye_left" in obs:
                e = f.vision(obs["eye_left"], obs["eye_right"])
                parts.append(ln(e, e.shape)); names.append("vision")
    global _BLOCKS
    if _BLOCKS is None or len(_BLOCKS) != len(parts):
        _BLOCKS = []
        o = 0
        for nm, p in zip(names, parts):
            _BLOCKS.append((o, o + int(p.shape[-1]), nm))
            o += int(p.shape[-1])
    return torch.cat(parts, dim=-1).detach() if len(parts) > 1 else parts[0]


def run(seed, n_train=3600, K=100, ckpt=600, n_eval=80):
    torch.manual_seed(seed); np.random.seed(seed)
    os.makedirs(LOG_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(LOG_DIR, f"ac_metrics_seed{seed}_{stamp}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        csv.writer(fp).writerow(CSV_COLUMNS)

    def log_row(row):
        with open(csv_path, "a", newline="", encoding="utf-8") as fp:
            csv.writer(fp).writerow(row)

    invtraj_path = os.path.join(LOG_DIR, f"inv_traj_seed{seed}_{stamp}.csv")
    if _INVTRAJ:
        with open(invtraj_path, "w", newline="", encoding="utf-8") as fp:
            csv.writer(fp).writerow(["train_step", "recover_corr", "infer_over_random",
                                     "star_over_random", "model_err_star"])

    _age_kw = {"age": float(_AGE)} if _AGE else {}  # 【taro-C5】月齢指定で成長モジュール適用
    # 【2026-07-25】体型補正（新生児プロポーション）を**学習でも**適用する。
    # 【なぜ今まで効いていなかったか】体型補正は `e_toy_env.py`（おもちゃ環境）の中に
    #   書かれており、**おもちゃ環境でしか効かなかった**。学習は仰向け環境
    #   （SupineMimoEnv）を使うので、太郎は**成人プロポーションのまま学習していた**
    #   （mimoGrowth の age=0 は「大きさは新生児・比率は成人」。上肢/下肢=0.77 に対し
    #    人間の新生児は 1.07＝腕の方が長い＝**逆転**）。
    #   Viewerで「足が根元からすごい動く」とユーザーが指摘したのが発覚のきっかけ
    #   （脚が相対的に長いと、股関節が少し回るだけで足先が大きく振れる）。
    # 体型の定義は core（`taro_core/src/body/infant_body.py`）へ移設済み。ここは
    # 環境変数の読み取り（`e_body_config.py`）を経由して受け取るだけ。
    # 注意：E_SHAPE=0 で補正を切れる（＝素のmimoGrowth体型＝アブレーション／過去実験の再現）。
    # 注意：【2026-07-25 配線の修正】ここは以前 `body_scale_custom_from_env` と
    #   `head_elongation_from_env` を**個別に呼んでいた**ため、あとから
    #   `e_body_config.body_kwargs_from_env`（身体設定の唯一の入口）に足された引数が
    #   **この学習ループには一切届いていなかった**。実害：
    #     ・`distal_mass`（末端の質量）… 感度分析4条件のモデルの重みが完全一致（＝無効）
    #     ・`limb_scale`（四肢の筋力）… 同じく、振った条件が同一モデルだった（＝無効）
    #     ・`flexion`（生理的屈曲）… Viewerに届かず、ユーザーの目視「膝が曲がっていない」で発覚
    #   ＝「身体の設定が散らばる」問題を core と e_body_config で解決したのに、
    #     **この呼び出し側だけが古い経路のまま取り残されていた**。
    #   → 入口を使う形に統一する。以後、身体の引数が増えてもここは直さなくてよい。
    #   注意：条件を振る実験をしたら **同じシードの2条件でモデルの重みが違うことを必ず確認する**
    #     （E/scripts/e_condition_check.py。チェックリスト項43）。
    # 【2026-07-29】月齢から体の引数を組み立てる部分を関数にした。学習の途中で
    #   体を作り直す（age_at で月齢を進める）ときに、**同じ組み立てを使い回す**ため。
    #   注意：ここを2箇所に書くと、片方だけ引数が増えて条件が食い違う（実際に2026-07-25 に
    #     `body_scale_custom_from_env` を個別に呼んでいたせいで感度分析が無効になった）。
    def _body_kwargs_for_age(age):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from e_body_config import body_kwargs_from_env, shape_enabled
        kw = {"age": float(age)}
        bk = body_kwargs_from_env(float(age))
        if not shape_enabled():
            # E_SHAPE=0＝素のmimoGrowth体型（アブレーション／過去実験の再現）。
            # 注意：旧版はこの print が別の条件の else に付いていたため、おもちゃ環境では
            #   補正が効いているのに「補正OFF」と**誤表示**していた。
            bk.pop("custom_measurements", None)
            bk["head_elongation"] = 1.0
            print("[body] 体型補正OFF（E_SHAPE=0）＝素のmimoGrowth体型", flush=True)
        if not _SUPINE:
            # おもちゃ環境（ToySupineEnv）は自前で楕円化するので渡さない（二重適用を防ぐ）。
            bk["head_elongation"] = 1.0
        kw.update(bk)
        # 体を作り直す実験では「同じ月齢なら同じ引数か」が生命線なので必ず出す
        print("[body-kw] age=%s: %s custom=%s" % (
            age, {k: v for k, v in kw.items() if k != "custom_measurements"},
            bool(kw.get("custom_measurements"))), flush=True)
        return kw

    if _AGE is not None:
        _age_kw = _body_kwargs_for_age(float(_AGE))
    # 注意：【2026-07-30 撤回】ここで SpringDamperModel を既定として渡していた。
    #   「学習ループの既定に合わせる」という判断だったが、それは
    #   **学習ループの既定そのものが逸脱していた**ので、分裂を固定するだけだった。
    #   → 駆動モデルを渡さない（None）ときは e_scene.build の既定＝筋肉モードに従う。
    _act_kw = {}
    if _MUSCLE:   # 【筋肉モデル】拮抗筋2本/関節・活性化ダイナミクス・引くだけ
        from mimoActuation.muscle import MuscleModel
        _act_kw = {"actuation_model": MuscleModel}
        print("[actuation] MuscleModel（拮抗筋2本/関節、action=180次元[0,1]）")
    elif _SMOOTH:  # 【taro-C5】①活性化ダイナミクス
        sys.path.insert(0, os.path.join(_CORE, "src", "body"))
        from smooth_actuation import SmoothTorqueModel
        _act_kw = {"actuation_model": SmoothTorqueModel}
    _vres = 0
    if _E1:      # 目標E1：おもちゃ環境（ToySupineEnv）。視覚は既定ONで**入力にだけ**入る
        sys.path.insert(0, os.path.join(_BRIDGE, "scripts"))   # = Taro/E/scripts（このファイルと同じ場所）
        from e_toy_env import ToySupineEnv, infant_vision_params, VISION_RES
        _vp = infant_vision_params() if _E1_VISION else None
        _vres = VISION_RES if _E1_VISION else 0

    # 【2026-07-29】環境を作る部分も関数に。学習の途中で体を作り直すときに使い回す。
    #
    # 【2026-07-30 追加】E_SCENE でシーンから環境を作れるようにした。
    #   【なぜ要るか】学習ループは環境変数だけで環境を組み立てていたので、
    #     リクライニング角・椅子の固定・四肢の筋緊張・おもちゃの位置が**指定できず**、
    #     太郎は常に「平らな床に仰向け」でしか学習できなかった。
    #     ＝リーチング実験（リクライニング60度）を学習ループで回せなかった。
    #   注意：E_SCENE が空なら従来と完全に同じ動作（既存実験は壊れない）。
    #   注意：月齢はシーンの age_months ではなく**学習ループ側の月齢**で上書きする。
    #     体を育てる（E_AGE_TO）と月齢が学習中に変わるため、シーンの固定値では合わない。
    #     ＝シーンは「月齢以外の環境」を担い、月齢は成長スケジュールが担う。
    def _make_env(age_kw):
        if _SCENE_NAME:
            import e_scene
            sc = e_scene.load(_SCENE_NAME)
            sc["body"]["age_months"] = float(age_kw.get("age", sc["body"]["age_months"]))
            sc["fingerprint"] = None      # 月齢を差し替えたので保存時の指紋とは一致しない
            env0, _hands = e_scene.build(
                sc, orient=False, vor=True, seed=seed, verbose=False,
                actuation_model=_act_kw.get("actuation_model"))
            return HybridEnv(env0)
        if _E1:
            return HybridEnv(ToySupineEnv(vision_params=_vp, touch_params=_touch_params(),
                                          **age_kw, **_act_kw))
        return HybridEnv(gym.make(_ENV_ID, vision_params=None, touch_params=_touch_params(),
                                  **age_kw, **_act_kw))

    env = _make_env(_age_kw)
    if _E1:
        print(f"[E1] 環境=ToySupineEnv／視覚={'ON' if _E1_VISION else 'OFF'}"
              f"／視覚を予測対象に={'YES(段階2)' if _E1_TARGET else 'NO(段階1)'}")
    # 触覚の次元数は「センサ点の総数」＝モデル構築時点で確定しており、reset不要で取れる。
    # ここで touch_dim を知るために env.reset() を足すと**乱数を1回余計に消費して学習の
    # 乱数列がずれる**（落とし穴チェック項3）。触覚なし条件が従来のCと比較不能になるので厳禁。
    # 観測空間はenv構築時に確定しているのでresetなしで読める。
    # 注意：get_sensor_count()は「センサ点の数」(1202)で、観測は1点あたり力の3成分＝3606。
    touch_dim = int(env.observation_space["touch"].shape[0]) if _TOUCH else 0
    # 固有感覚(observation)の次元は身体・アクチュエータで変わる（SpringDamper=621, MuscleModel=801）。
    # env.observation_space から実測してfusionに渡す（reset前に読めるので乱数列も汚さない）。
    _prop_dim = int(env.observation_space["observation"].shape[0])
    # 【体性感覚系の脳内経路化】E_SOMATOSENSORY=1 のとき、触覚センサの部位別配置(視床VPL相当)を
    # 環境から取り出してMinimalFusionに渡す。SomatosensoryCortexが1枚の巨大変換層を置き換える。
    _touch_map = None
    if _SOMATOSENSORY and _TOUCH:
        from somatosensory_cortex import build_touch_map_from_env
        _touch_map = build_touch_map_from_env(env)
        assert _touch_map.total_dim == touch_dim, \
            f"触覚の地図 {_touch_map.total_dim} != touch_dim {touch_dim}"
        print(f"[体性感覚系] SomatosensoryCortex を有効化：部位数="
              f"{len(_touch_map.group_names)}, 触覚総次元={touch_dim}", flush=True)
    fusion = MinimalFusion(touch_dim, vision_res=_vres, proprio_dim=_prop_dim,
                            touch_map=_touch_map)
    target_fusion = MinimalFusion(touch_dim, vision_res=_vres, proprio_dim=_prop_dim,
                                   touch_map=_touch_map).freeze()
    if _SOMATOSENSORY and _TOUCH and fusion.touch is not None:
        print(fusion.touch.summary(), flush=True)
    global _TGT_FUSION
    _TGT_FUSION = target_fusion      # ln_prop が視覚を予測対象に足すのに使う（E_E1_TARGET=1）
    n_env_act = env.action_space.shape[0]
    # 拮抗筋モード：脳・CPG・prev_a・log_probはすべて n_joint 次元、環境には to_env_action で
    # 2*n_joint に写像して渡す。既定は n_act == n_env_act で従来と一致。
    n_act = n_env_act // 2 if (_MUSCLE and _ANTAGONIST) else n_env_act
    # 【バグ修正・2026-07-15】最初のresetに必ずseedを渡す。
    # 環境の乱数(`env.unwrapped.np_random`)は gym が別に管理しており、torch.manual_seed も
    # np.random.seed も効かない。仰向け環境は reset のたびに np_random で初期姿勢を揺らす
    # （jitter）ので、**シードなしresetだと毎回ちがう姿勢から始まり、シードで再現できない**。
    # 立位Cはjitterが無いので影響を受けず、この穴は仰向けを足すまで露呈しなかった。
    # 症状：同一シード・同一条件のはずの2ランで life=0min が 58.99 vs 58.20 とズレる。
    obs, _ = env.reset(seed=seed)
    sdim = fusion.encode(obs).shape[0]; prop_dim = to_tensor(obs["observation"]).shape[0]
    # nat_headが吐く次元＝予測対象の次元。touch_mode=input なら触覚は予測対象に入らない。
    # 予測ヘッドの出力次元＝予測対象の次元。実際に ln_prop を1回通して**測る**
    # （手計算だと視覚64次元の足し忘れ等でズレる。ここは合わせないと学習が壊れる）。
    global _PROP_DIM
    _PROP_DIM = prop_dim          # ブロック分割の境目（固有感覚の次元数）
    out_dim = int(ln_prop(obs).shape[0])
    if _E1_TARGET:
        _bd = "／".join(f"{nm}:{e-s_}" for (s_, e, nm) in (_BLOCKS or []))
        print(f"[E1] 予測対象={_E1_TGT} → 全{out_dim}次元  内訳 {_bd}")
        print(f"     誤差はブロックごとに平均してから足す（次元数の影響を除く。λ_v={_LAM_V}）")
    # 【2026-07-25】proprio_dim（予測対象の次元）を渡す＝core の forward_model_head が
    # この目標に合った出力次元で作られる。従来は目標側で nat_head を別に作っていた。
    brain = TaroBrainWithMotor(vocab_size=3, sensory_dim=sdim, n_actuators=n_act,
                               proprio_dim=out_dim)
    # 【運動性喃語（脊髄CPG）】E_NOISE=colored のとき太郎の中で色付き探索を有効化する。
    # 既定 white では呼ばれない＝spinal_cpg=None＝白色ガウス＝従来と数値完全一致。
    if _E_NOISE == "colored":
        # 【2026-07-25・シナジーを筋肉モードに配線】従来はここで
        #   `_use_syn = _E_SYNERGY and not _MUSCLE` として**強制OFF**していた。
        #   理由は「_LEG_R等が90-actuator indexで、筋肉モードは180次元だから」という
        #   技術的なもの。しかし学習は常に筋肉モードで回しており、結果として
        #   文献が一致して言う新生児の特徴「まとめてしか動かせない」が
        #     一度も効いていない状態で全実験をしてきた（本日の目視から発覚）。
        #   → CPG側に pair_offset を入れて、対になる筋（伸ばす側）に符号反転で
        #     同じシナジーを混ぜるようにした（詳細は spinal_cord/cpg.py）。
        # 注意：拮抗筋モード(_ANTAGONIST)ではCPGが関節次元で動き explore 側で写像するので
        #   pair_offset は不要（0のまま＝従来の挙動）。
        _use_syn = bool(_E_SYNERGY)
        _pair_offset = (n_act // 2) if (_MUSCLE and not _ANTAGONIST) else 0
        brain.enable_spinal_babble(n_act, leg_r=_LEG_R, leg_l=_LEG_L, arm_r=_ARM_R, arm_l=_ARM_L,
                                   beta=_E_BETA, synergy=_use_syn, syn_w=_E_SYN_W, seed=seed,
                                   antagonist=(_MUSCLE and _ANTAGONIST),
                                   co_activation=_COACTIVATION, pair_offset=_pair_offset)
        print(f"[脊髄CPG] 色付き探索ON: β={_E_BETA} synergy={_use_syn} syn_w={_E_SYN_W}"
              f"{f' pair_offset={_pair_offset}(筋肉モードの拮抗筋ペアに符号反転で適用)' if _pair_offset else ''}"
              f"{' 【拮抗筋モードON】coactivation=' + str(_COACTIVATION) if _MUSCLE and _ANTAGONIST else ''}", flush=True)
    emb_dim = brain.sensory_proj.out_features  # GRUの入力次元(=64)

    # 【2026-07-25】D-a/D-b の層を**太郎の中（core）のものに一本化**した。
    # 従来はここで emb_proj / nat_head を別に作っており、core にある
    # motor_input_proj / forward_model_head は**作られるだけで一度も使われていなかった**
    # ＝太郎の脳が二重に存在し、実際に動いていたのは目標側という状態だった（構造監査で発覚）。
    # 構造・初期化とも core 側と完全に同型（Linear / MLP+LayerNorm、hidden=128）。
    # 注意：層の名前が変わるので**保存済みチェックポイントは読めなくなる**。既存モデルは
    #   欠陥のある報酬(predict)・補正なしの体・劣化した睡眠リプレイで学習したもので
    #   どのみち作り直しなので、学習しなおし前提で進める（ユーザー判断 2026-07-25）。
    emb_proj = brain.motor_input_proj      # 旧名を別名として残す（参照箇所が多いため）
    nat_head = brain.forward_model_head
    learner = TaroLearner(CombinedParams(brain, fusion), lr=_LR)
    dop = Dopamine(); ne = LocusCoeruleus(relative=_NE_REL); homeo = HomeostaticScaling(dim=sdim)
    dev_clock = DevelopmentalClock()  # ③発達年齢（累積学習回数）。sim秒(②)とは別軸。
    # 運動小脳。ON/OFFで乱数列を揃えるため、_CEREBに関わらず常に構築する（使う/学習する
    # のは_CEREB時のみ。gate/imitationは乱数を消費しないので、初期化以降はON/OFFで乱数が一致）。
    cereb = MotorCerebellum(brain.latent_dim, n_act)
    cere_opt = torch.optim.Adam(cereb.parameters(), lr=_LR)

    # 【taro-C5】継続学習：学習済みモデルを読み込む（脳をリセットしない）。形が合う層だけロード。
    if _LOADMODEL:
        def _load_matching(module, sd, tag):
            own = module.state_dict()
            matched = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
            module.load_state_dict(matched, strict=False)
            print(f"  [{tag}] ロード{len(matched)}/{len(own)}層", flush=True)
        _blob = torch.load(_LOADMODEL, map_location="cpu", weights_only=False)
        print(f"継続学習：{os.path.basename(_LOADMODEL)} を読み込み", flush=True)
        _load_matching(brain, _blob["brain"], "脳")
        fusion.insula.load_state_dict(_blob["fusion_insula"])
        fusion.proprio.load_state_dict(_blob["fusion_proprio"])
        fusion.vestibular.load_state_dict(_blob["fusion_vestibular"])
        # 【2026-07-25】emb_proj/nat_head は brain.motor_input_proj /
        # brain.forward_model_head になった＝上の _load_matching(brain, ...) に含まれる。
        # 旧チェックポイント（別名で保存されたもの）は層名が違うので読めない＝学習しなおし。
        if "emb_proj" in _blob:
            print("  [注意] 旧形式のチェックポイント（emb_proj/nat_headが別層）です。"
                  "層の構成が変わったため、その2層は読み込まれません＝学習しなおしになります。",
                  flush=True)
        if _CEREB and "cereb" in _blob:
            _load_matching(cereb, _blob["cereb"], "小脳")

    state = {"obs": obs, "hidden": brain.init_motor_hidden(),
             "prev_a": torch.zeros(n_act)}
    t0 = time.time()

    # 【Viewer】E_VIEW=1のときは学習せず、リアルタイム再生に入る（E_LOADMODEL済み前提）。
    # 連続制御（E4_CONTINUOUS=1）で1秒ホールドを解消できる（動きが滑らかに）。既定OFFなら
    # 学習と同じ1秒ホールドで再生（学習挙動をそのまま見る）。d_c5_motor_quality.pyのrun_viewを
    # e_growth_train用に適応した最小版：探索なし・報酬計算なし・純粋な再生のみ。
    # ヘルパー zc/act_mean/step_k/reset_state はまだ下で定義されるので、そこまで進んでから
    # 分岐する（現状 return はコード後半のヘルパー定義後に置く）。
    eff_accum, act_accum = [], []  # 【taro-C5】努力コストと|行動|の記録（フリーズ監視用）
    # 【階層化・変更②】‖a_t − a_{t-1}‖² の記録＝**CAPSが効いているかの直接証拠**
    # （設計メモ §7「②の効果」）。値が下がる＝行動を変えなくなった＝「何もしない」を選べている。
    caps_accum = []
    # 【taro-C5】努力コストの重み：筋力(最大トルク)が大きい筋ほど動かすとコストが高い（代謝の
    # 標準：活性化²×筋サイズ）。activation は正規化された行動 a∈[-1,1] を使う（cost()はトルク単位を
    # 二乗し実質トルク³で桁が狂うため不採用）。重みは合計1に正規化＝effort∈[0,1]で扱いやすい。
    _mimo_gear = np.abs(env.unwrapped.model.actuator_gear[:n_act, 0]).astype(np.float32)
    eff_w = torch.tensor(_mimo_gear / (_mimo_gear.sum() + 1e-8))

    def step_k(a):
        o, term = state["obs"], False
        for _ in range(K):
            o, r, te, tr, info = env.step(a)
            if te or tr:
                term = True; break
        return o, term

    def zc(sv, prev_a, cf, h):
        """【2026-07-25】太郎の infer_latent を呼ぶだけ（core へ一元化）。
        旧実装は同じ処理（[感覚,前回行動]→射影→GRU→pc_latent）を手書きしていた。"""
        return brain.infer_latent(sv, prev_a, cf, h)

    def reset_state():
        state["obs"], _ = env.reset()
        state["hidden"] = brain.init_motor_hidden()
        state["prev_a"] = torch.zeros(n_act)

    def act_mean(z):
        # 決定的な行動平均（評価・agency用）＝太郎の act_deterministic を呼ぶだけ。
        return brain.act_deterministic(z, cerebellum=(cereb if _CEREB else None))

    def infer_goal_action(z, clp, init_mean, g, n_steps=_pd.GOAL_INFER_STEPS, lr_inf=_pd.GOAL_INFER_LR):
        # Goal Babbling: 凍結した順モデル(nat_head)を反転し、望む感覚 g に届く行動を推論。
        # 逆モデル(Stage1)の機構をオンラインで使う＝目標指向の行動生成。
        target = (g - clp).detach()
        raw = torch.atanh(torch.clamp(init_mean, -0.999, 0.999)).detach().requires_grad_(True)
        opt = torch.optim.Adam([raw], lr=lr_inf)
        for _ in range(n_steps):
            opt.zero_grad()
            ((nat_head(torch.cat([z, torch.tanh(raw)], dim=-1)) - target) ** 2).mean().backward()
            opt.step()
        return torch.tanh(raw).detach()

    # 【Viewer】E_VIEW=1のときは学習せず、共通の motor_viewer.run_viewer に委譲する
    # （速度オーバーレイ・キー操作・実時間追従などが揃っている、共通測定器）。
    # 学習済みチェックポイント(E_LOADMODEL)を見る目的なので、探索なしの決定的な行動(act_mean)。
    #
    # E_VIEW_EXPLORE=1 で「運動性喃語そのもの」を見る（2026-07-25 追加）。
    #   既定(=0)は決定的な行動なので**ゆらぎが一切出ない**＝自発運動を目視できない。
    #   学習ループと同じ `brain.explore(mean, std)` を通すので、色付きノイズ(1/f^β)も
    #   シナジーも拮抗筋モードも**太郎の中の実装がそのまま効く**。
    #   注意：`E/scripts/e_ctrl_freq_probe.py` のViewerは脳を通らないダミー方策で、
    #     振幅も比較用の固定値（筋活性化 0.3±0.3）＝**本物より3倍以上激しい**。
    #     運動の激しさを評価するときは必ずこちら（本物）を見ること。
    if _VIEW:
        sys.path.insert(0, os.path.join(_CORE, "tools"))
        from motor_viewer import run_viewer

        _VIEW_EXPLORE = os.environ.get("E_VIEW_EXPLORE", "0") == "1"
        # 学習中の std は `0.05 + ne*0.45`（motor_drive）。学習初期は ne≈0.275 なので
        # std≈0.174 になる＝既定値の由来。E_VIEW_STD で振れば振幅の比較ができる。
        _VIEW_STD = float(os.environ.get("E_VIEW_STD", "0.174"))
        if _VIEW_EXPLORE:
            print(f"[Viewer] 運動性喃語ON（探索）: std={_VIEW_STD} "
                  f"（学習初期の実効値≒0.174／noise=neで変動）", flush=True)

        # 【2026-07-30 追加】E_VIEW_GOALBABBLE=1 で Goal Babbling の動きを目視する。
        #   【なぜ要るか】Goal Babbling を有効にすると、おもちゃに触れる回数が
        #     25回→17回（7分以降0回）に減り、自己モデルも +30.7%→+17.2% に崩れた。
        #     数字だけでは「何が起きているか」が分からない（退化を構造的に高評価する
        #     指標があるので、必ず目で見る＝feedback-watch-dont-just-measure）。
        #   注意：再生モードでは学習しないので goal_buf は空。ここで溜めながら使う。
        #     順モデルは読み込んだまま固定なので「いまのモデルが逆算する動き」が見える。
        #   注意：学習ループの切り替えは「予測誤差＋ノルアドレナリン」だが、再生では
        #     予測誤差を計算しないので、**割合を直接指定**する（E_VIEW_GB_RATE、既定0.5＝半分）。
        #     ＝学習ループと完全に同じ挙動ではない。動きの性質を見るための近似。
        # 可変にしておく＝Viewer実行中に g キーで ON/OFF を往復できる。
        #   別プロセスで2回起動して比べると乱数も姿勢も違うので、
        #   「どちらが速いか」のような主観の比較が当てにならない（同じ状態で切り替える）。
        _VIEW_GB = [os.environ.get("E_VIEW_GOALBABBLE", "0") == "1"]
        _VIEW_GB_RATE = float(os.environ.get("E_VIEW_GB_RATE", "0.5"))
        _view_gbuf = []
        _view_gb_stat = {"goal": 0, "explore": 0}

        def _toggle_gb():
            _VIEW_GB[0] = not _VIEW_GB[0]
            print(f"  [GoalBabbling] {'ON' if _VIEW_GB[0] else 'OFF'}"
                  f"（目標指向 {_view_gb_stat['goal']} 回 / 探索 {_view_gb_stat['explore']} 回）",
                  flush=True)

        def _gb_status():
            # 画面右上に出す文字。Viewerだけ見ていても条件と発動回数が分かるように。
            if not _VIEW_GB[0]:
                return "GoalBabbling OFF"
            n = len(_view_gbuf)
            if n < 64:
                return f"GoalBabbling ON (warmup {n}/64)"
            _n = max(_view_gb_stat['goal'], 1)
            _avg = _view_gb_stat.get('dsum', 0.0) / _n
            return (f"GoalBabbling ON goal={_view_gb_stat['goal']} "
                    f"dAction avg={_avg:.4f} max={_view_gb_stat.get('dmax', 0.0):.3f}")

        print(f"[Viewer] Goal Babbling {'ON' if _VIEW_GB[0] else 'OFF'}"
              f"（目標指向の割合 {_VIEW_GB_RATE:.0%}）／目標は経験から一様ランダム"
              f"＝学習ループと同じ選び方／ ; か ' か / キーで切り替え", flush=True)
        # 注意：g は使えない：MuJoCo の組み込みキーと衝突して「世界が暗くなる」
        #   （BACKSPACE が mj_resetData を呼ぶのと同じ構造。落とし穴チェックリスト参照）
        #   記号キーは組み込みで使われていないので安全側に3つ用意する。

        def _policy_fn(obs, prev_a, hidden, *, recompute, frac):
            # 連続制御の frac は今の学習済みモデルには使わない(policyはboundaryでのみ再計算、
            # 間は同じctrl保持)。将来 policyが frac を扱う場合はここで使う。
            if not recompute:
                return prev_a, hidden   # 再計算しない：呼び出し側で prev_a=action がctrlに使われる
            sv = fusion.encode(obs); cf = target_fusion.encode(obs).detach()
            z, _kl, _rc, hn = zc(sv, prev_a, cf, hidden)
            z = z.detach()
            mean = act_mean(z)
            # 注意：経験のバッファは**常に**溜める（ON/OFF に関係なく）。
            #   学習ループも goal_step の外で append しているので、そちらに合わせる。
            #   【2026-07-30 に踏んだ】ONのときだけ溜める実装にしたため、
            #   g キーで切り替えるたびに64件たまる前にOFFになり、
            #   **Goal Babbling が一度も発動しなかった**（ログ上は「目標指向 0 回」が続いた）。
            #   ユーザーの「g押してもなんも変わってない気がする」で発覚。
            _clp = ln_prop(obs).detach()
            _view_gbuf.append(_clp)
            if len(_view_gbuf) > 2000:
                _view_gbuf.pop(0)
            if _VIEW_GB[0]:
                if len(_view_gbuf) >= 64 and torch.rand(1).item() < _VIEW_GB_RATE:
                    g = _view_gbuf[torch.randint(len(_view_gbuf), (1,)).item()]
                    # 行動をどれだけ変えたかを測る。ユーザーの目視
                    #   「ONにしてもOFFでも動きが全く変わらない」を数値で確かめるため。
                    #   Goal Babbling が発動していても mean がほぼ動かないなら、
                    #   順モデルの反転が効いていない（調査④の仮説：反転が的外れ）。
                    _before = mean.detach().clone()
                    mean = infer_goal_action(z, _clp, mean, g)
                    _d = float((mean - _before).abs().mean())
                    _view_gb_stat["dsum"] = _view_gb_stat.get("dsum", 0.0) + _d
                    _view_gb_stat["dmax"] = max(_view_gb_stat.get("dmax", 0.0), _d)
                    _view_gb_stat["goal"] += 1
                else:
                    _view_gb_stat["explore"] += 1
                if sum(_view_gb_stat.values()) % 100 == 0:
                    print(f"[Viewer] 目標指向 {_view_gb_stat['goal']} 回 / "
                          f"探索 {_view_gb_stat['explore']} 回", flush=True)
            if _VIEW_EXPLORE:
                a, _lp = brain.explore(mean, torch.full_like(mean, _VIEW_STD))
                return a.detach(), hn.detach()
            a = torch.clamp(mean, -1.0, 1.0).detach()
            return a, hn.detach()

        banner = f"E_LOADMODEL={os.path.basename(_LOADMODEL) if _LOADMODEL else '(白紙)'}"
        run_viewer(env, brain, _policy_fn, rescale_action,
                   K=K, n_act=n_act, banner=banner,
                   extra_keys={";": _toggle_gb, "'": _toggle_gb, "/": _toggle_gb},
                   status_fn=_gb_status)
        return  # 再生モードは学習・保存・録画に進まない

    # 【姿勢の測定・2026-07-25】E_MEASURE_POSTURE=1 で、学習せず**学習済みモデルの姿勢**を測る。
    # 【なぜ必要か】制御頻度プローブ(e_ctrl_freq_probe.py)で「学習なしのノイズだけだと
    #   K100で36%・K10で65%の時間うつぶせになる」と判明した（Viewerでユーザーが
    #   「すぐうつぶせになる」と目視→測定器に体幹回転を追加して確認）。
    #   実際の新生児は寝返りできない（4-6ヶ月から）ので明確な逸脱。
    #   では**学習済みモデル(+58.8)はどうなのか**＝「仰向けで学習した」が成立しているのか、
    #   という +58.8 の解釈そのものに関わる問いなので、同じ物差しで測る。
    # 探索なし・決定的な行動（act_mean）で回す＝Viewerと同じ条件。
    if os.environ.get("E_MEASURE_POSTURE", "0") == "1":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import mujoco   # このファイルの他所では使っていないのでここでimportする
        from e_ctrl_freq_probe import _trunk_rotation_deg, _head_tilt_deg, _hand_dist

        n_meas = int(os.environ.get("E_MEASURE_STEPS", "6000"))
        _m, _d = env.unwrapped.model, env.unwrapped.data
        _dt = _m.opt.timestep * env.unwrapped.frame_skip
        _hinges = [i for i in range(_m.njnt) if _m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
        _dofs = [_m.jnt_dofadr[i] for i in _hinges]

        # 【部位別の測定・2026-07-25】全関節の平均だけでは「どこが動いているか」が見えない。
        # Viewerでユーザーが「足が根元からすごい動く。何回リセットしても」と指摘して発覚
        # （[[feedback-watch-dont-just-measure]]＝目視が測定器の穴を見つけた、本日2回目）。
        # 注意：MIMoの関節名は紛らわしい：`hip_lean1`等（左右が付かない）は**体幹の曲げ**、
        #   `right_hip1`等は**脚の付け根（股関節）**。先に体幹側を判定しないと混ざる。
        def _joint_group(nm):
            n = nm.replace("robot:", "")
            if n.startswith(("left_eye", "right_eye")):        return "眼"
            if n.startswith("head"):                            return "頭/首"
            if n.startswith("chest") or n.startswith("hip_"):   return "体幹"   # ←先に判定
            if "shoulder" in n:                                 return "肩"
            if "elbow" in n:                                    return "肘"
            if n.startswith(("left_hand", "right_hand")):       return "手首"
            if any(k in n for k in ("_th_", "_ff_", "_mf_", "_rf_", "_lf_")): return "手指"
            if any(k in n for k in ("hip1", "hip2", "hip3")):   return "脚の付け根"
            if "knee" in n:                                     return "膝"
            if "foot" in n:                                     return "足首"
            if "toe" in n:                                      return "つま先"
            return "その他"

        _grp_idx = {}
        for _pos, _ji in enumerate(_hinges):
            _grp_idx.setdefault(_joint_group(_m.jnt(_ji).name), []).append(_pos)
        _grp_qvel = {g: [] for g in _grp_idx}

        # 【軸別の分解・2026-07-25】ユーザーが新生児の自発運動の動画を見て
        # 「足は伸ばす・縮めるの繰り返しで、横方向の動きがない」と指摘したのがきっかけ。
        # 寝返りは体を長軸まわりに回す動きなので、**回旋と外転**がないと物理的に起きない。
        #   屈曲/伸展（伸ばす・縮める）だけでは、どれだけ激しくても回転モーメントが立たない。
        # → 太郎の動きを軸ごとに分けて、学習前後でどこが増えたかを見る。
        # MIMoの関節軸の意味（直立を基準にした解剖学的な向き）：
        #   x軸(1,0,0)まわり = 外転/内転・側屈   y軸(0,1,0)まわり = 屈曲/伸展
        #   z軸(0,0,1)まわり = 回旋（ひねり）
        # 注意：主成分で分類する近似（例：shoulder_horizontal は axis=(0,1,-4) で z が支配的
        #   なので「回旋」に入る）。厳密な解剖学的分類ではない。
        _AXIS_NAMES = ["外転/内転・側屈(横)", "屈曲/伸展(前後)", "回旋(ひねり)"]

        def _axis_kind(ji):
            return _AXIS_NAMES[int(np.argmax(np.abs(_m.jnt_axis[ji])))]

        _ax_idx, _trunk_ax, _leg_ax = {}, {}, {}
        for _pos, _ji in enumerate(_hinges):
            _nm = _m.jnt(_ji).name
            _k = _axis_kind(_ji)
            # 指・つま先・眼は数が多く、寝返りとは無関係なので全身の集計から外す
            if _joint_group(_nm) not in ("手指", "つま先", "眼"):
                _ax_idx.setdefault(_k, []).append(_pos)
            if _joint_group(_nm) == "体幹":
                _trunk_ax.setdefault(_k, []).append(_pos)
            if _joint_group(_nm) in ("脚の付け根", "膝"):
                _leg_ax.setdefault(_k, []).append(_pos)
        _ax_qvel = {k: [] for k in _ax_idx}
        _trunk_qvel = {k: [] for k in _trunk_ax}
        _leg_qvel = {k: [] for k in _leg_ax}
        _act_model = getattr(env.unwrapped, "actuation_model", None)

        obs, _ = env.reset(seed=seed)
        R0 = _d.body("upper_body").xmat.reshape(3, 3).copy()
        hidden = brain.init_motor_hidden()
        prev_a = torch.zeros(n_act)
        rots, qvels, jerks, tilts, hands, acts, coact, f_tot, f_net = [], [], [], [], [], [], [], [], []
        prev_qacc = None

        for _tick in range(max(1, n_meas // K)):
            # Viewer(_VIEW)の _policy_fn と同じ処理＝探索なしの決定的な行動。
            # （_policy_fn は _VIEW ブロック内のローカル定義なのでここからは見えない）
            _sv = fusion.encode(obs); _cf = target_fusion.encode(obs).detach()
            _z, _kl, _rc, hidden = zc(_sv, prev_a, _cf, hidden)
            a = torch.clamp(act_mean(_z.detach()), -1.0, 1.0).detach()
            hidden = hidden.detach()
            prev_a = a
            a_env = brain.to_env_action(a) if hasattr(brain, "to_env_action") else a
            ctrl = rescale_action(a_env, env.action_space)
            acts.append(float(np.abs(np.asarray(a, dtype=np.float64)).mean()))
            for _k in range(K):
                obs, _r, _te, _tr, _info = env.step(ctrl)
                qacc = _d.qacc[_dofs].copy()
                if prev_qacc is not None:
                    jerks.append(float(np.abs((qacc - prev_qacc) / _dt).mean()))
                prev_qacc = qacc
                _qv = np.abs(_d.qvel[_dofs])
                qvels.append(float(_qv.mean()))
                for _g, _ix in _grp_idx.items():
                    _grp_qvel[_g].append(float(_qv[_ix].mean()))
                for _k, _ix in _ax_idx.items():
                    _ax_qvel[_k].append(float(_qv[_ix].mean()))
                for _k, _ix in _trunk_ax.items():
                    _trunk_qvel[_k].append(float(_qv[_ix].mean()))
                for _k, _ix in _leg_ax.items():
                    _leg_qvel[_k].append(float(_qv[_ix].mean()))
                rots.append(_trunk_rotation_deg(_d, R0))
                tilts.append(_head_tilt_deg(_m, _d))
                hands.append(_hand_dist(_m, _d))
                if _act_model is not None and hasattr(_act_model, "muscle_activations"):
                    ma = np.asarray(_act_model.muscle_activations, dtype=np.float64)
                    mf = np.abs(np.asarray(_act_model.muscle_forces, dtype=np.float64))
                    nj = len(ma) // 2
                    coact.append(float(np.minimum(ma[:nj], ma[nj:]).mean()))
                    f_tot.append(float(mf.mean()))
                    f_net.append(float(np.abs(mf[:nj] - mf[nj:]).mean()))
                if _te or _tr:
                    obs, _ = env.reset(); prev_qacc = None
                    break

        rots_a = np.asarray(rots)
        eff = (np.mean(f_net) / np.mean(f_tot)) if f_tot and np.mean(f_tot) > 1e-12 else float("nan")
        print(f"\n=== 姿勢の測定（学習済みモデル） K={K} step={len(qvels)} ===")
        print(f"  model={os.path.basename(_LOADMODEL) if _LOADMODEL else '(白紙)'}")
        print(f"  |qvel|={np.mean(qvels):.4f}  jerk={np.mean(jerks) if jerks else float('nan'):.1f}  "
              f"|act|={np.mean(acts) if acts else float('nan'):.3f}")
        print(f"  同時活性化={np.mean(coact) if coact else float('nan'):.4f}  力の効率={eff:.3f}")
        print(f"  頭の傾き={np.nanmean(tilts):.1f}度  手-体幹={np.nanmean(hands):.3f}m")
        print(f"  体幹の回転={np.nanmean(rots_a):.1f}度(最大{np.nanmax(rots_a):.1f})  "
              f"仰向けでない時間={float(np.mean(rots_a > 90.0))*100:.1f}%")
        # 【2026-07-25 閾値の訂正】90度は「うつ伏せ」ではなく**ちょうど側臥位**だった。
        #   McGraw 1941（満期産児75名・1,840回の観察）＝**新生児は立ち直り反射で
        #   側臥位までは自発的に転がれる**＝90度超えは正常。異常なのは**真のうつ伏せ**。
        #   162度は Philipp et al. 2026（MIMoで寝返りを学習させた ICDL 2026 論文）の
        #   成功判定 rho>=0.95 に相当（rho = 体幹前方軸と重力軸の内積の平均）。
        #   実測（K=10, 3シード）: >90度 20.7% に対し >162度 7.2%＝**2.9倍の過大評価**。
        #   → 検証の落とし穴チェックリスト 項41（閾値の意味を文献で確認する）。
        #   注意：両方を併記する（過去のログと比較できるように90度も残す）。
        print(f"  真のうつ伏せ(>162度)={float(np.mean(rots_a > 162.0))*100:.1f}%  "
              f"(>120度={float(np.mean(rots_a > 120.0))*100:.1f}%  "
              f">150度={float(np.mean(rots_a > 150.0))*100:.1f}%)"
              f"  [90度=側臥位＝新生児でも正常／162度=Philipp2026のrho>=0.95相当]")
        print("  （比較：学習なしノイズだけの実測＝K100でうつ伏せ36.2%、K10で65.0%）")

        # 「一度うつ伏せになって戻れない」のか「行ったり来たり」なのかを数える。
        # 【なぜ要るか、2026-07-25・ユーザー指摘】
        #   「赤ちゃんも実際ひっくり返ることはあるんじゃない？一回うつぶせになると戻れないから
        #     親が戻してるんじゃないかな」＝**うつ伏せ%が高い＝悪い、という前提そのものへの疑い**。
        #   人間の発達では 寝返り（仰向け→うつ伏せ）が生後4〜6ヶ月、うつ伏せ→仰向けに戻る
        #   「寝返り返り」はさらに後。＝新生児は**うつ伏せになれないし、なったら戻れない**。
        #   だから区別が要る：
        #     ①一度きりで戻らない → 太郎の動きは人間に近く、**親が戻す仕組みが無い**のが逸脱（環境側）
        #     ②行ったり来たり     → 新生児にできない**寝返り返り**をしている（もっと重い逸脱）
        _prone = rots_a > 90.0
        _trans = int(np.sum(_prone[1:] != _prone[:-1]))       # 仰向け⇄うつ伏せの切り替わり回数
        _to_prone = int(np.sum(_prone[1:] & ~_prone[:-1]))    # 仰向け→うつ伏せ
        _to_supine = int(np.sum(~_prone[1:] & _prone[:-1]))   # うつ伏せ→仰向け（＝寝返り返り）
        _first = int(np.argmax(_prone)) if _prone.any() else -1
        print(f"  姿勢の切り替わり: 計{_trans}回 (仰向け->うつ伏せ {_to_prone}回 / "
              f"うつ伏せ->仰向け {_to_supine}回)  初めてうつ伏せになったtick={_first}")
        if _to_supine == 0 and _to_prone <= 1:
            print("     -> [一度きり] 人間の新生児と同じ「戻れない」挙動。"
                  "戻す親が居ないことが環境側の逸脱。")
        elif _to_supine > 0:
            print(f"     -> [行ったり来たり] うつ伏せから自力で{_to_supine}回戻っている＝"
                  "新生児にできない「寝返り返り」。運動側の逸脱。")
        if os.environ.get("E_MEASURE_TRACE", "0") == "1":
            _tp = os.path.join(LOG_DIR, f"posture_trace_seed{seed}_{stamp}.csv")
            with open(_tp, "w", newline="", encoding="utf-8") as _fp:
                _w = csv.writer(_fp); _w.writerow(["tick", "trunk_rot_deg", "is_prone"])
                for _i, _v in enumerate(rots_a):
                    _w.writerow([_i, f"{_v:.2f}", int(_v > 90.0)])
            print(f"  姿勢の時系列: {_tp}")
        print("  ── 部位別の |関節角速度|（どこが動いているか）──")
        _rank = sorted(((np.mean(v), g, len(_grp_idx[g])) for g, v in _grp_qvel.items() if v),
                       reverse=True)
        _tot = sum(r[0] * r[2] for r in _rank) or 1.0
        for _v, _g, _n in _rank:
            _bar = "█" * max(0, min(30, int(_v / (_rank[0][0] or 1) * 30)))
            print(f"    {_g:12s} {_v:7.4f} ({_n:2d}関節) {_bar}")

        # 【軸別・2026-07-25】寝返りは「体を長軸まわりに回す」動き。回旋・外転が無ければ
        # 物理的に起きない。新生児の自発運動は屈曲/伸展が支配的（動画での観察）なので、
        # 太郎の横方向・回旋の量が人間より多いなら、それが寝返りの原因の候補になる。
        def _axis_block(title, idx, qv):
            if not any(qv.values()):
                return
            print(f"  -- {title} --")
            _s = {k: float(np.mean(v)) for k, v in qv.items() if v}
            _flex = _s.get("屈曲/伸展(前後)", 0.0)
            for _k in _AXIS_NAMES:
                if _k not in _s:
                    continue
                _bar = "#" * max(0, min(24, int(_s[_k] / (max(_s.values()) or 1) * 24)))
                _ratio = (_s[_k] / _flex) if _flex > 1e-9 else float("nan")
                print(f"    {_k:20s} {_s[_k]:7.4f} ({len(idx[_k]):2d}関節) "
                      f"屈伸比={_ratio:5.2f} {_bar}")
        print("  軸別の |関節角速度|（関節の主軸で分類・指/つま先/眼は除外）")
        _axis_block("全身", _ax_idx, _ax_qvel)
        _axis_block("体幹だけ（寝返りに直結）", _trunk_ax, _trunk_qvel)
        _axis_block("脚（股関節+膝）", _leg_ax, _leg_qvel)
        return  # 測定モードは学習・保存・録画に進まない

    # 【測定器の分離・2026-07-23】evaluate/agency_probe/inverse_probe/inverse_exec_probe/
    # closed_loop_probe は e_probes.py へ切り出した（太郎の外から測る道具＝学習ループ本体から独立）。
    # 上で定義したモデル操作ヘルパー（zc/act_mean/step_k/ln_prop/reset_state/infer_goal_action）と
    # モデル・環境・設定を束ねて probe へ渡す。挙動は元と完全に同一。
    ctx = e_probes.ProbeContext(
        brain=brain, fusion=fusion, target_fusion=target_fusion, nat_head=nat_head,
        env=env, rescale_action=rescale_action,
        zc=zc, act_mean=act_mean, step_k=step_k, ln_prop=ln_prop,
        reset_state=reset_state, infer_goal_action=infer_goal_action,
        state=state, n_act=n_act, n_eval=n_eval, K=K, DT=DT, seed=seed, log_dir=LOG_DIR)

    def checkpoint(step):
        cl, mg, co, pr = e_probes.evaluate(ctx)
        ag, magr = e_probes.agency_probe(ctx)
        life_min = step * K * DT / 60.0
        real_min = (time.time() - t0) / 60.0
        # E1の主指標：直近区間で手が視野に入っていた割合（段階1では記録のみ・報酬に不使用）
        _hv = (100.0 * hv["hit"] / hv["tot"]) if hv["tot"] else float("nan")
        log_row([f"{life_min:.1f}", step, f"{cl:.2f}", f"{mg:.2f}", f"{co:.4f}",
                 f"{pr:.2f}", f"{ag:.2f}", f"{magr:.1f}", f"{real_min:.1f}", f"{_hv:.2f}"])
        hv["hit"] = 0; hv["tot"] = 0        # 区間ごとにリセット＝推移が見える
        noise = 0.05 + ne.get_ne_level() * 0.45
        cereb_tag = f" cereb=on(err={cereb.err_ema.item():.2f})" if _CEREB else " cereb=off"
        # 【taro-C5】|行動|＝力の出し具合（0.86が全力偏重。低下＝加減を学習。0付近＝フリーズ警告）。
        act_tag = ""
        if act_accum:
            act_tag = f" |act|={np.mean(act_accum[-200:]):.3f}"
            if _EFFORT and eff_accum:
                act_tag += f" effort={np.mean(eff_accum[-200:]):.3f}(λ={_EFFORT})"
        # 【階層化・変更②】CAPSが効いているかの直接証拠＝‖a_t−a_{t-1}‖² の推移。
        # 下がる＝行動を変えなくなった。注意下がりすぎ（≒0）は「固まった」＝失敗のサイン。
        # 注意：タグはASCIIのみ（Windowsのcp932で出力できない文字を混ぜると print が例外を投げ、
        #   学習が途中で落ちる。上付き2・絵文字で実際に2回踏んだ）。
        if caps_accum:
            act_tag += f" da2={np.mean(caps_accum[-200:]):.4f}(lam={_CAPS})"
        # おもちゃへの接触（リーチングの主指標）。通しの累積なので毎回同じ形で出る。
        if _touch_probe is not None:
            act_tag += " " + _touch_probe.line(K * DT)
        print(f"[AC seed{seed} rew={_REWARD} ne={'rel' if _NE_REL else 'abs'} touch={_TOUCH_MODE if _TOUCH else 'off'}] life={life_min:.0f}min | classify={cl:.1f}% margin={mg:+.1f}% "
              f"corr={co:.3f} persist={pr:.1f}% agency={ag:.1f}%(mag {magr:.0f}%) | "
              f"noise={noise:.3f}(mat={ne.maturation:.2f}){cereb_tag}{act_tag} real={real_min:.0f}min", flush=True)

    # 経験バッファ（睡眠中リプレイ用）。各ステップの予測に必要な材料を貯める。
    # 【2026-07-25】睡眠リプレイのバッファを太郎の海馬（core: brain/hippocampus.py の
    # MotorHippocampus）に一元化。**旧実装は独自のdictで、core にある FIFO容量上限(3600)も
    # clear() も無く、学習全期間ぶん無制限に増え続けていた**＝「直近の覚醒経験を再生する」
    # という睡眠リプレイの意味から外れた劣化コピーだった（構造監査で発覚）。
    # 海馬はバッファに徹し、定着（重み更新）のロジックは下の consolidate が持つ、という
    # 分担は core の設計どおり。
    hippo = brain.hippocampus
    # Goal Babbling 用の目標バッファ＝Self-Priorの最小版（過去に経験した固有感覚の分布）。
    goal_buf = []
    # 予測誤差の速い/遅い走行平均＝学習進度の材料（"いつもより驚いたか"の自己正規化にも使う）。
    # 【2026-07-25】太郎の中（core）へ一元化。時定数(0.9/0.99)も core が持つ。
    _lp = LearningProgress()
    pe_fast, pe_slow = _lp.pe_fast, _lp.pe_slow
    hv = {"hit": 0, "tot": 0}    # E1：手が視野内だったtickの数（checkpointごとにリセット）
    # 【2026-07-30】おもちゃに手が触れた回数（リーチングの主指標）。
    #   注意：hv と違い**通しの累積**にする。リーチングは希少な事象で、
    #     区間ごとにリセットすると「0回」が並ぶだけで推移が読めない。
    _touch_probe = None
    if _E1:
        from e_toy_touch import ToyTouchProbe
        _touch_probe = ToyTouchProbe(env.unwrapped.model, env.unwrapped.data)
        if not _touch_probe.ok:
            print("[toy_touch] おもちゃが無い環境＝接触は測らない", flush=True)
            _touch_probe = None
    mj = env.unwrapped           # モデル/データへの参照（hand_in_view_rate 用）
    reach_goal, reach_prev_dist = None, 0.0  # 閉ループreaching訓練：保持中の目標と直前の距離

    def consolidate(n_batches=200, bs=128):
        """睡眠中の記憶定着：貯めた経験をバッチで再生し、自己モデル(予測経路)を復習で固める。"""
        _eps = hippo.replay()
        N = len(_eps)
        if N < bs:
            return
        SV = torch.stack([e[0] for e in _eps]); PA = torch.stack([e[1] for e in _eps])
        AA = torch.stack([e[2] for e in _eps]); CF = torch.stack([e[3] for e in _eps])
        CLP = torch.stack([e[4] for e in _eps]); NLP = torch.stack([e[5] for e in _eps])
        H = torch.cat([e[6] for e in _eps], dim=1)  # (layers, N, hidden)
        for _ in range(n_batches):
            idx = torch.randint(0, N, (bs,))
            hb = H[:, idx].contiguous()
            emb = emb_proj(torch.cat([SV[idx], PA[idx]], dim=-1)).unsqueeze(1)  # (bs,1,emb) batch_first
            out, _ = brain.motor_gru(emb, hb)  # out (bs,1,hidden)
            z, kl, rc = brain.pc_latent.infer(hb[-1], out[:, 0], CF[idx])
            pred = CLP[idx] + nat_head(torch.cat([z, AA[idx]], dim=-1))
            loss = block_pe(pred, NLP[idx]) + kl + rc   # 学習ループと同じ基準
            learner.optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(learner.brain.parameters(), learner.grad_clip)
            learner.optimizer.step()

    # 【2026-07-29】体を育てる：保存を挟まずに env だけ作り直す。
    #   脳(brain/fusion/cereb)・オプティマイザ・経験バッファ(goal_buf)・馴化は
    #   Python の変数のまま残る＝**何回体を変えても内部状態は消えない**（項75）。
    _cur_age = age_at(0, n_train)

    def _regrow(new_age):
        nonlocal env
        old = (int(env.observation_space["observation"].shape[0]),
               int(env.action_space.shape[0]))
        env.close()
        env = _make_env(_body_kwargs_for_age(new_age))
        new = (int(env.observation_space["observation"].shape[0]),
               int(env.action_space.shape[0]))
        # 注意：触覚ONではセンサ点が月齢で変わる（0ヶ月1734→4ヶ月4274）ので次元が合わなくなる。
        #   黙って壊れると「体を育てたのに学習が進まない」と読み違えるので、ここで止める。
        assert old == new, (f"体を作り直したら観測/行動の次元が変わった {old} -> {new}。"
                            "触覚ONではこの方式は使えない（落とし穴チェックリスト 項75）")
        ctx.env = env          # 測定器も新しい体を見る（e_probes の _to_env_ctrl は self.env 参照）
        reset_state()          # 新しい体で立て直す。脳の重みと経験は保持されたまま

    if _AGE_TO:
        print(f"[body-growth] 月齢 {float(_AGE or 0.0)} → {float(_AGE_TO)}／"
              f"開始{_AGE_START}回目・{_AGE_RAMP or '一瞬'}回かけて・{_AGE_EVERY}回ごとに見直し",
              flush=True)

    checkpoint(0)
    for i in range(n_train):
        # 体を育てる（E_AGE_TO 未指定なら何もしない＝従来と完全に同じ）
        if _AGE_TO and _AGE_EVERY > 0 and i > 0 and i % _AGE_EVERY == 0:
            _na = age_at(i, n_train)
            if abs(_na - _cur_age) > 1e-9:
                _regrow(_na)
                print(f"[body-growth] {i}回目：月齢 {_cur_age:.3f} → {_na:.3f} ヶ月", flush=True)
                _cur_age = _na
        sv = fusion.encode(state["obs"]); cf = target_fusion.encode(state["obs"]).detach(); clp = ln_prop(state["obs"])
        z, kl, rc, hn = zc(sv, state["prev_a"], cf, state["hidden"].detach())
        # 行動生成は太郎の中に一元化（brain.motor_drive）。運動野の精密制御＋小脳の自動化
        # ブレンドで mean/std を作る（従来のスクリプト側実装と数値完全一致）。e_c は小脳の
        # 学習(observe)に使う。std は 0.05 + ne*0.45（motor_drive の min/max_std 既定と一致）。
        mean, std, _w_c, e_c = brain.motor_drive(
            z, ne.get_ne_level(), cerebellum=(cereb if _CEREB else None))
        # Goal Babbling：一部のステップで、経験(Self-Prior)から目標をサンプルし、逆算した
        # 行動を平均に据える（＝目標に手を伸ばす）。残りは今まで通りの探索（ハイブリッド）。
        # 切り替え：fixed=i%2固定(暫定)／ne=NE(青斑核=探索/活用の神経調節, Aston-Jones &
        # Cohen)から創発＝NEが高い(探索したい)ほど探索寄り、低い(落ち着き)ほど目標指向。
        goal_step = False
        if _GOALBABBLE and len(goal_buf) >= 64:
            if _GB_SWITCH == "fixed":
                goal_step = (i % 2 == 0)
            elif _GB_SWITCH == "ne":
                goal_step = torch.rand(1).item() < (1.0 - ne.get_ne_level())
            else:  # "pe"：予測誤差(驚き)を主役＋NEを下駄、で探索/目標を創発
                # 【逸脱/工学近似 注意】向き（驚き大・NE大→探索、＝分かる所は狙い分からぬ所は探る）
                # はEFE/LC-NE/予測符号化に基づく人間模倣だが、"足し算・等重み・この正規化・確率への
                # 写像"という具体式には生物学的根拠なし＝恣意的。逸脱リスト参照。要感度確認・アブレーション。
                rel = min(pe_fast / (pe_slow + 1e-6), 2.0) / 2.0     # [0,1], 0.5=平常の驚き
                explore_drive = min(ne.get_ne_level() + rel, 1.0)
                goal_step = torch.rand(1).item() < (1.0 - explore_drive)
        if goal_step:
            if _CLTRAIN:
                # 閉ループreaching：1つの目標を"届く/停滞"まで保持してにじり寄る（held goal）
                if reach_goal is None:
                    reach_goal = goal_buf[torch.randint(len(goal_buf), (1,)).item()].clone()
                    reach_prev_dist = mse(clp, reach_goal).item()
                g = reach_goal
            else:
                g = goal_buf[torch.randint(len(goal_buf), (1,)).item()]  # 毎回新しい目標（reachしない）
            mean = infer_goal_action(z.detach(), clp, mean, g)
        else:
            reach_goal = None  # 探索に切替 → リーチ終了
        a, lp = brain.explore(mean, std)   # 探索も太郎の中（将来ここに色付き喃語を集約）
        goal_buf.append(clp.detach())
        if len(goal_buf) > 2000:
            goal_buf.pop(0)
        pred = clp + nat_head(torch.cat([z, a.detach()], dim=-1))
        # 拮抗筋モード：a(n_joint) → to_env_action で n_env_act次元の筋活性化へ写像してから env に送る。
        # 拮抗筋OFFなら a_env==a（従来と1バイト差なし）。
        a_env = brain.to_env_action(a)
        state["obs"], term = step_k(rescale_action(a_env, env.action_space)); nlp = ln_prop(state["obs"])
        if _CLTRAIN and reach_goal is not None:
            nd = mse(nlp, reach_goal).item()
            if nd >= reach_prev_dist:
                reach_goal = None  # 近づかなくなった→リーチ終了（次は新しい目標）＝停止条件
            else:
                reach_prev_dist = nd
        if _REPLAY:
            hippo.record(sv.detach(), state["prev_a"].detach(), a.detach(), cf.detach(),
                         clp.detach(), nlp.detach(), state["hidden"].detach())
        if _E1:      # E1：手が視野に入っているかを毎tick数える（記録のみ。報酬には効かない）
            hv["hit"] += hand_in_view_rate(mj.model, mj.data)
            hv["tot"] += 1
            # 【2026-07-30】おもちゃに手が触れたかも数える（記録のみ）。
            #   リーチングが起きたかの主指標。判定の実体は e_toy_touch.py に一本化。
            if _touch_probe is not None:
                _touch_probe.update(mj.model, mj.data)
        pe = block_pe(pred, nlp)   # 次元数の影響を除く（段階1）
        # 【2026-07-25】学習進度を太郎の中（core: brain/learning_progress.py）へ一元化。
        # 旧実装は同じ式・同じ時定数(0.9/0.99)を3ファイルにコピペ＝数値は完全に同一。
        _progress = _lp.update(pe.item())
        pe_fast, pe_slow = _lp.pe_fast, _lp.pe_slow   # 既存の参照箇所のために同期
        # 内発的動機。progress＝学習進度（誤差が減っていれば正）／predict＝従来の予測しやすさ。
        # rew_task＝タスクの出来そのもの（努力コストを引く前）。NE（探索）にはこちらを見せる。
        rew_task = _progress if _REWARD == "progress" else brain.sensorimotor_reward(pe.item())
        # 【taro-C5】努力コスト：活性化²の筋力重み付き平均（∈[0,1]）を報酬から引く。＝大きな力ほど
        # 損→自分で加減する（Selinger 2015等の代謝最小化。注意二乗・λ・重みは近似＝感度確認対象）。既定OFF。
        rew = rew_task
        if _EFFORT:
            effort = float((a.detach() ** 2 * eff_w).sum())
            rew = rew_task - _EFFORT * effort
            eff_accum.append(effort)
        # 【階層化・変更②】action smoothing（CAPS, Mysore et al. 2021）：
        # 「さっきと違うことを急にするのは損」＝行動の急変にペナルティ。
        # state["prev_a"] は902行で更新されるので、ここでは**1つ前の行動**を指す。
        # 注意：effort_cost（大きさ）とは別物で、こちらは変化量に効く＝ゆっくり大きく動くのは咎めない。
        # λ=0（CAPS OFF）でも**必ず記録する**。理由：OFF側の値が無いと
        #   「λが弱くて効かなかった」のか「元からこの値なのか」を切り分けられない。
        #   実際に最初その設計にしてしまい、λ=0.01 の da2 が増えていく現象を
        #   評価できなかった（2026-07-25）。＝対照条件の測定値を捨てない。
        smooth = smoothness_cost(a.detach(), state["prev_a"].detach())
        caps_accum.append(smooth)
        if _CAPS:
            rew = rew - _CAPS * smooth
        act_accum.append(float(a.detach().abs().mean().item()))
        # 方策の学習（ドーパミン）は努力コスト込みの報酬rewを見る＝「疲れは損」を学ぶ。
        pl = learner.learn_action([lp], dop.compute_rpe(rew))
        hl = homeo.homeostatic_loss(sv); homeo.observe(sv)
        learner.update(pe + hl + kl + rc, pl)
        if _CEREB:
            # 小脳は方策とは別に、実際に行った運動を教師なしで真似て自動化パターンを固める。
            closs = cereb.imitation_loss(z.detach(), a.detach())
            cere_opt.zero_grad(); closs.backward(); cere_opt.step()
            cereb.observe(e_c)  # 馴染み度の基準を更新（自己正規化）
        dev_clock.tick()  # ③発達年齢を進める（覚醒中の学習1回）。consolidate側では進めない。
        # 【taro-C5・NE decoupling】NE（探索）にはタスク報酬rew_taskだけを見せる（努力コスト抜き）。
        # 理由：LC-NEの利得信号(Aston-Jones & Cohen 2005)は「タスクの成否」を追うもので、代謝コストは
        # 別系統。努力コストで下がった報酬をNEに見せると「失敗した→探索せよ」と誤読し探索を暴走させ、
        # 大振幅ノイズが努力コストを打ち消す自滅ループになる（実測：noise0.095→0.5）。rew_taskで断つ。
        ne.observe_reward(rew_task); ne.release_ne()
        if term:
            reach_goal = None  # エピソード終了→リーチも終了
        if _MATURE:
            # 成熟は sim秒でなく発達年齢(学習回数)で駆動。n_train学習で完全成熟。
            ne.mature(dev_clock.progress(n_train))
        state["hidden"] = hn.detach(); state["prev_a"] = a.detach()
        if term:
            reset_state()
        if _REPLAY and (i + 1) % ckpt == 0:
            consolidate()  # 睡眠：この間の経験を再生して定着
        if (i + 1) % ckpt == 0:
            checkpoint(i + 1)
            if _INVTRAJ:
                _ip = e_probes.inverse_probe(ctx); _ie = e_probes.inverse_exec_probe(ctx)  # 逆モデルの天井軌跡
                with open(invtraj_path, "a", newline="", encoding="utf-8") as _fp:
                    csv.writer(_fp).writerow([i + 1, _ip["recover_corr"], _ip["infer_over_random"],
                                             _ie["star_over_random"], _ie["model_err"]])

    if _INVPROBE:
        e_probes.inverse_probe(ctx)  # 学習後に逆モデルStage1診断を1回
    if _INVEXEC:
        e_probes.inverse_exec_probe(ctx)  # Stage1.5＝推論a*の実行テスト
    if _CLPROBE:
        e_probes.closed_loop_probe(ctx, goal_buf)  # C4＝閉ループ制御 vs 開ループの到達比較
    if os.environ.get("E_SAVEMODEL"):
        # 確立した自己モデルを保存（D等の下流で"最高モデル"を再利用するため）。opt-in。
        mp = os.environ["E_SAVEMODEL"]; os.makedirs(os.path.dirname(mp), exist_ok=True)
        blob = {"brain": brain.state_dict(),
                "fusion_insula": fusion.insula.state_dict(),
                "fusion_proprio": fusion.proprio.state_dict(),
                "fusion_vestibular": fusion.vestibular.state_dict(),
                # 【2026-07-25】emb_proj/nat_head は brain の一部になったので
                # "brain" に含まれる（個別保存は不要）。
                "cereb": cereb.state_dict(),
                "config": {"sdim": sdim, "prop_dim": prop_dim, "touch_dim": touch_dim,
                           "out_dim": out_dim, "n_act": n_act, "K": K,
                           "seed": seed, "n_train": n_train, "replay": _REPLAY,
                           "cereb": _CEREB, "supine": _SUPINE, "touch": _TOUCH, "touch_mode": _TOUCH_MODE,
                           "reward": _REWARD, "ne_relative": _NE_REL, "env_id": _ENV_ID,
                           "effort": _EFFORT, "loadmodel": os.path.basename(_LOADMODEL) if _LOADMODEL else "",
                           "fusion": "MinimalFusion(interoception+proprio621+vestibular"
                                     + ("+touch)" if _TOUCH else ")"),
                           # 【2026-07-25 追加】身体・探索の設定を記録する。
                           #   これが無かったため、感度分析のモデルが同一だと分かっても
                           #   「どの設定で保存されたか」を後から確認できなかった
                           #   （e_condition_check.py の config diff が none になる）。
                           #   注意：条件を振る実験では、ここに条件が反映されていることを確認する。
                           "age": _AGE, "muscle": _MUSCLE, "antagonist": _ANTAGONIST,
                           "noise": _E_NOISE, "beta": _E_BETA,
                           "synergy": _E_SYNERGY, "syn_w": _E_SYN_W,
                           "body_kwargs": {k: v for k, v in _age_kw.items()
                                           if k != "custom_measurements"},
                           "shape_custom": bool(_age_kw.get("custom_measurements"))}}
        if fusion.touch is not None:
            blob["fusion_touch"] = fusion.touch.state_dict()
        torch.save(blob, mp)
        print(f"SAVED MODEL {mp}", flush=True)
    if os.environ.get("E_RECORD"):
        # 学習後の太郎を等速で録画。学習と同じ環境で録らないと「学習時と違う太郎」を見せる
        # ことになる（**_age_kw と **_act_kw を渡し忘れると SpringDamperModel/デフォルト月齢で
        # 録画されてしまい、MuscleModel等の非デフォルト設定と action/observation 次元が
        # 合わずエラー＝2026-07-24に MUSCLE_5000で踏んだ）。
        def _make_render_env():
            return HybridEnv(gym.make(_ENV_ID, vision_params=None,
                                      touch_params=_touch_params(), render_mode="rgb_array",
                                      **_age_kw, **_act_kw))
        e_probes.record_video(ctx, os.environ["E_RECORD"], _make_render_env)
    print(f"DONE seed={seed} log={csv_path}", flush=True)


if __name__ == "__main__":
    seed = int(sys.argv[1])
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 3600
    # 【2026-07-25 既定を 100 → 10 に変更】上位（脳）が運動指令を出す間隔。
    # K=100 は 1Hz＝**人間の最遅神経発火（α運動ニューロン7Hz）の1/7以下**で生物学的に
    # 成立しない。K=10 は 10Hz で皮質μ律動に対応する（設計メモ §3）。
    # 注意：「同じ学習回数なら K=100 のほうが margin は高い」（51.5 vs 45.4）。
    #   それでも K=10 を既定にするのは**人間に近いから**であって性能のためではない。
    #   経験量（sim時間）を揃えると K=10 が圧勝する（45.4 vs 26.6）。
    # 注意：既定を変える理由：同じ日に「報酬の既定が predict のままで、それに気づかず
    #   学習していた」という失敗をしている（追記2）。**正しい設定を既定にする**。
    K = int(os.environ.get("E_K", "10"))
    if K >= 100:
        print(f"[!] E_K={K}（{100/K:.0f}Hz）＝人間の最遅神経発火7Hzより遅い。"
              f"比較・再現目的でなければ K=10 を使うこと。", flush=True)
    ckpt = int(os.environ.get("E_CKPT", "600"))  # 【taro-C5】評価/ログ間隔（短い実験用に小さくできる）
    run(seed, n_train=n_train, K=K, ckpt=ckpt)
