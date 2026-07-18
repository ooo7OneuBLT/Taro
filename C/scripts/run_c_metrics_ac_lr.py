"""
目標C 指標ランナー【action-conditioned 版（改修プロトタイプ）】。
run_c_metrics_seed.py との違いは2点だけ（共有クラスは触らない）：
  D-a: GRU入力を [感覚] → [感覚, 前回の行動] に（遠心性コピーを再帰の中へ）
  D-b: 予測ヘッドを線形 → 非線形MLPに（行動→感覚の線形上限0.22を破る）
根拠：World Model/RSSM・Tani系 MTRNN/PV-RNN の標準形（行動条件づけを再帰に入れて次観測予測）。
狙い：persist>100（過大予測）と自己復元0.22の根＝「行動→感覚の弱い結びつき」を強める。

出力は logs/C/ に ac_metrics_seed{seed}_{日時}.csv。10分刻み・agency含む。
使い方: python run_c_metrics_ac_seed.py <seed> [n_train]  (既定 n_train=3600=1時間)
"""
import os, sys, csv, time, datetime, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)

_BRIDGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # = Taro/C（logs出力はC側に残す）
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
from homeostatic_scaling import HomeostaticScaling
from test_phase8_motor_learning import CombinedParams, rescale_action, to_tensor
from sensory_encoders import ProprioceptionEncoder, VestibularEncoder, TouchEncoder
from insula import Insula

# 仰向け環境（D側で定義）。C_SUPINE=1 のときだけ使う。
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir, "D", "scripts"))
from gymnasium.envs.registration import register
if os.environ.get("C_SUPINE", "0") == "1":
    from d_supine_env import SupineMimoEnv  # noqa
    register(id="TaroSupine-v0", entry_point="d_supine_env:SupineMimoEnv", max_episode_steps=6000)
elif os.environ.get("C_LEAN", "1") == "1":
    from mimo_lean import LeanMimoEnv  # noqa
    # max_episode_steps=6000 は MIMoBenchV2-v0 の登録値と同一（＝太郎の人生の長さを変えない）
    register(id="TaroBenchV2Lean-v0", entry_point="mimo_lean:LeanMimoEnv", max_episode_steps=6000)

mse = torch.nn.functional.mse_loss
DT = 0.01
LOG_DIR = os.environ.get("C_LOGDIR", os.path.join(_BRIDGE, "logs", "C", "ac_prototype", "patched_layernorm"))
_LR = float(os.environ.get("C_LR", "0.005"))
_MATURE = os.environ.get("C_MATURE", "0") == "1"  # 1=学習進行に合わせて探索を結晶化
_REPLAY = os.environ.get("C_REPLAY", "1") == "1"  # 既定ON＝睡眠中の経験リプレイ（記憶定着）。自己モデル確立の本命機構(taro-C2＝margin+11→+48の頭打ち突破)。C_REPLAY=0で無効化可（アブレーション用）
# 【注意すべき機能】既定ON＝運動小脳（自動化/結晶化）。良性は検証済み（発散・フリーズ・
# agency崩壊なし）だが、現指標では効果ほぼ不変・常時稼働（約26%ブレンド）。将来の壁の
# 切り分け時は「小脳が効いている可能性」を必ず確認する（`注意すべき機能リスト.md` 参照）。
_CEREB = os.environ.get("C_CEREBELLUM", "1") == "1"  # 0で無効化可
# 【taro-C5】努力コスト＝運動の代謝コストを報酬から引く。報酬=予測のうまさ − C_EFFORT×(筋活動)²。
# 既定0＝OFF＝従来と完全に同一。人間は代謝エネルギー最小になるよう動く（Selinger 2015 等）＝
# 大きな力＝損、を入れると自分で力を加減する。⚠️二乗の形・λの値は恣意的（cost()=Σu²·Tmaxを流用、
# 生理の正確な代謝式ではない）＝感度確認の対象。フリーズ（motor collapse, 逸脱リストB1）と背中
# 合わせなので、|行動|の低下と自己モデル(margin/agency)の生存を必ず併せて確認する。
_EFFORT = float(os.environ.get("C_EFFORT", "0"))
# 【taro-C5】学習済みモデルから継続学習する（脳をリセットしない方針）。C_LOADMODEL=<pt path>。
# 形が合う層だけロード（転移学習）。既定なし＝従来どおりゼロから学習。
_LOADMODEL = os.environ.get("C_LOADMODEL", "")
_INVPROBE = os.environ.get("C_INVPROBE", "0") == "1"  # 1=逆モデルStage1診断（学習後に1回）
_INVEXEC = os.environ.get("C_INVEXEC", "0") == "1"  # 1=逆モデルStage1.5＝推論a*の実行テスト
_GOALBABBLE = os.environ.get("C_GOALBABBLE", "0") == "1"  # 1=Goal Babbling(目標指向の探索)
_GB_SWITCH = os.environ.get("C_GB_SWITCH", "pe")  # 探索/目標の切替: fixed(i%2) / ne(NE) / pe(予測誤差+NE, 既定)
_CLPROBE = os.environ.get("C_CLPROBE", "0") == "1"  # 1=C4診断＝閉ループ制御 vs 開ループの到達比較
_CLTRAIN = os.environ.get("C_CLTRAIN", "0") == "1"  # 1=閉ループreaching訓練（held goal＋停止条件）
_INVTRAJ = os.environ.get("C_INVTRAJ", "0") == "1"  # 1=各チェックポイントで逆probeも回し、逆モデルの"天井"軌跡(recover_corr/star)を記録
# 内発的動機の切替（既定＝従来の「予測しやすさ」）。Cは触覚なし・margin+51が確立した唯一の
# "うまくいくと分かっている環境"なので、ここで動機だけを差し替えれば**触覚という交絡なしに
# 動機の良し悪しだけ**を判定できる（D0では予測対象の89%が触覚で交絡していた）。
#   predict  : 1/(1+誤差)＝予測しやすい状態を求める（従来）
#   progress : pe_slow−pe_fast＝学習進度（Oudeyerの好奇心。誤差が"減っている"ことを求める）
_REWARD = os.environ.get("C_REWARD", "predict")
# NEを相対化するか（既定＝従来の絶対閾値）。従来は報酬の絶対値に固定閾値(<0.1で探索/>0.3で活用)を
# 当てるので、値域の違う報酬関数を入れると壊れる（学習進度≒0.04は常に「報酬ゼロ」と誤認される）。
# relative=Trueは「長期基準線と比べていつもより良いか」で決める＝尺度非依存（D0で必要性が判明）。
_NE_REL = os.environ.get("C_NE_RELATIVE", "0") == "1"
# 【2026-07-15】姿勢と触覚の切替。どちらも既定OFF＝従来の立位・触覚なしのCと完全に同じ。
#   C_SUPINE=1 : 仰向けで開始する。録画で判明した通り、既定(立位)のCは**開始3秒で転倒し、
#     以降ずっと床でもがいている**。margin+51はその状態で出た数字。仮説＝「Cが成功したのは
#     転んで偶然"手足が自由に振れる状態"になったから」。仰向けはそれを意図してやる。
#     シーンはCと同一(benchmarkv2)のまま姿勢だけ変える＝比較で変わる要素は姿勢1つだけ。
#   C_TOUCH=1  : 触覚を足す（乳児acuity＝somatotopy比を保ったまま解像度を2倍粗く）。
#     予測対象が固有感覚621→固有感覚+触覚 になるので、評価は必ず分けて見る（合計だけ見ると
#     触覚の次元数に薄められて何も分からない＝D0で踏んだ罠）。
_SUPINE = os.environ.get("C_SUPINE", "0") == "1"
_TOUCH = os.environ.get("C_TOUCH", "0") == "1"
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
_TOUCH_MODE = os.environ.get("C_TOUCH_MODE", "target")
# 省メモリ版（絵を落とす。物理は不変・視覚ONなら自動で素に戻る）。詳細は D/scripts/mimo_lean.py。
# 1本 2.64GB→0.28GB＝同時実行 6本→約22本。既定ON（仰向けは常に省メモリ版の上に載る）。
# C_LEAN=0 で従来の素のモデルに戻せる（アブレーション/描画品質が要るとき用）。
_LEAN = os.environ.get("C_LEAN", "1") == "1"
_ENV_ID = "TaroSupine-v0" if _SUPINE else ("TaroBenchV2Lean-v0" if _LEAN else "MIMoBenchV2-v0")


def _touch_params():
    if not _TOUCH:
        return None
    from d_supine_env import infant_touch_params
    return infant_touch_params(2.0)
CSV_COLUMNS = ["life_min", "train_step", "classify", "margin", "corr", "persist",
               "agency", "mag_ratio", "real_min"]


# MinimalFusion は taro_core/src/senses/fusion.py へ抽出済み（doc/移行記録_taro_core化_2026-07-17.md）。
# ここでは import して従来どおり使う（`from run_c_metrics_ac_lr import MinimalFusion` する
# D側スクリプトの後方互換のため、この名前で再エクスポートする）。視覚を足す拡張(vision_res)も
# 抽出先に入っているので、本番の視覚ONはそちらを使う。
from fusion import MinimalFusion  # noqa: F401  （再エクスポート）


def ln_prop(obs):
    """予測対象。既定は固有感覚のみ（従来のC）。

    C_TOUCH=1 かつ C_TOUCH_MODE=target のときだけ触覚を予測対象に加える。
    C_TOUCH_MODE=input なら触覚は fusion の入力にだけ入り、予測対象は固有感覚のまま。
    """
    v = to_tensor(obs["observation"])
    if _TOUCH and _TOUCH_MODE == "target":
        v = torch.cat([v, to_tensor(obs["touch"])])
    return torch.nn.functional.layer_norm(v, v.shape).detach()


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

    env = HybridEnv(gym.make(_ENV_ID, vision_params=None, touch_params=_touch_params()))
    # 触覚の次元数は「センサ点の総数」＝モデル構築時点で確定しており、reset不要で取れる。
    # ここで touch_dim を知るために env.reset() を足すと**乱数を1回余計に消費して学習の
    # 乱数列がずれる**（落とし穴チェック項3）。触覚なし条件が従来のCと比較不能になるので厳禁。
    # 観測空間はenv構築時に確定しているのでresetなしで読める。
    # 注意：get_sensor_count()は「センサ点の数」(1202)で、観測は1点あたり力の3成分＝3606。
    touch_dim = int(env.observation_space["touch"].shape[0]) if _TOUCH else 0
    fusion = MinimalFusion(touch_dim); target_fusion = MinimalFusion(touch_dim).freeze()
    n_act = env.action_space.shape[0]
    # 【バグ修正・2026-07-15】最初のresetに必ずseedを渡す。
    # 環境の乱数(`env.unwrapped.np_random`)は gym が別に管理しており、torch.manual_seed も
    # np.random.seed も効かない。仰向け環境は reset のたびに np_random で初期姿勢を揺らす
    # （jitter）ので、**シードなしresetだと毎回ちがう姿勢から始まり、シードで再現できない**。
    # 立位Cはjitterが無いので影響を受けず、この穴は仰向けを足すまで露呈しなかった。
    # 症状：同一シード・同一条件のはずの2ランで life=0min が 58.99 vs 58.20 とズレる。
    obs, _ = env.reset(seed=seed)
    sdim = fusion.encode(obs).shape[0]; prop_dim = to_tensor(obs["observation"]).shape[0]
    # nat_headが吐く次元＝予測対象の次元。touch_mode=input なら触覚は予測対象に入らない。
    out_dim = prop_dim + (touch_dim if _TOUCH_MODE == "target" else 0)
    brain = TaroBrainWithMotor(vocab_size=3, sensory_dim=sdim, n_actuators=n_act)
    emb_dim = brain.sensory_proj.out_features  # GRUの入力次元(=64)

    # D-a: [感覚, 前回行動] → GRU入力。brain.sensory_proj の代わりに使う自前の射影。
    emb_proj = nn.Linear(sdim + n_act, emb_dim)
    # D-b: 非線形MLPの予測ヘッド（[z, 今の行動] → 固有感覚の変化）。
    # 【パッチ 2026-07-13】中間に LayerNorm を追加。アブレーションで発散の原因が
    # 「ヘッドが入力を無制限に増幅（pred爆発）」と確定したため（zの膨張は濡れ衣）。
    # 活性を正規化して出力爆発を防ぐ（深層ネット/世界モデルの標準的な安定化）。
    nat_head = nn.Sequential(nn.Linear(brain.latent_dim + n_act, 128), nn.SiLU(),
                             nn.LayerNorm(128), nn.Linear(128, out_dim))
    learner = TaroLearner(CombinedParams(brain, fusion, emb_proj, nat_head), lr=_LR)
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
        _load_matching(emb_proj, _blob["emb_proj"], "emb_proj")
        _load_matching(nat_head, _blob["nat_head"], "nat_head")
        if _CEREB and "cereb" in _blob:
            _load_matching(cereb, _blob["cereb"], "小脳")

    state = {"obs": obs, "hidden": brain.init_motor_hidden(),
             "prev_a": torch.zeros(n_act)}
    t0 = time.time()
    eff_accum, act_accum = [], []  # 【taro-C5】努力コストと|行動|の記録（フリーズ監視用）
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
        # D-a: 感覚と前回行動を結合して射影→GRU
        emb = emb_proj(torch.cat([sv, prev_a], dim=-1)).unsqueeze(0).unsqueeze(0)
        out, nh = brain.motor_gru(emb, h)
        z, kl, rc = brain.pc_latent.infer(h[-1, 0], out[0, -1], cf)
        return z, kl, rc, nh

    def reset_state():
        state["obs"], _ = env.reset()
        state["hidden"] = brain.init_motor_hidden()
        state["prev_a"] = torch.zeros(n_act)

    def act_mean(z):
        # 決定的な行動平均（評価・agency用、ノイズなし）。小脳ONなら自動化ブレンドを適用。
        pm = torch.tanh(brain.motor_head(z))
        if not _CEREB:
            return pm
        w, cere_a, _ = cereb.gate(z, pm)
        return (1.0 - w) * pm + w * cere_a

    def infer_goal_action(z, clp, init_mean, g, n_steps=15, lr_inf=0.1):
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

    def evaluate():
        Zs, acts, nx, cu, self_err, ep = [], [], [], [], [], []
        pdel, adel = [], []
        for _ in range(n_eval):
            sv = fusion.encode(state["obs"]); cf = target_fusion.encode(state["obs"]).detach(); clp = ln_prop(state["obs"])
            z, _, _, hn = zc(sv, state["prev_a"], cf, state["hidden"]); z = z.detach()
            a = torch.clamp(act_mean(z), -1.0, 1.0).detach()
            pd = nat_head(torch.cat([z, a], dim=-1)).detach()
            state["obs"], term = step_k(rescale_action(a, env.action_space))
            nlp = ln_prop(state["obs"])
            self_err.append(mse(clp + pd, nlp).item()); ep.append(mse(clp, nlp).item())
            pdel.append(pd.numpy()); adel.append((nlp - clp).numpy())
            Zs.append(z); acts.append(a); nx.append(nlp); cu.append(clp)
            state["hidden"] = hn.detach(); state["prev_a"] = a
            if term:
                reset_state()
        N = len(Zs); correct = total = 0; other_errs = []
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                eo = mse((cu[i] + nat_head(torch.cat([Zs[i], acts[j]], dim=-1))).detach(), nx[i]).item()
                other_errs.append(eo)
                if self_err[i] < eo:
                    correct += 1
                total += 1
        classify = correct / total * 100
        margin = (np.mean(other_errs) - np.mean(self_err)) / np.mean(other_errs) * 100
        persist = np.mean(self_err) / np.mean(ep) * 100
        P = np.concatenate([p.flatten() for p in pdel]); A = np.concatenate([a.flatten() for a in adel])
        corr = float(np.corrcoef(P, A)[0, 1])
        return classify, margin, corr, persist

    def agency_probe(n=30):
        # 遠心性コピー(=太郎の意図a_self)は両トライアル共通。体が自分の意図どおり(自己)か
        # 外部指令(外因)かだけが違う。GRUに渡す前回行動は常に「太郎自身の意図」。
        self_errs, ext_errs, self_mag, ext_mag, self_acts = [], [], [], [], []
        for _ in range(n):
            sv = fusion.encode(state["obs"]); cf = target_fusion.encode(state["obs"]).detach(); clp = ln_prop(state["obs"])
            z, _, _, hn = zc(sv, state["prev_a"], cf, state["hidden"]); z = z.detach()
            a = torch.clamp(act_mean(z), -1.0, 1.0).detach()
            pred = nat_head(torch.cat([z, a], dim=-1)).detach()
            state["obs"], term = step_k(rescale_action(a, env.action_space))
            nlp = ln_prop(state["obs"])
            self_errs.append(mse(clp + pred, nlp).item())
            self_mag.append((nlp - clp).abs().mean().item())
            self_acts.append(a); state["hidden"] = hn.detach(); state["prev_a"] = a
            if term:
                reset_state()
        perm = np.random.permutation(len(self_acts))
        for k in range(n):
            sv = fusion.encode(state["obs"]); cf = target_fusion.encode(state["obs"]).detach(); clp = ln_prop(state["obs"])
            z, _, _, hn = zc(sv, state["prev_a"], cf, state["hidden"]); z = z.detach()
            a_self = torch.clamp(act_mean(z), -1.0, 1.0).detach()
            a_ext = self_acts[perm[k]]
            pred = nat_head(torch.cat([z, a_self], dim=-1)).detach()
            state["obs"], term = step_k(rescale_action(a_ext, env.action_space))
            nlp = ln_prop(state["obs"])
            ext_errs.append(mse(clp + pred, nlp).item())
            ext_mag.append((nlp - clp).abs().mean().item())
            state["hidden"] = hn.detach(); state["prev_a"] = a_self  # 遠心性コピーは意図
            if term:
                reset_state()
        correct = total = 0
        for se in self_errs:
            for ee in ext_errs:
                correct += int(se < ee); total += 1
        agency = correct / total * 100
        mag_ratio = np.mean(ext_mag) / max(np.mean(self_mag), 1e-9) * 100
        return agency, mag_ratio

    def inverse_probe(n=60, n_steps=40, n_restarts=3, lr_inf=0.1):
        """逆モデルStage1診断：凍結した順モデル(nat_head)を"反転"して、望む次感覚に当てる
        行動を推論できるか。goal＝実際に到達した次固有感覚(到達可能)。
        主眼＝①行動レバレッジ(err_random − err_infer：行動が予測にどれだけ効くか)、
        ②順精度の天井(err_areal)。副次＝復元相関(a*とa_realの一致、§5線形0.22と比較)。"""
        ei, ea, er, astar_all, areal_all = [], [], [], [], []
        for _ in range(n):
            sv = fusion.encode(state["obs"]); cf = target_fusion.encode(state["obs"]).detach()
            clp = ln_prop(state["obs"])
            z, _, _, hn = zc(sv, state["prev_a"], cf, state["hidden"]); z = z.detach()
            a_real = torch.clamp(act_mean(z), -1.0, 1.0).detach()
            state["obs"], term = step_k(rescale_action(a_real, env.action_space))
            target = (ln_prop(state["obs"]) - clp).detach()  # 望む変化＝実際の次感覚−今

            def ferr(a):
                return ((nat_head(torch.cat([z, a], dim=-1)) - target) ** 2).mean()

            best_a, best_e = a_real, ferr(a_real).item()
            for r in range(n_restarts):
                raw = (torch.atanh(torch.clamp(a_real, -0.999, 0.999)).clone() if r == 0
                       else torch.empty(n_act).uniform_(-1.0, 1.0)).detach().requires_grad_(True)
                opt = torch.optim.Adam([raw], lr=lr_inf)
                for _ in range(n_steps):
                    opt.zero_grad(); ferr(torch.tanh(raw)).backward(); opt.step()
                af = torch.tanh(raw).detach(); ef = ferr(af).item()
                if ef < best_e:
                    best_e, best_a = ef, af
            ei.append(best_e); ea.append(ferr(a_real).item())
            er.append(ferr(torch.empty(n_act).uniform_(-1.0, 1.0)).item())
            astar_all.append(best_a.numpy()); areal_all.append(a_real.numpy())
            state["hidden"] = hn.detach(); state["prev_a"] = a_real
            if term:
                reset_state()
        A = np.concatenate(astar_all); B = np.concatenate(areal_all)
        rec = float(np.corrcoef(A, B)[0, 1])
        mi, ma, mr = float(np.mean(ei)), float(np.mean(ea)), float(np.mean(er))
        print(f"[INV seed{seed}] err_infer={mi:.4f} err_areal(順精度天井)={ma:.4f} "
              f"err_random={mr:.4f} | leverage(rand-infer)={mr - mi:.4f} "
              f"infer/random={mi / max(mr, 1e-9):.2f} infer/areal={mi / max(ma, 1e-9):.2f} "
              f"recover_corr(a*,a_real)={rec:.3f}", flush=True)
        with open(os.path.join(LOG_DIR, f"inv_probe_seed{seed}.txt"), "w", encoding="utf-8") as fp:
            fp.write(f"err_infer,{mi}\nerr_areal,{ma}\nerr_random,{mr}\nleverage,{mr - mi}\n"
                     f"infer_over_random,{mi / max(mr, 1e-9)}\ninfer_over_areal,{mi / max(ma, 1e-9)}\n"
                     f"recover_corr,{rec}\n")
        return {"infer_over_random": mi / max(mr, 1e-9), "recover_corr": rec}

    def inverse_exec_probe(n=40, n_steps=40, n_restarts=3, lr_inf=0.1):
        """逆モデルStage1.5＝実行テスト：推論a*を"実際にMIMoで実行"し、現実の次感覚が
        目標に届くか。同じ物理状態(qpos/qvel)から a_real / a* / random を実行して現実の
        到達誤差を比較（MuJoCo state save/restore でカウンターファクト）。
        goal＝基準行動a_realを実行した現実の次感覚。d_real2＝a_real再実行＝決定性の床(≈0期待)。"""
        mj = env.unwrapped
        d_star, d_rand, d_floor, mi_model = [], [], [], []

        def real_rollout(a, qpos, qvel):
            mj.set_state(qpos.copy(), qvel.copy())
            o, _ = step_k(rescale_action(a, env.action_space))
            return ln_prop(o)

        for _ in range(n):
            sv = fusion.encode(state["obs"]); cf = target_fusion.encode(state["obs"]).detach()
            clp = ln_prop(state["obs"])
            z, _, _, hn = zc(sv, state["prev_a"], cf, state["hidden"]); z = z.detach()
            a_real = torch.clamp(act_mean(z), -1.0, 1.0).detach()
            qpos = mj.data.qpos.copy(); qvel = mj.data.qvel.copy()
            g = real_rollout(a_real, qpos, qvel)          # 目標＝a_realの現実の結果
            g2 = real_rollout(a_real, qpos, qvel)         # 再実行＝決定性の床
            target = (g - clp).detach()

            def ferr(a):
                return ((nat_head(torch.cat([z, a], dim=-1)) - target) ** 2).mean()

            best_a, best_e = a_real, ferr(a_real).item()
            for r in range(n_restarts):
                raw = (torch.atanh(torch.clamp(a_real, -0.999, 0.999)).clone() if r == 0
                       else torch.empty(n_act).uniform_(-1.0, 1.0)).detach().requires_grad_(True)
                opt = torch.optim.Adam([raw], lr=lr_inf)
                for _ in range(n_steps):
                    opt.zero_grad(); ferr(torch.tanh(raw)).backward(); opt.step()
                af = torch.tanh(raw).detach(); ef = ferr(af).item()
                if ef < best_e:
                    best_e, best_a = ef, af
            a_rand = torch.empty(n_act).uniform_(-1.0, 1.0)
            o_star = real_rollout(best_a, qpos, qvel)     # 推論a*を実行
            o_rand = real_rollout(a_rand, qpos, qvel)     # ランダムを実行
            d_star.append(mse(o_star, g).item())
            d_rand.append(mse(o_rand, g).item())
            d_floor.append(mse(g2, g).item())
            mi_model.append(best_e)
            # 実トラジェクトリを a_real で1歩進める（他プローブと同様に状態を歩かせる）
            state["obs"], term = step_k(rescale_action(a_real, env.action_space))
            state["hidden"] = hn.detach(); state["prev_a"] = a_real
            if term:
                reset_state()
        ds, dr, df = float(np.mean(d_star)), float(np.mean(d_rand)), float(np.mean(d_floor))
        mm = float(np.mean(mi_model))
        print(f"[INVEXEC seed{seed}] d_star(a*実行)={ds:.4f} d_random={dr:.4f} "
              f"d_floor(a_real再実行)={df:.4f} | star/random={ds / max(dr, 1e-9):.2f} "
              f"star/floor={ds / max(df, 1e-9):.2f} model_err(a*)={mm:.4f}", flush=True)
        with open(os.path.join(LOG_DIR, f"inv_exec_seed{seed}.txt"), "w", encoding="utf-8") as fp:
            fp.write(f"d_star,{ds}\nd_random,{dr}\nd_floor,{df}\n"
                     f"star_over_random,{ds / max(dr, 1e-9)}\nstar_over_floor,{ds / max(df, 1e-9)}\n"
                     f"model_err_star,{mm}\n")
        return {"star_over_random": ds / max(dr, 1e-9), "model_err": mm}

    def closed_loop_probe(n=30, max_reach=10):
        """C4診断：閉ループ制御 vs 開ループ で、目標姿勢へどれだけ届くか。
        目標g＝過去に経験した姿勢(goal_buf)。同じ物理状態(MuJoCo save/restore)から比較。
        補正の刻み k_inner ＝ K（順モデルの予測幅に合わせる。第1版はK//8でミスマッチ→負けた）。
        【リーチ長は固定しない＝根拠づけ】"補正しても目標に近づかなくなったら止める"（能動的
        推論＝誤差が減らせなくなったら停止）。max_reach は暴走防止の安全上限のみ（挙動を決める
        数字ではない）。開ループは閉ループが要した長さと同じで比較（公平）。"""
        mj = env.unwrapped
        k_inner = K  # 補正の刻み＝順モデルの予測幅（ミスマッチ解消）
        dc, do, dr, dn, steps = [], [], [], [], []

        def const_rollout(a, nsteps, qpos, qvel):
            mj.set_state(qpos.copy(), qvel.copy())
            ra = rescale_action(a, env.action_space); o = state["obs"]
            for _ in range(nsteps):
                o, _, te, tr, _ = env.step(ra)
                if te or tr:
                    break
            return ln_prop(o)

        def closed_rollout(g, z0, clp0, h0, pa0, qpos, qvel):
            mj.set_state(qpos.copy(), qvel.copy())
            z_now, clp_now, hcl, pacl = z0, clp0, h0, pa0
            o = state["obs"]; prev_d = mse(clp_now, g).item(); nstep = 0
            for _ in range(max_reach):
                a = infer_goal_action(z_now, clp_now, torch.clamp(act_mean(z_now), -1.0, 1.0), g)
                ra = rescale_action(a, env.action_space)
                for _ in range(k_inner):
                    o, _, te, tr, _ = env.step(ra)
                    if te or tr:
                        break
                nstep += 1
                clp_now = ln_prop(o)  # ← 実際に観測した状態から補正（閉ループの"閉じ"）
                d = mse(clp_now, g).item()
                if d >= prev_d:  # これ以上近づけない→停止（誤差最小化の自然な停止条件＝根拠）
                    break
                prev_d = d
                sv_now = fusion.encode(o); cf_now = target_fusion.encode(o).detach()
                zt, _, _, hn = zc(sv_now, pacl, cf_now, hcl)
                z_now = zt.detach(); hcl = hn.detach(); pacl = a
            return ln_prop(o), nstep

        for _ in range(n):
            sv = fusion.encode(state["obs"]); cf = target_fusion.encode(state["obs"]).detach()
            clp0 = ln_prop(state["obs"])
            z0, _, _, hn0 = zc(sv, state["prev_a"], cf, state["hidden"]); z0 = z0.detach()
            g = goal_buf[torch.randint(len(goal_buf), (1,)).item()]
            qpos = mj.data.qpos.copy(); qvel = mj.data.qvel.copy()
            o_closed, nstep = closed_rollout(g, z0, clp0, state["hidden"], state["prev_a"], qpos, qvel)
            a_open = infer_goal_action(z0, clp0, torch.clamp(act_mean(z0), -1.0, 1.0), g)
            o_open = const_rollout(a_open, max(1, nstep) * k_inner, qpos, qvel)  # 閉ループと同じ長さで公平比較
            o_rand = const_rollout(torch.empty(n_act).uniform_(-1.0, 1.0), max(1, nstep) * k_inner, qpos, qvel)
            do.append(mse(o_open, g).item()); dc.append(mse(o_closed, g).item())
            dr.append(mse(o_rand, g).item()); dn.append(mse(clp0, g).item()); steps.append(nstep)
            mj.set_state(qpos, qvel)  # 実トラジェクトリを1リーチぶん進める
            state["obs"], term = step_k(rescale_action(a_open, env.action_space))
            state["hidden"] = hn0.detach(); state["prev_a"] = a_open
            if term:
                reset_state()
        mc, mo, mr, mn, ms = (float(np.mean(dc)), float(np.mean(do)), float(np.mean(dr)),
                              float(np.mean(dn)), float(np.mean(steps)))
        print(f"[CLOSEDLOOP seed{seed}] closed={mc:.4f} open={mo:.4f} random={mr:.4f} "
              f"nothing={mn:.4f} | closed/open={mc / max(mo, 1e-9):.2f} "
              f"closed/nothing={mc / max(mn, 1e-9):.2f} reach_len_avg={ms:.1f}", flush=True)
        with open(os.path.join(LOG_DIR, f"closed_loop_seed{seed}.txt"), "w", encoding="utf-8") as fp:
            fp.write(f"closed,{mc}\nopen,{mo}\nrandom,{mr}\nnothing,{mn}\nreach_len_avg,{ms}\n"
                     f"closed_over_open,{mc / max(mo, 1e-9)}\nclosed_over_nothing,{mc / max(mn, 1e-9)}\n")

    def checkpoint(step):
        cl, mg, co, pr = evaluate()
        ag, magr = agency_probe()
        life_min = step * K * DT / 60.0
        real_min = (time.time() - t0) / 60.0
        log_row([f"{life_min:.1f}", step, f"{cl:.2f}", f"{mg:.2f}", f"{co:.4f}",
                 f"{pr:.2f}", f"{ag:.2f}", f"{magr:.1f}", f"{real_min:.1f}"])
        noise = 0.05 + ne.get_ne_level() * 0.45
        cereb_tag = f" cereb=on(err={cereb.err_ema.item():.2f})" if _CEREB else " cereb=off"
        # 【taro-C5】|行動|＝力の出し具合（0.86が全力偏重。低下＝加減を学習。0付近＝フリーズ警告）。
        act_tag = ""
        if act_accum:
            act_tag = f" |act|={np.mean(act_accum[-200:]):.3f}"
            if _EFFORT and eff_accum:
                act_tag += f" effort={np.mean(eff_accum[-200:]):.3f}(λ={_EFFORT})"
        print(f"[AC seed{seed} rew={_REWARD} ne={'rel' if _NE_REL else 'abs'} touch={_TOUCH_MODE if _TOUCH else 'off'}] life={life_min:.0f}min | classify={cl:.1f}% margin={mg:+.1f}% "
              f"corr={co:.3f} persist={pr:.1f}% agency={ag:.1f}%(mag {magr:.0f}%) | "
              f"noise={noise:.3f}(mat={ne.maturation:.2f}){cereb_tag}{act_tag} real={real_min:.0f}min", flush=True)

    # 経験バッファ（睡眠中リプレイ用）。各ステップの予測に必要な材料を貯める。
    buf = {k: [] for k in ("sv", "prev_a", "a", "cf", "clp", "nlp", "h")}
    # Goal Babbling 用の目標バッファ＝Self-Priorの最小版（過去に経験した固有感覚の分布）。
    goal_buf = []
    pe_fast, pe_slow = 1.0, 1.0  # 予測誤差の速い/遅い走行平均（"いつもより驚いたか"の自己正規化用）
    reach_goal, reach_prev_dist = None, 0.0  # 閉ループreaching訓練：保持中の目標と直前の距離

    def consolidate(n_batches=200, bs=128):
        """睡眠中の記憶定着：貯めた経験をバッチで再生し、自己モデル(予測経路)を復習で固める。"""
        N = len(buf["sv"])
        if N < bs:
            return
        SV = torch.stack(buf["sv"]); PA = torch.stack(buf["prev_a"]); AA = torch.stack(buf["a"])
        CF = torch.stack(buf["cf"]); CLP = torch.stack(buf["clp"]); NLP = torch.stack(buf["nlp"])
        H = torch.cat(buf["h"], dim=1)  # (layers, N, hidden)
        for _ in range(n_batches):
            idx = torch.randint(0, N, (bs,))
            hb = H[:, idx].contiguous()
            emb = emb_proj(torch.cat([SV[idx], PA[idx]], dim=-1)).unsqueeze(1)  # (bs,1,emb) batch_first
            out, _ = brain.motor_gru(emb, hb)  # out (bs,1,hidden)
            z, kl, rc = brain.pc_latent.infer(hb[-1], out[:, 0], CF[idx])
            pred = CLP[idx] + nat_head(torch.cat([z, AA[idx]], dim=-1))
            loss = mse(pred, NLP[idx]) + kl + rc
            learner.optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(learner.brain.parameters(), learner.grad_clip)
            learner.optimizer.step()

    checkpoint(0)
    for i in range(n_train):
        sv = fusion.encode(state["obs"]); cf = target_fusion.encode(state["obs"]).detach(); clp = ln_prop(state["obs"])
        z, kl, rc, hn = zc(sv, state["prev_a"], cf, state["hidden"].detach())
        policy_m = torch.tanh(brain.motor_head(z.detach()))
        std = 0.05 + ne.get_ne_level() * 0.45
        if _CEREB:
            # 自動化ブレンド：馴染んだ状態ほど小脳の滑らかな出力で置換＋探索ノイズ減（結晶化）
            w_c, cere_a, e_c = cereb.gate(z.detach(), policy_m)
            mean = (1.0 - w_c) * policy_m + w_c * cere_a
            std = std * (1.0 - w_c)
        else:
            mean = policy_m
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
                # 【逸脱/工学近似 ⚠️】向き（驚き大・NE大→探索、＝分かる所は狙い分からぬ所は探る）
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
        dist = torch.distributions.Normal(mean, std)
        a = torch.clamp(dist.sample(), -1.0, 1.0); lp = dist.log_prob(a).sum()
        goal_buf.append(clp.detach())
        if len(goal_buf) > 2000:
            goal_buf.pop(0)
        pred = clp + nat_head(torch.cat([z, a.detach()], dim=-1))
        state["obs"], term = step_k(rescale_action(a, env.action_space)); nlp = ln_prop(state["obs"])
        if _CLTRAIN and reach_goal is not None:
            nd = mse(nlp, reach_goal).item()
            if nd >= reach_prev_dist:
                reach_goal = None  # 近づかなくなった→リーチ終了（次は新しい目標）＝停止条件
            else:
                reach_prev_dist = nd
        if _REPLAY:
            buf["sv"].append(sv.detach()); buf["prev_a"].append(state["prev_a"].detach())
            buf["a"].append(a.detach()); buf["cf"].append(cf.detach())
            buf["clp"].append(clp.detach()); buf["nlp"].append(nlp.detach())
            buf["h"].append(state["hidden"].detach())
        pe = mse(pred, nlp)
        pe_fast = 0.9 * pe_fast + 0.1 * pe.item()   # 次ステップの切替判断に使う（因果的に過去の驚き）
        pe_slow = 0.99 * pe_slow + 0.01 * pe.item()
        # 内発的動機。progress＝学習進度（誤差が減っていれば正）／predict＝従来の予測しやすさ。
        rew = (pe_slow - pe_fast) if _REWARD == "progress" else brain.sensorimotor_reward(pe.item())
        # 【taro-C5】努力コスト：活性化²の筋力重み付き平均（∈[0,1]）を報酬から引く。＝大きな力ほど
        # 損→自分で加減する（Selinger 2015等の代謝最小化。⚠️二乗・λ・重みは近似＝感度確認対象）。既定OFF。
        if _EFFORT:
            effort = float((a.detach() ** 2 * eff_w).sum())
            rew = rew - _EFFORT * effort
            eff_accum.append(effort)
        act_accum.append(float(a.detach().abs().mean().item()))
        pl = learner.learn_action([lp], dop.compute_rpe(rew))
        hl = homeo.homeostatic_loss(sv); homeo.observe(sv)
        learner.update(pe + hl + kl + rc, pl)
        if _CEREB:
            # 小脳は方策とは別に、実際に行った運動を教師なしで真似て自動化パターンを固める。
            closs = cereb.imitation_loss(z.detach(), a.detach())
            cere_opt.zero_grad(); closs.backward(); cere_opt.step()
            cereb.observe(e_c)  # 馴染み度の基準を更新（自己正規化）
        dev_clock.tick()  # ③発達年齢を進める（覚醒中の学習1回）。consolidate側では進めない。
        ne.observe_reward(rew); ne.release_ne()
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
                _ip = inverse_probe(); _ie = inverse_exec_probe()  # 逆モデルの天井軌跡
                with open(invtraj_path, "a", newline="", encoding="utf-8") as _fp:
                    csv.writer(_fp).writerow([i + 1, _ip["recover_corr"], _ip["infer_over_random"],
                                             _ie["star_over_random"], _ie["model_err"]])

    if _INVPROBE:
        inverse_probe()  # 学習後に逆モデルStage1診断を1回
    if _INVEXEC:
        inverse_exec_probe()  # Stage1.5＝推論a*の実行テスト
    if _CLPROBE:
        closed_loop_probe()  # C4＝閉ループ制御 vs 開ループの到達比較
    if os.environ.get("C_SAVEMODEL"):
        # 確立した自己モデルを保存（D等の下流で"最高モデル"を再利用するため）。opt-in。
        mp = os.environ["C_SAVEMODEL"]; os.makedirs(os.path.dirname(mp), exist_ok=True)
        blob = {"brain": brain.state_dict(),
                "fusion_insula": fusion.insula.state_dict(),
                "fusion_proprio": fusion.proprio.state_dict(),
                "fusion_vestibular": fusion.vestibular.state_dict(),
                "emb_proj": emb_proj.state_dict(), "nat_head": nat_head.state_dict(),
                "cereb": cereb.state_dict(),
                "config": {"sdim": sdim, "prop_dim": prop_dim, "touch_dim": touch_dim,
                           "out_dim": out_dim, "n_act": n_act, "K": K,
                           "seed": seed, "n_train": n_train, "replay": _REPLAY,
                           "cereb": _CEREB, "supine": _SUPINE, "touch": _TOUCH, "touch_mode": _TOUCH_MODE,
                           "reward": _REWARD, "ne_relative": _NE_REL, "env_id": _ENV_ID,
                           "effort": _EFFORT, "loadmodel": os.path.basename(_LOADMODEL) if _LOADMODEL else "",
                           "fusion": "MinimalFusion(interoception+proprio621+vestibular"
                                     + ("+touch)" if _TOUCH else ")")}}
        if fusion.touch is not None:
            blob["fusion_touch"] = fusion.touch.state_dict()
        torch.save(blob, mp)
        print(f"SAVED MODEL {mp}", flush=True)
    if os.environ.get("C_RECORD"):
        # 【必須級・2026-07-15】学習後の太郎を等速で録画する（C_RECORD=<出力mp4>で有効）。
        # 理由＝太郎の指標は「予測がうまいか」を測るが、予測を最もうまくする方法は
        # 「何も面白いことをしない」こと。つまり**指標は行動の退化をむしろ高評価する**。
        # 実測(D0)：margin+52/corr0.76が過去最高値なのに中身は筋肉の痙攣、corr0.998の
        # 正体は環境の連続リセット。数字だけでは構造的に検出できないので目視を残す。
        # 注意：脳はK=100(0.5秒)に1回判断するので、判断ごとに1枚だと7.5倍速になる。
        # 判断の"間"のコマも撮って等速にする（render_everyステップ毎）。
        try:
            import cv2
            # 学習と同じ環境で録画する（姿勢・触覚を揃え忘れると"学習時と違う太郎"を
            # 見せることになる。D0の録画で実際にやらかした＝録画側だけリセットが残った）。
            renv = HybridEnv(gym.make(_ENV_ID, vision_params=None,
                                      touch_params=_touch_params(), render_mode="rgb_array"))
            ro, _ = renv.reset(seed=seed)
            rh = brain.init_motor_hidden(); rpa = torch.zeros(n_act)
            frames = []; render_every = 4
            for _ in range(20):   # 20判断＝10秒ぶん
                sv = fusion.encode(ro); cf = target_fusion.encode(ro).detach()
                z, _, _, rhn = zc(sv, rpa, cf, rh); z = z.detach()
                a = torch.clamp(act_mean(z), -1.0, 1.0).detach()
                ctrl = rescale_action(a, renv.action_space)
                for k in range(K):
                    ro, _, te, tr, _ = renv.step(ctrl)
                    if k % render_every == 0:
                        f = renv.render()
                        if f is not None:
                            frames.append(f)
                    if te or tr:
                        break
                rh = rhn.detach(); rpa = a
                if te or tr:
                    ro, _ = renv.reset(); rh = brain.init_motor_hidden(); rpa = torch.zeros(n_act)
            renv.close()
            if frames:
                mp4 = os.environ["C_RECORD"]; os.makedirs(os.path.dirname(mp4), exist_ok=True)
                fps = (1.0 / DT) / render_every   # 等速再生になるfps
                hh, ww, _ = frames[0].shape
                vw = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ww, hh))
                for f in frames:
                    vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                vw.release()
                print(f"RECORDED {mp4} ({len(frames)}フレーム, 等速{fps:.0f}fps)", flush=True)
        except Exception as e:
            print(f"[警告] 録画に失敗（学習結果は保存済み）: {type(e).__name__}: {e}", flush=True)
    print(f"DONE seed={seed} log={csv_path}", flush=True)


if __name__ == "__main__":
    seed = int(sys.argv[1])
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 3600
    K = int(os.environ.get("C_K", "100"))
    ckpt = int(os.environ.get("C_CKPT", "600"))  # 【taro-C5】評価/ログ間隔（短い実験用に小さくできる）
    run(seed, n_train=n_train, K=K, ckpt=ckpt)
