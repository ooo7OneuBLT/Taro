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

_BRIDGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # = Taro/C（src/・tests/を持つ）
for sub in ("wrapper", "senses", "brain"):
    sys.path.insert(0, os.path.join(_BRIDGE, "src", sub))
sys.path.insert(0, os.path.join(_BRIDGE, "tests"))
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
from sensory_encoders import ProprioceptionEncoder, VestibularEncoder
from insula import Insula

mse = torch.nn.functional.mse_loss
DT = 0.01
LOG_DIR = os.environ.get("C_LOGDIR", os.path.join(_BRIDGE, "logs", "C", "ac_prototype", "patched_layernorm"))
_LR = float(os.environ.get("C_LR", "0.005"))
_MATURE = os.environ.get("C_MATURE", "0") == "1"  # 1=学習進行に合わせて探索を結晶化
_REPLAY = os.environ.get("C_REPLAY", "0") == "1"  # 1=睡眠中の経験リプレイ（記憶定着）を行う
# 【注意すべき機能】既定ON＝運動小脳（自動化/結晶化）。良性は検証済み（発散・フリーズ・
# agency崩壊なし）だが、現指標では効果ほぼ不変・常時稼働（約26%ブレンド）。将来の壁の
# 切り分け時は「小脳が効いている可能性」を必ず確認する（`注意すべき機能リスト.md` 参照）。
_CEREB = os.environ.get("C_CEREBELLUM", "1") == "1"  # 0で無効化可
CSV_COLUMNS = ["life_min", "train_step", "classify", "margin", "corr", "persist",
               "agency", "mag_ratio", "real_min"]


class MinimalFusion:
    def __init__(self):
        self.insula = Insula(state_dim=4, embedding_dim=64)
        self.proprio = ProprioceptionEncoder(input_dim=621)
        self.vestibular = VestibularEncoder(input_dim=6)

    def parameters(self):
        import itertools
        return itertools.chain(self.insula.parameters(), self.proprio.parameters(), self.vestibular.parameters())

    def encode(self, obs):
        f = torch.cat([self.insula(to_tensor(obs["interoception"])),
                       self.proprio(to_tensor(obs["observation"])),
                       self.vestibular(to_tensor(obs["vestibular"]))], dim=-1)
        return torch.nn.functional.layer_norm(f, f.shape)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)
        return self


def ln_prop(obs):
    v = to_tensor(obs["observation"])
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

    env = HybridEnv(gym.make("MIMoBenchV2-v0", vision_params=None, touch_params=None))
    fusion = MinimalFusion(); target_fusion = MinimalFusion().freeze()
    n_act = env.action_space.shape[0]
    obs, _ = env.reset()
    sdim = fusion.encode(obs).shape[0]; prop_dim = to_tensor(obs["observation"]).shape[0]
    brain = TaroBrainWithMotor(vocab_size=3, sensory_dim=sdim, n_actuators=n_act)
    emb_dim = brain.sensory_proj.out_features  # GRUの入力次元(=64)

    # D-a: [感覚, 前回行動] → GRU入力。brain.sensory_proj の代わりに使う自前の射影。
    emb_proj = nn.Linear(sdim + n_act, emb_dim)
    # D-b: 非線形MLPの予測ヘッド（[z, 今の行動] → 固有感覚の変化）。
    # 【パッチ 2026-07-13】中間に LayerNorm を追加。アブレーションで発散の原因が
    # 「ヘッドが入力を無制限に増幅（pred爆発）」と確定したため（zの膨張は濡れ衣）。
    # 活性を正規化して出力爆発を防ぐ（深層ネット/世界モデルの標準的な安定化）。
    nat_head = nn.Sequential(nn.Linear(brain.latent_dim + n_act, 128), nn.SiLU(),
                             nn.LayerNorm(128), nn.Linear(128, prop_dim))
    learner = TaroLearner(CombinedParams(brain, fusion, emb_proj, nat_head), lr=_LR)
    dop = Dopamine(); ne = LocusCoeruleus(); homeo = HomeostaticScaling(dim=sdim)
    dev_clock = DevelopmentalClock()  # ③発達年齢（累積学習回数）。sim秒(②)とは別軸。
    # 運動小脳。ON/OFFで乱数列を揃えるため、_CEREBに関わらず常に構築する（使う/学習する
    # のは_CEREB時のみ。gate/imitationは乱数を消費しないので、初期化以降はON/OFFで乱数が一致）。
    cereb = MotorCerebellum(brain.latent_dim, n_act)
    cere_opt = torch.optim.Adam(cereb.parameters(), lr=_LR)
    state = {"obs": obs, "hidden": brain.init_motor_hidden(),
             "prev_a": torch.zeros(n_act)}
    t0 = time.time()

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

    def checkpoint(step):
        cl, mg, co, pr = evaluate()
        ag, magr = agency_probe()
        life_min = step * K * DT / 60.0
        real_min = (time.time() - t0) / 60.0
        log_row([f"{life_min:.1f}", step, f"{cl:.2f}", f"{mg:.2f}", f"{co:.4f}",
                 f"{pr:.2f}", f"{ag:.2f}", f"{magr:.1f}", f"{real_min:.1f}"])
        noise = 0.05 + ne.get_ne_level() * 0.45
        cereb_tag = f" cereb=on(err={cereb.err_ema.item():.2f})" if _CEREB else " cereb=off"
        print(f"[AC seed{seed} mat={_MATURE}] life={life_min:.0f}min | classify={cl:.1f}% margin={mg:+.1f}% "
              f"corr={co:.3f} persist={pr:.1f}% agency={ag:.1f}%(mag {magr:.0f}%) | "
              f"noise={noise:.3f}(mat={ne.maturation:.2f}){cereb_tag} real={real_min:.0f}min", flush=True)

    # 経験バッファ（睡眠中リプレイ用）。各ステップの予測に必要な材料を貯める。
    buf = {k: [] for k in ("sv", "prev_a", "a", "cf", "clp", "nlp", "h")}

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
        dist = torch.distributions.Normal(mean, std)
        a = torch.clamp(dist.sample(), -1.0, 1.0); lp = dist.log_prob(a).sum()
        pred = clp + nat_head(torch.cat([z, a.detach()], dim=-1))
        state["obs"], term = step_k(rescale_action(a, env.action_space)); nlp = ln_prop(state["obs"])
        if _REPLAY:
            buf["sv"].append(sv.detach()); buf["prev_a"].append(state["prev_a"].detach())
            buf["a"].append(a.detach()); buf["cf"].append(cf.detach())
            buf["clp"].append(clp.detach()); buf["nlp"].append(nlp.detach())
            buf["h"].append(state["hidden"].detach())
        pe = mse(pred, nlp)
        rew = brain.sensorimotor_reward(pe.item())
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

    print(f"DONE seed={seed} log={csv_path}", flush=True)


if __name__ == "__main__":
    seed = int(sys.argv[1])
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 3600
    K = int(os.environ.get("C_K", "100"))
    run(seed, n_train=n_train, K=K)
