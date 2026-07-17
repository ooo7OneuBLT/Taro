"""【放棄・2026-07-15】この系統は仮説が否定されたため使っていない。記録として残す。

当時の仮説：Dでmarginが伸びない真因は「体が行動でどれだけ動かせるか（可制御性）」の差である。
そこで「行動による固有感覚の差 ÷ 行動ゼロの受動ドリフト」を代理指標にして測った。

否定した根拠（実測）：Cの比 1.17 < Dの比 1.30 なのに、marginはCが圧勝(+51 vs +11)。
＝この代理指標は目的の量をまるで測れておらず、**判断を積極的に誤らせた**。

本当の原因（同日判明）：
  ①margin +11 の正体は **C_REPLAY(睡眠リプレイ)が既定OFFだった**こと（Cも同条件なら+11）。
  ②D0が学べなかったのは **借りた環境が3〜500stepで太郎の人生を打ち切っていた**こと。
  ③自己モデルの成否を分けるのは **体が行動で実際にどれだけ動けるか**（立位Cは転倒して手足が
    自由になり+51、座位D0は腰を世界に溶接され腕も畳まれて学べず）。仮説の"方向"は近かったが、
    上の代理指標では捉えられていなかった。

教訓：代理指標は「既知の正解ケース」で先に較正する（検証の落とし穴チェックリスト 項6）。
現在の後継：d_supine_check.py / d_supine_touch_truth.py（接触ペアと基準線で直接測る）
"""
"""
D準備(単体/C方式)：太郎A の自己モデルを「他者なし」で確立し、保存する。
＝人間の順序（自己モデルをまず自分の運動で確立→のちに他者へ転用）に忠実。

d_train_self.py と同一アーキ・同一obs次元(proprio93+touch56=128)のまま、
Bを遠方(sep=3.0)に固定(ctrl=0)して**接触ゼロ＝実質単体**にする。これで
Bの体当たり汚染が消え、margin が Cと同水準(+40〜50%)まで立つはず。
保存物はD2にそのまま load 可能（次元一致を保証するため2体envを流用）。

使い方: python d_train_self_solo.py [seed] [n_train=5000]
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "taro_core"))
import paths
paths.setup_brain_path()
from two_agent_env import TwoAgentMIMo
from taro_brain_motor import TaroBrainWithMotor
from basal_ganglia import TaroLearner
from dopamine import Dopamine
from locus_coeruleus import LocusCoeruleus
from homeostatic_scaling import HomeostaticScaling
from test_phase8_motor_learning import CombinedParams
from sensory_encoders import ProprioceptionEncoder, TouchEncoder

mse = torch.nn.functional.mse_loss


def to_t(x):
    return torch.tensor(np.asarray(x), dtype=torch.float32)


class TouchFusion:
    """固有感覚(自分)＋触覚(相手B)。単体学習では触覚は常に0（他者なし）。"""
    def __init__(self, prop_dim, touch_dim, emb=64):
        self.proprio = ProprioceptionEncoder(input_dim=prop_dim, embedding_dim=emb)
        self.touch = TouchEncoder(input_dim=touch_dim, hidden_dim=emb, embedding_dim=emb)

    def parameters(self):
        import itertools
        return itertools.chain(self.proprio.parameters(), self.touch.parameters())

    def encode(self, obs):
        f = torch.cat([self.proprio(to_t(obs["proprio_qpos"])), self.touch(to_t(obs["touch_of_B"]))], dim=-1)
        return torch.nn.functional.layer_norm(f, f.shape)


def ln_prop(obs):
    v = to_t(obs["proprio_qpos"])
    return torch.nn.functional.layer_norm(v, v.shape).detach()


def rescale(a01, lo, hi):
    return lo + (a01 + 1.0) / 2.0 * (hi - lo)


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    K = 100
    torch.manual_seed(seed); np.random.seed(seed)

    env = TwoAgentMIMo(sep=3.0)          # ★Bを遠方に(接触ゼロ＝実質単体)
    obs = env.reset()
    prop_dim = len(obs["proprio_qpos"]); touch_dim = len(obs["touch_of_B"])
    fusion = TouchFusion(prop_dim, touch_dim); tfusion = TouchFusion(prop_dim, touch_dim)
    for p in tfusion.parameters():
        p.requires_grad_(False)
    sdim = fusion.encode(obs).shape[0]
    brain = TaroBrainWithMotor(vocab_size=3, sensory_dim=sdim, n_actuators=env.na)
    emb_dim = brain.sensory_proj.out_features
    emb_proj = nn.Linear(sdim + env.na, emb_dim)
    nat_head = nn.Sequential(nn.Linear(brain.latent_dim + env.na, 128), nn.SiLU(),
                             nn.LayerNorm(128), nn.Linear(128, prop_dim))
    learner = TaroLearner(CombinedParams(brain, fusion, emb_proj, nat_head), lr=0.005)
    dop = Dopamine(); ne = LocusCoeruleus(); homeo = HomeostaticScaling(dim=sdim)
    hidden = brain.init_motor_hidden(); prev_a = torch.zeros(env.na)
    m = env.model
    lo = np.array([m.actuator_ctrlrange[i, 0] for i in env.aid]); hi = np.array([m.actuator_ctrlrange[i, 1] for i in env.aid])
    b_frozen = np.zeros(env.nb)           # ★B固定(ctrl=0で静止)

    def zc(sv, pa, cf, h):
        emb = emb_proj(torch.cat([sv, pa], dim=-1)).unsqueeze(0).unsqueeze(0)
        out, nh = brain.motor_gru(emb, h)
        z, kl, rc = brain.pc_latent.infer(h[-1, 0], out[0, -1], cf)
        return z, kl, rc, nh

    def step_env(a01):
        a_ctrl = rescale(a01, lo, hi)
        return env.step(a_ctrl, b_frozen, K=K)   # Bは常に固定・遠方

    print(f"=== D自己モデル学習(単体/C方式) (seed={seed}, n_train={n_train}, K={K}, sep=3.0 B固定) ===", flush=True)
    print(f"proprio={prop_dim} touch={touch_dim} sensory_dim={sdim} na={env.na}", flush=True)

    ckpt = 600  # 覚醒600ステップごとに「睡眠」＝リプレイで定着（Cのreplay_on設定に一致）
    buf = {k: [] for k in ("sv", "prev_a", "a", "cf", "clp", "nlp", "h")}

    def consolidate(n_batches=200, bs=128):
        """睡眠中の記憶定着：貯めた経験をバッチ再生し自己モデルを固める（Cのconsolidateと同一）。
        ＝margin頭打ち(+11)を破って+48へ到達させる本命機構（taro-C2 睡眠リプレイ）。"""
        N = len(buf["sv"])
        if N < bs:
            return
        SV = torch.stack(buf["sv"]); PA = torch.stack(buf["prev_a"]); AA = torch.stack(buf["a"])
        CF = torch.stack(buf["cf"]); CLP = torch.stack(buf["clp"]); NLP = torch.stack(buf["nlp"])
        H = torch.cat(buf["h"], dim=1)  # (layers, N, hidden)
        for _ in range(n_batches):
            idx = torch.randint(0, N, (bs,))
            hb = H[:, idx].contiguous()
            emb = emb_proj(torch.cat([SV[idx], PA[idx]], dim=-1)).unsqueeze(1)
            out, _ = brain.motor_gru(emb, hb)
            z, kl, rc = brain.pc_latent.infer(hb[-1], out[:, 0], CF[idx])
            pred = CLP[idx] + nat_head(torch.cat([z, AA[idx]], dim=-1))
            loss = mse(pred, NLP[idx]) + kl + rc
            learner.optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(learner.brain.parameters(), learner.grad_clip)
            learner.optimizer.step()

    for i in range(n_train):
        sv = fusion.encode(obs); cf = tfusion.encode(obs).detach(); clp = ln_prop(obs)
        z, kl, rc, hn = zc(sv, prev_a, cf, hidden.detach())
        mean = torch.tanh(brain.motor_head(z.detach()))
        dist = torch.distributions.Normal(mean, 0.05 + ne.get_ne_level() * 0.45)
        a = torch.clamp(dist.sample(), -1.0, 1.0); lp = dist.log_prob(a).sum()
        pred = clp + nat_head(torch.cat([z, a.detach()], dim=-1))
        obs = step_env(a.detach().numpy()); nlp = ln_prop(obs)
        pe = mse(pred, nlp); rew = brain.sensorimotor_reward(pe.item())
        pl = learner.learn_action([lp], dop.compute_rpe(rew))
        hl = homeo.homeostatic_loss(sv); homeo.observe(sv)
        learner.update(pe + hl + kl + rc, pl)
        ne.observe_reward(rew); ne.release_ne()
        buf["sv"].append(sv.detach()); buf["prev_a"].append(prev_a.detach())
        buf["a"].append(a.detach()); buf["cf"].append(cf.detach())
        buf["clp"].append(clp.detach()); buf["nlp"].append(nlp.detach())
        buf["h"].append(hidden.detach())
        hidden = hn.detach(); prev_a = a.detach()
        if (i + 1) % ckpt == 0:
            consolidate()  # 睡眠：この間の経験を再生して定着
            mg, co = evaluate(env, fusion, tfusion, brain, emb_proj, nat_head, zc, step_env, {"h": hidden, "pa": prev_a, "obs": obs})
            print(f"  step {i+1}: margin={mg:+.1f}% corr={co:.3f}  noise={0.05+ne.get_ne_level()*0.45:.3f}  [sleep+replay]", flush=True)

    # 保存（D2が load する確立済み自己モデル）
    mdir = os.path.join(_HERE, os.pardir, "models"); os.makedirs(mdir, exist_ok=True)
    path = os.path.join(mdir, f"self_A_seed{seed}.pt")
    torch.save({"brain": brain.state_dict(), "fusion_proprio": fusion.proprio.state_dict(),
                "fusion_touch": fusion.touch.state_dict(), "emb_proj": emb_proj.state_dict(),
                "nat_head": nat_head.state_dict(), "config": {"prop_dim": prop_dim, "touch_dim": touch_dim,
                "sdim": sdim, "na": env.na, "K": K, "seed": seed, "n_train": n_train, "mode": "solo_sep3"}}, path)
    print(f"SAVED {path}", flush=True)


def evaluate(env, fusion, tfusion, brain, emb_proj, nat_head, zc, step_env, st, n_eval=60):
    """margin(入替)とcorr(変化相関)。Cのevaluateと同型。"""
    h, pa, obs = st["h"], st["pa"], st["obs"]
    Zs, acts, cu, nx, self_err = [], [], [], [], []
    pdel, adel = [], []
    for _ in range(n_eval):
        sv = fusion.encode(obs); cf = tfusion.encode(obs).detach(); clp = ln_prop(obs)
        z, _, _, hn = zc(sv, pa, cf, h); z = z.detach()
        a = torch.clamp(torch.tanh(brain.motor_head(z)), -1.0, 1.0).detach()
        pd = nat_head(torch.cat([z, a], dim=-1)).detach()
        obs = step_env(a.numpy()); nlp = ln_prop(obs)
        self_err.append(mse(clp + pd, nlp).item())
        pdel.append(pd.numpy()); adel.append((nlp - clp).numpy())
        Zs.append(z); acts.append(a); cu.append(clp); nx.append(nlp)
        h = hn.detach(); pa = a
    N = len(Zs); oe = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            oe.append(mse((cu[i] + nat_head(torch.cat([Zs[i], acts[j]], dim=-1))).detach(), nx[i]).item())
    margin = (np.mean(oe) - np.mean(self_err)) / np.mean(oe) * 100
    P = np.concatenate([p.flatten() for p in pdel]); A = np.concatenate([a.flatten() for a in adel])
    corr = float(np.corrcoef(P, A)[0, 1])
    st["h"], st["pa"], st["obs"] = h, pa, obs
    return margin, corr


if __name__ == "__main__":
    main()
