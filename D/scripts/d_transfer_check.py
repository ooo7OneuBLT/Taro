"""
移植検証：Cで確立した最高モデル(+51)を、そのままDの2体環境に載せて性能が保たれるか。

同じ重みで
  ① 単体C環境  HybridEnv(MIMoBenchV2-v0)        ← 学習した時と同じ環境（基準）
  ② D2体環境   HybridEnv(TwoMimoEnv, ベータ遠方) ← 作り直した環境（干渉ゼロで純粋な移植テスト）
を評価し、margin/corr/classify が一致すれば「観測フォーマットが同形＝土台の移植成功」。

学習は一切しない（重みは凍結）。使い方: python d_transfer_check.py [model_path]
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "taro_core"))
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "C", "scripts"))
import paths
paths.setup_brain_path()

import gymnasium as gym
import mimoEnv  # noqa
from hybrid_env import HybridEnv
from taro_brain_motor import TaroBrainWithMotor
from cerebellum_motor import MotorCerebellum
from test_phase8_motor_learning import rescale_action, to_tensor
from run_c_metrics_ac_lr import MinimalFusion, ln_prop   # Cの感覚融合をそのまま流用
from d_env import TwoMimoEnv

mse = torch.nn.functional.mse_loss
K = 100


def build(sdim, n_act, prop_dim, latent_dim_src=None):
    brain = TaroBrainWithMotor(vocab_size=3, sensory_dim=sdim, n_actuators=n_act)
    emb_proj = nn.Linear(sdim + n_act, brain.sensory_proj.out_features)
    nat_head = nn.Sequential(nn.Linear(brain.latent_dim + n_act, 128), nn.SiLU(),
                             nn.LayerNorm(128), nn.Linear(128, prop_dim))
    cereb = MotorCerebellum(brain.latent_dim, n_act)
    return brain, emb_proj, nat_head, cereb


def load_into(ck, brain, fusion, emb_proj, nat_head, cereb):
    brain.load_state_dict(ck["brain"])
    fusion.insula.load_state_dict(ck["fusion_insula"])
    fusion.proprio.load_state_dict(ck["fusion_proprio"])
    fusion.vestibular.load_state_dict(ck["fusion_vestibular"])
    emb_proj.load_state_dict(ck["emb_proj"])
    nat_head.load_state_dict(ck["nat_head"])
    cereb.load_state_dict(ck["cereb"])


def evaluate(env, fusion, tfusion, brain, emb_proj, nat_head, cereb, n_eval=80):
    """Cのevaluate()と同型：classify/margin/corr/persist。学習なし・重み凍結。"""
    obs, _ = env.reset()
    h = brain.init_motor_hidden(); pa = torch.zeros(env.action_space.shape[0])

    def zc(sv, prev_a, cf, hh):
        emb = emb_proj(torch.cat([sv, prev_a], dim=-1)).unsqueeze(0).unsqueeze(0)
        out, nh = brain.motor_gru(emb, hh)
        z, kl, rc = brain.pc_latent.infer(hh[-1, 0], out[0, -1], cf)
        return z, kl, rc, nh

    def act_mean(z):
        pm = torch.tanh(brain.motor_head(z))
        w, cere_a, _ = cereb.gate(z, pm)      # 保存モデルは小脳ONで学習済み
        return (1.0 - w) * pm + w * cere_a

    def step_k(a):
        o, term = obs, False
        for _ in range(K):
            o, r, te, tr, info = env.step(a)
            if te or tr:
                term = True; break
        return o, term

    Zs, acts, nx, cu, self_err, ep, pdel, adel = [], [], [], [], [], [], [], []
    for _ in range(n_eval):
        sv = fusion.encode(obs); cf = tfusion.encode(obs).detach(); clp = ln_prop(obs)
        z, _, _, hn = zc(sv, pa, cf, h); z = z.detach()
        a = torch.clamp(act_mean(z), -1.0, 1.0).detach()
        pd = nat_head(torch.cat([z, a], dim=-1)).detach()
        obs, term = step_k(rescale_action(a, env.action_space))
        nlp = ln_prop(obs)
        self_err.append(mse(clp + pd, nlp).item()); ep.append(mse(clp, nlp).item())
        pdel.append(pd.numpy()); adel.append((nlp - clp).numpy())
        Zs.append(z); acts.append(a); nx.append(nlp); cu.append(clp)
        h = hn.detach(); pa = a
        if term:
            obs, _ = env.reset(); h = brain.init_motor_hidden(); pa = torch.zeros_like(pa)
    N = len(Zs); correct = total = 0; oe = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            e = mse((cu[i] + nat_head(torch.cat([Zs[i], acts[j]], dim=-1))).detach(), nx[i]).item()
            oe.append(e); correct += int(self_err[i] < e); total += 1
    classify = correct / total * 100
    margin = (np.mean(oe) - np.mean(self_err)) / np.mean(oe) * 100
    persist = np.mean(self_err) / np.mean(ep) * 100
    P = np.concatenate([p.flatten() for p in pdel]); A = np.concatenate([a.flatten() for a in adel])
    corr = float(np.corrcoef(P, A)[0, 1])
    return classify, margin, corr, persist


def run_on(env, ck, tag):
    torch.manual_seed(0); np.random.seed(0)
    fusion = MinimalFusion(); tfusion = MinimalFusion().freeze()
    obs, _ = env.reset()
    sdim = fusion.encode(obs).shape[0]; prop_dim = to_tensor(obs["observation"]).shape[0]
    n_act = env.action_space.shape[0]
    brain, emb_proj, nat_head, cereb = build(sdim, n_act, prop_dim)
    load_into(ck, brain, fusion, emb_proj, nat_head, cereb)
    for m in (brain, emb_proj, nat_head, cereb):
        for p in m.parameters():
            p.requires_grad_(False)
    for p in fusion.parameters():
        p.requires_grad_(False)
    cl, mg, co, pe = evaluate(env, fusion, tfusion, brain, emb_proj, nat_head, cereb)
    print(f"[{tag}] sdim={sdim} prop={prop_dim} n_act={n_act} | "
          f"classify={cl:.1f}% margin={mg:+.1f}% corr={co:.3f} persist={pe:.1f}%", flush=True)
    return mg, co


def main():
    mp = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, os.pardir, "models", "self_A_best_seed0.pt")
    ck = torch.load(mp, weights_only=False)
    print(f"=== 最高モデルの移植検証 ===\nmodel={os.path.basename(mp)} config={ck['config']}\n", flush=True)

    print("① 単体C環境（学習時と同じ＝基準）", flush=True)
    e1 = HybridEnv(gym.make("MIMoBenchV2-v0", vision_params=None, touch_params=None))
    m1, c1 = run_on(e1, ck, "単体C")

    print("\n② D 2体環境（ベータ遠方=干渉ゼロ）", flush=True)
    e2 = HybridEnv(TwoMimoEnv(sep=3.0, vision_params=None, touch_params=None))
    m2, c2 = run_on(e2, ck, "D2体")

    print(f"\n=== 判定 ===\nmargin 単体={m1:+.1f}% → 2体={m2:+.1f}%（差 {m2-m1:+.1f}）", flush=True)
    print(f"corr   単体={c1:.3f} → 2体={c2:.3f}", flush=True)
    print("=> 差が小さければ『観測が同形＝土台の移植成功』。Dはこの環境でCの資産をそのまま使える。", flush=True)


if __name__ == "__main__":
    main()
