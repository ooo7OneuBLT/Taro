"""
B1：相手を"本物のエージェント"に格上げして認識が効くか。
相手＝第2の太郎（別seed・簡易学習）の行動列を、太郎①の体に外力として適用。
太郎①が「どの行動が動きを起こしたか」を2AFCで認識できるか（＝Dの構造版）。
比較：ランダム力のD（~84%）。相手が"構造ある実エージェント"でも効くか。
対照：未学習floor（≈50）。
使い方: python d_b1.py <seed> [n_train]
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "C"))
import paths
paths.setup_brain_path()
import gymnasium as gym
import mimoEnv  # noqa
from hybrid_env import HybridEnv
from taro_brain_motor import TaroBrainWithMotor
from basal_ganglia import TaroLearner
from dopamine import Dopamine
from locus_coeruleus import LocusCoeruleus
from homeostatic_scaling import HomeostaticScaling
from test_phase8_motor_learning import CombinedParams, rescale_action, to_tensor
from sensory_encoders import ProprioceptionEncoder, VestibularEncoder
from insula import Insula

mse = torch.nn.functional.mse_loss


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


def ln_prop(obs):
    v = to_tensor(obs["observation"])
    return torch.nn.functional.layer_norm(v, v.shape).detach()


def build(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    env = HybridEnv(gym.make("MIMoBenchV2-v0", vision_params=None, touch_params=None))
    fusion = MinimalFusion(); tfusion = MinimalFusion()
    for p in tfusion.parameters():
        p.requires_grad_(False)
    n_act = env.action_space.shape[0]
    obs, _ = env.reset()
    sdim = fusion.encode(obs).shape[0]; prop_dim = to_tensor(obs["observation"]).shape[0]
    brain = TaroBrainWithMotor(vocab_size=3, sensory_dim=sdim, n_actuators=n_act)
    emb_dim = brain.sensory_proj.out_features
    emb_proj = nn.Linear(sdim + n_act, emb_dim)
    nat_head = nn.Sequential(nn.Linear(brain.latent_dim + n_act, 128), nn.SiLU(),
                             nn.LayerNorm(128), nn.Linear(128, prop_dim))
    learner = TaroLearner(CombinedParams(brain, fusion, emb_proj, nat_head), lr=0.005)
    return dict(env=env, fusion=fusion, tfusion=tfusion, brain=brain, emb_proj=emb_proj,
                nat_head=nat_head, learner=learner, dop=Dopamine(), ne=LocusCoeruleus(),
                homeo=HomeostaticScaling(dim=sdim), n_act=n_act, obs=obs,
                hidden=brain.init_motor_hidden(), prev_a=torch.zeros(n_act))


def zc(m, sv, prev_a, cf, h):
    emb = m["emb_proj"](torch.cat([sv, prev_a], dim=-1)).unsqueeze(0).unsqueeze(0)
    out, nh = m["brain"].motor_gru(emb, h)
    z, kl, rc = m["brain"].pc_latent.infer(h[-1, 0], out[0, -1], cf)
    return z, kl, rc, nh


def step_k(m, a, K=100):
    o, term = m["obs"], False
    for _ in range(K):
        o, r, te, tr, info = m["env"].step(a)
        if te or tr:
            term = True; break
    return o, term


def train(m, n_train, K=100):
    for i in range(n_train):
        sv = m["fusion"].encode(m["obs"]); cf = m["tfusion"].encode(m["obs"]).detach(); clp = ln_prop(m["obs"])
        z, kl, rc, hn = zc(m, sv, m["prev_a"], cf, m["hidden"].detach())
        mean = torch.tanh(m["brain"].motor_head(z.detach()))
        dist = torch.distributions.Normal(mean, 0.05 + m["ne"].get_ne_level() * 0.45)
        a = torch.clamp(dist.sample(), -1.0, 1.0); lp = dist.log_prob(a).sum()
        pred = clp + m["nat_head"](torch.cat([z, a.detach()], dim=-1))
        m["obs"], term = step_k(m, rescale_action(a, m["env"].action_space)); nlp = ln_prop(m["obs"])
        pe = mse(pred, nlp); rew = m["brain"].sensorimotor_reward(pe.item())
        pl = m["learner"].learn_action([lp], m["dop"].compute_rpe(rew))
        hl = m["homeo"].homeostatic_loss(sv); m["homeo"].observe(sv)
        m["learner"].update(pe + hl + kl + rc, pl)
        m["ne"].observe_reward(rew); m["ne"].release_ne()
        m["hidden"] = hn.detach(); m["prev_a"] = a.detach()
        if term:
            m["obs"], _ = m["env"].reset(); m["hidden"] = m["brain"].init_motor_hidden(); m["prev_a"] = torch.zeros(m["n_act"])


def rollout_actions(m, n, K=100):
    """第2の太郎(m)を自分の体で走らせ、方策が出す行動列（＝構造ある実エージェントの振る舞い）を記録。"""
    acts = []
    for _ in range(n):
        sv = m["fusion"].encode(m["obs"]); cf = m["tfusion"].encode(m["obs"]).detach()
        z, _, _, hn = zc(m, sv, m["prev_a"], cf, m["hidden"])
        a = torch.clamp(torch.tanh(m["brain"].motor_head(z.detach())), -1.0, 1.0).detach()
        acts.append(a)
        m["obs"], term = step_k(m, rescale_action(a, m["env"].action_space))
        m["hidden"] = hn.detach(); m["prev_a"] = a.detach()
        if term:
            m["obs"], _ = m["env"].reset(); m["hidden"] = m["brain"].init_motor_hidden(); m["prev_a"] = torch.zeros(m["n_act"])
    return acts


def apply_and_recognize(m, other_acts):
    """相手の行動列を太郎①の体に適用し、認識2AFC。"""
    Zs, clps, nlps = [], [], []
    for a_ext in other_acts:
        sv = m["fusion"].encode(m["obs"]); cf = m["tfusion"].encode(m["obs"]).detach(); clp = ln_prop(m["obs"])
        z, _, _, hn = zc(m, sv, m["prev_a"], cf, m["hidden"]); z = z.detach()
        m["obs"], term = step_k(m, rescale_action(a_ext, m["env"].action_space)); nlp = ln_prop(m["obs"])
        Zs.append(z); clps.append(clp); nlps.append(nlp)
        m["hidden"] = hn.detach(); m["prev_a"] = a_ext.detach()
        if term:
            m["obs"], _ = m["env"].reset(); m["hidden"] = m["brain"].init_motor_hidden(); m["prev_a"] = torch.zeros(m["n_act"])
    N = len(Zs); correct = total = 0; se_all, oe_all = [], []
    for i in range(N):
        with torch.no_grad():
            se = mse(clps[i] + m["nat_head"](torch.cat([Zs[i], other_acts[i]], dim=-1)), nlps[i]).item()
        se_all.append(se)
        for j in range(N):
            if i == j:
                continue
            with torch.no_grad():
                oe = mse(clps[i] + m["nat_head"](torch.cat([Zs[i], other_acts[j]], dim=-1)), nlps[i]).item()
            oe_all.append(oe)
            if se < oe:
                correct += 1
            total += 1
    return correct / total * 100, (np.mean(oe_all) - np.mean(se_all)) / np.mean(oe_all) * 100


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 2400
    n = 40
    print(f"=== B1 相手=第2の太郎 (seed={seed}) ===", flush=True)

    # 相手（第2の太郎）：別seedで簡易学習し、行動列を記録
    m2 = build(seed + 100); train(m2, 1200)
    other_acts = rollout_actions(m2, n); m2["env"].close()

    # 観測者（太郎①）
    mu = build(seed)  # 未学習floor
    d0, g0 = apply_and_recognize(mu, other_acts); mu["env"].close()
    mt = build(seed); train(mt, n_train)
    d1, g1 = apply_and_recognize(mt, other_acts); mt["env"].close()
    print(f"floor 未学習   recog={d0:.1f}%  margin={g0:+.1f}%", flush=True)
    print(f"RESULT seed={seed} 学習後  recog={d1:.1f}%  margin={g1:+.1f}%  (floor {d0:.1f}%)", flush=True)
    print(f"判定: 学習後 recog が偶然50%を明確に超え floorを上回れば「実エージェントの行動認識に転用成立」", flush=True)


if __name__ == "__main__":
    main()
