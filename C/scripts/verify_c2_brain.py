"""
本実装の検証：TaroBrainWithMotor の公式メソッド（infer_latent / predict_proprio /
hippocampus / consolidate）だけで、C1（改良自己モデル）と C2（睡眠リプレイ）が
再現するかを確かめる。実験script内のinline版ではなく、脳本体の実装を叩く。

期待：リプレイOFF ≈ C1（持続予測比 persist ~79% / 変化相関 corr ~0.45）、
      リプレイON  ≈ C2（persist ~69% / corr ~0.56）。
使い方: python verify_c2_brain.py <seed>   （環境変数 C_REPLAY=1 でリプレイON）
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # = Taro/C
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
_REPLAY = os.environ.get("C_REPLAY", "0") == "1"


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
    env = HybridEnv(gym.make("MIMoBenchV2-v0", vision_params=None, touch_params=None))
    fusion = MinimalFusion(); target_fusion = MinimalFusion().freeze()
    n_act = env.action_space.shape[0]
    obs, _ = env.reset()
    sdim = fusion.encode(obs).shape[0]; prop_dim = to_tensor(obs["observation"]).shape[0]
    # 脳本体（改良自己モデル＋海馬を内蔵）
    brain = TaroBrainWithMotor(vocab_size=3, sensory_dim=sdim, n_actuators=n_act, proprio_dim=prop_dim)
    learner = TaroLearner(CombinedParams(brain, fusion), lr=0.005)  # emb_proj/nat_headは脳に入ったので不要
    dop = Dopamine(); ne = LocusCoeruleus(); homeo = HomeostaticScaling(dim=sdim)
    hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)

    def step_k(a):
        o, term = obs, False
        for _ in range(K):
            o, r, te, tr, info = env.step(a)
            if te or tr:
                term = True; break
        return o, term

    def evaluate():
        nonlocal obs, hidden, prev_a
        self_err, ep, pdel, adel = [], [], [], []
        for _ in range(n_eval):
            sv = fusion.encode(obs); cf = target_fusion.encode(obs).detach(); clp = ln_prop(obs)
            z, _, _, hn = brain.infer_latent(sv, prev_a, cf, hidden); z = z.detach()
            a = torch.clamp(torch.tanh(brain.motor_head(z)), -1.0, 1.0).detach()
            pd = brain.predict_proprio(z, a, clp).detach()
            obs, term = step_k(rescale_action(a, env.action_space)); nlp = ln_prop(obs)
            self_err.append(mse(pd, nlp).item()); ep.append(mse(clp, nlp).item())
            pdel.append((pd - clp).numpy()); adel.append((nlp - clp).numpy())
            hidden = hn.detach(); prev_a = a
            if term:
                obs2, _ = env.reset(); obs = obs2; hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)
        persist = np.mean(self_err) / np.mean(ep) * 100
        P = np.concatenate([p.flatten() for p in pdel]); A = np.concatenate([a.flatten() for a in adel])
        corr = float(np.corrcoef(P, A)[0, 1])
        return persist, corr

    for i in range(n_train):
        sv = fusion.encode(obs); cf = target_fusion.encode(obs).detach(); clp = ln_prop(obs)
        z, kl, rc, hn = brain.infer_latent(sv, prev_a, cf, hidden.detach())
        mean = torch.tanh(brain.motor_head(z.detach()))
        dist = torch.distributions.Normal(mean, 0.05 + ne.get_ne_level() * 0.45)
        a = torch.clamp(dist.sample(), -1.0, 1.0); lp = dist.log_prob(a).sum()
        pred = brain.predict_proprio(z, a.detach(), clp)
        obs, term = step_k(rescale_action(a, env.action_space)); nlp = ln_prop(obs)
        if _REPLAY:
            brain.hippocampus.record(sv.detach(), prev_a.detach(), a.detach(),
                                     cf.detach(), clp.detach(), nlp.detach(), hidden.detach())
        pe = mse(pred, nlp); rew = brain.sensorimotor_reward(pe.item())
        pl = learner.learn_action([lp], dop.compute_rpe(rew))
        hl = homeo.homeostatic_loss(sv); homeo.observe(sv)
        learner.update(pe + hl + kl + rc, pl)
        ne.observe_reward(rew); ne.release_ne()
        hidden = hn.detach(); prev_a = a.detach()
        if term:
            obs, _ = env.reset(); hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)
        if _REPLAY and (i + 1) % ckpt == 0:
            brain.consolidate(learner)  # 睡眠：貯めた経験を再生して定着

    persist, corr = evaluate()
    env.close()
    tag = "リプレイON(C2期待)" if _REPLAY else "リプレイOFF(C1期待)"
    print(f"RESULT seed={seed} {tag}: persist={persist:.1f}% corr={corr:.3f}", flush=True)


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
