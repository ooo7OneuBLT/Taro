"""
B2：先読み（anticipation）。相手（第2の太郎）の"次の行動"を太郎①が予測できるか。
相手の行動列を太郎①に適用→状態zを記録→「z→次の行動」の読み出しヘッドを学習→held-outで2AFC。
＝「太郎の状態に"次を読む情報"が在るか」の情報存在テスト。

対照：
  shuffle  … 次行動ラベルを混ぜて学習（情報が本物なら≈50に崩れる）
  persist  … next≈current（相手の滑らかさだけで当たる分の参照）
  floor    … 未学習の太郎①の状態から（≈50のはず）
判定：anticipation が 50% と floor と shuffle を明確に超え、かつ persist(滑らかさ)を上回れば
      「自己モデルの状態から、滑らかさ以上の先読みが出た（最小の理解）」。
使い方: python d_b2.py <seed> [n_train]
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
                hidden=brain.init_motor_hidden(), prev_a=torch.zeros(n_act), latent_dim=brain.latent_dim)


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


def observe_states(m, other_acts):
    """相手の行動列を適用し、各行動を体験した"後"の状態z_postを記録。"""
    zs_post = []
    for a_ext in other_acts:
        m["obs"], term = step_k(m, rescale_action(a_ext, m["env"].action_space))
        sv = m["fusion"].encode(m["obs"]); cf = m["tfusion"].encode(m["obs"]).detach()
        z, _, _, hn = zc(m, sv, m["prev_a"], cf, m["hidden"])
        zs_post.append(z.detach())
        m["hidden"] = hn.detach(); m["prev_a"] = a_ext.detach()
        if term:
            m["obs"], _ = m["env"].reset(); m["hidden"] = m["brain"].init_motor_hidden(); m["prev_a"] = torch.zeros(m["n_act"])
    return zs_post


def twoafc_pred(preds, targets):
    """各iで pred_i が target_i に、他の target_j より近いか（MSE距離）。"""
    N = len(preds); c = t = 0
    for i in range(N):
        di = ((preds[i] - targets[i]) ** 2).mean().item()
        for j in range(N):
            if i == j:
                continue
            dj = ((preds[i] - targets[j]) ** 2).mean().item()
            if di < dj:
                c += 1
            t += 1
    return c / t * 100


def anticipation(latent_dim, n_act, zs_post, acts, shuffle=False):
    """z_post[t] → a[t+1] を学習し、held-outで2AFC。shuffle=Trueで対照。"""
    Z = torch.stack(zs_post[:-1])          # 状態(after a_t)
    A = torch.stack(acts[1:])              # 次の行動 a_{t+1}
    M = Z.shape[0]; S = M // 2
    Ztr, Atr, Zte, Ate = Z[:S], A[:S], Z[S:], A[S:]
    if shuffle:
        Atr = Atr[torch.randperm(Atr.shape[0])]
    head = nn.Linear(latent_dim, n_act)
    opt = torch.optim.Adam(head.parameters(), lr=0.01)
    for _ in range(300):
        opt.zero_grad(); loss = mse(torch.tanh(head(Ztr)), Atr); loss.backward(); opt.step()
    with torch.no_grad():
        preds = torch.tanh(head(Zte))
    return twoafc_pred([preds[i] for i in range(preds.shape[0])], [Ate[i] for i in range(Ate.shape[0])])


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 2400
    N = 100
    print(f"=== B2 先読み (seed={seed}) ===", flush=True)

    m2 = build(seed + 100); train(m2, 1200)
    other_acts = rollout_actions(m2, N); m2["env"].close()

    # 持続参照：next≈current（相手の滑らかさだけで当たる分）
    persist = twoafc_pred([other_acts[t] for t in range(N // 2, N - 1)],
                          [other_acts[t + 1] for t in range(N // 2, N - 1)])

    # floor：未学習の太郎①の状態から
    mu = build(seed)
    zsu = observe_states(mu, other_acts); mu["env"].close()
    fl = anticipation(mu["latent_dim"], mu["n_act"], zsu, other_acts)

    # 学習後
    mt = build(seed); train(mt, n_train)
    zst = observe_states(mt, other_acts)
    ant = anticipation(mt["latent_dim"], mt["n_act"], zst, other_acts)
    shuf = anticipation(mt["latent_dim"], mt["n_act"], zst, other_acts, shuffle=True)
    mt["env"].close()

    print(f"persist(滑らかさ参照 next=current) = {persist:.1f}%", flush=True)
    print(f"floor(未学習の状態から)           = {fl:.1f}%", flush=True)
    print(f"shuffle(ラベル混ぜ)               = {shuf:.1f}%", flush=True)
    print(f"RESULT seed={seed} anticipation(先読み) = {ant:.1f}%  "
          f"(persist {persist:.1f} / floor {fl:.1f} / shuffle {shuf:.1f})", flush=True)
    print(f"判定: ant が 50%・floor・shuffle を明確に超え、persist(滑らかさ)も上回れば「最小の先読み成立」", flush=True)


if __name__ == "__main__":
    main()
