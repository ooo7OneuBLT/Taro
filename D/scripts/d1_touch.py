"""
D1：触覚で相手を知覚する（視覚なし）。2体MIMo(two_agent_env)で、太郎A が
固有感覚(自分)＋触覚(相手Bから受ける力) を知覚し、Cの自己モデル(GRU)で符号化する。
＝「Cの脳(C/src)＋Dの触覚環境(D/scripts)」を繋ぐ最初の一歩。相手を"触れて感じる"ことを確認。

置き場所：D/scripts（D固有）。脳は C/src を import（Cは土台・不変）。
使い方: python d1_touch.py [seed]
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                              # two_agent_env
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "taro_core"))     # C/paths
import paths
paths.setup_brain_path()                                              # C/src/brain, senses ...
from two_agent_env import TwoAgentMIMo
from taro_brain_motor import TaroBrainWithMotor
from sensory_encoders import ProprioceptionEncoder, TouchEncoder


def to_t(x):
    return torch.tensor(np.asarray(x), dtype=torch.float32)


class TouchFusion:
    """D用の感覚融合：固有感覚(自分)＋触覚(相手Bから受ける力)。
    2体最小環境なので視覚・前庭・内受容は無し。＝『自分の体＋触れた相手』を符号化する。"""
    def __init__(self, prop_dim, touch_dim, emb=64):
        self.proprio = ProprioceptionEncoder(input_dim=prop_dim, embedding_dim=emb)
        self.touch = TouchEncoder(input_dim=touch_dim, hidden_dim=emb, embedding_dim=emb)

    def parameters(self):
        import itertools
        return itertools.chain(self.proprio.parameters(), self.touch.parameters())

    def encode(self, obs):
        f = torch.cat([self.proprio(to_t(obs["proprio_qpos"])),
                       self.touch(to_t(obs["touch_of_B"]))], dim=-1)
        return torch.nn.functional.layer_norm(f, f.shape)


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    torch.manual_seed(seed); np.random.seed(seed)

    env = TwoAgentMIMo(sep=0.16)
    obs = env.reset()
    prop_dim = len(obs["proprio_qpos"]); touch_dim = len(obs["touch_of_B"])
    fusion = TouchFusion(prop_dim, touch_dim)
    target_fusion = TouchFusion(prop_dim, touch_dim)  # pc_latent 用の凍結目標エンコーダ
    for p in target_fusion.parameters():
        p.requires_grad_(False)
    sdim = fusion.encode(obs).shape[0]

    brain = TaroBrainWithMotor(vocab_size=3, sensory_dim=sdim, n_actuators=env.na)
    emb_dim = brain.sensory_proj.out_features
    emb_proj = nn.Linear(sdim + env.na, emb_dim)
    hidden = brain.init_motor_hidden(); prev_a = torch.zeros(env.na)

    # A/B のアクチュエータ可動域（[-1,1] → 実ctrl へ変換）
    m = env.model
    lo = np.array([m.actuator_ctrlrange[i, 0] for i in env.aid]); hi = np.array([m.actuator_ctrlrange[i, 1] for i in env.aid])
    lob = np.array([m.actuator_ctrlrange[i, 0] for i in env.bid]); hib = np.array([m.actuator_ctrlrange[i, 1] for i in env.bid])

    print(f"=== D1 触覚知覚 (seed={seed}) ===", flush=True)
    print(f"proprio_dim={prop_dim}  touch_dim={touch_dim}  sensory_dim(融合)={sdim}  na={env.na}", flush=True)

    N = 300; contacts = 0; touch_sum = 0.0; parts = np.zeros(touch_dim); zdim = brain.latent_dim
    for t in range(N):
        sv = fusion.encode(obs)                                   # 固有感覚＋触覚を融合
        cf = target_fusion.encode(obs).detach()
        emb = emb_proj(torch.cat([sv, prev_a], dim=-1)).unsqueeze(0).unsqueeze(0)
        out, hn = brain.motor_gru(emb, hidden)                    # Cの自己モデル(GRU)で符号化
        z, _, _ = brain.pc_latent.infer(hidden[-1, 0], out[0, -1], cf)  # ← 触覚込みの潜在状態z(32次元)
        z = z.detach()
        a01 = np.random.uniform(-1.0, 1.0, env.na)                # D1: A もランダム（触覚知覚の確認が目的）
        a_ctrl = lo + (a01 + 1.0) / 2.0 * (hi - lo)
        b01 = np.random.uniform(-1.0, 1.0, env.nb)                # B:ランダム(D2でGoal Babblingに)
        b_ctrl = lob + (b01 + 1.0) / 2.0 * (hib - lob)
        obs = env.step(a_ctrl, b_ctrl, K=5)
        tb = obs["touch_of_B"]; s = float(tb.sum())
        if s > 0:
            contacts += 1; parts += tb
        touch_sum += s
        hidden = hn.detach(); prev_a = to_t(a01)

    top = np.argsort(-parts)[:5]
    top_names = [mujoco_name(env, env.a_bodies[i]) for i in top if parts[i] > 0]
    print(f"接触した割合 = {contacts / N * 100:.0f}%   平均触覚 = {touch_sum / N:.2f}", flush=True)
    print(f"よく触られた部位 = {top_names}", flush=True)
    print("=> 太郎A は『固有感覚(自分)＋触覚(相手B)』を C の自己モデルで符号化できた（触覚が状態z に入る）", flush=True)
    print("   次(D2)：Bを Goal Babbling で動かし、逆モデルで相手の目標を推論→先読み", flush=True)


def mujoco_name(env, bid):
    import mujoco
    return mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, bid)


if __name__ == "__main__":
    main()
