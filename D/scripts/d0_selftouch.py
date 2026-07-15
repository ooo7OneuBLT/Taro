"""
D0：自己接触で「触覚を含む自己モデル」を確立する（1体）。

なぜD0が要るか：Cの自己モデル(+50)は**触覚を一度も感じたことがない**（内受容+固有感覚+前庭のみ）。
触覚を知らないモデルに他者の触覚を渡しても解釈できない＝共鳴マップが成立しない。
人間も**触覚が最初の自己探索の感覚**で、視覚成熟より前に身体意識の基礎を作る（BabyBench/MIMoの
標準課題も self-touch）。＝自己接触で触覚を接地してから他者(D1〜)へ。

環境：MIMo公式 `MIMoSelfBody-v0`（座位・weldで固定＝倒れない・触覚ON・**右腕のみ自由**）
      ＋ HybridEnv（太郎の内部状態=interoceptionを追加）
学習：Cの学習ループをそのまま流用（運動性喃語＋予測誤差＋方策勾配＋**睡眠リプレイ**）
予測：**固有感覚＋触覚**を予測（＝触覚的身体スキーマ。後の「予期せぬ触覚＝他者」の土台）

使い方: python d0_selftouch.py [seed] [n_train]   出力: D/models/self_touch_seed{seed}.pt
"""
import os, sys, time, csv, datetime, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "C"))
import paths
paths.setup_brain_path()
sys.path.insert(0, paths.MIMO_DIR)

import gymnasium as gym
import mimoEnv  # noqa
from mimoActuation.actuation import SpringDamperModel
from mimoActuation.muscle import MuscleModel
from hybrid_env import HybridEnv

# 駆動モデル。既定＝筋肉（人間模倣に忠実）。D0_MUSCLE=0 でバネダンパー（アブレーション用）。
# 実測：バネダンパーは関節速度 平均8.3/最大15.8で腕が吹っ飛ぶ（人間的でない暴れ方）。
# 筋肉は 平均2.0/最大8.5＝4分の1。筋肉は力-速度関係を持ち瞬時に叩きつけられないため。
# 行動次元も 8→16（主動筋/拮抗筋のペア＝人間の筋配置）。筋パラメータはXMLから読まれる。
_MUSCLE = os.environ.get("D0_MUSCLE", "1") == "1"
ACTUATION = MuscleModel if _MUSCLE else SpringDamperModel

# 内発的動機。既定＝学習進度（好奇心）。D0_REWARD=predict で旧来の「予測しやすさ」（アブレーション用）。
#
# 【なぜ変えたか＝実測に基づく】旧来の報酬 1/(1+予測誤差) は「予測しやすい状態」を求める。
# その結果、太郎は**筋肉を全部最大活性にして腕を固定**した（飽和率100%・自己接触0%・
# 触覚が一定値で張り付き）。動きを消せば予測は完璧に当たる＝高報酬、という
# **暗い部屋問題(dark room problem)**（自由エネルギー原理への古典的批判）が実際に起きた。
# 指標は margin+52/corr0.76 と最高値を出すのに、行動は人間から最も遠い（ランダムより退化）。
#
# 【人間模倣】乳児は「予測しやすいもの」でなく「**今まさに学べるもの**」を探す。
# 学習進度＝予測誤差の"減り具合"を動機にするのが発達ロボティクスの標準
# （Oudeyer & Kaplan, Intelligent Adaptive Curiosity / Schmidhuber の圧縮進度）。
# これは暗い部屋問題（学べない＝進度0＝退屈）と、ノイズTV問題（永遠に学べない＝進度0）
# の両方を同時に避ける。
_REWARD = os.environ.get("D0_REWARD", "progress")  # "progress"（学習進度） / "predict"（旧・予測しやすさ）
from taro_brain_motor import TaroBrainWithMotor
from basal_ganglia import TaroLearner
from dopamine import Dopamine
from locus_coeruleus import LocusCoeruleus
from homeostatic_scaling import HomeostaticScaling
from cerebellum_motor import MotorCerebellum
from test_phase8_motor_learning import CombinedParams, rescale_action, to_tensor
from sensory_encoders import ProprioceptionEncoder, TouchEncoder
from insula import Insula

mse = torch.nn.functional.mse_loss
K = 100
LOG_DIR = os.path.join(_HERE, os.pardir, "logs", "D0")


class SelfTouchFusion:
    """D0の感覚融合＝内受容(太郎の内部状態)＋固有感覚＋**触覚**。
    Cの MinimalFusion の前庭を触覚に差し替えた形（selfbodyは前庭なし・触覚ONのため）。"""
    def __init__(self, prop_dim, touch_dim, emb=64):
        self.insula = Insula(state_dim=4, embedding_dim=emb)
        self.proprio = ProprioceptionEncoder(input_dim=prop_dim, embedding_dim=emb)
        self.touch = TouchEncoder(input_dim=touch_dim, hidden_dim=256, embedding_dim=emb)

    def parameters(self):
        import itertools
        return itertools.chain(self.insula.parameters(), self.proprio.parameters(), self.touch.parameters())

    def encode(self, obs):
        f = torch.cat([self.insula(to_tensor(obs["interoception"])),
                       self.proprio(to_tensor(obs["observation"])),
                       self.touch(to_tensor(obs["touch"]))], dim=-1)
        return torch.nn.functional.layer_norm(f, f.shape)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)
        return self


def ln_sens(obs):
    """予測対象＝固有感覚＋触覚（＝自分の動きの感覚的帰結すべて）。"""
    v = torch.cat([to_tensor(obs["observation"]), to_tensor(obs["touch"])])
    return torch.nn.functional.layer_norm(v, v.shape).detach()


def touch_sum(obs):
    return float(np.abs(np.asarray(obs["touch"])).sum())


def hand_touch_sum(env):
    """手・指の触覚だけを合計する（＝自己接触の指標）。

    座位では床に触れているのは尻・脚で、手は床から浮いている。だから
    「手が感じた触覚 > 0」＝自分の体に触れた、を意味する（somatotopyで発信元を切り分ける）。
    """
    t = env.unwrapped.touch
    m = env.unwrapped.model
    total = 0.0
    for bid, out in t.sensor_outputs.items():
        name = (m.body(bid).name or "")
        if ("hand" in name) or ("finger" in name) or ("distal" in name) or ("thumb" in name):
            total += float(np.abs(np.asarray(out)).sum())
    return total


def run(seed=0, n_train=3600, ckpt=600, n_eval=60):
    torch.manual_seed(seed); np.random.seed(seed)
    os.makedirs(LOG_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(LOG_DIR, f"d0_seed{seed}_{stamp}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        csv.writer(fp).writerow(["train_step", "classify", "margin", "corr", "persist", "touch_mean",
                                 "hand_touch_pct", "a_saturation", "real_min"])

    env = HybridEnv(gym.make("MIMoSelfBody-v0", actuation_model=ACTUATION))
    obs, _ = env.reset()
    n_act = env.action_space.shape[0]
    prop_dim = to_tensor(obs["observation"]).shape[0]; tch_dim = to_tensor(obs["touch"]).shape[0]
    fusion = SelfTouchFusion(prop_dim, tch_dim); tfusion = SelfTouchFusion(prop_dim, tch_dim).freeze()
    sdim = fusion.encode(obs).shape[0]
    out_dim = prop_dim + tch_dim
    brain = TaroBrainWithMotor(vocab_size=3, sensory_dim=sdim, n_actuators=n_act)
    emb_proj = nn.Linear(sdim + n_act, brain.sensory_proj.out_features)
    nat_head = nn.Sequential(nn.Linear(brain.latent_dim + n_act, 128), nn.SiLU(),
                             nn.LayerNorm(128), nn.Linear(128, out_dim))
    learner = TaroLearner(CombinedParams(brain, fusion, emb_proj, nat_head), lr=0.005)
    dop = Dopamine(); ne = LocusCoeruleus(); homeo = HomeostaticScaling(dim=sdim)
    cereb = MotorCerebellum(brain.latent_dim, n_act); cere_opt = torch.optim.Adam(cereb.parameters(), lr=0.005)
    state = {"obs": obs, "hidden": brain.init_motor_hidden(), "prev_a": torch.zeros(n_act)}
    t0 = time.time()
    print(f"=== D0 自己接触 (seed={seed}, n_train={n_train}, K={K}, "
          f"駆動={'筋肉' if _MUSCLE else 'バネダンパー'}, 動機={'学習進度' if _REWARD == 'progress' else '予測しやすさ'}) ===", flush=True)
    print(f"固有感覚={prop_dim} 触覚={tch_dim} 内受容=4 → sdim={sdim} / 行動={n_act}(右腕) / 予測={out_dim}", flush=True)

    def zc(sv, prev_a, cf, h):
        emb = emb_proj(torch.cat([sv, prev_a], dim=-1)).unsqueeze(0).unsqueeze(0)
        out, nh = brain.motor_gru(emb, h)
        z, kl, rc = brain.pc_latent.infer(h[-1, 0], out[0, -1], cf)
        return z, kl, rc, nh

    def act_mean(z):
        pm = torch.tanh(brain.motor_head(z))
        w, cere_a, _ = cereb.gate(z, pm)
        return (1.0 - w) * pm + w * cere_a

    def step_k(a):
        o, term = state["obs"], False
        for _ in range(K):
            o, r, te, tr, info = env.step(a)
            if te or tr:
                term = True; break
        return o, term

    def reset_state():
        state["obs"], _ = env.reset()
        state["hidden"] = brain.init_motor_hidden(); state["prev_a"] = torch.zeros(n_act)

    def evaluate():
        Zs, acts, nx, cu, self_err, ep, pdel, adel, tsum = [], [], [], [], [], [], [], [], []
        hand_hits, sat = 0, []
        for _ in range(n_eval):
            sv = fusion.encode(state["obs"]); cf = tfusion.encode(state["obs"]).detach(); clp = ln_sens(state["obs"])
            z, _, _, hn = zc(sv, state["prev_a"], cf, state["hidden"]); z = z.detach()
            a = torch.clamp(act_mean(z), -1.0, 1.0).detach()
            sat.append(float((a.abs() > 0.9).float().mean()))   # 飽和＝限界に振り切り(固まりの兆候)
            pd = nat_head(torch.cat([z, a], dim=-1)).detach()
            state["obs"], term = step_k(rescale_action(a, env.action_space))
            nlp = ln_sens(state["obs"]); tsum.append(touch_sum(state["obs"]))
            if hand_touch_sum(env) > 0:
                hand_hits += 1
            self_err.append(mse(clp + pd, nlp).item()); ep.append(mse(clp, nlp).item())
            pdel.append(pd.numpy()); adel.append((nlp - clp).numpy())
            Zs.append(z); acts.append(a); nx.append(nlp); cu.append(clp)
            state["hidden"] = hn.detach(); state["prev_a"] = a
            if term:
                reset_state()
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
        return (classify, margin, corr, persist, float(np.mean(tsum)),
                hand_hits / max(N, 1) * 100, float(np.mean(sat)) * 100)

    # 予測誤差の速い/遅い走行平均＝学習進度の材料（Cのrunnerと同じ係数）
    pe_fast, pe_slow = 1.0, 1.0

    def intrinsic_reward(pe_value):
        """内発的動機。progress＝学習進度（誤差が減っていれば正）／predict＝旧来の予測しやすさ。"""
        nonlocal pe_fast, pe_slow
        pe_fast = 0.9 * pe_fast + 0.1 * pe_value      # 直近の誤差
        pe_slow = 0.99 * pe_slow + 0.01 * pe_value    # 長期の誤差
        if _REWARD == "predict":
            return brain.sensorimotor_reward(pe_value)      # 1/(1+誤差)＝暗い部屋へ行く旧動機
        # 学習進度＝「長期の誤差」−「直近の誤差」。減っていれば正＝"今学べている"＝報酬。
        # 動きを消すと誤差は下がりきって進度0＝退屈になるので、暗い部屋に留まれない。
        # 生の差分をそのまま使う（恣意的な定数・正規化を持ち込まない。Oudeyerの定式化どおり）。
        return pe_slow - pe_fast

    buf = {k: [] for k in ("sv", "prev_a", "a", "cf", "clp", "nlp", "h")}

    def consolidate(n_batches=200, bs=128):
        """睡眠リプレイ（Cと同一）。自己モデル確立の本命機構。"""
        N = len(buf["sv"])
        if N < bs:
            return
        SV = torch.stack(buf["sv"]); PA = torch.stack(buf["prev_a"]); AA = torch.stack(buf["a"])
        CF = torch.stack(buf["cf"]); CLP = torch.stack(buf["clp"]); NLP = torch.stack(buf["nlp"])
        H = torch.cat(buf["h"], dim=1)
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
        sv = fusion.encode(state["obs"]); cf = tfusion.encode(state["obs"]).detach(); clp = ln_sens(state["obs"])
        z, kl, rc, hn = zc(sv, state["prev_a"], cf, state["hidden"].detach())
        pm = torch.tanh(brain.motor_head(z.detach()))
        std = 0.05 + ne.get_ne_level() * 0.45
        w_c, cere_a, e_c = cereb.gate(z.detach(), pm)
        mean = (1.0 - w_c) * pm + w_c * cere_a; std = std * (1.0 - w_c)
        dist = torch.distributions.Normal(mean, std)
        a = torch.clamp(dist.sample(), -1.0, 1.0); lp = dist.log_prob(a).sum()
        pred = clp + nat_head(torch.cat([z, a.detach()], dim=-1))
        state["obs"], term = step_k(rescale_action(a, env.action_space)); nlp = ln_sens(state["obs"])
        buf["sv"].append(sv.detach()); buf["prev_a"].append(state["prev_a"].detach())
        buf["a"].append(a.detach()); buf["cf"].append(cf.detach())
        buf["clp"].append(clp.detach()); buf["nlp"].append(nlp.detach()); buf["h"].append(state["hidden"].detach())
        pe = mse(pred, nlp); rew = intrinsic_reward(pe.item())
        pl = learner.learn_action([lp], dop.compute_rpe(rew))
        hl = homeo.homeostatic_loss(sv); homeo.observe(sv)
        learner.update(pe + hl + kl + rc, pl)
        closs = cereb.imitation_loss(z.detach(), a.detach())
        cere_opt.zero_grad(); closs.backward(); cere_opt.step(); cereb.observe(e_c)
        ne.observe_reward(rew); ne.release_ne()
        state["hidden"] = hn.detach(); state["prev_a"] = a.detach()
        if term:
            reset_state()
        if (i + 1) % ckpt == 0:
            consolidate()
            cl, mg, co, pr, tm, hp, sa = evaluate()
            rm = (time.time() - t0) / 60
            with open(csv_path, "a", newline="", encoding="utf-8") as fp:
                csv.writer(fp).writerow([i + 1, f"{cl:.2f}", f"{mg:.2f}", f"{co:.4f}", f"{pr:.2f}",
                                         f"{tm:.1f}", f"{hp:.1f}", f"{sa:.1f}", f"{rm:.1f}"])
            print(f"  step {i+1}: margin={mg:+.1f}% corr={co:.3f} persist={pr:.1f}% | "
                  f"自己接触={hp:.0f}% 飽和={sa:.0f}% 触覚={tm:.0f} ({rm:.0f}分)", flush=True)

    mdir = os.path.join(_HERE, os.pardir, "models"); os.makedirs(mdir, exist_ok=True)
    path = os.path.join(mdir, f"self_touch_{'muscle' if _MUSCLE else 'spring'}_{_REWARD}_seed{seed}.pt")
    torch.save({"brain": brain.state_dict(), "fusion_insula": fusion.insula.state_dict(),
                "fusion_proprio": fusion.proprio.state_dict(), "fusion_touch": fusion.touch.state_dict(),
                "emb_proj": emb_proj.state_dict(), "nat_head": nat_head.state_dict(), "cereb": cereb.state_dict(),
                "config": {"sdim": sdim, "prop_dim": prop_dim, "touch_dim": tch_dim, "out_dim": out_dim,
                           "n_act": n_act, "K": K, "seed": seed, "n_train": n_train, "env": "MIMoSelfBody-v0",
                           "actuation": "MuscleModel" if _MUSCLE else "SpringDamperModel", "reward": _REWARD,
                           "fusion": "SelfTouchFusion(interoception+proprio+touch)"}}, path)
    print(f"SAVED {path}\nlog={csv_path}", flush=True)
    return path


if __name__ == "__main__":
    s = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 3600
    run(seed=s, n_train=n)
