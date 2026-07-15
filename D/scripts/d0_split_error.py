"""
診断：D0の予測誤差を「固有感覚（＝自分の体のマッピング）」と「触覚」に分けて測る。

背景：D0はCと同じ学習・同じ睡眠リプレイなのに persist 95〜100%（＝「何もしない予測」と互角）で
自己モデルが確立しない。Cは同条件で persist 68〜70%・margin+51 を出していた。
最大の構造差＝**予測対象の89%(1908/2150)が触覚**になったこと。触覚はほぼゼロの疎な信号なので、
 ①「何もしない予測」が触覚に対して異常に強い → persistが壊れて見える
 ②損失が触覚に支配され、固有感覚（体のマッピング）の学習が埋もれる
のどちらか（or両方）が起きている疑い。分けて測れば切り分けできる。

使い方: python d0_split_error.py [model_path] [n_eval]
"""
import os, sys, warnings
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
from hybrid_env import HybridEnv
from taro_brain_motor import TaroBrainWithMotor
from cerebellum_motor import MotorCerebellum
from test_phase8_motor_learning import rescale_action, to_tensor
from d0_selftouch import SelfTouchFusion, ln_sens, K

mse = torch.nn.functional.mse_loss


def main():
    mp = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _HERE, os.pardir, "models", "self_touch_muscle_predict_seed0.pt")
    n_eval = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    ck = torch.load(mp, weights_only=False); cfg = ck["config"]
    P, T = cfg["prop_dim"], cfg["touch_dim"]
    print(f"=== 予測誤差の内訳診断 ===\nmodel={os.path.basename(mp)} "
          f"動機={cfg.get('reward','?')} 駆動={cfg.get('actuation','?')}")
    print(f"予測対象: 固有感覚{P} + 触覚{T} = {P+T}  （触覚が{T/(P+T)*100:.0f}%を占める）\n")

    from mimoActuation.actuation import SpringDamperModel
    from mimoActuation.muscle import MuscleModel
    am = MuscleModel if cfg.get("actuation") == "MuscleModel" else SpringDamperModel
    env = HybridEnv(gym.make("MIMoSelfBody-v0", actuation_model=am,
                             done_active=False, max_episode_steps=6000))
    obs, _ = env.reset(seed=0)
    na = env.action_space.shape[0]
    fusion = SelfTouchFusion(P, T); tfusion = SelfTouchFusion(P, T).freeze()
    brain = TaroBrainWithMotor(vocab_size=3, sensory_dim=cfg["sdim"], n_actuators=na)
    emb_proj = nn.Linear(cfg["sdim"] + na, brain.sensory_proj.out_features)
    nat_head = nn.Sequential(nn.Linear(brain.latent_dim + na, 128), nn.SiLU(),
                             nn.LayerNorm(128), nn.Linear(128, cfg["out_dim"]))
    cereb = MotorCerebellum(brain.latent_dim, na)
    brain.load_state_dict(ck["brain"]); fusion.insula.load_state_dict(ck["fusion_insula"])
    fusion.proprio.load_state_dict(ck["fusion_proprio"]); fusion.touch.load_state_dict(ck["fusion_touch"])
    emb_proj.load_state_dict(ck["emb_proj"]); nat_head.load_state_dict(ck["nat_head"]); cereb.load_state_dict(ck["cereb"])
    for m in (brain, emb_proj, nat_head, cereb):
        for p in m.parameters():
            p.requires_grad_(False)
    fusion.freeze()

    h = brain.init_motor_hidden(); pa = torch.zeros(na)
    Zs, acts, cu, nx = [], [], [], []
    for _ in range(n_eval):
        sv = fusion.encode(obs); cf = tfusion.encode(obs).detach(); clp = ln_sens(obs)
        emb = emb_proj(torch.cat([sv, pa], dim=-1)).unsqueeze(0).unsqueeze(0)
        out, hn = brain.motor_gru(emb, h)
        z, _, _ = brain.pc_latent.infer(h[-1, 0], out[0, -1], cf); z = z.detach()
        pm = torch.tanh(brain.motor_head(z)); w, ca, _ = cereb.gate(z, pm)
        a = torch.clamp((1 - w) * pm + w * ca, -1, 1).detach()
        ctrl = rescale_action(a, env.action_space)
        for _ in range(K):
            obs, r, te, tr, info = env.step(ctrl)
            if te or tr:
                break
        nlp = ln_sens(obs)
        Zs.append(z); acts.append(a); cu.append(clp); nx.append(nlp)
        h = hn.detach(); pa = a
        if te or tr:
            obs, _ = env.reset(); h = brain.init_motor_hidden(); pa = torch.zeros(na)

    def report(sl, name, total_dim):
        self_err, naive_err, other_err = [], [], []
        N = len(Zs)
        for i in range(N):
            pd = nat_head(torch.cat([Zs[i], acts[i]], dim=-1)).detach()
            self_err.append(mse((cu[i] + pd)[sl], nx[i][sl]).item())
            naive_err.append(mse(cu[i][sl], nx[i][sl]).item())     # 「何もしない」予測
            for j in range(N):
                if i == j:
                    continue
                po = nat_head(torch.cat([Zs[i], acts[j]], dim=-1)).detach()
                other_err.append(mse((cu[i] + po)[sl], nx[i][sl]).item())
        se, ne_, oe = np.mean(self_err), np.mean(naive_err), np.mean(other_err)
        persist = se / max(ne_, 1e-12) * 100
        margin = (oe - se) / max(oe, 1e-12) * 100
        # 「何もしない」がどれだけ強いか＝信号がどれだけ動かないか
        print(f"[{name:6s}] 次元={total_dim:5d}({total_dim/(P+T)*100:2.0f}%) | "
              f"太郎の誤差={se:.5f}  なまけ予測の誤差={ne_:.5f} | "
              f"persist={persist:6.1f}%  margin={margin:+6.1f}%")
        return persist, margin

    print("             ↓persist<100で「なまけ予測」に勝ち。margin高いほど行動依存＝自己モデルが本物")
    pp, mp_ = report(slice(0, P), "固有感覚", P)
    tp, tm = report(slice(P, P + T), "触覚", T)
    ap, am_ = report(slice(0, P + T), "合計", P + T)
    print(f"\n=== 判定 ===")
    if pp < 100 and tp > 100:
        print(f"→ **固有感覚は学べている(persist {pp:.0f}%)のに、触覚が壊している(persist {tp:.0f}%)**。")
        print(f"   合計{ap:.0f}%は触覚に引きずられた見かけの値＝『体のマッピングは失敗していない』。")
    elif pp > 100:
        print(f"→ **固有感覚そのものが なまけ予測に負けている(persist {pp:.0f}%)**＝体のマッピング自体が未確立。")
    else:
        print(f"→ 固有感覚 persist={pp:.0f}% / 触覚 persist={tp:.0f}%。")
    print(f"   参考：Cの自己モデルは固有感覚のみで persist 68〜70% / margin +51 だった。")


if __name__ == "__main__":
    main()
