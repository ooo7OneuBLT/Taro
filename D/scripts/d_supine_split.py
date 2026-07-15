"""
診断：仰向け・触覚ありの太郎の予測誤差を「固有感覚(621)」と「触覚(3606)」に分けて測る。

【なぜ必要か】
仰向け3シードの結果は 触覚なし margin +40〜53 / 触覚あり +15〜24 で、範囲が重ならない。
しかし触覚ありの予測対象は 621+3606=4227次元で**85%が触覚**なので、合計のmarginは
触覚に薄められている可能性がある。
**2026-07-15の午前、D0でまさにこれに騙された**：合計 persist 86% を見て「自己モデルが確立
できていない」と報告したが、分けて測ると 固有感覚 persist 77.7% / margin +39.8（＝確立して
いる）で、触覚 persist 96.3%（＝学べていない）が合計を引きずっていただけだった。

同じ轍を踏まないため、学習済みモデル（保存済み）を読んで分けて測る。学習し直しは不要。

判定の意味：
  persist  = 太郎の誤差 / 「何も変わらない」と予測した誤差 × 100。100未満で"なまけ予測"に勝ち。
  margin   = 行動を入れ替えたときに誤差がどれだけ増えるか＝**行動依存＝自己モデルの本体**。

使い方: python d_supine_split.py [model_path] [n_eval]
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("C_TOUCH", "1")     # ln_prop を「固有感覚＋触覚」にするため import 前に必要
os.environ.setdefault("C_SUPINE", "1")
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "C", "scripts"))
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
from run_c_metrics_ac_lr import MinimalFusion, ln_prop, _ENV_ID, _touch_params

mse = torch.nn.functional.mse_loss
K = 100


def main():
    mp = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _HERE, os.pardir, "models", "supine_touch1_seed0.pt")
    n_eval = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    ck = torch.load(mp, weights_only=False); cfg = ck["config"]
    P, T = cfg["prop_dim"], cfg["touch_dim"]
    print(f"=== 予測誤差の内訳（仰向け・触覚あり）===")
    print(f"model={os.path.basename(mp)}  seed={cfg['seed']}  姿勢={'仰向け' if cfg['supine'] else '立位'}")
    print(f"予測対象: 固有感覚{P} + 触覚{T} = {P+T}  （触覚が{T/(P+T)*100:.0f}%を占める）\n")

    env = HybridEnv(gym.make(_ENV_ID, vision_params=None, touch_params=_touch_params()))
    fusion = MinimalFusion(T); tfusion = MinimalFusion(T).freeze()
    n_act = env.action_space.shape[0]
    obs, _ = env.reset(seed=cfg["seed"])
    brain = TaroBrainWithMotor(vocab_size=3, sensory_dim=cfg["sdim"], n_actuators=n_act)
    emb_proj = nn.Linear(cfg["sdim"] + n_act, brain.sensory_proj.out_features)
    nat_head = nn.Sequential(nn.Linear(brain.latent_dim + n_act, 128), nn.SiLU(),
                             nn.LayerNorm(128), nn.Linear(128, cfg["out_dim"]))
    cereb = MotorCerebellum(brain.latent_dim, n_act)
    brain.load_state_dict(ck["brain"])
    fusion.insula.load_state_dict(ck["fusion_insula"])
    fusion.proprio.load_state_dict(ck["fusion_proprio"])
    fusion.vestibular.load_state_dict(ck["fusion_vestibular"])
    fusion.touch.load_state_dict(ck["fusion_touch"])
    emb_proj.load_state_dict(ck["emb_proj"]); nat_head.load_state_dict(ck["nat_head"])
    cereb.load_state_dict(ck["cereb"])
    for m in (brain, emb_proj, nat_head, cereb):
        for p in m.parameters():
            p.requires_grad_(False)
    fusion.freeze()

    # 注意：pc_latent.infer は推論時に内部で勾配を使う（予測符号化の誤差回帰）ので
    # torch.no_grad() で囲んではいけない。重みは上で凍結済みなので学習は起きない。
    h = brain.init_motor_hidden(); pa = torch.zeros(n_act)
    Zs, acts, cu, nx = [], [], [], []
    for _ in range(n_eval):
        sv = fusion.encode(obs); cf = tfusion.encode(obs).detach(); clp = ln_prop(obs)
        emb = emb_proj(torch.cat([sv, pa], dim=-1)).unsqueeze(0).unsqueeze(0)
        out, hn = brain.motor_gru(emb, h)
        z, _, _ = brain.pc_latent.infer(h[-1, 0], out[0, -1], cf); z = z.detach()
        pm = torch.tanh(brain.motor_head(z)); w, ca, _ = cereb.gate(z, pm)
        a = torch.clamp((1 - w) * pm + w * ca, -1, 1).detach()
        ctrl = rescale_action(a, env.action_space)
        te = tr = False
        for _ in range(K):
            obs, r, te, tr, info = env.step(ctrl)
            if te or tr:
                break
        Zs.append(z); acts.append(a); cu.append(clp); nx.append(ln_prop(obs))
        h = hn.detach(); pa = a
        if te or tr:
            obs, _ = env.reset(); h = brain.init_motor_hidden(); pa = torch.zeros(n_act)

    def report(sl, name, dim):
        se_, ne_, oe_ = [], [], []
        N = len(Zs)
        for i in range(N):
            pd = nat_head(torch.cat([Zs[i], acts[i]], dim=-1)).detach()
            se_.append(mse((cu[i] + pd)[sl], nx[i][sl]).item())
            ne_.append(mse(cu[i][sl], nx[i][sl]).item())          # 「何も変わらない」予測
            for j in range(N):
                if i == j:
                    continue
                po = nat_head(torch.cat([Zs[i], acts[j]], dim=-1)).detach()
                oe_.append(mse((cu[i] + po)[sl], nx[i][sl]).item())
        se, nev, oe = np.mean(se_), np.mean(ne_), np.mean(oe_)
        persist = se / max(nev, 1e-12) * 100
        margin = (oe - se) / max(oe, 1e-12) * 100
        print(f"[{name:6s}] 次元={dim:5d}({dim/(P+T)*100:2.0f}%) | 太郎の誤差={se:.6f} "
              f"なまけ予測={nev:.6f} | persist={persist:6.1f}%  margin={margin:+6.1f}%")
        return persist, margin

    print("        ↓persist<100で「なまけ予測」に勝ち／margin高いほど行動依存＝自己モデルが本物")
    pp, pm_ = report(slice(0, P), "固有感覚", P)
    tp, tm = report(slice(P, P + T), "触覚", T)
    ap, am_ = report(slice(0, P + T), "合計", P + T)
    print("\n=== 判定 ===")
    print(f"参考：仰向け・触覚なしの3シードは margin +40〜53 / persist 66〜81 だった。")
    if pp < 100 and pm_ > 35:
        print(f"→ **固有感覚は触覚なしと同等に学べている**(persist {pp:.0f}% / margin {pm_:+.0f})。"
              f"合計{am_:+.0f}は触覚{T}次元に薄められた見かけの値。")
    elif pp < 100:
        print(f"→ 固有感覚は なまけ予測には勝つ(persist {pp:.0f}%)が、行動依存が弱い(margin {pm_:+.0f})。"
              f"＝触覚を足したこと自体が体のマッピングを劣化させている。")
    else:
        print(f"→ **固有感覚そのものが なまけ予測に負けている**(persist {pp:.0f}%)＝体のマッピングが壊れた。")
    print(f"   触覚は persist {tp:.0f}% / margin {tm:+.0f}"
          f"{'（＝学べていない。触れっぱなしで変化しない信号）' if tp > 90 else ''}")
    env.close()


if __name__ == "__main__":
    main()
