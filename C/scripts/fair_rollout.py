"""
公平対決：同じ「1秒後の固有感覚」を、一発 vs 0.1秒×10連鎖(ロールアウト)で予測。
土俵を1秒に固定＝標的・テストデータ・物差しを完全に同一にして、"予測の構造"だけを比べる。
（Taroの全脳は配線せず、構造の是非だけを蒸留して測る軽い診断。）

方式A(一発):  p̂ = p0 + fA([p0, a])                 … 1秒を一飛び
方式B(連鎖):  p=p0; ×10: p = p + fB([p, a]);  p̂=p   … 0.1秒を10回、自分の予測を戻して繋ぐ
両方とも fA/fB は同一アーキ(Linear→SiLU→LayerNorm→Linear)。行動aは1秒間固定(現Taroと同じ)。

対照/物差し：
  persist比 = MSE(予測, 実1秒後) / MSE("何もしない"=p0, 実1秒後) × 100  （<100で"何もしない"に勝ち）
  corr      = 予測した変化 vs 実際の変化 の相関
使い方: python fair_rollout.py <seed>
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)

_BRIDGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # = Taro/C
for sub in ("wrapper", "senses", "brain"):
    sys.path.insert(0, os.path.join(_BRIDGE, "src", sub))
sys.path.insert(0, os.path.join(_BRIDGE, "tests"))
import gymnasium as gym
import mimoEnv  # noqa
from hybrid_env import HybridEnv
from test_phase8_motor_learning import rescale_action, to_tensor

mse = torch.nn.functional.mse_loss
SUB = 10          # 0.1秒 = 10 mj_step
NSUB = 10         # 1秒 = 0.1秒 ×10
def ln_prop_obs(obs):
    v = to_tensor(obs["observation"])
    return torch.nn.functional.layer_norm(v, v.shape).detach()


def head(inp, out):
    return nn.Sequential(nn.Linear(inp, 128), nn.SiLU(), nn.LayerNorm(128), nn.Linear(128, out))


def collect(env, n_act, n_traj, rng):
    """各軌道＝1秒間ひとつの固定行動。0.1秒ごとの固有感覚 sub[0..10] を記録。"""
    obs, _ = env.reset()
    for _ in range(20):
        obs, *_ = env.step(np.zeros(n_act))
    trajs = []
    while len(trajs) < n_traj:
        a = rng.uniform(-1, 1, n_act).astype(np.float32)
        ra = rescale_action(torch.tensor(a), env.action_space)
        sub = [ln_prop_obs(obs)]; term = False
        for _m in range(NSUB):
            for _ in range(SUB):
                obs, r, te, tr, info = env.step(ra)
                if te or tr:
                    term = True; break
            sub.append(ln_prop_obs(obs))
            if term:
                break
        if not term and len(sub) == NSUB + 1:
            trajs.append((torch.tensor(a), sub))
        else:
            obs, _ = env.reset()
            for _ in range(20):
                obs, *_ = env.step(np.zeros(n_act))
    return trajs


def train_head(f, X, Y, epochs=500, lr=1e-3):
    opt = torch.optim.Adam(f.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad(); loss = mse(f(X), Y); loss.backward(); opt.step()
    return f


def run(seed, n_traj=600):
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    env = HybridEnv(gym.make("MIMoBenchV2-v0", vision_params=None, touch_params=None))
    n_act = env.action_space.shape[0]
    trajs = collect(env, n_act, n_traj, rng); env.close()
    pdim = trajs[0][1][0].shape[0]
    ntr = len(trajs); S = int(ntr * 0.6)
    tr, te = trajs[:S], trajs[S:]

    # 学習データ：A=1秒ペア(1本/軌道)、B=0.1秒ペア(10本/軌道)。両方 [p,a]→Δ。
    def pairsA(ts):
        X = [torch.cat([s[0], a]) for a, s in ts]; Y = [s[NSUB] - s[0] for a, s in ts]
        return torch.stack(X), torch.stack(Y)

    def pairsB(ts):
        X, Y = [], []
        for a, s in ts:
            for k in range(NSUB):
                X.append(torch.cat([s[k], a])); Y.append(s[k + 1] - s[k])
        return torch.stack(X), torch.stack(Y)

    XA, YA = pairsA(tr); XB, YB = pairsB(tr)
    fA = train_head(head(pdim + n_act, pdim), XA, YA)
    fB = train_head(head(pdim + n_act, pdim), XB, YB)

    # 評価（held-outで同じ標的=1秒後）
    predA_d, predB_d, act_d, persist_d = [], [], [], []
    with torch.no_grad():
        for a, s in te:
            p0, tgt = s[0], s[NSUB]
            pA = p0 + fA(torch.cat([p0, a]))
            p = p0.clone()
            for _ in range(NSUB):
                p = p + fB(torch.cat([p, a]))
            predA_d.append((pA - p0).numpy()); predB_d.append((p - p0).numpy())
            act_d.append((tgt - p0).numpy())
            persist_d.append((pA, p, tgt, p0))
    # 誤差・persist比・corr
    seA = np.mean([((pA - tgt) ** 2).mean().item() for pA, pB, tgt, p0 in persist_d])
    seB = np.mean([((pB - tgt) ** 2).mean().item() for pA, pB, tgt, p0 in persist_d])
    sep = np.mean([((p0 - tgt) ** 2).mean().item() for pA, pB, tgt, p0 in persist_d])
    PA = np.concatenate([d.flatten() for d in predA_d]); PB = np.concatenate([d.flatten() for d in predB_d])
    AC = np.concatenate([d.flatten() for d in act_d])
    corrA = float(np.corrcoef(PA, AC)[0, 1]); corrB = float(np.corrcoef(PB, AC)[0, 1])
    persistA = seA / sep * 100; persistB = seB / sep * 100

    print(f"=== 公平対決 seed={seed} (n_traj={ntr}, test={len(te)}, A_pairs={len(XA)}, B_pairs={len(XB)}) ===", flush=True)
    print(f"方式A 一発      : persist={persistA:5.1f}%  corr={corrA:.3f}", flush=True)
    print(f"方式B 連鎖(x10) : persist={persistB:5.1f}%  corr={corrB:.3f}", flush=True)
    win = "B(連鎖)" if (persistB < persistA and corrB > corrA) else ("A(一発)" if (persistA < persistB and corrA > corrB) else "混合/差なし")
    print(f"RESULT seed={seed} 勝者={win} | A(persist{persistA:.1f}/corr{corrA:.3f}) vs B(persist{persistB:.1f}/corr{corrB:.3f})", flush=True)


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
