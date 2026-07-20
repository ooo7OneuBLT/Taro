"""【E1の関門・切り分け版】接触が太郎の信号のどこで消えているかを特定する。

【前提＝第1版の結果】
接触は31.5%のtickで起きているのに、固有感覚621次元の予測誤差 pe は
  触れているとき 1.00107 / いないとき 1.00201（比0.999、Cohen's d=-0.005）
＝**まったく動かなかった**。この鎖のどこで切れているかを切り分ける。

【切り分ける4点】
 (0) 基準線との差で見る（ユーザーの仮説）
     床との接地など**常に一定の背景**があるなら、絶対値のpeを比べても埋もれる。
     太郎自身が既にこの発想を持っている（progress報酬 = pe_slow − pe_fast ＝
     遅い移動平均を基準線にした差、相対NE も同じ）。同じやり方で接触の効果を見る。
 (a) 部位を絞る：621次元全体でなく**腕の関節だけ**の予測誤差を見る。
     ②「他の次元に薄められている」ならここで差が出る。
     （D0で踏んだ「触覚が次元数に薄められる」罠と同じ構造）
 (b) 生の物理量：接触時に**関節の位置・速度・アクチュエータ反力**が実際に変化しているか。
     ③「そもそも物理的に反力が返っていない」ならここが動かない。
 (c) 質量スイープ：おもちゃを重くすると差が出るか。①「手応えが小さすぎる」の検証。

使い方: python e_contact_diag.py [n_ticks]
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "D", "scripts"))
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "C", "scripts"))
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "taro_core"))
import paths  # noqa: E402
paths.setup_brain_path()
sys.path.insert(0, paths.MIMO_DIR)

import mimoEnv  # noqa: F401,E402
import d_c5_motor_quality as mq  # noqa: E402
from fusion import to_tensor  # noqa: E402
import torch.nn as nn  # noqa: E402

HAND_PARTS = ("hand", "fingers", "ff", "lf", "th", "mf", "rf")


def ln_prop(obs):
    v = to_tensor(obs["observation"])
    return torch.nn.functional.layer_norm(v, v.shape).detach()


def arm_dof_indices(model):
    """腕（肩・肘・手・指）の自由度インデックス。部位を絞って見るため。"""
    keys = ("shoulder", "elbow", "hand", "finger", "ff", "lf", "th", "mf", "rf", "wrist")
    dofs = []
    for j in range(model.njnt):
        nm = model.joint(j).name
        if any(k in nm for k in keys):
            adr = int(model.jnt_dofadr[j])
            dofs.append(adr)
    return np.array(sorted(set(dofs)), dtype=int)


def run(n_ticks, toy_mass=None):
    env, brain, fusion, emb_proj, cereb, n_act = mq.build("off", age=0)
    policy = mq.make_policy(brain, fusion, emb_proj, cereb, n_act, babble=True)
    raw = env.unwrapped
    if toy_mass is not None:                      # (c) 質量スイープ
        bid = raw.model.body("test_object1").id
        raw.model.body_mass[bid] = toy_mass

    obs, _ = env.reset(seed=0)
    prop_dim = to_tensor(obs["observation"]).shape[0]
    nat_head = nn.Sequential(nn.Linear(brain.latent_dim + n_act, 128), nn.SiLU(),
                             nn.LayerNorm(128), nn.Linear(128, prop_dim))
    blob = torch.load(mq.CKPT, map_location="cpu", weights_only=False)
    nat_head.load_state_dict({k: v for k, v in blob["nat_head"].items()
                              if k in nat_head.state_dict()}, strict=False)

    arm_dofs = arm_dof_indices(raw.model)
    hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)
    rec = dict(pe=[], pe_arm=[], touch=[], hand=[], qacc=[], qfrc=[], armvel=[])

    for t in range(n_ticks):
        sv = fusion.encode(obs); cf = fusion.encode(obs).detach(); clp = ln_prop(obs)
        a, hidden = policy(obs, prev_a, hidden)
        emb = emb_proj(torch.cat([sv, prev_a], dim=-1)).unsqueeze(0).unsqueeze(0)
        out, _ = brain.motor_gru(emb, hidden)
        z, _, _ = brain.pc_latent.infer(hidden[-1, 0], out[0, -1], cf)
        with torch.no_grad():
            delta = nat_head(torch.cat([z.detach(), a], dim=-1))
            pred = clp + delta
        ctrl = mq.rescale_action(a, env.action_space); prev_a = a

        touched = handed = False
        qacc_arm = qfrc_arm = vel_arm = 0.0
        for k in range(mq.K):
            obs, r, te, tr, info = env.step(ctrl)
            cs = [c for c in raw.toy_contacts() if c != "world"]
            if cs:
                touched = True
                if any(any(h in c for h in HAND_PARTS) for c in cs):
                    handed = True
            qacc_arm = max(qacc_arm, float(np.abs(raw.data.qacc[arm_dofs]).max()))
            qfrc_arm = max(qfrc_arm, float(np.abs(raw.data.qfrc_constraint[arm_dofs]).max()))
            vel_arm = max(vel_arm, float(np.abs(raw.data.qvel[arm_dofs]).max()))
            if te or tr:
                break
        nlp = ln_prop(obs)
        with torch.no_grad():
            err = (pred - nlp) ** 2
        rec["pe"].append(float(err.mean()))
        rec["pe_arm"].append(float(err[arm_dofs[arm_dofs < err.shape[0]]].mean()))
        rec["touch"].append(touched); rec["hand"].append(handed)
        rec["qacc"].append(qacc_arm); rec["qfrc"].append(qfrc_arm); rec["armvel"].append(vel_arm)
        if te or tr:
            obs, _ = env.reset()
            hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)
    env.close()
    return {k: np.asarray(v) for k, v in rec.items()}


def compare(name, vals, mask, baseline_rel=False):
    """接触あり/なしで比較。baseline_rel=True なら直前の遅い移動平均との差で見る。"""
    x = vals.astype(float)
    if baseline_rel:
        slow = np.zeros_like(x); s = x[0]
        for i, v in enumerate(x):
            slow[i] = s
            s = 0.99 * s + 0.01 * v      # progress報酬と同じ遅い移動平均＝基準線
        x = x - slow                      # 「いつもと比べてどうか」
    a, b = x[mask], x[~mask]
    if len(a) < 3 or len(b) < 3:
        print(f"  {name:26s} (サンプル不足)")
        return
    sp = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1))
                 / max(len(a)+len(b)-2, 1))
    d = (a.mean() - b.mean()) / max(sp, 1e-12)
    print(f"  {name:26s} touch={a.mean():+.6f}  no={b.mean():+.6f}  "
          f"ratio={a.mean()/b.mean() if abs(b.mean())>1e-12 else float('nan'):+.3f}  d={d:+.3f}")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    print("=== (0)(a)(b) 通常の質量(30g)で切り分け ===")
    r = run(n)
    m = r["touch"]
    print(f"  contact {m.sum()}/{len(m)} ({m.mean()*100:.1f}%)  hand {r['hand'].sum()}")
    print("\n [絶対値]")
    compare("pe (全621次元)", r["pe"], m)
    compare("pe (腕の次元のみ)", r["pe_arm"], m)
    print("\n [基準線との差＝progress流]")
    compare("pe (全621次元)", r["pe"], m, baseline_rel=True)
    compare("pe (腕の次元のみ)", r["pe_arm"], m, baseline_rel=True)
    print("\n [生の物理量＝そもそも反力が返っているか]")
    compare("腕のqacc(最大)", r["qacc"], m)
    compare("腕の拘束力qfrc(最大)", r["qfrc"], m)
    compare("腕のqvel(最大)", r["armvel"], m)

    print("\n=== (c) おもちゃの質量スイープ（手応えが足りないのか） ===")
    for mass in (0.3, 3.0):
        rr = run(max(n // 2, 60), toy_mass=mass)
        mm = rr["touch"]
        if mm.sum() < 3:
            print(f"  mass={mass*1000:.0f}g : 接触不足({mm.sum()})")
            continue
        print(f"  --- mass={mass*1000:.0f}g  contact {mm.mean()*100:.1f}% ---")
        compare("pe(腕/基準線差)", rr["pe_arm"], mm, baseline_rel=True)
        compare("腕の拘束力qfrc", rr["qfrc"], mm)


if __name__ == "__main__":
    main()
