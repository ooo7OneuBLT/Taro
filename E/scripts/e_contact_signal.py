"""【E1の関門】おもちゃに触れたとき、太郎の固有感覚の予測誤差は動くか。

【なぜ最初にこれを測るか】
E1の設計はこうなっている：
    おもちゃを押す → 反作用で関節の感覚が変わる → それは予測対象(固有感覚)なので学べる
    → 予測が上達する → **progress報酬(pe_slow − pe_fast)が発生** → リーチが強化される
この鎖の最初の環（**触れたことが太郎の予測誤差に現れるか**）を、**まだ一度も確かめていない**。
ここが動かなければ、学習を何時間回しても報酬が発生せず**無駄**になる。数分で判定できる。

【測り方】
C5の学習済みモデル（＝実際の太郎の脳）で仰向け・おもちゃ環境を走らせ、各tickで
  ・MuJoCoの実接触でおもちゃに触れているか（距離ではなく接触で判定）
  ・予測誤差 pe = mse(clp + nat_head([z,a]), nlp)   ＝run_c_metricsと同じ定義
を記録し、**接触あり/なしで pe の分布を比べる**。

【判定】
  接触時に pe が有意に大きい → 触れることが「学べる出来事」になっている＝学習を回す価値あり
  変わらない               → おもちゃが太郎の感覚に影響していない＝設計変更が必要
                             （手応え/質量を上げる・接触機会を増やす・視覚を入れる 等）

⚠️注意：peが大きいこと自体は報酬ではない。progress報酬は「peが**減っていく**こと」。
ここで見たいのは「そもそも接触が信号として存在するか」＝学習の材料があるか。

使い方: python e_contact_signal.py [n_ticks]
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

mse = torch.nn.functional.mse_loss
HAND_PARTS = ("hand", "fingers", "ff", "lf", "th", "mf", "rf")


def ln_prop(obs):
    """予測対象＝固有感覚（run_c_metrics_ac_lr.ln_prop と同じ定義）。"""
    v = to_tensor(obs["observation"])
    return torch.nn.functional.layer_norm(v, v.shape).detach()


def main():
    n_ticks = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    env, brain, fusion, emb_proj, cereb, n_act = mq.build("off", age=0)
    policy = mq.make_policy(brain, fusion, emb_proj, cereb, n_act, babble=True)
    raw = env.unwrapped
    if not hasattr(raw, "toy_contacts"):
        print("!! E_TOY=1 で起動してください（おもちゃ環境が要る）")
        return

    # 予測ヘッド nat_head は build() が作らないので、ここで用意して学習済み重みを読む
    import torch.nn as nn
    obs, _ = env.reset(seed=0)
    sdim = fusion.encode(obs).shape[0]
    prop_dim = to_tensor(obs["observation"]).shape[0]
    nat_head = nn.Sequential(nn.Linear(brain.latent_dim + n_act, 128), nn.SiLU(),
                             nn.LayerNorm(128), nn.Linear(128, prop_dim))
    blob = torch.load(mq.CKPT, map_location="cpu", weights_only=False)
    mq.load_matching(nat_head, blob["nat_head"], "nat_head")

    hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)
    pes, touched, by_hand, toy_moved = [], [], [], []
    prev_toy = raw.data.body("test_object1").xpos.copy()

    for t in range(n_ticks):
        sv = fusion.encode(obs)
        cf = fusion.encode(obs).detach()
        clp = ln_prop(obs)
        a, hidden = policy(obs, prev_a, hidden)
        # pc_latent.infer は内部で torch.autograd.grad を使う（予測符号化の推論そのものが
        # 勾配降下）ため、no_grad の中では動かない。勾配は使うが学習はしない＝結果をdetachする。
        emb = emb_proj(torch.cat([sv, prev_a], dim=-1)).unsqueeze(0).unsqueeze(0)
        out, _ = brain.motor_gru(emb, hidden)
        z, _, _ = brain.pc_latent.infer(hidden[-1, 0], out[0, -1], cf)
        with torch.no_grad():
            pred = clp + nat_head(torch.cat([z.detach(), a], dim=-1))
        ctrl = mq.rescale_action(a, env.action_space)
        prev_a = a
        contact_this_tick, hand_this_tick = False, False
        move = 0.0
        for k in range(mq.K):
            obs, r, te, tr, info = env.step(ctrl)
            cs = [c for c in raw.toy_contacts() if c != "world"]
            if cs:
                contact_this_tick = True
                if any(any(h in c for h in HAND_PARTS) for c in cs):
                    hand_this_tick = True
            p = raw.data.body("test_object1").xpos
            move += float(np.linalg.norm(p - prev_toy))
            prev_toy = p.copy()
            if te or tr:
                break
        nlp = ln_prop(obs)
        pes.append(float(mse(pred, nlp)))
        touched.append(contact_this_tick)
        by_hand.append(hand_this_tick)
        toy_moved.append(move)
        if te or tr:
            obs, _ = env.reset()
            hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)
            prev_toy = raw.data.body("test_object1").xpos.copy()

    pes = np.asarray(pes); touched = np.asarray(touched)
    by_hand = np.asarray(by_hand); toy_moved = np.asarray(toy_moved)

    print(f"\n=== contact -> prediction error ({len(pes)} ticks) ===")
    print(f"  ticks with toy contact      : {touched.sum()} ({touched.mean()*100:.1f}%)")
    print(f"  ticks with HAND contact     : {by_hand.sum()} ({by_hand.mean()*100:.1f}%)")
    if touched.sum() == 0 or (~touched).sum() == 0:
        print("  !! 片方の群が空。ティック数を増やすか、おもちゃを近づける必要がある")
        env.close(); return

    a_pe, b_pe = pes[touched], pes[~touched]
    print(f"\n  pe when TOUCHING : mean={a_pe.mean():.5f}  median={np.median(a_pe):.5f}  n={len(a_pe)}")
    print(f"  pe when NOT      : mean={b_pe.mean():.5f}  median={np.median(b_pe):.5f}  n={len(b_pe)}")
    ratio = a_pe.mean() / max(b_pe.mean(), 1e-12)
    print(f"  ratio (touch/not): {ratio:.3f}x")
    # 効果量（Cohen's d）＝分散を考慮した差の大きさ
    sp = np.sqrt(((len(a_pe)-1)*a_pe.var(ddof=1) + (len(b_pe)-1)*b_pe.var(ddof=1))
                 / max(len(a_pe)+len(b_pe)-2, 1))
    d = (a_pe.mean() - b_pe.mean()) / max(sp, 1e-12)
    print(f"  Cohen's d        : {d:+.3f}  (|d|>0.2 small, >0.5 medium, >0.8 large)")

    mv = toy_moved > 1e-4
    if mv.sum() > 3 and (~mv).sum() > 3:
        print(f"\n  pe when toy MOVED: mean={pes[mv].mean():.5f} (n={mv.sum()})")
        print(f"  pe when toy still: mean={pes[~mv].mean():.5f} (n={(~mv).sum()})")

    print("\n=== verdict ===")
    if abs(d) > 0.2 and ratio > 1.05:
        print("  OK: 接触は予測誤差に現れている＝『学べる出来事』になっている")
        print("      → 学習を回す価値がある")
    else:
        print("  NG: 接触しても予測誤差がほぼ変わらない")
        print("      → おもちゃが太郎の感覚にほとんど影響していない")
        print("      → 手応え(質量/抵抗)を上げる・接触機会を増やす・視覚を入れる等の設計変更が要る")
    env.close()


if __name__ == "__main__":
    main()
