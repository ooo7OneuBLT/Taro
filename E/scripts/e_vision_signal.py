"""【E1の関門②】視覚を予測対象に入れたとき、それは**信号として存在するか**を測る。

【なぜ先に測るか＝前回この確認を飛ばして失敗した】
おもちゃ案では「触れば反力が返る→予測誤差が動く→progress報酬が出る」と考えて設計したが、
実際に測ると **621次元の中で効果量 d=-0.005＝まったく動かなかった**（希釈）。
学習を回す前にこれを測っていれば、無駄な設計をせずに済んだ。同じ失敗を繰り返さないため、
**視覚についても学習前に「信号があるか」を確かめる**。

【測ること】
 (1) **視覚の予測誤差は固有感覚と同じくらいのスケールか**（＝MSEに埋もれないか）
 (2) **手が視野に入っているtickで、視覚の予測誤差は変わるか**
     手は自分の運動と対応して動くので、見えているときは「予測しがいがある」はず。
     ⚠️ここで差が出なければ、太郎にとって手は他の背景と区別がつかない＝創発の材料が無い。
 (3) **視覚が変化したtickで予測誤差が動くか**（そもそも視覚の変化を検出できているか）

【判定】
 (2)で有意差（|d|>0.2）が出れば「手を見ることは学べる出来事」＝学習を回す価値がある。
 出なければ、視覚64次元への圧縮で手の情報が失われている可能性＝設計変更が要る。

使い方: python e_vision_signal.py [n_ticks]
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn

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
import e_target as et  # noqa: E402
import e_toy_env as te  # noqa: E402
from e_hand_in_view import eye_angles, EYES, HANDS  # noqa: E402


def cohens_d(a, b):
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    sp = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1))
                 / max(len(a)+len(b)-2, 1))
    return float((a.mean() - b.mean()) / max(sp, 1e-12))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    torch.manual_seed(seed); np.random.seed(seed)
    env, brain, fusion, emb_proj, cereb, n_act = mq.build("off", age=0)
    policy = mq.make_policy(brain, fusion, emb_proj, cereb, n_act, babble=True)
    raw = env.unwrapped
    m, d = raw.model, raw.data
    half_fov = te.VISION_FOVY / 2.0

    obs, _ = env.reset(seed=seed)
    frozen = et.make_frozen_fusion(fusion, touch_dim=0, vision_res=te.VISION_RES)
    target = et.PredictionTarget(frozen, use_vision=True)
    tgt0 = target(obs)
    prop_dim = int(np.asarray(obs["observation"]).shape[0])
    print(f"\n予測対象: {target.describe()}  → 全{tgt0.shape[0]}次元"
          f"（固有感覚{prop_dim} + 視覚{tgt0.shape[0]-prop_dim}）")

    # 予測ヘッド。出力を新しい次元に合わせて作り直す（C5の重みは形が違うので読めない＝新規）
    nat_head = nn.Sequential(nn.Linear(brain.latent_dim + n_act, 128), nn.SiLU(),
                             nn.LayerNorm(128), nn.Linear(128, tgt0.shape[0]))

    hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)
    rec = dict(pe_all=[], pe_prop=[], pe_vis=[], hand=[], visdiff=[])
    prev_vis = None

    for t in range(n):
        sv = fusion.encode(obs); cf = fusion.encode(obs).detach()
        cur = target(obs)
        a, hidden = policy(obs, prev_a, hidden)
        emb = emb_proj(torch.cat([sv, prev_a], dim=-1)).unsqueeze(0).unsqueeze(0)
        out, _ = brain.motor_gru(emb, hidden)
        z, _, _ = brain.pc_latent.infer(hidden[-1, 0], out[0, -1], cf)
        with torch.no_grad():
            pred = cur + nat_head(torch.cat([z.detach(), a], dim=-1))
        ctrl = mq.rescale_action(a, env.action_space); prev_a = a
        for k in range(mq.K):
            obs, r, term, trunc, info = env.step(ctrl)
            if term or trunc:
                break
        nxt = target(obs)
        with torch.no_grad():
            err = (pred - nxt) ** 2
        rec["pe_all"].append(float(err.mean()))
        rec["pe_prop"].append(float(err[:prop_dim].mean()))
        rec["pe_vis"].append(float(err[prop_dim:].mean()))
        # 手が両目の視野内にあるか
        inview = False
        for h in HANDS:
            ang = eye_angles(m, d, np.array(d.body(h).xpos, dtype=float))
            if all(ang[c] <= half_fov for c in EYES):
                inview = True
                break
        rec["hand"].append(inview)
        v = np.asarray(raw.get_vision_obs()["eye_left"], dtype=float)
        rec["visdiff"].append(0.0 if prev_vis is None
                              else float(np.abs(v - prev_vis).mean()))
        prev_vis = v
        if term or trunc:
            obs, _ = env.reset()
            hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)
            prev_vis = None

    R = {k: np.asarray(v, dtype=float) for k, v in rec.items()}
    hand = R["hand"].astype(bool)
    print(f"\n--- {n} tick ---")
    print("(1) スケール比較（視覚が固有感覚に埋もれていないか）")
    print(f"    固有感覚の予測誤差 mean {R['pe_prop'].mean():.5f}")
    print(f"    視覚の予測誤差     mean {R['pe_vis'].mean():.5f}"
          f"   （比 {R['pe_vis'].mean()/max(R['pe_prop'].mean(),1e-12):.2f}倍）")
    nv = R['pe_vis'].mean() * (len(R['pe_vis']) and 1)
    print(f"    → MSE全体への寄与は次元数比のまま: 視覚 {(tgt0.shape[0]-prop_dim)}/{tgt0.shape[0]}"
          f" = {100*(tgt0.shape[0]-prop_dim)/tgt0.shape[0]:.1f}%")

    print("\n(2) ★手が視野に入っているとき、視覚の予測誤差は変わるか")
    print(f"    手が視野内: {hand.sum()}/{len(hand)} tick ({hand.mean()*100:.1f}%)")
    if hand.sum() >= 3 and (~hand).sum() >= 3:
        for key, name in (("pe_vis", "視覚の予測誤差"), ("pe_prop", "固有感覚の予測誤差")):
            a, b = R[key][hand], R[key][~hand]
            print(f"    {name}: 視野内 {a.mean():.5f} / 外 {b.mean():.5f}"
                  f"  比 {a.mean()/max(b.mean(),1e-12):.3f}  d={cohens_d(a,b):+.3f}")
    else:
        print("    ⚠️サンプル不足（手が視野に入るtickが少なすぎる）＝この時点で判定不能")

    print("\n(3) 視覚が変化したtickで予測誤差が動くか")
    mv = R["visdiff"] > np.median(R["visdiff"])
    if mv.sum() >= 3 and (~mv).sum() >= 3:
        a, b = R["pe_vis"][mv], R["pe_vis"][~mv]
        print(f"    視覚の予測誤差: 変化大 {a.mean():.5f} / 小 {b.mean():.5f}"
              f"  d={cohens_d(a,b):+.3f}")

    print("\n=== 判定 ===")
    if hand.sum() >= 3 and (~hand).sum() >= 3:
        dd = cohens_d(R["pe_vis"][hand], R["pe_vis"][~hand])
        if abs(dd) > 0.2:
            print("  OK: 手が視野にあるかどうかが視覚の予測誤差に現れている")
            print("      → 「手を見る」は学べる出来事＝学習を回す価値がある")
        else:
            print("  NG: 手が見えていてもいなくても視覚の予測誤差が変わらない")
            print("      → 64次元への圧縮で手の情報が失われている疑い＝設計変更が要る")
    else:
        print("  判定不能: 手が視野に入るtickが少なすぎる（ベースライン0.7%）")
        print("      → tick数を増やすか、手が視野に入りやすい条件で測り直す必要がある")
    env.close()


if __name__ == "__main__":
    main()
