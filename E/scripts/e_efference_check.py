"""egomotion実装の前段：太郎ひとりだけで2つを検証する（親も外部の他者も使わない）。

【背景】egomotion割引（自己運動を差し引いて外の本当の動きを読む）を太郎の脳に実装する前に、
合成データで確認した「自己運動信号があれば読める」という前提が、**太郎自身の体で成り立つか**
を先に確かめる。ユーザー提案の2段階：

  ①指令 ↔ 前庭感覚：首に出した指令(遠心性コピー)通りに、実際の前庭感覚(角速度・加速度)が
    ついてくるか。ズレていれば「指令がそもそも信用できない」（VORで実際に見つかった問題＝
    利得目標1.03に対し実効0.55〜0.64で頭打ち＝トルク飽和、と同じ構造の可能性）。
  ②指令 → 視覚：同じ指令だけで、視界の変化量（コマ間の画素差）を予測できるか。

【なぜ物理版(親を動かす実験)を避けるか】
d_ego_leftright.py を目視したところ、太郎の頭は回転(yaw)、親は平行移動で、両者の見かけの
動きが幾何学的に対応していない（ユーザー指摘）。この実験は太郎ひとりの自己運動と自己の視界
だけを見るので、その不一致は原理的に起きない。

【判定】単純な小型ネット（train_eval, d_ego_synthetic.py と同じ構成）で予測させ、
「予測しない（平均だけ答える）」場合と比較する。

使い方: python e_efference_check.py [n_tick]
"""

# ⚠️★古い方式（2026-07-30 に整理）。新しい実験は `run/main.py` を通す。
#   【経緯】目標Eの実験スクリプトが118本あり、うち66本が**独立に環境を組み立てていた**。
#     そのため「学習は関節モード（90関節を独立に駆動＝逸脱リスト 逸脱5）、
#     測定とViewerは筋肉モード（拮抗筋2本/関節）」という**別の体で動く**事故が起きた
#     （ユーザーの目視「視線誘導反射の実験の時とは動きが全然違う」で発覚。
#      実測で動きが人間の新生児の約3.3倍速かった）。
#   【設計と移行計画】`E/docs/実行基盤_設計.md`
#   ⚠️このファイルは**記録として残す**（削除しない方針）。
#     中の測り方は再利用できるので、プラグインへ移すときの元にする。
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
import mujoco  # noqa: E402
import d_c5_motor_quality as mq  # noqa: E402

torch.set_num_threads(4)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HEAD_ACTS = ("head_swivel", "head_tilt", "head_tilt_side")   # 首3軸（遠心性コピーの対象）


def collect(n_tick, seed=0, noise_mode="white", beta=0.8, w_mean=1.0):
    """babble方策で動かし、毎tickの (首への指令, 前庭感覚, 視界変化量) を記録する。
    noise_mode="colored" で色付きノイズ(colored_noise.py)に切替（既定は従来の白色雑音）。
    w_mean<1.0 でmean（Cで学習した閉ループ制御の出力）の影響を下げる（既定1.0=従来と同一）。"""
    env, brain, fusion, emb_proj, cereb, n_act = mq.build("off", age=0)
    policy = mq.make_policy(brain, fusion, emb_proj, cereb, n_act, babble=True,
                             noise_mode=noise_mode, beta=beta, noise_seed=seed, w_mean=w_mean)
    raw = env.unwrapped
    m = raw.model
    head_idx = [i for i in range(m.nu) if any(k in m.actuator(i).name for k in HEAD_ACTS)]
    print(f"  首アクチュエータ: {[m.actuator(i).name for i in head_idx]}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    obs, _ = env.reset(seed=seed)
    hidden = brain.init_motor_hidden()
    prev_a = torch.zeros(n_act)

    cmds, vests, dvis = [], [], []
    prev_eye = None
    for t in range(n_tick):
        a, hidden = policy(obs, prev_a, hidden)
        ctrl = mq.rescale_action(a, env.action_space)
        prev_a = a
        for k in range(mq.K):
            obs, r, term, trunc, info = env.step(ctrl)
            if term or trunc:
                break
        cmds.append(np.asarray(ctrl)[head_idx].copy())
        vests.append(np.asarray(obs["vestibular"], dtype=np.float32).copy())
        eye = np.asarray(obs.get("eye_left"))
        if eye is not None:
            eye_small = eye.astype(np.float32)
            if prev_eye is not None:
                dvis.append(float(np.abs(eye_small - prev_eye).mean()))
            else:
                dvis.append(0.0)
            prev_eye = eye_small
        else:
            dvis.append(0.0)
        if term or trunc:
            obs, _ = env.reset()
            hidden = brain.init_motor_hidden()
            prev_a = torch.zeros(n_act)
            prev_eye = None
    env.close()
    return (np.asarray(cmds, dtype=np.float32), np.asarray(vests, dtype=np.float32),
            np.asarray(dvis, dtype=np.float32))


def standardize(a, tr_idx):
    mean = a[tr_idx].mean(0, keepdims=True)
    std = a[tr_idx].std(0, keepdims=True) + 1e-6
    return (a - mean) / std, mean, std


def train_eval_reg(X, y, tr, te, tag, steps=2000):
    """回帰。R^2（平均だけ答える場合との比較）で報告する。"""
    net = nn.Sequential(nn.Linear(X.shape[1], 32), nn.SiLU(), nn.LayerNorm(32), nn.Linear(32, y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.MSELoss()
    Xtr = torch.tensor(X[tr], dtype=torch.float32)
    ytr = torch.tensor(y[tr], dtype=torch.float32)
    for _ in range(steps):
        idx = torch.randperm(len(Xtr))[: min(128, len(Xtr))]
        loss = lossf(net(Xtr[idx]), ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = net(torch.tensor(X[te], dtype=torch.float32)).numpy()
    yte = y[te]
    ss_res = float(((yte - pred) ** 2).sum())
    ss_tot = float(((yte - yte.mean(0, keepdims=True)) ** 2).sum()) + 1e-9
    r2 = 1.0 - ss_res / ss_tot
    print(f"[{tag:32s}] R^2 = {r2:6.3f}（1.0=完全予測, 0.0=平均を答えるのと同じ, 負=それより悪い）")
    return r2


def split(n, frac=0.8, seed=1):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return perm[: int(n * frac)], perm[int(n * frac):]


def lag1_autocorr(cmd):
    """首指令の大きさ(各tickのL2ノルム)のlag-1自己相関。0=白色雑音、1に近いほど時間的に粘る。"""
    mag = np.linalg.norm(cmd, axis=1)
    if mag.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(mag[:-1], mag[1:])[0, 1])


def main():
    n_tick = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    noise_mode = os.environ.get("E_NOISE_MODE", "white")
    beta = float(os.environ.get("E_NOISE_BETA", "0.8"))
    w_mean = float(os.environ.get("E_W_MEAN", "1.0"))
    seed = int(os.environ.get("E_SEED", "0"))
    tag = f"{noise_mode}" + (f"_b{beta}" if noise_mode == "colored" else "") + f"_wm{w_mean}_seed{seed}"
    print(f"=== 太郎ひとりで検証：指令→前庭感覚、指令→視覚（{n_tick}tick, ノイズ={tag}）===\n")
    cmd, vest, dvis = collect(n_tick, seed=seed, noise_mode=noise_mode, beta=beta, w_mean=w_mean)
    print(f"  収集完了: 指令{cmd.shape} 前庭感覚{vest.shape} 視界変化{dvis.shape}")
    np.savez(os.path.join(_HERE, os.pardir, "logs", f"efference_raw_{tag}.npz"),
             cmd=cmd, vest=vest, dvis=dvis)  # 目視分析用に生データを保存
    print(f"  首指令の標準偏差: {np.round(cmd.std(0), 4)}")
    print(f"  前庭感覚の標準偏差: {np.round(vest.std(0), 4)}")
    print(f"  視界変化量: mean={dvis.mean():.5f} std={dvis.std():.5f}")
    ac = lag1_autocorr(cmd)
    print(f"  首指令(大きさ)のlag-1自己相関: {ac:.3f}"
          f"（0=白色雑音／今回の目標: β={beta}相当に応じて0より大きい）\n")

    tr, te = split(n_tick)
    Xc, *_ = standardize(cmd, tr)

    print("--- ①指令 → 前庭感覚（指令通りに体が動いているか）---")
    Yv, *_ = standardize(vest, tr)
    r2_v = train_eval_reg(Xc, Yv, tr, te, "指令→前庭感覚")

    print("\n--- ②指令 → 視界の変化量 ---")
    Yd = dvis.reshape(-1, 1)
    r2_d_cmd = train_eval_reg(Xc, Yd, tr, te, "指令→視界変化")

    print("\n--- （参考）前庭感覚 → 視界の変化量 ---")
    r2_d_vest = train_eval_reg(Yv, Yd, tr, te, "前庭感覚→視界変化")

    print("\n=== 解釈 ===")
    if any(np.isnan(x) for x in (r2_v, r2_d_cmd, r2_d_vest)):
        print("⚠️ NaNが混じっている＝視覚が無効（E_TOY=1でToySupineEnvを使う必要がある）など"
              "収集そのものに不備がある。結果を解釈せず、まず収集を直すこと。")
        return
    if r2_v < 0.2:
        print("① 指令と前庭感覚の対応が弱い＝指令通りに体が動いていない（VORと同じ構造の疑い）")
    else:
        print("① 指令から前庭感覚がある程度予測できる＝指令は概ね信用できる")
    if r2_d_cmd < 0.1 and r2_d_vest > r2_d_cmd + 0.1:
        print("② 指令だけでは視界変化を読めないが前庭感覚を使えば読める＝①のズレが原因")
    elif r2_d_cmd < 0.1 and r2_d_vest < 0.1:
        print("② 前庭感覚を使っても視界変化を読めない＝自己運動と視界の関係がまだ学習しにくい"
              "（運動がもっと成熟するまで待つ根拠になりうる）")
    else:
        print("② 指令から視界変化がある程度読める＝今の運動性喃語の段階でも土台がある")


if __name__ == "__main__":
    main()
