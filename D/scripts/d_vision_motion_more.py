"""
★視覚の門番チェック③：左右の向き・速度も、複数コマから読めるか。

【なぜ・2026-07-16】門番②（`d_vision_motion.py`）で「近づく/遠ざかる」がシャッフル対照つきで
確定した（順番あり100%→バラバラ62.5%＝時間の順序が本物）。仕組みが機能すると分かったので、
**正解の作り方だけを変えて**左右・速度に広げる（写真・複数コマ・シャッフル対照は流用）。

【★予想＝2種類の"時間情報"は、シャッフル対照で挙動が違うはず】
- **左右**（右へ動く/左へ動く）＝「向き」の仲間。**順番が命**（右→左と左→右はコマの順序でしか
  区別できない）。＝シャッフルで落ちるはず。
- **速度**（速く近づく/ゆっくり近づく）＝**順番は関係ない**。「どれだけ広い範囲を動いたか」で
  決まり、それはコマをバラバラにしても分かる（最大コマと最小コマの差）。＝シャッフルしても
  落ちないはず。ただし**1コマでは分からない**（1枚に速度は無い）。
この違いが出れば、「向き＝順序情報／速度＝範囲情報」という別種の時間情報を両方取り出せている
と示せる。＝シャッフル対照は"順序を使ったか"の検査であって"時間を使ったか"の検査ではない、
という機微も確認できる（速度は時間情報だが順序は不要）。

使い方: python d_vision_motion_more.py [leftright|speed] [n_clip]
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "C"))
import paths
paths.setup_brain_path()
sys.path.insert(0, paths.MIMO_DIR)

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)

import gymnasium as gym
import mimoEnv  # noqa
from gymnasium.envs.registration import register

from d1_carer_vision_env import CarerVisionEnv, lean_vision_params
from d_supine_env import infant_touch_params

register(id="CarerVisMore-v0", entry_point="d1_carer_vision_env:CarerVisionEnv", max_episode_steps=100000)

RES = 32
FACE_X = 0.37
FACE_Z = 0.32          # 左右のときの固定の高さ（視界に入る）
K = 4


def record_clip(env, u, ys, zs):
    """養育者を (FACE_X, ys[t], zs[t]) の列に沿って動かし、太郎の左目を K コマ記録。実測距離も返す。"""
    a = np.zeros(u.action_space.shape[0])
    frames, dist = [], []
    for t in range(K):
        for _ in range(25):
            u.set_hand_target([FACE_X, ys[t], zs[t]]); obs, r, te, tr, info = env.step(a)
        eye = u.data.body("left_eye").xpos
        dist.append(float(np.linalg.norm(u.hand_pos - eye)))
        frames.append(obs["eye_left"].astype(np.float32).reshape(-1) / 255.0)
    return np.concatenate(frames), np.asarray(dist)


def collect(mode, n_clip):
    env = gym.make("CarerVisMore-v0", vision_params=lean_vision_params(RES),
                   touch_params=infant_touch_params(2.0))
    env.reset(seed=0)
    u = env.unwrapped
    rng = np.random.default_rng(0)
    X, Y = [], []
    for i in range(n_clip):
        label = int(rng.integers(0, 2))
        if mode == "leftright":
            # 左右に動く。中心yはランダム（1コマでは向きが分からないよう範囲を重ねる）。
            mid = float(rng.uniform(-0.04, 0.04)); span = float(rng.uniform(0.05, 0.10))
            zs = np.full(K, FACE_Z)
            ys = np.linspace(mid - span, mid + span, K) if label else np.linspace(mid + span, mid - span, K)
            # label=1: y増加=右へ（+y方向）/ label=0: y減少=左へ
        elif mode == "speed":
            # 両方とも「近づく」。label=1=速い（広く動く）/ label=0=遅い（狭く動く）。
            # 開始位置はランダム（絶対位置で速度がバレないよう）。
            span = (float(rng.uniform(0.11, 0.15)) if label else float(rng.uniform(0.02, 0.05)))
            start = float(rng.uniform(0.40, 0.46))
            zs = np.linspace(start, start - span, K)
            ys = np.zeros(K)
        else:
            raise ValueError(mode)
        xi, _ = record_clip(env, u, ys, zs)
        X.append(xi); Y.append(label)
        if (i + 1) % 40 == 0:
            print(f"  クリップ {i+1}/{n_clip}", flush=True)
    env.close()
    return np.stack(X), np.asarray(Y)


def train_eval(X, y, tr, te, tag, steps=3000):
    net = nn.Sequential(nn.Linear(X.shape[1], 128), nn.SiLU(), nn.LayerNorm(128), nn.Linear(128, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    Xtr = torch.tensor(X[tr], dtype=torch.float32); ytr = torch.tensor(y[tr], dtype=torch.float32).unsqueeze(1)
    for _ in range(steps):
        idx = torch.randperm(len(Xtr))[:128]
        loss = lossf(net(Xtr[idx]), ytr[idx]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = (net(torch.tensor(X[te], dtype=torch.float32)).squeeze(1) > 0).numpy().astype(int)
    acc = float((pred == y[te]).mean())
    print(f"[{tag:34s}] 正解率 {acc*100:5.1f}%")
    return acc


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "leftright"
    n_clip = int(sys.argv[2]) if len(sys.argv) > 2 else 240
    label_name = {"leftright": "右へ/左へ", "speed": "速い/遅い"}[mode]
    print(f"=== ★視覚の門番チェック③：{label_name} を複数コマから読めるか ===")
    print(f"太郎の左目 {RES}x{RES} を {K}コマ/クリップ、計{n_clip}クリップ。太郎は無動作・脳は使わない。\n")

    X, Y = collect(mode, n_clip)
    n = len(X); rng = np.random.default_rng(1)
    perm = rng.permutation(n); tr = perm[:int(n * 0.8)]; te = perm[int(n * 0.8):]
    print(f"\nクリップ{n}本（学習{len(tr)} / 本番{len(te)}）\n")

    print("--- 健康診断 ---")
    one = X.reshape(n, K, -1)[:, K // 2, :]
    train_eval(one, Y, tr, te, "対照：1コマだけ（1枚では無理のはず）")
    R = rng.standard_normal((n, 32)).astype(np.float32)
    train_eval(R, Y, tr, te, "対照：でたらめ", steps=1500)

    print("\n--- 本題と、順番シャッフル対照 ---")
    acc = train_eval(X, Y, tr, te, "本題：Kコマの画像")
    Xr = X.reshape(n, K, -1).copy(); rsh = np.random.default_rng(7)
    for i in range(n):
        rsh.shuffle(Xr[i])
    acc_sh = train_eval(Xr.reshape(n, -1), Y, tr, te, "対照：順番をバラバラ")

    print("\n=== 判定 ===")
    print(f"本題 {acc*100:.1f}%  /  順番バラバラ {acc_sh*100:.1f}%  （差 {(acc-acc_sh)*100:+.1f}pt）")
    if mode == "leftright":
        if acc > 0.8 and acc - acc_sh > 0.15:
            print("→ ★予想どおり：左右は**順番が命**。シャッフルで落ちた＝順序情報を使っている。")
        elif acc > 0.8:
            print("→ 読めたが、シャッフルでも落ちない＝1枚ずつの位置で解いた疑い（設計に漏れ）。")
        else:
            print("→ 読めない。")
    else:  # speed
        if acc > 0.8 and acc - acc_sh < 0.15:
            print("→ ★予想どおり：速度は**順番不要・範囲で決まる**。シャッフルでも落ちない＝正しい挙動。")
            print("   （1コマでは無理・複数コマなら順番なしでも読める＝順序でなく『広がり』の情報）")
        elif acc > 0.8:
            print("→ 読めたが、シャッフルで落ちた＝順序も使っている（想定より複雑）。")
        else:
            print("→ 読めない。")


if __name__ == "__main__":
    main()
