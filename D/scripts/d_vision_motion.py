"""
★視覚の門番チェック②：複数コマから「近づく最中か・遠ざかる最中か」読めるか。

【なぜ・2026-07-16】門番①（`d_vision_depth.py`）で「1枚の画像から距離が読める」(91.5%)を
確認した。次は**動きの向き**。ただしこれは1枚では原理的に解けない（写真1枚に速度の情報は
無い＝今日確立）。そこで**複数コマをまとめて渡す仕組みが機能するか**を、一番シンプルな2択
（近づく/遠ざかる）だけで確かめる。左右・速度は、これが通ってから別途追加する
（一度に測る新しいことを1つに絞る＝今週の型）。

【1枚では解けないように作る＝ここが肝】
近づくクリップと遠ざかるクリップで、**通る距離の範囲を重ねる**。各クリップは
ランダムな中心距離の周りを、ランダムな速さで動く。＝**どの1枚を取り出しても、近づく/遠ざかる
どちらのクリップから来たか区別がつかない**（大きさの分布が両者で同じ）。向きは
「大きくなっていく/小さくなっていく」という**コマ間の変化**にしか現れない。
＝単一コマの対照が~50%になるはず（ならなければ、絶対位置が漏れている）。

【健康診断】
・正例：本当の距離の数列(K個)を渡す → 差の符号で向きが分かる → ~100%でなければ採点が壊れ
・単一コマ対照：1枚だけ渡す → ~50%のはず（1枚では原理的に向きが解けない＝この課題の核心）
・でたらめ対照：乱数 → ~50%
・本題：Kコマの画像をまとめて渡す → 通れば「複数コマの仕組みは機能する」

使い方: python d_vision_motion.py [n_clip]
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "taro_core"))
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

register(id="CarerVisMotion-v0", entry_point="d1_carer_vision_env:CarerVisionEnv", max_episode_steps=100000)

RES = 32
FACE_X = 0.37
K = 4                    # 1クリップのコマ数
Z_MID_LO, Z_MID_HI = 0.34, 0.44   # クリップの中心距離の範囲（両クラス共通＝1枚では区別不能に）
SPAN_LO, SPAN_HI = 0.04, 0.10     # 中心からどれだけ動くか（速さのばらつき）


def record_clip(env, u, z_start, z_end):
    """養育者を z_start→z_end へ動かしながら、太郎の左目を K コマ記録。距離の数列も返す。"""
    a = np.zeros(u.action_space.shape[0])
    frames, zs = [], []
    for step in range(K):
        z = z_start + (z_end - z_start) * step / (K - 1)
        for _ in range(25):                       # 各コマの前に少し進める（動いている状態を作る）
            u.set_hand_target([FACE_X, 0.0, z]); obs, r, te, tr, info = env.step(a)
        eye = u.data.body("left_eye").xpos
        zs.append(float(np.linalg.norm(u.hand_pos - eye)))
        frames.append(obs["eye_left"].astype(np.float32).reshape(-1) / 255.0)
    return np.concatenate(frames), np.asarray(zs)


def collect(n_clip):
    env = gym.make("CarerVisMotion-v0", vision_params=lean_vision_params(RES),
                   touch_params=infant_touch_params(2.0))
    env.reset(seed=0)
    u = env.unwrapped
    rng = np.random.default_rng(0)
    X, Zs, Y = [], [], []
    for i in range(n_clip):
        mid = float(rng.uniform(Z_MID_LO, Z_MID_HI))
        span = float(rng.uniform(SPAN_LO, SPAN_HI))
        approach = bool(rng.integers(0, 2))
        # 近づく＝遠→近＝z大→z小 / 遠ざかる＝近→遠＝z小→z大。中心midは両者共通なので
        # どの1枚の距離も同じ分布から来る＝1枚では向きが分からない。
        if approach:
            xi, zi = record_clip(env, u, mid + span, mid - span)
        else:
            xi, zi = record_clip(env, u, mid - span, mid + span)
        X.append(xi); Zs.append(zi); Y.append(int(approach))
        if (i + 1) % 40 == 0:
            print(f"  クリップ {i+1}/{n_clip}", flush=True)
    env.close()
    return np.stack(X), np.stack(Zs), np.asarray(Y)


def train_eval(X, y, tr, te, tag, steps=3000):
    net = nn.Sequential(nn.Linear(X.shape[1], 128), nn.SiLU(), nn.LayerNorm(128), nn.Linear(128, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    Xtr = torch.tensor(X[tr], dtype=torch.float32); ytr = torch.tensor(y[tr], dtype=torch.float32).unsqueeze(1)
    for _ in range(steps):
        idx = torch.randperm(len(Xtr))[:128]
        loss = lossf(net(Xtr[idx]), ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = (net(torch.tensor(X[te], dtype=torch.float32)).squeeze(1) > 0).numpy().astype(int)
    acc = float((pred == y[te]).mean())
    print(f"[{tag:36s}] 正解率 {acc*100:5.1f}%   （50%=当てずっぽう / 100%=完璧）")
    return acc


def main():
    n_clip = int(sys.argv[1]) if len(sys.argv) > 1 else 240
    print("=== ★視覚の門番チェック②：複数コマから『近づく/遠ざかる』が読めるか ===")
    print(f"太郎の左目 {RES}x{RES} を {K}コマ/クリップ、計{n_clip}クリップ。太郎は無動作・脳は使わない。")
    print("※近づく/遠ざかるで通る距離の範囲を重ねてある＝1枚では向きが分からないよう設計\n")

    X, Zs, Y = collect(n_clip)
    n = len(X); rng = np.random.default_rng(1)
    perm = rng.permutation(n); tr = perm[:int(n * 0.8)]; te = perm[int(n * 0.8):]
    print(f"\nクリップ{n}本（学習{len(tr)} / 本番{len(te)}）  近づく{int(Y.sum())}本 / 遠ざかる{n-int(Y.sum())}本\n")

    print("--- 健康診断 ---")
    # 正例：本当の距離の数列(K個)。差の符号で向きが分かる → 採点が正しければ~100%
    Zn = (Zs - Zs.mean()) / (Zs.std() + 1e-9)
    train_eval(Zn, Y, tr, te, "正例：本当の距離の数列→向き(採点確認)", steps=1500)
    # 単一コマ対照：真ん中の1コマだけ。1枚に速度は無い → ~50%のはず（この課題の核心）
    one = X.reshape(n, K, -1)[:, K // 2, :]
    train_eval(one, Y, tr, te, "対照：1コマだけ→向き(1枚では無理)")
    # でたらめ対照
    R = rng.standard_normal((n, 32)).astype(np.float32)
    train_eval(R, Y, tr, te, "対照：でたらめ→向き", steps=1500)

    print("\n--- 本題：Kコマの画像をまとめて ---")
    acc = train_eval(X, Y, tr, te, "太郎の目の連続画像→近づく/遠ざかる")

    # ★シャッフル対照（決定的）：Kコマの**順番だけ**を1クリップ内でバラバラにする（中身は同じ）。
    # 時間の情報（大きくなっていく/小さくなっていく）を使っているなら、順番を壊すと成績が落ちる。
    # 落ちなければ、順番に依らない＝**1枚ずつの絶対位置のカンニング**で解いていた。
    # ＝「本題 − シャッフル」が、純粋に時間の順序から得た分。
    Xr = X.reshape(n, K, -1).copy()
    rsh = np.random.default_rng(7)
    for i in range(n):
        rsh.shuffle(Xr[i])                        # クリップごとにコマの順を入れ替え
    Xr = Xr.reshape(n, -1)
    acc_sh = train_eval(Xr, Y, tr, te, "対照：順番をバラバラにした連続画像")

    print("\n=== 判定 ===")
    print(f"本題（順番あり）        {acc*100:5.1f}%")
    print(f"順番をバラバラ          {acc_sh*100:5.1f}%   ← これが本題と同じなら、時間でなくカンニング")
    print(f"（差 {(acc-acc_sh)*100:+.1f} ポイント＝純粋に『順番』から得た分）")
    if acc - acc_sh > 0.15 and acc > 0.80:
        print("\n→ ★**順番を壊すと落ちた＝本当に時間の情報を使っている**。複数コマの仕組みは機能する。")
        print("   次：左右・速度を追加（写真と複数コマの仕組みは流用・正解の作り方を変えるだけ）。")
    elif acc - acc_sh <= 0.15:
        print("\n→ **順番を壊しても落ちない＝時間でなく、1枚ずつの絶対位置で解いていた**（カンニング）。")
        print("   クリップの作り方を直す必要がある（各コマの距離分布を両クラスで完全に揃える）。")
        print("   ＝この設計では『時間を読めた』とは言えない。要やり直し。")
    else:
        print(f"\n→ 弱い（本題{acc*100:.0f}%）。コマ数Kや解像度を検討。")


if __name__ == "__main__":
    main()
