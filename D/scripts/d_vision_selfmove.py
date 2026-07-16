"""
★視覚の頑健性チェック：太郎自身が動いても、養育者の動きを読めるか。

【なぜ・2026-07-16】今日の門番①〜③は全て**太郎が完全に静止**（a=0）していた。だが本物の
太郎は手足を動かす。触覚に見切りをつけた理由は「相手の信号が床(82.5%)と自己(13.6%)に埋もれ、
相手は3.9%」だった。＝**触覚で踏んだ"自分に埋もれる"罠の視覚版**が起きないかを確かめる。
太郎が手を動かせば自分の手が視界に入りうるし、頭が揺れれば目の向きも変わる。それでも
「養育者が近づく/遠ざかる」を読めるか。

【設計＝門番②(近づく/遠ざかる)を、太郎の行動だけ変えて2条件で比較】
  ①静止：a=0（今日の門番と同じ。基準）
  ②自己運動：a=ランダムな探索行動（手足がバタつく＝自分の手が視界に入りうる）
両方で「近づく/遠ざかる」をシャッフル対照つきで測る。②が①並に読めれば、視覚は自己運動に
埋もれない＝触覚の罠は視覚では起きない。②で落ちれば、自分の手が邪魔している＝対策が要る
（手を視界から外す・養育者の色を変える等、直せる問題）。

【あわせて、判定器に渡す実際のクリップ(4コマ)を動画に保存】＝人間が中身を目で確認できるよう。

使い方: python d_vision_selfmove.py [n_clip]
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
import cv2

torch.set_num_threads(4)

import gymnasium as gym
import mimoEnv  # noqa
from gymnasium.envs.registration import register

from d1_carer_vision_env import CarerVisionEnv, lean_vision_params
from d_supine_env import infant_touch_params

register(id="CarerVisSelf-v0", entry_point="d1_carer_vision_env:CarerVisionEnv", max_episode_steps=100000)

RES = 32
FACE_X = 0.37
K = 4
Z_MID_LO, Z_MID_HI = 0.34, 0.44
SPAN_LO, SPAN_HI = 0.04, 0.10


def collect(self_move, n_clip, save_clips=0):
    """近づく/遠ざかるクリップを集める。self_move=Trueなら太郎はランダムに手足を動かす。

    save_clips>0 のとき、最初の数クリップを動画用のフレーム(拡大)として貯めて返す。
    """
    env = gym.make("CarerVisSelf-v0", vision_params=lean_vision_params(RES),
                   touch_params=infant_touch_params(2.0))
    env.reset(seed=0)
    u = env.unwrapped
    na = u.action_space.shape[0]
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    X, Y = [], []
    clip_frames = []                       # 見せる用（拡大した生画像）
    for i in range(n_clip):
        mid = float(rng.uniform(Z_MID_LO, Z_MID_HI)); span = float(rng.uniform(SPAN_LO, SPAN_HI))
        approach = bool(rng.integers(0, 2))
        zs = (np.linspace(mid + span, mid - span, K) if approach
              else np.linspace(mid - span, mid + span, K))
        frames = []
        keep_big = (save_clips and len(clip_frames) < save_clips)
        big_this = []
        for t in range(K):
            for _ in range(25):
                # ★ここが条件差：静止(a=0) か、探索的にバタつく(a=正しい範囲のランダム行動)
                ctrl = env.action_space.sample() if self_move else np.zeros(na)
                u.set_hand_target([FACE_X, 0.0, zs[t]])
                obs, r, te, tr, info = env.step(ctrl)
            frames.append(obs["eye_left"].astype(np.float32).reshape(-1) / 255.0)
            if keep_big:
                big_this.append(cv2.resize(obs["eye_left"], (RES * 4, RES * 4),
                                           interpolation=cv2.INTER_NEAREST))
        X.append(np.concatenate(frames)); Y.append(int(approach))
        if keep_big:
            clip_frames.append((approach, big_this))
        if (i + 1) % 40 == 0:
            print(f"  クリップ {i+1}/{n_clip}", flush=True)
    env.close()
    return np.stack(X), np.asarray(Y), clip_frames


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
    print(f"[{tag:30s}] 正解率 {acc*100:5.1f}%")
    return acc


def evaluate(X, Y, label):
    n = len(X); rng = np.random.default_rng(1)
    perm = rng.permutation(n); tr = perm[:int(n * 0.8)]; te = perm[int(n * 0.8):]
    acc = train_eval(X, Y, tr, te, f"{label}：近づく/遠ざかる")
    Xr = X.reshape(n, K, -1).copy(); rsh = np.random.default_rng(7)
    for i in range(n):
        rsh.shuffle(Xr[i])
    acc_sh = train_eval(Xr.reshape(n, -1), Y, tr, te, f"{label}：順番バラバラ")
    return acc, acc_sh


def save_clip_video(clips, out):
    """判定器に渡す実際のクリップ(4コマ)を、間に区切りを入れて1本の動画にする。"""
    frames = []
    for approach, imgs in clips:
        for j, im in enumerate(imgs):
            f = im.copy()
            txt = ("chikaduku" if approach else "toozakaru") + f" {j+1}/{K}"
            cv2.putText(f, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            for _ in range(8):                       # 各コマを長めに表示
                frames.append(f)
        sep = np.zeros_like(imgs[0])
        for _ in range(6):
            frames.append(sep)
    h, w, _ = frames[0].shape
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 12, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()


def main():
    n_clip = int(sys.argv[1]) if len(sys.argv) > 1 else 240
    print("=== ★視覚の頑健性：太郎自身が動いても養育者の動きを読めるか ===")
    print(f"門番②(近づく/遠ざかる)を、太郎の行動だけ変えて2条件で比較。{n_clip}クリップずつ。\n")

    print("--- ①太郎は静止（基準・今日の門番と同じ）---")
    Xs, Ys, clips = collect(False, n_clip, save_clips=6)
    a1, s1 = evaluate(Xs, Ys, "静止")

    print("\n--- ②太郎は手足を動かす（自己運動）---")
    Xm, Ym, _ = collect(True, n_clip)
    a2, s2 = evaluate(Xm, Ym, "自己運動")

    # 判定器に渡す実際のクリップを動画に保存（静止条件の6クリップ）
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                       "logs", "video", "quiz_clips.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_clip_video(clips, out)

    print("\n=== 判定 ===")
    print(f"{'条件':10s} {'本題':>8s} {'順番バラバラ':>12s} {'差(順序分)':>10s}")
    print(f"{'①静止':10s} {a1*100:7.1f}% {s1*100:11.1f}% {(a1-s1)*100:+9.1f}pt")
    print(f"{'②自己運動':10s} {a2*100:7.1f}% {s2*100:11.1f}% {(a2-s2)*100:+9.1f}pt")
    if a2 > 0.80 and (a2 - s2) > 0.15:
        print("\n→ ★**太郎が動いても読める**。自分の手に埋もれない＝触覚の罠(相手が3.9%)は視覚では起きない。")
    elif a2 > 0.65:
        print(f"\n→ 太郎が動くと弱まる（静止{a1*100:.0f}%→自己運動{a2*100:.0f}%）。自分の手が一部邪魔。対策検討。")
    else:
        print(f"\n→ 太郎が動くと読めない（{a2*100:.0f}%）。自分の手が視界を占領＝対策が要る（手を外す・色を変える等）。")
    print(f"\n判定器に渡す実際のクリップ動画: {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
