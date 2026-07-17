"""
★視覚の門番チェック：太郎の目の画像から「どっちが近いか」当てられるか。

【なぜ・2026-07-16】触覚だけの他者理解が壁に当たり(相手は触覚の3.9%・接触は不連続で
AUC0.71が天井)、視覚に切り替えた。描画バグも直した(眼球を生APIで直接描画)。
そこでまず**「太郎の目に映る画像に、相手を読み取る情報が入っているか」だけ**を確かめる。

【この実験の位置づけ＝門番。太郎の脳は使わない】
今週の型（触覚で確立）と同じ：太郎の本物の脳に挑ませる前に、**別の小さいネット**で
「そもそも解ける問題か」を先に確かめる。
  解ける → 画像に情報がある＝太郎に挑ませる価値がある（次段階＝本物の視覚エンコーダ接続）
  解けない → 画像に情報が無い or 与え方が悪い＝太郎に挑ませても無駄
これは太郎の能力テストではない。**問題が成立しているかのテスト**。

【一番易しい問い＝2枚のうちどっちが近いか（2択）】
・動きの情報は要らない（各画像は単独で「養育者がどれだけ大きく写っているか」だけ使う）。
・「近い/遠い」の境界を人が決めなくていい（大きい方が近い、で正解が自動で決まる）。
・カプセルは常に同じ大きさなので、画像上のサイズが距離の手がかりになる。
＝視覚が成立するなら、まずこれは通るはず（通らなければ描画か配線が壊れている）。

【健康診断（落とし穴チェック項6）】
・でたらめ入力→50%付近でなければ、カンニング経路がある（データの作り方が漏れている）
・易しすぎる正例（サイズの数字を直接渡す）→ほぼ100%でなければ、正解ラベルか採点が壊れている

使い方: python d_vision_depth.py [n_shot]
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

register(id="CarerVisDepth-v0", entry_point="d1_carer_vision_env:CarerVisionEnv", max_episode_steps=100000)

RES = 32          # 門番チェックは軽く。32x32で十分「どっちが大きいか」は見える
FACE_X = 0.37     # 顔の真上
Z_NEAR, Z_FAR = 0.28, 0.55   # 近い〜遠いの範囲（顔の上・視界に入る高さ）


def collect_shots(n_shot):
    """養育者をいろんな距離(高さz)に置き、その都度 太郎の左目の画像と本当の距離を記録。

    太郎は何もしない（無動作）。養育者だけをテレポートさせ、落ち着かせてから撮る。
    距離＝眼から養育者までの実距離（zだけでなく実測の3D距離）。
    """
    env = gym.make("CarerVisDepth-v0", vision_params=lean_vision_params(RES),
                   touch_params=infant_touch_params(2.0))
    obs, _ = env.reset(seed=0)
    u = env.unwrapped
    a = np.zeros(u.action_space.shape[0])
    rng = np.random.default_rng(0)

    imgs, dists, sizes = [], [], []
    for i in range(n_shot):
        z = float(rng.uniform(Z_NEAR, Z_FAR))
        y = float(rng.uniform(-0.06, 0.06))          # 少し左右にも散らす（サイズだけが手がかりになるよう位置は変える）
        for _ in range(40):                          # 落ち着かせる
            u.set_hand_target([FACE_X, y, z]); obs, r, te, tr, info = env.step(a)
        eye = u.data.body("left_eye").xpos
        hand = u.hand_pos
        d = float(np.linalg.norm(hand - eye))         # 眼から養育者までの本当の距離
        img = obs["eye_left"].astype(np.float32) / 255.0
        # 「サイズの数字」＝赤く写っている画素数（易しい正例の入力に使う）
        r_ = (obs["eye_left"][:, :, 0].astype(int)
              - (obs["eye_left"][:, :, 1].astype(int) + obs["eye_left"][:, :, 2].astype(int)) // 2)
        imgs.append(img.reshape(-1)); dists.append(d); sizes.append(float((r_ > 40).sum()))
        if (i + 1) % 40 == 0:
            print(f"  撮影 {i+1}/{n_shot}", flush=True)
    env.close()
    return np.stack(imgs), np.asarray(dists), np.asarray(sizes)


def make_pairs(n_img, n_pairs, rng, min_gap):
    """2枚組を作る。距離差が min_gap 未満の紛らわしい組は避ける（易しい問いから始める）。"""
    pairs = []
    while len(pairs) < n_pairs:
        i, j = rng.integers(0, n_img, 2)
        if i != j:
            pairs.append((i, j))
    return np.asarray(pairs)


def train_and_eval(X, pairs_tr, y_tr, pairs_te, y_te, steps=3000, tag=""):
    """2枚組を見せて「1枚目の方が近いか(1) 2枚目か(0)」を当てさせる。教師あり学習。

    正解が最初から分かっている（距離を記録済み）ので、方策勾配は不要。機械の答えと
    正解のズレを直接見て、まっすぐ重みを直す（BCE損失）。
    """
    din = X.shape[1]
    net = nn.Sequential(nn.Linear(din * 2, 128), nn.SiLU(), nn.LayerNorm(128), nn.Linear(128, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X, dtype=torch.float32)

    def batch(pairs):
        a = Xt[pairs[:, 0]]; b = Xt[pairs[:, 1]]
        return torch.cat([a, b], dim=1)

    Btr = batch(pairs_tr); ytr = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    n = len(Btr)
    for _ in range(steps):
        idx = torch.randperm(n)[:256]
        loss = lossf(net(Btr[idx]), ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = (net(batch(pairs_te)).squeeze(1) > 0).numpy().astype(int)
    acc = float((pred == y_te).mean())
    print(f"[{tag:34s}] 正解率 {acc*100:5.1f}%   （50%=当てずっぽう / 100%=完璧）")
    return acc


def main():
    n_shot = int(sys.argv[1]) if len(sys.argv) > 1 else 240
    print("=== ★視覚の門番チェック：画像から『どっちが近いか』読めるか ===")
    print(f"太郎の左目 {RES}x{RES}。養育者を顔の上・距離{Z_NEAR}〜{Z_FAR}mに置いて{n_shot}枚撮影。")
    print("太郎は無動作。太郎の脳は使わない（別の小さいネットで『問題が成立するか』を確かめる）。\n")

    X, dist, size = collect_shots(n_shot)
    # 撮影を「学習用」と「本番用」に分ける。**同じ写真が両方に入らないよう写真単位で分割**
    n = len(X); rng = np.random.default_rng(1)
    perm = rng.permutation(n); tr_idx = perm[:int(n * 0.8)]; te_idx = perm[int(n * 0.8):]
    print(f"\n撮影{n}枚（学習用{len(tr_idx)} / 本番用{len(te_idx)}・写真単位で分割）")
    print(f"距離の範囲: {dist.min():.3f}〜{dist.max():.3f}m  赤画素の範囲: {size.min():.0f}〜{size.max():.0f}\n")

    def pairs_from(idx, n_pairs):
        pr = make_pairs(len(idx), n_pairs, rng, 0)
        gi, gj = idx[pr[:, 0]], idx[pr[:, 1]]
        y = (dist[gi] < dist[gj]).astype(int)      # 正解＝1枚目が近ければ1
        return np.stack([gi, gj], 1), y

    ptr, ytr = pairs_from(tr_idx, 4000)
    pte, yte = pairs_from(te_idx, 1000)

    print("--- 健康診断 ---")
    # 正例A（採点そのものの確認）：入力に**本当の距離の数字**を渡す。正解も距離で決めているので
    # これが~100%にならなければ、採点かペアの作り方が壊れている（＝下の結果は全部無効）。
    D = (dist.reshape(-1, 1) - dist.mean()) / (dist.std() + 1e-9)
    train_and_eval(D, ptr, ytr, pte, yte, steps=1500, tag="正例A：本当の距離→近い方(採点確認)")
    # 正例B（手作り特徴の参考）：赤画素数。カプセルが狭い視野から外れると0になり距離の目安に
    # ならないので、低くても採点の異常ではない（特徴が弱いだけ）。参考値として出す。
    S = size.reshape(-1, 1) / max(size.max(), 1)
    train_and_eval(S, ptr, ytr, pte, yte, steps=1500, tag="参考B：赤画素数→近い方(弱い特徴)")
    # 負例：でたらめ入力。当てずっぽう(50%)付近でなければカンニング経路がある
    R = rng.standard_normal((n, 16)).astype(np.float32)
    train_and_eval(R, ptr, ytr, pte, yte, steps=1500, tag="負例：でたらめな数字→近い方")

    print("\n--- 本題：太郎の目の画像そのものから ---")
    acc = train_and_eval(X, ptr, ytr, pte, yte, tag="太郎の目の画像→どっちが近い")

    print("\n=== 判定 ===")
    if acc > 0.80:
        print(f"→ ★**画像から距離が読める**（{acc*100:.0f}%）。問題として成立している。")
        print("   ＝視覚は触覚と違い、相手の情報がちゃんと画像に入っている。")
        print("   次段階：太郎の本物の脳（Phase5の視覚エンコーダ）に繋ぐ価値がある。")
    elif acc > 0.65:
        print(f"→ 偶然は超える（{acc*100:.0f}%）が弱い。解像度({RES})やCNN化を検討。")
    else:
        print(f"→ 画像から距離が読めない（{acc*100:.0f}%）。描画・配線・与え方のどれかを疑う")
        print("   （健康診断が両方通っているなら、画像の与え方＝1列に潰す方式が悪い可能性）。")


if __name__ == "__main__":
    main()
