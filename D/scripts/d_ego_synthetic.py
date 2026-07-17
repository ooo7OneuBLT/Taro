"""
★egomotion割引・合成版：MuJoCoの物理制約(視野外・衝突・描画バグ)を排除し、
「見かけの動き」と「自己運動信号」を直接作ってテストする。

【なぜ・2026-07-17】ユーザー提案：検証なのだから、赤丸が動く映像＋自己運動信号を直接
合成して送りつけ、判定器が「見かけ」と「自己運動」から「真の動き」を復元できるかを
見ればいい。物理エンジンで反転を起こそうとすると、視野外・自分の腕の写り込み・生存
バイアスが絡み合い続けた。本質（egomotion割引）だけを取り出してクリーンにテストする。

【合成の仕組み】各クリップ：
  真の動き(true_dir, true_amt)と自己運動(self_dir, self_amt)を"独立に"ランダム抽選
  見かけの位置(t) = 中心 + true_dir*true_amt*t/(K-1) - self_dir*self_amt*t/(K-1)
  （自己運動が真の動きと同じ向き・同等以上に強ければ、見かけは反転する）
  画像＝見かけの位置に赤丸を描いた4コマ。体の感覚＝self_dir*self_amt（と前庭覚相当の系列）。
  ラベル＝true_dir（真の動きの向き）。

【比較する5条件】（今日の本物の実験と同じ構造）
  ①静止相当（self_amt=0）画像だけ＝天井
  ②自己運動・画像だけ＝床（反転で埋もれるはず）
  ③自己運動・自己運動信号だけ＝~50%であるべき（真の動きと自己運動は独立だから）
  ④自己運動・画像＋自己運動信号＝本命
  ⑤④のシャッフル

使い方: python d_ego_synthetic.py [n_clip]
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import cv2

torch.set_num_threads(4)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RES = 128                     # 動きが目で見えるようキャンバスを拡大(64→128)
K = 4
CENTER = RES // 2
TRUE_LO, TRUE_HI = 15, 28     # 真の動きの大きさ(px相当の単位)。前回(6-14)は小さすぎ可視化が弱かった
SELF_LO, SELF_HI = 0, 35      # 自己運動の大きさ（0も含む＝時々ほぼ自己運動なし）
RADIUS = 9                    # 丸も大きく見やすく
VDIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "logs", "video"))


def render_frame(x, size=RES):
    """位置xに赤丸を描いた画像(size,size,3)。背景は青系(egomotion実験と同じ配色)。"""
    img = np.full((size, size, 3), (60, 90, 130), dtype=np.uint8)
    x = int(np.clip(x, RADIUS, size - RADIUS - 1))
    cv2.circle(img, (x, size // 2), RADIUS, (200, 40, 40), -1)
    return img


def make_clip(rng, self_amt_range=(SELF_LO, SELF_HI)):
    true_dir = float(rng.choice([-1.0, 1.0]))
    true_amt = float(rng.uniform(TRUE_LO, TRUE_HI))
    self_dir = float(rng.choice([-1.0, 1.0]))
    self_amt = float(rng.uniform(*self_amt_range))
    imgs, body = [], []
    for t in range(K):
        frac = t / (K - 1)
        apparent = CENTER + true_dir * true_amt * frac - self_dir * self_amt * frac
        imgs.append(render_frame(apparent))
        body.append(self_dir * self_amt * frac)   # 自己運動信号（遠心性コピー/前庭覚相当）
    return np.stack(imgs), np.array(body, dtype=np.float32), true_dir, (true_dir, true_amt, self_dir, self_amt)


def extract_x(img):
    """赤丸のx座標を画像から抽出（物理版の_red_pixels重心と同じ発想）。
    生画像49152〜196608次元をそのまま渡すと、わずか4次元の自己運動信号が埋もれて
    学習が不安定になる（実測：シャッフル後の方が高くなる逆転が発生）。位置を先に抽出し、
    「位置の系列＋自己運動信号」という小さな問題に変えることで学習を安定させる。
    """
    r = img[..., 0].astype(np.int16); g = img[..., 1].astype(np.int16); b = img[..., 2].astype(np.int16)
    mask = (r - (g + b) // 2) > 40
    xs = np.where(mask)[1]
    return float(xs.mean()) if len(xs) > 0 else float(CENTER)


def collect(n_clip, self_move, seed=0, raw_image=False):
    rng = np.random.default_rng(seed)
    imgs, bodies, Y = [], [], []
    self_range = (SELF_LO, SELF_HI) if self_move else (0.0, 0.0)
    for _ in range(n_clip):
        im, bd, true_dir, _ = make_clip(rng, self_range)
        if raw_image:
            imgs.append(im.astype(np.float32).reshape(K, -1) / 255.0)
        else:
            imgs.append(np.array([extract_x(im[t]) for t in range(K)], dtype=np.float32))
        bodies.append(bd)
        Y.append(int(true_dir > 0))
    return dict(img=np.stack(imgs), body=np.stack(bodies), y=np.asarray(Y))


def build_X(data, keys, tr_idx, shuffle=False):
    n = len(data["y"])
    mats = []
    for k in keys:
        a = data[k].astype(np.float32)
        if a.ndim == 2:
            a = a[:, :, None]
        if k != "img" or a.shape[-1] == 1:   # 生画像(H*W*3次元)以外は標準化。位置系列imgも対象。
            flat = a[tr_idx].reshape(-1, a.shape[-1])
            mean = flat.mean(0, keepdims=True); std = flat.std(0, keepdims=True) + 1e-6
            a = (a - mean) / std
        mats.append(a)
    X = np.concatenate(mats, axis=2)
    if shuffle:
        X = X.copy(); rsh = np.random.default_rng(7)
        for i in range(n):
            rsh.shuffle(X[i])
    return X.reshape(n, -1)


def train_eval(X, y, tr, te, tag, steps=3000):
    net = nn.Sequential(nn.Linear(X.shape[1], 64), nn.SiLU(), nn.LayerNorm(64), nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    Xtr = torch.tensor(X[tr], dtype=torch.float32); ytr = torch.tensor(y[tr], dtype=torch.float32).unsqueeze(1)
    for _ in range(steps):
        idx = torch.randperm(len(Xtr))[:128]
        loss = lossf(net(Xtr[idx]), ytr[idx]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = (net(torch.tensor(X[te], dtype=torch.float32)).squeeze(1) > 0).numpy().astype(int)
    acc = float((pred == y[te]).mean())
    print(f"[{tag:34s}] 正解率 {acc*100:5.1f}%", flush=True)
    return acc


def split(n):
    rng = np.random.default_rng(1)
    perm = rng.permutation(n)
    return perm[:int(n * 0.8)], perm[int(n * 0.8):]


def save_canonical_demo():
    """ユーザー提案そのものの実例：見かけは中央→左、だが自己運動信号は右に強い→真実は右。"""
    K_ = K
    true_dir, true_amt = 1.0, 16.0    # 真実：右へ
    self_dir, self_amt = 1.0, 35.0    # 自己運動：右向きに強く（見かけを反転させる）
    frames, body = [], []
    for t in range(K_):
        frac = t / (K_ - 1)
        apparent = CENTER + true_dir * true_amt * frac - self_dir * self_amt * frac
        img = render_frame(apparent)
        big = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
        cv2.putText(big, f"apparent(image only)", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(big, f"self-motion signal={self_dir*self_amt*frac:+.1f}", (6, 44),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(big, f"TRUE = RIGHT", (6, 236), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        frames.append(big); body.append(self_dir * self_amt * frac)
    os.makedirs(VDIR, exist_ok=True)
    out_mp4 = os.path.join(VDIR, "synthetic_canonical.mp4")
    vw = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), 2, (256, 256))
    for f in frames:
        for _ in range(15):
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    sheet = np.concatenate(frames, axis=1)
    out_png = os.path.join(VDIR, "synthetic_canonical_sheet.png")
    cv2.imwrite(out_png, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    return os.path.abspath(out_mp4), os.path.abspath(out_png)


def save_random_demos(n=6, seed=42):
    """ランダムな組み合わせのクリップをいくつか動画・シートで保存。"""
    rng = np.random.default_rng(seed)
    frames_all = []
    for i in range(n):
        im, bd, true_dir, (td, ta, sd, sa) = make_clip(rng, (SELF_LO, SELF_HI))
        for t in range(K):
            big = cv2.resize(im[t], (200, 200), interpolation=cv2.INTER_NEAREST)
            lab = "RIGHT" if true_dir > 0 else "LEFT"
            cv2.putText(big, f"clip{i} true={lab} self_sig={bd[t]:+.1f}", (4, 16),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            frames_all.append(big)
    os.makedirs(VDIR, exist_ok=True)
    out_mp4 = os.path.join(VDIR, "synthetic_random_demos.mp4")
    vw = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), 3, (200, 200))
    for f in frames_all:
        for _ in range(6):
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    rows = [np.concatenate(frames_all[i*K:(i+1)*K], axis=1) for i in range(n)]
    sheet = np.concatenate(rows, axis=0)
    out_png = os.path.join(VDIR, "synthetic_random_demos_sheet.png")
    cv2.imwrite(out_png, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    return os.path.abspath(out_mp4), os.path.abspath(out_png)


def main():
    n_clip = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    print("=== egomotion割引・合成版：見かけ＋自己運動信号を直接テスト ===\n")

    print("--- 具体例（ユーザー提案そのもの）を保存 ---")
    mp4, png = save_canonical_demo()
    print(f"  mp4: {mp4}")
    print(f"  png: {png}")

    print("\n--- ランダムな組み合わせを保存 ---")
    mp4r, pngr = save_random_demos()
    print(f"  mp4: {mp4r}")
    print(f"  png: {pngr}")

    print(f"\n--- 判定器テスト（n={n_clip}）---")
    st = collect(n_clip, self_move=False)
    mv = collect(n_clip, self_move=True)
    tr_s, te_s = split(n_clip); tr_m, te_m = split(n_clip)

    c_top = train_eval(build_X(st, ["img"], tr_s), st["y"], tr_s, te_s, "①静止相当：画像だけ(天井)")
    a_img = train_eval(build_X(mv, ["img"], tr_m), mv["y"], tr_m, te_m, "②自己運動：画像だけ(床)")
    a_body = train_eval(build_X(mv, ["body"], tr_m), mv["y"], tr_m, te_m, "③自己運動：自己運動信号だけ")
    a_all = train_eval(build_X(mv, ["img", "body"], tr_m), mv["y"], tr_m, te_m, "④自己運動：画像＋自己運動信号")
    a_sh = train_eval(build_X(mv, ["img", "body"], tr_m, shuffle=True), mv["y"], tr_m, te_m, "⑤④のシャッフル")

    print("\n=== まとめ ===")
    print(f"  ①天井（静止相当・画像だけ）    : {c_top*100:5.1f}%")
    print(f"  ②床  （自己運動・画像だけ）      : {a_img*100:5.1f}%")
    print(f"  ③検査（自己運動・自己運動信号）  : {a_body*100:5.1f}%  ←~50%であるべき")
    print(f"  ④本命（自己運動・画像＋信号）    : {a_all*100:5.1f}%")
    print(f"  ⑤④のシャッフル                : {a_sh*100:5.1f}%  ←下がるべき")

    print("\n=== 解釈 ===")
    if a_body > 0.62:
        print("警告：自己運動信号だけで当たる＝設計に漏れ（真の動きと自己運動が独立でない）")
    elif a_all - a_img > 0.15:
        print("★自己運動信号を足すと読める＝egomotionは体の感覚で割り引ける、と合成データで確認。")
    else:
        print("→ 信号を足しても改善せず。合成データでも埋もれる場合、根本的な再検討が必要。")


if __name__ == "__main__":
    main()
