# -*- coding: utf-8 -*-
"""egomotion割引を「画像の連続」を直接渡して確かめ直す（2026-08-02）。

【なぜ作ったか】
d_ego_synthetic.py は、赤丸のx座標を先に抽出（extract_x）してから判定器に渡している。
これは「視覚処理を人間が代わりにやってあげている」ことであり、
ユーザーの指摘「映像を渡さないと意味なくない？」はそのとおり。

2026-07-17 の記録では、生画像を渡した版は失敗している：
    ②画像だけ 81.2% に対し、④画像＋自己運動信号 52.5%（混ぜたのに悪化した）
その原因は「生画像19万6608次元に、わずか4次元の自己運動信号が埋もれる」と
解釈されたが、**その解釈は検証されていない**。

そこで、原因が本当に次元数の不均衡なのかを切り分ける。

【比べる4条件】いずれも「見ている側が動く」設定で、②と④だけを測る
    A  生画像そのまま                     ＝ 2026-07-17 の再現
    B  生画像を標準化する                  ＝ スケールを揃えるだけ
    C  自己運動信号を画像と同じ次元数まで複製  ＝ 次元数を揃える
    D  画像を小さくする（32x32）           ＝ 画像側の次元を落とす

【判定】④ が ② を上回れば「画像の連続でも egomotion 割引ができる」と言える。
        どの条件で上回るかで、失敗の原因が分かる。
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d_ego_synthetic as ego                                    # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
torch.set_num_threads(4)


def collect_raw(n_clip, seed=0, res=None):
    """生画像のまま集める。res を指すと、その大きさに縮めてから渡す。"""
    import cv2
    rng = np.random.default_rng(seed)
    imgs, bodies, Y = [], [], []
    for _ in range(n_clip):
        im, bd, true_dir, _ = ego.make_clip(rng, (ego.SELF_LO, ego.SELF_HI))
        if res is not None and res != ego.RES:
            im = np.stack([cv2.resize(im[t], (res, res),
                                      interpolation=cv2.INTER_AREA) for t in range(ego.K)])
        imgs.append(im.astype(np.float32).reshape(ego.K, -1) / 255.0)
        bodies.append(bd)
        Y.append(int(true_dir > 0))
    return dict(img=np.stack(imgs), body=np.stack(bodies), y=np.asarray(Y))


def make_X(data, use_body, tr_idx, normalize_img=False, body_repeat=1):
    """判定器に渡す行列を組む。

    normalize_img  生画像を訓練集合の平均・分散で標準化するか
    body_repeat    自己運動信号を何回複製するか（次元数を揃えるため）
    """
    img = data["img"].astype(np.float32)                       # (n, K, D)
    if normalize_img:
        flat = img[tr_idx].reshape(-1, img.shape[-1])
        mean = flat.mean(0, keepdims=True)
        std = flat.std(0, keepdims=True) + 1e-6
        img = (img - mean) / std
    mats = [img]
    if use_body:
        body = data["body"].astype(np.float32)[:, :, None]     # (n, K, 1)
        flat = body[tr_idx].reshape(-1, 1)
        body = (body - flat.mean()) / (flat.std() + 1e-6)
        if body_repeat > 1:
            body = np.repeat(body, body_repeat, axis=2)
        mats.append(body)
    X = np.concatenate(mats, axis=2)
    return X.reshape(len(data["y"]), -1)


def run(n_clip=400, steps=3000):
    print(f"=== 画像の連続を直接渡す（n={n_clip}）===\n")
    print("見ている側は常に動く設定。②画像だけ と ④画像＋自分の動き を比べる。")
    print("④ が ② を上回れば、画像のままでも引き算ができたことになる。\n")

    tr, te = ego.split(n_clip)

    conds = [
        ("A  生画像そのまま（2026-07-17の再現）", dict(res=None, norm=False, rep=1)),
        ("B  生画像を標準化", dict(res=None, norm=True, rep=1)),
        ("C  自分の動きを画像と同じ次元まで複製", dict(res=None, norm=True, rep=None)),
        ("D  画像を32x32に縮める", dict(res=32, norm=True, rep=1)),
    ]

    print(f"{'条件':44s} {'②画像だけ':>10s} {'④画像＋動き':>12s} {'差':>8s}")
    print("-" * 80)
    results = []
    for name, c in conds:
        data = collect_raw(n_clip, seed=0, res=c["res"])
        dim = data["img"].shape[-1]
        rep = dim if c["rep"] is None else c["rep"]

        X2 = make_X(data, use_body=False, tr_idx=tr, normalize_img=c["norm"])
        X4 = make_X(data, use_body=True, tr_idx=tr, normalize_img=c["norm"], body_repeat=rep)

        a2 = ego.train_eval(X2, data["y"], tr, te, f"{name} ②", steps=steps)
        a4 = ego.train_eval(X4, data["y"], tr, te, f"{name} ④", steps=steps)
        results.append((name, dim, rep, a2, a4))
        print(f"{name:44s} {a2*100:9.1f}% {a4*100:11.1f}% {(a4-a2)*100:+7.1f}pt")

    print("\n=== まとめ ===")
    for name, dim, rep, a2, a4 in results:
        judge = "上回った" if a4 - a2 > 0.05 else ("変わらず" if abs(a4 - a2) <= 0.05 else "下回った")
        print(f"  {name:44s} 画像{dim:6d}次元/コマ  自分の動き{rep:6d}次元  → {judge}")

    print("\n=== 読み方 ===")
    print("  Aで下回り、B〜Dで上回る  → 原因はスケールか次元数の不均衡。画像でもできる")
    print("  A〜Dすべて下回る        → 原因は別にある。画像からは読めていない")
    print("  Aでも上回る             → 2026-07-17 の失敗は再現しない（当時の別の要因）")


def run_full(n_clip=400, steps=3000, repeat=3, res=32):
    """①〜④の4条件を、映像をそのまま渡す形で測る（2026-08-02 追加）。

    条件D（32x32に縮めて標準化）を採用する。理由：条件Cの100%が最良だが
    自分の動きを49152次元に複製するため1本あたりが重すぎ、3回まわすと現実的でない。
    Dでも④は97.5%で、結論（映像のままでも引き算できる）は変わらない。

    項88の教訓に従い、1回の値を記録に残さず repeat 回まわして幅で出す。
    """
    print(f"=== ①〜④を映像のまま測る（n={n_clip}・{repeat}回・{res}x{res}）===\n")
    acc = {k: [] for k in "①②③④"}
    for r in range(repeat):
        tr, te = ego.split(n_clip)
        # ① 見ている側が動かない（自己運動0）＋映像だけ

        st = collect_raw_still(n_clip, seed=r, res=res)
        X1 = make_X(st, use_body=False, tr_idx=tr, normalize_img=True)
        acc["①"].append(ego.train_eval(X1, st["y"], tr, te, f"[{r+1}] ①動かない・映像", steps))
        # ②③④ 見ている側が動く
        mv = collect_raw(n_clip, seed=100 + r, res=res)
        X2 = make_X(mv, use_body=False, tr_idx=tr, normalize_img=True)
        acc["②"].append(ego.train_eval(X2, mv["y"], tr, te, f"[{r+1}] ②動く・映像", steps))
        X3 = make_X_bodyonly(mv, tr)
        acc["③"].append(ego.train_eval(X3, mv["y"], tr, te, f"[{r+1}] ③動く・動きだけ", steps))
        X4 = make_X(mv, use_body=True, tr_idx=tr, normalize_img=True)
        acc["④"].append(ego.train_eval(X4, mv["y"], tr, te, f"[{r+1}] ④動く・映像＋動き", steps))
        print()

    名 = {"①": "太郎は動かない・映像", "②": "太郎が動く・映像",
          "③": "太郎が動く・動きの情報だけ", "④": "太郎が動く・映像＋動き"}
    print("=== まとめ（3回の幅）===")
    for k in "①②③④":
        v = [x * 100 for x in acc[k]]
        print(f"  {k} {名[k]:26s} {min(v):5.1f}〜{max(v):5.1f}%   （{', '.join(f'{x:.1f}' for x in v)}）")


def collect_raw_still(n_clip, seed=0, res=None):
    """見ている側が動かない（自己運動=0）設定で、生画像のまま集める。"""
    import cv2
    rng = np.random.default_rng(seed)
    imgs, bodies, Y = [], [], []
    for _ in range(n_clip):
        im, bd, true_dir, _ = ego.make_clip(rng, (0.0, 0.0))
        if res is not None and res != ego.RES:
            im = np.stack([cv2.resize(im[t], (res, res),
                                      interpolation=cv2.INTER_AREA) for t in range(ego.K)])
        imgs.append(im.astype(np.float32).reshape(ego.K, -1) / 255.0)
        bodies.append(bd)
        Y.append(int(true_dir > 0))
    return dict(img=np.stack(imgs), body=np.stack(bodies), y=np.asarray(Y))


def make_X_bodyonly(data, tr_idx):
    """自分の動きの情報だけを渡す（映像を使わない）。"""
    body = data["body"].astype(np.float32)[:, :, None]
    flat = body[tr_idx].reshape(-1, 1)
    body = (body - flat.mean()) / (flat.std() + 1e-6)
    return body.reshape(len(data["y"]), -1)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    st = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    if len(sys.argv) > 3 and sys.argv[3] == "full":
        run_full(n, st)
    else:
        run(n, st)
