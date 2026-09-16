# -*- coding: utf-8 -*-
"""物と背景を分けられるか（2026-09-03・下見）。

今の太郎は DINOv2 の CLS トークン（画像1枚の要約384個）だけを使っている。
DINOv2 は内部で画像を16×16のマス目に分け、**マスごとの特徴**も持っている。
それを取り出せば、新しいモデルを足さずに「物と背景」を分けられるかもしれない。

分け方：マスごとの特徴を2群に分ける（k平均・k=2）。周囲のマスが多い方を背景とする。
種類は出さない。「ここに何かがある」だけ。名前は太郎が学ぶ部分なので奪わない。

    .venv/Scripts/python.exe F/scripts/f59_object_segment_probe.py
出力: F/logs/F2-59_物の切り出し/図_物と背景の分離.png
"""
import os, sys, io, json, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
os.chdir(r"C:\claude\AI\Taro")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "run")
sys.path.insert(0, os.path.abspath("taro_core/src")); sys.path.insert(0, os.path.abspath("F/scripts"))
import numpy as np, torch, mujoco
from PIL import Image, ImageDraw, ImageFont
import f_gen_f49 as G
import f49_test as T

OUT = "F/logs/F2-59_物の切り出し"


def patch_features(model, img, mean, std, device="cpu"):
    """DINOv2 のマスごとの特徴を取り出す（CLSではなくパッチ）。"""
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
    x = (x - mean) / std
    with torch.no_grad():
        out = model.forward_features(x.to(device))
    p = out["x_norm_patchtokens"][0].cpu().numpy()      # (マス数, 384)
    n = int(np.sqrt(p.shape[0]))
    return p, n


def split_fg_bg(p, n, thresh=0.55):
    """視野の中心のマスを種にして、それに似たマスを「物」とする。
    人間も注視している対象が図（figure）になるので、中心を種にするのは不自然ではない。
    0=背景, 1=物。"""
    v = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-9)
    g = v.reshape(n, n, -1)
    c0, c1 = n // 2, n // 2
    seed_vec = g[c0 - 1:c0 + 1, c1 - 1:c1 + 1].reshape(-1, g.shape[-1]).mean(axis=0)
    seed_vec = seed_vec / (np.linalg.norm(seed_vec) + 1e-9)
    sim = (v @ seed_vec).reshape(n, n)
    # 似ている度の分布から自動でしきい値（中心の似方と外周の似方の中間）
    edge = np.concatenate([sim[0], sim[-1], sim[:, 0], sim[:, -1]])
    lo, hi = float(edge.mean()), float(sim.max())
    t = lo + (hi - lo) * (1.0 - thresh)
    return (sim >= t).astype(np.uint8)


def main():
    os.makedirs(OUT, exist_ok=True)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    stim = T.render_stimuli()
    seen = {}
    for it in stim:                                     # 8語を1枚ずつ（最初に出た角度）
        if it[0] not in seen:
            seen[it[0]] = it
    pick = list(seen.values())
    font = ImageFont.truetype("C:/Windows/Fonts/meiryo.ttc", 13)
    W = 116
    canvas = Image.new("RGB", (W * len(pick), 300), (255, 255, 255)); dr = ImageDraw.Draw(canvas)
    for i, (word, idx, unseen, yaw, img, name) in enumerate(pick):
        im = Image.fromarray(img).resize((224, 224))
        p, n = patch_features(model, np.asarray(im), mean, std)
        mask = split_fg_bg(p, n)
        big = np.kron(mask, np.ones((224 // n, 224 // n), dtype=np.uint8))
        big = np.pad(big, ((0, 224 - big.shape[0]), (0, 224 - big.shape[1])), mode="edge")
        over = np.asarray(im, dtype=np.float32).copy()
        over[..., 0] = np.where(big > 0, np.minimum(over[..., 0] + 70, 255), over[..., 0])
        over[..., 2] = np.where(big > 0, over[..., 2] * 0.6, over[..., 2])
        frac = float(mask.mean())
        x = i * W
        dr.text((x + 4, 4), word, fill=(0, 0, 120), font=font)
        canvas.paste(Image.fromarray(img).resize((108, 108)), (x + 4, 24))
        canvas.paste(Image.fromarray(over.astype(np.uint8)).resize((108, 108)), (x + 4, 140))
        dr.text((x + 4, 252), "物とされた割合", fill=(0, 0, 0), font=font)
        dr.text((x + 4, 270), "%.0f%%（マス %d×%d）" % (frac * 100, n, n), fill=(0, 0, 0), font=font)
        print("%-6s 物とされたマスの割合 %.0f%%（%d×%d マス）" % (word, frac * 100, n, n))
    dr.text((4, 122), "↓ 赤いところが「物」と判定された場所", fill=(120, 0, 0), font=font)
    p = OUT + "/図_物と背景の分離.png"; canvas.save(p); print("図:", p)


if __name__ == "__main__":
    main()
