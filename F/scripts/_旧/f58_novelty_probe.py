# -*- coding: utf-8 -*-
"""「新しさ」が測れるか（2026-09-03・下見）。

太郎には馴化（今見ているものへの飽き・数値1つ）と IOR（さっき見た場所を避ける・地図）は
あるが、**記憶にない物に惹かれる**仕組みが無い。作る前に、
「知っている物」と「見たことのない物」で値が分かれるかを確かめる。

測り方：語彙が持つ見た目の記憶（`lexicon.proto`＝語ごとの384次元ベクトル・25件）と
今の見えを比べ、**一番近い記憶との遠さ**を新しさとする（1 − 最大コサイン類似度）。

    .venv/Scripts/python.exe F/scripts/f58_novelty_probe.py [モデル.pt]
出力: F/logs/F2-58_新しさ/図_新しさの分かれ方.png
"""
import os, sys, io, glob, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
os.chdir(r"C:\claude\AI\Taro")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "run")
sys.path.insert(0, os.path.abspath("taro_core/src")); sys.path.insert(0, os.path.abspath("F/scripts"))
import numpy as np, torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import f_gen_f49 as G
import f49_test as T
from senses.vision_backends import get_backend

OUT = "F/logs/F2-58_新しさ"
PHOTOS = sorted(glob.glob("F/assets/user_photos/converted/*.jpg"))


def novelty(vec, protos):
    """一番近い記憶との遠さ。0＝記憶とそっくり、1に近いほど新しい。"""
    v = np.asarray(vec, dtype=np.float64); v = v / (np.linalg.norm(v) + 1e-9)
    sims = [float(v @ (p / (np.linalg.norm(p) + 1e-9))) for p in protos]
    return 1.0 - max(sims), max(sims)


def main(model_path):
    os.makedirs(OUT, exist_ok=True)
    fp = "C:/Windows/Fonts/meiryo.ttc"
    font_manager.fontManager.addfont(fp)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
    st = torch.load(model_path, map_location="cpu", weights_only=False)
    protos = [np.asarray(v, dtype=np.float64) for v in st["lexicon"]["proto"].values()]
    print("見た目の記憶 %d 件（384次元）" % len(protos))

    be = get_backend({"backend": "dinov2_vits14", "fovea_px": 10 ** 9})
    groups = {}
    # ① 知っている語の物（初見の個体）＝ f49_test の48枚
    stim = T.render_stimuli()
    vals = []
    for word, idx, unseen, yaw, img, name in stim:     # render_stimuli の並び
        vec = np.asarray(be.encode(img, img), dtype=np.float32)
        n, s = novelty(vec, protos); vals.append(n)
    groups["知っている8語\n（初見の個体）"] = vals
    # ② 語彙に無い物＝ユーザーの写真（猫・人物・マグ）
    for p in PHOTOS:
        im = Image.open(p).convert("RGB"); w, h = im.size; s0 = min(w, h)
        im = im.crop(((w - s0) // 2, (h - s0) // 2, (w - s0) // 2 + s0, (h - s0) // 2 + s0)).resize((224, 224))
        arr = np.asarray(im, dtype=np.uint8)
        vec = np.asarray(be.encode(arr, arr), dtype=np.float32)
        n, s = novelty(vec, protos)
        groups["写真：" + os.path.splitext(os.path.basename(p))[0]] = [n]
        print("  写真 %-12s 新しさ %.3f（一番近い記憶との似方 %.3f）" % (os.path.basename(p), n, s))
    m = float(np.mean(groups["知っている8語\n（初見の個体）"]))
    sd = float(np.std(groups["知っている8語\n（初見の個体）"]))
    print("知っている8語（初見個体48枚）：新しさ 平均 %.3f ± %.3f" % (m, sd))

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    names = list(groups.keys()); data = [groups[k] for k in names]
    ax.boxplot([d for d in data if len(d) > 1] or [[0]], positions=[0], widths=0.5)
    ax.scatter([0] * len(data[0]), data[0], alpha=.45, s=18, color="#2b6cb0", label="1枚ずつ")
    for i, (nm, d) in enumerate(list(groups.items())[1:], start=1):
        ax.scatter([i] * len(d), d, s=70, color="#c53030", zorder=3)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("新しさ（1 − 一番近い記憶との似方）")
    ax.set_title("「新しさ」は知っている物と知らない物で分かれるか", fontsize=12)
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    p = OUT + "/図_新しさの分かれ方.png"; fig.savefig(p, dpi=110); print("図:", p)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "F/models/F2-49c_r3_seed93_%s.pt" % G.DATE)
