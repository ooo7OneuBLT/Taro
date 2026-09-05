# -*- coding: utf-8 -*-
"""F2-65: object_detector.detect()にMobileSAM経路を追加した後の入口出口確認
（2026-09-04）。本走行ではなく、書き捨てレベルの動作確認（CLAUDE.md「作る前に
10〜30行の書き捨てスクリプトで入口と出口を確かめる」に対応・今回は実装後の確認）。

既存画像（F2-64・F2-63・F2-49）に対して、旧方式(figure_mask)とMobileSAM方式の
両方でdetect()を呼び、件数・位置・面積を比べるだけ。

    .venv/Scripts/python.exe F/scripts/f65_mobilesam_detect_probe.py
"""
import os, sys, warnings, time
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
os.chdir(r"C:\claude\AI\Taro")
sys.path.insert(0, os.path.abspath("taro_core/src"))
import numpy as np
from PIL import Image
from senses.object_detector import patch_features, detect, load_mobilesam

IMAGES = [
    ("F/logs/F2-64_視野に物が無い場合/raw_empty.png", "物なし"),
    ("F/logs/F2-64_視野に物が無い場合/raw_with_object.png", "コップ1個"),
    ("F/logs/F2-49_実物スキャン/test/stimuli48/img03.png", "コップ(別個体)"),
]

print("DINOv2を読み込み中...")
import torch
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).eval()
print("MobileSAMを読み込み中...")
mgen = load_mobilesam()

for path, name in IMAGES:
    img = np.array(Image.open(path).convert("RGB").resize((224, 224)))
    p, n = patch_features(model, img)

    dets_old = detect(p, n, thresh=0.55)
    t0 = time.time()
    dets_new = detect(p, n, img=img, mask_generator=mgen)
    dt = time.time() - t0

    print("\n[%s] %s" % (name, path))
    print("  旧方式(figure_mask): %d件  面積=%s" % (
        len(dets_old), [round(d["area"], 3) for d in dets_old]))
    print("  新方式(MobileSAM)  : %d件  面積=%s  位置=%s  所要%.2fs" % (
        len(dets_new), [round(d["area"], 3) for d in dets_new],
        [tuple(round(c, 1) for c in d["pos"]) for d in dets_new], dt))
    for d in dets_new:
        v = d["appearance"]
        print("    見た目ベクトル: shape=%s norm=%.3f" % (v.shape, np.linalg.norm(v)))

print("\n完了")
