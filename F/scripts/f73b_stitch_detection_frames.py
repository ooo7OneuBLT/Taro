# -*- coding: utf-8 -*-
"""検出フレーム（`検出フレーム/frame_*.png`）を1秒1コマの動画へつなぐ道具（2026-09-05）。

【なぜ】`object_files.py`のframes_outが吐くPNG連番は、検出のたび（interval_s=1.0秒ごと）
  に1枚。目視しやすいよう、1コマを2倍に拡大し、1コマ=1秒（10fps×10回書く）でmp4へ
  まとめる。仕様：F/docs/二語文/仕様_M1.5b_視界動画_検出と物体ファイルの重ね描き_2026-09-05.md。
  手本：D/scripts/d_record.py（cv2.VideoWriterの使い方）。

【使い方】
    .venv\\Scripts\\python.exe F/scripts/f73b_stitch_detection_frames.py <frames_dir> <out_mp4>
  引数省略時は F2-73b の既定パスを使う。
"""
import glob
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np


def _imread_unicode(path):
    """cv2.imreadはWindowsで日本語パスを読めない（内部でANSI fopen）ので、
    np.fromfile+cv2.imdecodeで迂回する。"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))

_DEFAULT_FRAMES = os.path.join(_REPO_ROOT, "F", "logs", "F2-73b_M1.5_視界動画", "検出フレーム")
_DEFAULT_OUT = os.path.join(_REPO_ROOT, "F", "logs", "F2-73b_M1.5_視界動画", "動画_検出_1fps.mp4")


def stitch(frames_dir, out_path, scale=2, fps=10, hold_frames=10):
    """frames_dir配下のframe_*.pngを番号順に読み、各コマをscale倍に拡大して、
    1コマをhold_frames回（fps=10なら10回=1秒）書き込んだmp4を作る。"""
    paths = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    if not paths:
        print("フレームが見つからない: %s" % frames_dir)
        return None

    first = _imread_unicode(paths[0])
    h, w = first.shape[0] * scale, first.shape[1] * scale
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # cv2.VideoWriterも日本語パスを直接開けないので、ASCII一時ファイルに書いてから
    # os.replace（Python自前のファイル操作。日本語パスを正しく扱える）で本来の場所へ移す。
    tmp_out = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + ".mp4")
    vw = cv2.VideoWriter(tmp_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n_coma = 0
    for p in paths:
        img = _imread_unicode(p)
        if img is None:
            continue
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        for _ in range(hold_frames):
            vw.write(img)
        n_coma += 1
    vw.release()
    shutil.move(tmp_out, out_path)

    print("=== 検出動画 完成 ===")
    print("フレーム(コマ)数: %d" % n_coma)
    print("解像度          : %dx%d" % (w, h))
    print("保存先          : %s" % os.path.abspath(out_path))
    return out_path, n_coma


if __name__ == "__main__":
    frames_dir = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_FRAMES
    out_path = sys.argv[2] if len(sys.argv) > 2 else _DEFAULT_OUT
    stitch(frames_dir, out_path)
