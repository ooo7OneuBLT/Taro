"""
録画mp4を「チャットで直接見られる形」に変換する（スマホ等で動画ファイルを開けない時用）。

出力2種：
  ①GIF        … 動きがそのまま見える。幅を落として軽くする。
  ②連続コマ一覧 … 1枚の画像に等間隔のコマを並べる。時間経過が一目で分かる（GIFより確実に見える）。

使い方: python mp4_to_chat.py <mp4> [出力先ディレクトリ] [gif幅] [コマ数]
"""
import os, sys
import cv2
import numpy as np
from PIL import Image, ImageDraw


def load_frames(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def autocrop(frames, pad=30):
    """太郎の体だけに寄せて切り出す。

    MIMoの既定カメラは引きの画で、体は画面の2割程度しか占めない。そのまま縮小すると
    腕の動き（＝見たいもの）が潰れる。全コマを通して「動いた画素」の外接矩形＝体の
    活動範囲なので、そこにpadを足して切り出す。背景(床・チェッカー)は完全に静止して
    いるので、動き＝太郎だけを拾える。
    """
    a = np.stack(frames).astype(np.int16)
    motion = (a.max(axis=0) - a.min(axis=0)).max(axis=2).astype(np.float64)
    # 「動いた画素の外接矩形」だと失敗する：MIMoの床は鏡面で体が映り込むため、
    # 床の反射までが動きとして検出され、矩形が画面全体に広がる。
    # 動きの"量"の分布で見て、上下左右の外れ値(反射・影)を落とす＝体の本体だけが残る。
    def span(w_):
        cdf = np.cumsum(w_) / max(w_.sum(), 1e-9)
        return int(np.searchsorted(cdf, 0.03)), int(np.searchsorted(cdf, 0.97))
    y0, y1 = span(motion.sum(axis=1))
    x0, x1 = span(motion.sum(axis=0))
    h, w, _ = frames[0].shape
    y0, y1 = max(0, y0 - pad), min(h, y1 + pad)
    x0, x1 = max(0, x0 - pad), min(w, x1 + pad)
    if (y1 - y0) < 40 or (x1 - x0) < 40:
        return frames, None
    return [f[y0:y1, x0:x1] for f in frames], (x0, y0, x1, y1)


def make_gif(frames, fps, out, width=320, target_fps=12):
    # 元のfpsからtarget_fpsへ間引く（等速のまま軽くする）
    step = max(1, int(round(fps / target_fps)))
    sel = frames[::step]
    h, w, _ = sel[0].shape
    size = (width, int(h * width / w))
    imgs = [Image.fromarray(f).resize(size, Image.LANCZOS) for f in sel]
    imgs[0].save(out, save_all=True, append_images=imgs[1:],
                 duration=int(1000 * step / fps), loop=0, optimize=True)
    return out, len(imgs), size


def make_sheet(frames, fps, out, n=8, cols=4, width=280):
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    h, w, _ = frames[0].shape
    tw, th = width, int(h * width / w)
    rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * th), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    for i, fi in enumerate(idx):
        im = Image.fromarray(frames[fi]).resize((tw, th), Image.LANCZOS)
        x, y = (i % cols) * tw, (i // cols) * th
        sheet.paste(im, (x, y))
        # 何秒時点のコマかを焼き込む（等速なので fi/fps がsim秒）
        label = f"{fi / fps:.1f}s"
        dr.rectangle([x + 2, y + 2, x + 54, y + 20], fill=(0, 0, 0))
        dr.text((x + 6, y + 5), label, fill=(255, 255, 255))
        dr.rectangle([x, y, x + tw - 1, y + th - 1], outline=(200, 200, 200))
    sheet.save(out)
    return out, sheet.size


def main():
    src = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(src)
    gw = int(sys.argv[3]) if len(sys.argv) > 3 else 320
    ncell = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    os.makedirs(outdir, exist_ok=True)
    frames, fps = load_frames(src)
    if not frames:
        print(f"[エラー] コマが読めない: {src}")
        return
    frames, box = autocrop(frames)
    print(f"切り出し: {box}  → {frames[0].shape[1]}x{frames[0].shape[0]}" if box else "切り出し: なし（動きが検出できず）")
    base = os.path.splitext(os.path.basename(src))[0]
    g, ng, gsize = make_gif(frames, fps, os.path.join(outdir, base + ".gif"), width=gw)
    s, ssize = make_sheet(frames, fps, os.path.join(outdir, base + "_sheet.png"), n=ncell)
    print(f"元: {len(frames)}コマ / {fps:.0f}fps / {len(frames)/fps:.1f}秒(等速)")
    print(f"GIF   : {g}  ({ng}コマ, {gsize[0]}x{gsize[1]}, {os.path.getsize(g)/1024:.0f}KB)")
    print(f"コマ一覧: {s}  ({ssize[0]}x{ssize[1]}, {os.path.getsize(s)/1024:.0f}KB)")


if __name__ == "__main__":
    main()
