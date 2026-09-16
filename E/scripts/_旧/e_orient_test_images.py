"""視線誘導反射の単体テストで使う人工画像を可視化する。

各テストケースについて frame1（前フレーム）と frame2（現フレーム）を並べ、
期待される反射の出力方向を矢印で表示する。
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RES = 128   # MIMoと同じ

def blank():
    return np.zeros((RES, RES), dtype=np.float32)

def square(img, cx, cy, size=15, val=1.0):
    """(cx, cy)を中心に square(size)を描く。座標は [0, RES)。"""
    r = size // 2
    x0, x1 = max(0, cx-r), min(RES, cx+r)
    y0, y1 = max(0, cy-r), min(RES, cy+r)
    img[y0:y1, x0:x1] = val
    return img

def noise(img, amount=0.1, seed=0):
    """背景ノイズ。"""
    rng = np.random.default_rng(seed)
    img = img + rng.uniform(-amount, amount, size=img.shape)
    return np.clip(img, 0, 1)


# テストケース定義: (label, frame1, frame2, expected_hdir, expected_vdir, note)
def make_cases():
    cases = []

    f1 = square(blank(), 40, 64, 15); f2 = square(blank(), 80, 64, 15)
    cases.append(("1. move RIGHT", f1, f2, "+", "0", "h_dir positive"))
    f1 = square(blank(), 80, 64, 15); f2 = square(blank(), 40, 64, 15)
    cases.append(("2. move LEFT", f1, f2, "-", "0", "h_dir negative"))
    f1 = square(blank(), 64, 90, 15); f2 = square(blank(), 64, 40, 15)
    cases.append(("3. move UP", f1, f2, "0", "+", "v_dir positive"))
    f1 = square(blank(), 64, 40, 15); f2 = square(blank(), 64, 90, 15)
    cases.append(("4. move DOWN", f1, f2, "0", "-", "v_dir negative"))
    f1 = square(blank(), 80, 64, 15); f2 = f1.copy()
    cases.append(("5. NO motion", f1, f2, "0", "0", "both zero"))
    f1 = square(blank(), 60, 64, 10); f2 = square(blank(), 70, 64, 10)
    cases.append(("6. motion at CENTER", f1, f2, "+(weak)", "0", "current impl gives 0, new should be +"))
    f1 = square(blank(), 10, 64, 10); f2 = square(blank(), 25, 64, 10)
    cases.append(("7. motion at EDGE", f1, f2, "-", "0", "weaker due to center bias"))
    f1 = noise(square(blank(), 40, 64, 15), 0.15, seed=0)
    f2 = noise(square(blank(), 80, 64, 15), 0.15, seed=1)
    cases.append(("8. RIGHT + bg noise", f1, f2, "+", "0", "threshold rejects noise"))
    f1 = square(square(blank(), 30, 40, 10), 30, 90, 10)
    f2 = square(square(blank(), 60, 40, 10), 60, 90, 10)
    cases.append(("9. MULTIPLE points right", f1, f2, "+", "0", "centroid between points"))
    f1 = square(blank(), 40, 40, 15); f2 = square(blank(), 80, 80, 15)
    cases.append(("10. DIAGONAL down-right", f1, f2, "+", "-", "both components"))

    return cases


def plot_cases(cases, outpath):
    n = len(cases)
    cols = 5
    rows = n * 3 // cols + (1 if (n * 3) % cols else 0)
    # 各ケースを 3 枚（frame1, frame2, diff）で表示
    fig, axes = plt.subplots(n, 4, figsize=(11, 2.4*n))
    for i, (label, f1, f2, h, v, note) in enumerate(cases):
        diff = np.abs(f2 - f1)
        axes[i, 0].imshow(f1, cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title(f"{label}\nframe 1", fontsize=9)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(f2, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title("frame 2", fontsize=9)
        axes[i, 1].axis("off")
        axes[i, 2].imshow(diff, cmap="hot", vmin=0, vmax=1)
        axes[i, 2].set_title("|frame2 - frame1|", fontsize=9)
        axes[i, 2].axis("off")
        # 期待方向
        axes[i, 3].set_xlim(-1, 1)
        axes[i, 3].set_ylim(-1, 1)
        axes[i, 3].axhline(0, color="gray", linewidth=0.5)
        axes[i, 3].axvline(0, color="gray", linewidth=0.5)
        # 期待方向を矢印で
        hx = {"+": 0.7, "-": -0.7, "+(weak)": 0.35, "0": 0.0}.get(h, 0.0)
        vy = {"+": 0.7, "-": -0.7, "+(weak)": 0.35, "0": 0.0}.get(v, 0.0)
        if abs(hx) > 0.01 or abs(vy) > 0.01:
            axes[i, 3].annotate("", xy=(hx, vy), xytext=(0, 0),
                                 arrowprops=dict(arrowstyle="->", color="red", lw=2))
        else:
            axes[i, 3].plot(0, 0, "ro", markersize=8)
        axes[i, 3].set_title(f"expected: h={h}, v={v}\n{note}", fontsize=8)
        axes[i, 3].set_xlabel("h_dir")
        axes[i, 3].set_ylabel("v_dir")
        axes[i, 3].set_aspect("equal")

    fig.tight_layout()
    fig.savefig(outpath, dpi=110, bbox_inches="tight")
    print("saved:", outpath)


def main():
    cases = make_cases()
    out = os.path.join(os.path.dirname(__file__), os.pardir, "docs", "figures",
                       "orient_test_images_20260726.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plot_cases(cases, os.path.abspath(out))


if __name__ == "__main__":
    main()
