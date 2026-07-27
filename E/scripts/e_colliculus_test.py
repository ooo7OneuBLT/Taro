"""上丘の座標変換そのものの単体テスト（環境に持ち込む前に確かめる）。

【なぜ】環境で測ったら結果が悪化した（符号の一致 7/7 → 4/7）。
実装のバグか、変換の性質か、切り分けるには**変換だけ**を試すのが確実。
→ 落とし穴チェックリスト 項6（測定器そのものの健康診断）。

【確かめること】
  1. 往復：視野座標 → 上丘座標 → 視野座標 で元に戻るか
  2. 特異点：左視野（Φ≈π）で発散しないか
  3. 1点だけ光らせたとき、その位置を正しく返すか
  4. ★2点が離れているとき、重心がどこを指すか（潰れの度合い）
  5. 旧方式（平らな画像で重心）との比較
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(_ROOT, "taro_core", "src", "brain"))
try: sys.stdout.reconfigure(errors="replace")
except Exception: pass

import numpy as np
from superior_colliculus import (CollicularMap, visual_to_collicular,
                                 collicular_to_visual, magnification,
                                 OTTES_A, OTTES_BU, OTTES_BV)

W = H = 128
FOVY = 60.0
HALF = FOVY / 2.0


def flat_centroid(a):
    """旧方式：平らな画像で重心を取り、-1〜1 に正規化する。"""
    a = np.clip(np.asarray(a, dtype=float), 0, None)
    if a.max() <= 1e-12:
        return float("nan"), float("nan")
    ys, xs = np.nonzero(a > 0)
    w = a[a > 0]
    cx = float((xs * w).sum() / w.sum())
    cy = float((ys * w).sum() / w.sum())
    hx, hy = (W - 1) / 2.0, (H - 1) / 2.0
    return (cx - hx) / hx, -(cy - hy) / hy


def main():
    ok = 0
    total = 0
    print("=== 上丘の座標変換：単体テスト ===")
    print(f"  定数 A={OTTES_A}度  Bu={OTTES_BU}mm  Bv={OTTES_BV}mm/rad")
    print(f"  画像 {W}x{H}  視野 {FOVY}度（半角 {HALF}度）\n")

    # ---- 1. 往復 ----------------------------------------------------------
    print("--- 1. 往復（視野→上丘→視野）---")
    print(f"{'入力(x,y)[度]':>18}{'上丘(u,v)[mm]':>20}{'戻り(x,y)[度]':>20}{'誤差':>8}")
    for x, y in [(1, 0), (5, 0), (20, 0), (-5, 0), (-20, 0),
                 (0, 10), (10, 10), (-10, -10), (25, 25)]:
        R = np.hypot(x, y)
        P = np.arctan2(y, abs(x))          # 右側へ折り返す
        u, v = visual_to_collicular(R, P)
        bx, by = collicular_to_visual(u, v)
        bx = bx * (1 if x >= 0 else -1)    # 折り返しを戻す
        err = np.hypot(bx - x, by - y)
        total += 1
        if err < 0.01:
            ok += 1
        print(f"{f'({x:5.1f},{y:5.1f})':>18}{f'({u:6.3f},{v:6.3f})':>20}"
              f"{f'({bx:6.2f},{by:6.2f})':>20}{err:>8.4f}")

    # ---- 2. 特異点 --------------------------------------------------------
    print("\n--- 2. 特異点（左視野・R=A=3度 のあたり）---")
    bad = 0
    for x in (-3.0, -2.99, -3.01, -0.001, 0.0):
        R = abs(x)
        P = np.pi if x < 0 else 0.0
        u, v = visual_to_collicular(R, P)      # ★折り返さずに直接（旧実装の再現）
        uf, vf = visual_to_collicular(R, np.arctan2(0.0, abs(x)))   # 折り返しあり
        flag = "★発散" if (not np.isfinite(u)) or abs(u) > 100 else ""
        if flag:
            bad += 1
        print(f"  x={x:7.3f}  折り返しなし u={u:10.3f} {flag:8}  折り返しあり u={uf:8.3f}")
    total += 1
    if bad > 0:
        ok += 1
        print("  → ★折り返しなしでは発散する＝左右を分ける処理が必須（実装ずみ）")
    else:
        print("  → 発散しなかった（想定と違う。式を確認すること）")

    # ---- 3. 1点だけ光らせる ----------------------------------------------
    print("\n--- 3. 1点だけ光らせたとき、その位置を返すか ---")
    smap = CollicularMap(W, H, FOVY)
    print(f"{'置いた位置':>12}{'新方式(上丘)':>14}{'旧方式(平ら)':>14}{'真値':>10}")
    for frac in (-0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8):
        a = np.zeros((H, W))
        px = int(round((frac + 1) / 2 * (W - 1)))
        a[H // 2, px] = 1.0
        nh, nv = smap.direction(a)
        fh, fv = flat_centroid(a)
        # 真値：その画素の視野角を半角で割ったもの
        truth = np.degrees(np.arctan(frac * np.tan(np.radians(HALF)))) / HALF
        total += 1
        if abs(nh - truth) < 0.05:
            ok += 1
        print(f"{frac:>12.1f}{nh:>14.3f}{fh:>14.3f}{truth:>10.3f}")
    print("  ★1点なら新方式も真値と一致するはず（重心の潰れは複数点のときに出る）")

    # ---- 4. 広がりのある塊 ------------------------------------------------
    print("\n--- 4. 広がりのある塊（おもちゃ相当・幅40画素）---")
    print(f"{'中心の位置':>12}{'新方式(上丘)':>14}{'旧方式(平ら)':>14}{'真値':>10}")
    for frac in (-0.6, -0.3, 0.0, 0.3, 0.6):
        a = np.zeros((H, W))
        px = int(round((frac + 1) / 2 * (W - 1)))
        x0, x1 = max(0, px - 20), min(W, px + 20)
        a[H // 2 - 20:H // 2 + 20, x0:x1] = 1.0
        nh, _ = smap.direction(a)
        fh, _ = flat_centroid(a)
        truth = np.degrees(np.arctan(frac * np.tan(np.radians(HALF)))) / HALF
        print(f"{frac:>12.1f}{nh:>14.3f}{fh:>14.3f}{truth:>10.3f}")
    print("  ★新方式は中心寄りに潰れるはず（上丘の性質。Goossens & Van Opstal 2012）")

    # ---- 5. マグニフィケーション ------------------------------------------
    print("\n--- 5. マグニフィケーション M(R)=Bu/(R+A) [mm/度] ---")
    for R in (0, 1, 3, 5, 10, 20, 30):
        print(f"  R={R:3d}度  M={float(magnification(R)):.4f} mm/度"
              f"   中心の {float(magnification(R))/float(magnification(0))*100:5.1f}%")

    # ---- 6. ★格子方式（上丘の升目へ写して受容野でまとめる）----------------
    print("\n--- 6. 格子方式：升目の健全性 ---")
    from scipy.ndimage import gaussian_filter
    frac_ok = float(smap._grid_ok.mean())
    print(f"  升の数 {smap.nu}x{smap.nv}x2   有効な升 {frac_ok*100:.1f} %")
    print(f"  升の大きさ  u方向 {smap.du:.4f} mm   v方向 {smap.dv:.4f} mm")
    print(f"  受容野 σ={smap.rf_sigma_cells[1]:.1f} 升(u) / "
          f"{smap.rf_sigma_cells[0]:.1f} 升(v)")
    total += 1
    if 0.2 < frac_ok < 0.95:
        ok += 1
    else:
        print("  ★有効な升の割合がおかしい（写像か範囲の設定を確認）")

    print("\n--- 7. 格子方式：1点を光らせて位置が返るか ---")
    print(f"{'置いた位置':>12}{'格子方式':>12}{'重み方式':>12}{'旧(平ら)':>12}{'真値':>10}")
    for frac in (-0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8):
        a = np.zeros((H, W))
        px = int(round((frac + 1) / 2 * (W - 1)))
        a[H // 2, px] = 1.0
        g = smap.to_grid(a)
        gh, _ = smap.grid_direction(g)
        nh, _ = smap.direction(a * smap.mag_area / smap.mag_area.max())
        fh, _ = flat_centroid(a)
        truth = np.degrees(np.arctan(frac * np.tan(np.radians(HALF)))) / HALF
        total += 1
        if abs(gh - truth) < 0.08:
            ok += 1
        print(f"{frac:>12.1f}{gh:>12.3f}{nh:>12.3f}{fh:>12.3f}{truth:>10.3f}")

    print("\n--- 8. ★本番：大きな対象の『左右の縁だけ』が光り、競合を通す ---")
    print("  実際の動き検出はこうなる（物体の中身は色が変わらないので消える）。")
    print("  物体を視野の右 15度 に置き、幅（＝縁の間隔）を変える。")
    print("""
  ⚠️判定の基準について（2026-07-27 に訂正）
    最初は「常に物体の中心(+0.50)を指すこと」を正解にしたが、**これは文献に
    裏付けが無い**。人間は大きな対象では平均化が弱まり個別の縁へ着地する
    （van der Stigchel et al. 2012 Vision Res 62:108-115）。小さければ
    正確に中心へ行く（Kilpeläinen & Georgeson 2018、幅4度で偏差0.57度以内）。
    ★人間に無いのは「対象が存在しない方向へ飛ぶこと」。
      よって判定は**出力が2つの縁のあいだに収まっているか**とする。""")
    import e_orienting_v2 as OR

    def compete_grid(a):
        """格子方式：上丘へ写す → 受容野 → 上丘の上で競合 → 重心。"""
        g = smap.to_grid(a)
        sv, su = smap.rf_sigma_cells
        g = gaussian_filter(g, (0.0, sv, su))
        if g.max() < 1e-9:
            return float("nan")
        inp = g / g.max()
        u = inp.copy()
        se = (0.0, OR.LI_SIGMA_EXC_MM / smap.dv, OR.LI_SIGMA_EXC_MM / smap.du)
        si = (0.0, OR.LI_SIGMA_INH_MM / smap.dv, OR.LI_SIGMA_INH_MM / smap.du)
        for _ in range(OR.LI_N_ITER):
            f = np.clip(u, 0, None)
            u = u + OR.LI_RATE * (-u + inp
                                  + gaussian_filter(f, se) * OR.LI_W_EXC
                                  - gaussian_filter(f, si) * OR.LI_W_INH
                                  - OR.LI_W_GLOBAL * float(f.mean()))
        u = np.clip(u, 0, None)
        return smap.grid_direction(u, thresh_frac=OR.CENTROID_THRESH_FRAC)[0]

    def compete_flat(a):
        """旧方式：画像の座標のまま競合 → 上丘の座標で重心（受容野なし）。"""
        s_e = OR.LI_SIGMA_EXC_FRAC * max(H, W)
        s_i = OR.LI_SIGMA_INH_FRAC * max(H, W)
        inp = a / max(a.max(), 1e-12)
        u = inp.copy()
        for _ in range(OR.LI_N_ITER):
            f = np.clip(u, 0, None)
            u = u + OR.LI_RATE * (-u + inp
                                  + gaussian_filter(f, s_e) * OR.LI_W_EXC
                                  - gaussian_filter(f, s_i) * OR.LI_W_INH
                                  - OR.LI_W_GLOBAL * float(f.mean()))
        u = np.clip(u, 0, None)
        return smap.direction(u, thresh_frac=OR.CENTROID_THRESH_FRAC)[0]

    t = np.tan(np.radians(HALF))
    center_deg = 15.0
    truth = center_deg / HALF
    print(f"\n{'物体の幅':>10}{'縁の位置[度]':>16}{'格子方式':>11}{'旧方式':>11}"
          f"{'真値':>8}   判定")
    for sep_deg in (4, 8, 16, 26, 40):
        a = np.zeros((H, W))
        edges = []
        for sgn in (-1, +1):
            xd = center_deg + sgn * sep_deg / 2.0
            gx = np.tan(np.radians(xd)) / t
            px = int(np.clip(round((gx + 1) / 2 * (W - 1)), 0, W - 1))
            edges.append(xd)
            a[H // 2 - 12:H // 2 + 12, max(0, px - 1):px + 2] = 1.0
        gh = compete_grid(a)
        fh = compete_flat(a)
        lo, hi = edges[0] / HALF, edges[1] / HALF
        total += 1
        good = (lo - 0.05) <= gh <= (hi + 0.05)     # 2つの縁のあいだにあるか
        if good:
            ok += 1
        print(f"{sep_deg:>9}度{f'{edges[0]:+.0f} / {edges[1]:+.0f}':>16}"
              f"{gh:>11.3f}{fh:>11.3f}{truth:>8.2f}   "
              f"{'○ 縁の間' if good else '★縁の外へ飛んだ'}")

    print(f"\n=== 判定 {ok}/{total} 通過 ===")
    if ok < total:
        print("  ★通らなかった項目がある。実装を見直すこと")


if __name__ == "__main__":
    main()
