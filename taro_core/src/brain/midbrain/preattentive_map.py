# -*- coding: utf-8 -*-
"""段1 前注意の地図 ── 目を向ける前に「どこで何が起きているか」を並列に作る。

【仕様】F/docs/二語文/仕様_見る側の段構成_実装_2026-09-09.md 2節・
F/docs/二語文/仕様_見る側の設計_段構成_2026-09-09.md 後半A（段1）。

【役割】連続2コマ（段0が`shift_actual`でずれを引いた後）の差分から「動いた塊」
（blobs）の一覧と、目立ち度の地図（`static_sal`、段6の探索が使う）を作る。
Itti & Koch の目立ち地図・上丘表層の近似[Tier2]。人間の分節の最初の手がかりは
共通運動（Kellman & Spelke 1983 [Tier1]）という知見を、動いた塊＝1つの物の
候補、として近似する。
"""
import numpy as np
from scipy import ndimage


class PreattentiveMap:
    def __init__(self, diff_thresh=20.0, min_blob_area_px=30, max_blobs=8, cell=32):
        self.diff_thresh = float(diff_thresh)
        self.min_blob_area_px = int(min_blob_area_px)
        self.max_blobs = int(max_blobs)
        self.cell = int(cell)
        self._prev = None   # 前コマのグレースケール画像 (H, W) float32

    def update(self, img224_gray_f32, shift_actual, moving):
        """毎検出コマ呼ぶ。`moving`のコマは塊の一覧を空にする（前コマの保持だけ更新）。

        【2026-09-10・測定器の直し】以前は`moving`のコマで**何も計算せずに返して
        いた**ため、遠心性コピーの効きを見る診断（`motion_mean` /
        `motion_mean_noshift`）が、**目が動いたコマだけ常に 0.0** になっていた。
        遠心性コピーが要るのはまさにそのコマだけなので、この診断は一度も
        効いたことがなかった（F2-110pre/F2-111pre の実測で発覚。目が動いた
        98コマすべてで両方 0.00）。
        塊の一覧（blobs）と `valid` は従来どおり空・False のままなので、
        **走行の挙動は1ビットも変わらない**（`motion_mean` 系は記録専用。
        `run/plugins/common/object_files.py` の onset_extra だけが読む）。
        """
        img = np.asarray(img224_gray_f32, dtype=np.float32)
        h, w = img.shape[:2]
        prev = self._prev
        self._prev = img.copy()

        static_sal = self._static_saliency(img)

        if prev is None:
            return {"blobs": [], "motion_mean": 0.0, "motion_mean_noshift": 0.0,
                    "shift_meas": (0.0, 0.0),
                    "static_sal": static_sal, "valid": False}

        dx, dy = shift_actual
        idx = int(round(dx))
        idy = int(round(dy))
        # 【なぜ、2026-09-09】shift_actualは段0（遠心性コピー）が実際の目の
        #   角度差から計算した「画像の中身がどれだけ動いたか」の予告。前コマを
        #   その分だけ画素単位でずらしてから引くと、目が動いただけの見かけの
        #   差分が打ち消される（Duhamel 1992・Wurtz 2008、輪の歯止め4節）。
        prev_shifted = np.roll(prev, shift=(idy, idx), axis=(0, 1))
        valid_mask = np.ones((h, w), dtype=bool)
        if idy > 0:
            valid_mask[:idy, :] = False
        elif idy < 0:
            valid_mask[idy:, :] = False
        if idx > 0:
            valid_mask[:, :idx] = False
        elif idx < 0:
            valid_mask[:, idx:] = False

        diff_noshift = np.abs(img - prev)
        diff_shift = np.abs(img - prev_shifted)

        motion_mean_noshift = float(diff_noshift.mean())
        motion_mean = float(diff_shift[valid_mask].mean()) if valid_mask.any() else 0.0
        # 【2026-09-10】画像そのものから測った「中身が実際にどれだけ動いたか」。
        #   遠心性コピーが送ってきた shift_actual と突き合わせれば、
        #   予告が当たっているかを直接見られる（残差の比較より曖昧さが無い）。
        shift_meas = self._measure_shift(prev, img)

        if moving:
            # 目が動いている最中は塊の一覧を出さない（従来どおり）。
            #   診断の値だけは返す（上の docstring 参照）。
            return {"blobs": [], "motion_mean": motion_mean,
                    "motion_mean_noshift": motion_mean_noshift,
                    "shift_meas": shift_meas,
                    "static_sal": static_sal, "valid": False}

        blob_mask = (diff_shift > self.diff_thresh) & valid_mask
        blob_mask = ndimage.binary_opening(blob_mask, structure=np.ones((3, 3)))
        lab, n = ndimage.label(blob_mask)
        blobs = []
        for i in range(1, n + 1):
            ys, xs = np.where(lab == i)
            area = int(len(ys))
            if area < self.min_blob_area_px:
                continue
            cx = float(xs.mean())
            cy = float(ys.mean())
            mean_abs_diff = float(diff_shift[ys, xs].mean())
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            blobs.append({"cx": cx, "cy": cy, "area": area,
                           "bbox": (x0, y0, x1 - x0 + 1, y1 - y0 + 1),
                           "mean_abs_diff": mean_abs_diff})
        blobs.sort(key=lambda b: b["area"], reverse=True)
        blobs = blobs[:self.max_blobs]

        return {"blobs": blobs, "motion_mean": motion_mean,
                "motion_mean_noshift": motion_mean_noshift,
                "shift_meas": shift_meas,
                "static_sal": static_sal, "valid": True}

    @staticmethod
    def _measure_shift(prev, cur):
        """2コマの間に画像の中身が何画素動いたかを、位相相関で測る（測定専用）。

        返す (dx, dy) の向きは `np.roll(prev, shift=(dy, dx))` と同じ流儀＝
        `shift_actual` とそのまま比べられる（合成画像で符号と大きさを確認済み。
        `F/logs/_机上/遠心性コピーの測定器_机上確認_2026-09-10.txt`）。
        1画素刻み（副画素の補間はしない）。
        """
        h, w = prev.shape[:2]
        win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
        a = (prev - prev.mean()) * win
        b = (cur - cur.mean()) * win
        A = np.fft.rfft2(a)
        B = np.fft.rfft2(b)
        R = np.conj(A) * B
        R /= (np.abs(R) + 1e-9)
        r = np.fft.irfft2(R, s=(h, w))
        iy, ix = np.unravel_index(int(np.argmax(r)), r.shape)
        if ix > w // 2:
            ix -= w
        if iy > h // 2:
            iy -= h
        return float(ix), float(iy)

    def _static_saliency(self, img):
        """Laplacianの絶対値をcell×cellへ平均する（cv2は使わずnumpyだけで計算。
        仕様2節「static_sal：Laplacianの絶対値をcell×cellに平均」）。"""
        h, w = img.shape[:2]
        up = np.roll(img, 1, axis=0)
        down = np.roll(img, -1, axis=0)
        left = np.roll(img, 1, axis=1)
        right = np.roll(img, -1, axis=1)
        lap = np.abs(up + down + left + right - 4.0 * img)
        cell = self.cell
        ch = max(h / cell, 1e-9)
        cw = max(w / cell, 1e-9)
        out = np.zeros((cell, cell), dtype=np.float32)
        for i in range(cell):
            y0, y1 = int(i * ch), min(int((i + 1) * ch), h)
            if y1 <= y0:
                y1 = min(y0 + 1, h)
            for j in range(cell):
                x0, x1 = int(j * cw), min(int((j + 1) * cw), w)
                if x1 <= x0:
                    x1 = min(x0 + 1, w)
                out[i, j] = float(lap[y0:y1, x0:x1].mean())
        return out
