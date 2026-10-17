"""目の入口（入口（感覚））。

**段階3 作り直し（2026-10-16、ユーザー了承の設計図）：毎回の切り出しを、MobileSAM から
「床から浮いて見えるまとまり」（core_v2/floor_objects.py、目だけのテストの dino_floor_a）に替えた。**
目だけのテスト（G/scripts/eye_test.py、200枚＋別の100枚）で、全身が見えている物 96〜100%、
画像の端で切れた物 58〜96%（MobileSAM は0%）、床のマスを物と間違える数 1枚0.8〜1.0個（同5.4個）、
1枚約30ms（同約515ms）。MobileSAM は setup(answer_check=True) のときだけ作る（答え合わせ用、
detector="sam" で段階2までの切り出しに戻せる）。
人間の視覚野との対応（粗い対応）：
* 網膜〜V1：画像（左目）。明るさ・色・向き・動きの目立ちの地図（SalienceMap、上丘・V1 の役）
* V2：図と地を分ける。床（地）らしくないマスを図としてつなぎ、見た目の違う物どうしは分ける
* V4〜下側頭葉：DINOv2 のマスの特徴（見た目、何であるか）。物ごとの荷札の元
* MT：画面全体がどれだけずれたか（位相相関、image_shift）＝目が実際に動いた分。復帰抑制の印を
  目と一緒に動かすのに使う（目の動きの地図）
* 頭頂葉（物を1つ1つ追う）はここではなく orienteye の印の側［Tier3・モデル上の選択］

段階2・工程2：旧太郎の物の切り出し（MobileSAM）と見た目（DINOv2）、目立ちの地図
（SalienceMap）を借りて、書式に包む。
* eyein→gaze  : 物ごとの (荷札, 地図)。荷札＝見た目384を固定の乱数行列で64へ縮めて
                L2正規化し雑音を足したもの（tagging.py）。地図＝物の中心にガウスの山
* eyein→prio  : 目立ちの地図（28×28、0〜1）
* eyein→objfile: gaze と同じ物ごとの (荷札, 地図)＋最後に目の動きの地図（段階4、物の名簿へ）。
                段階3の eyein→semantic / orienteye は名簿へ移したので、今はゼロ
* それ以外    : 素通し（ゼロ）

**仕様書 §2-2 からの変更（2026-10-15、実測による）**：仕様書は旧太郎の「物体ファイル」
（result["files"]、毎コマ追跡して名簿を持つ仕組み）を読むことにしていたが、
① 旧太郎の設定（cohesion_gap_px=4）では3つの物と机のマスが1つにまとめられ、物が
  1つも取れなかった（3つの物の真ん中に1枚だけ）
② まとめを切っても、追跡は動く妨害物に2枚目3枚目の名簿を作り、見失った名簿が
  予測位置を漂い続けた（misses 3〜7）
ので、追跡は使わず**毎回その場で切り出す**（旧太郎の最初のコマと同じ全体切り出し）。
こうすると「今見えている物」だけが出る。20コマで3つの物は毎コマ取れた（points_per_side=12）。
机のマスなども物として出るので、物が MAX_OBJECTS（8）を超えたら目立ちの高い順に残す。

**軽くする（段階3、2026-10-16、ユーザー承認「両方やる」）**：全体の切り出し（MobileSAM、
1回 約0.4秒）は DETECT_EVERY（3）回に1回だけ。間の回は、前の回の物ごとに、前の位置の
まわり ±TRACK_PATCHES マス（DINOv2 の 16×16 マス、1マス14画素）で見た目が一番似たマスを
探し、前の回からそのマスがずれた分だけ位置を動かす（見た目の荷札はそのまま使う）。見た目が SAME_COS 未満なら見失った
として落とす。画面全体が前の回と大きく変わったとき（目が跳んだとき）は、間の回でも全体を
切り出し直す。DINOv2 と目立ちの地図は毎回計算する（軽い）。

**目が動いた分を画像から求める（2026-10-16）**：前の回と今の回の画像（灰色）の位相相関で、
画面全体がどれだけずれたか（画素）を毎回求める（image_shift）。使い道は2つ。
* 間の回の追跡で、前の物の位置をまずこのずれだけ動かしてから、まわりを探す。ずれが
  JUMP_PX（28画素）を超えたら、目が跳んだとみて全体を切り出し直す
* eyein→orienteye の最後に「目の動きの地図」を1つ足す（荷札はゼロ）。前の回の注視点が今どこに
  見えるかを、真ん中からのずれの半分の所に山として置く（半分にするのは、大きく跳んでも地図の
  外へ出ないため）。復帰抑制の印をこの分だけずらす。目を動かす指令の写しではなく、実際に目が
  動いた分なので、跳びが指令より小さくても印は物に残る［Tier3・モデル上の選択］

環境からの入力（inbox[ENV]）：左目の画像（224×224×3、uint8）。
setup() を呼ぶまでは素通し（骨組みのテストは重い視覚を作らずに回る）。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

from ..formats import GRID, MAX_OBJECTS, TAG_DIM, DTYPE
from ..tagging import appearance_tag, noisy
from .base import ENV, Box

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

# 旧太郎の _門2やり直し_2026-09-16 の設定から、物の切り出しと目立ちに要る所だけを残した。
# cohesion_gap_px は切る（上の①）。points_per_side=12：8だと 224ms・12だと 402ms・16だと
# 650ms（1コマ、RTX 3060 Ti、20コマの中央値）。3つとも全部のコマで取れた
DEFAULT_CONFIG = {
    "points_per_side": 12,
    "points_per_batch": 32,
    "empty_cache_after_detect": False,
    "interval_s": 0.09,
    "cohesion_gap_px": None,
    "attention": {"cell": GRID, "ior_tau_s": 0.7, "ior_gain": 1.0, "acc_tau_s": 0.11,
                  "dog_iters": 10},
    "gaze_from_attention": False,   # 目を動かすのは orienteye 箱だけ（仕様書 §2-2）
}
BUMP_SIGMA_CELLS = 1.0
DETECT_EVERY = 3          # 全体の切り出しを何回に1回にするか（1＝毎回、段階2と同じ）
TRACK_PATCHES = 2         # 間の回に物を探す範囲（DINOv2 のマス、±）
SAME_COS = 0.6            # これ未満なら見失った［仮置き］
SCENE_SAME_COS = 0.9      # 画面全体の似かたがこれ未満なら目が跳んだとみて切り出し直す［仮置き］
JUMP_PX = 28.0            # 画面全体のずれがこれを超えたら切り出し直す（DINOv2 の2マス）
EGO_SCALE = 0.5           # 目の動きの地図：真ん中からのずれ（マス）にこれを掛けた所に山を置く


def bump(cx_px: float, cy_px: float, sigma: float = BUMP_SIGMA_CELLS) -> torch.Tensor:
    """224画素の座標 (cx, cy) に、高さ1のガウスの山を置いた 28×28 の地図。"""
    step = 224.0 / GRID
    gx, gy = cx_px / step - 0.5, cy_px / step - 0.5     # マス c の中心は 8c+4 画素
    ys, xs = np.mgrid[0:GRID, 0:GRID]
    m = np.exp(-((xs - gx) ** 2 + (ys - gy) ** 2) / (2.0 * sigma ** 2))
    return torch.as_tensor(m, dtype=DTYPE)


_HANN = np.outer(np.hanning(224), np.hanning(224)).astype(np.float32)


def image_shift(prev_gray: np.ndarray, gray: np.ndarray) -> tuple[float, float, float]:
    """前の画像から今の画像へ、中身が (dx, dy) 画素ずれたかを位相相関で求める。
    3つ目は相関の山の高さ（0〜1、低いほど当てにならない）。"""
    f1 = np.fft.fft2(prev_gray * _HANN)
    f2 = np.fft.fft2(gray * _HANN)
    r = f2 * np.conj(f1)
    r /= np.abs(r) + 1e-8
    c = np.fft.ifft2(r).real
    py, px = np.unravel_index(int(np.argmax(c)), c.shape)
    h, w = c.shape
    dy = py - h if py > h // 2 else py
    dx = px - w if px > w // 2 else px
    return float(dx), float(dy), float(c[py, px])


def ego_map(dx_px: float, dy_px: float) -> torch.Tensor:
    """目の動きの地図：前の回の注視点が今見えている所（真ん中からのずれ×EGO_SCALE）に山。"""
    c = 112.0 + EGO_SCALE * dx_px
    r = 112.0 + EGO_SCALE * dy_px
    return bump(min(max(c, 0.0), 223.0), min(max(r, 0.0), 223.0))


def decode_ego(m: torch.Tensor) -> tuple[float, float]:
    """ego_map の山から、目の動き（中身のずれ、マス単位の (行, 列)）を戻す。"""
    w = (m >= 0.5 * float(m.max())).to(m.dtype) * m
    tot = float(w.sum())
    ys, xs = torch.meshgrid(torch.arange(GRID, dtype=m.dtype), torch.arange(GRID, dtype=m.dtype),
                            indexing="ij")
    r = float((w * ys).sum()) / tot
    c = float((w * xs).sum()) / tot
    center = (GRID - 1) / 2.0
    return (r - center) / EGO_SCALE, (c - center) / EGO_SCALE


def normalize_rows(p: np.ndarray) -> np.ndarray:
    return p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-8)


def scene_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """同じ場所のマスどうしの見た目の似かた（cos）の平均。目が跳ぶと下がる。"""
    return float(np.mean(np.sum(a * b, axis=1)))


def _match_center(a, feats, n, ci, cj, radius):
    """(ci, cj) のまわり ±radius マスで見た目 a に一番近いマス群の重心（マス単位）と最高の cos。"""
    cells = [(i, j) for i in range(max(ci - radius, 0), min(ci + radius, n - 1) + 1)
             for j in range(max(cj - radius, 0), min(cj + radius, n - 1) + 1)]
    cos = np.array([float(feats[i * n + j] @ a) for i, j in cells])
    best = float(cos.max())
    keep = cos >= best - 0.05
    w = cos[keep] - (best - 0.05) + 1e-3
    ii = np.array([c[0] for c in cells], dtype=np.float64)[keep]
    jj = np.array([c[1] for c in cells], dtype=np.float64)[keep]
    return float(np.sum(w * ii) / w.sum()), float(np.sum(w * jj) / w.sum()), best


def track(prev_dets: list[dict], feats: np.ndarray, n: int, prev_feats: np.ndarray | None = None,
          radius: int = TRACK_PATCHES, same_cos: float = SAME_COS,
          shift_px: tuple[float, float] = (0.0, 0.0)) -> list[dict]:
    """前の回の物を、今の DINOv2 のマス（正規化済み、(n*n, 384)）の中で探し直す。
    前の位置のまわり ±radius マスで、物の見た目との cos が最高に近いマスの重心を求め、
    前の回の画像で同じように求めた重心からのずれだけ位置を動かす（止まっている物は動かない）。
    shift_px（画面全体のずれ、目の動き）があれば、今の画像ではその分ずらした所のまわりを探す。
    最高の cos が same_cos 未満なら見失ったとして落とす。"""
    scale = 224.0 / n
    out = []
    for d in prev_dets:
        a = np.asarray(d["appearance"], dtype=np.float32)
        a = a / max(float(np.linalg.norm(a)), 1e-8)
        ci = min(max(int(d["pos"][1] // scale), 0), n - 1)
        cj = min(max(int(d["pos"][0] // scale), 0), n - 1)
        sx, sy = d["pos"][0] + shift_px[0], d["pos"][1] + shift_px[1]
        if not (0.0 <= sx < 224.0 and 0.0 <= sy < 224.0):
            continue                                  # 目が動いて視野の外へ出た
        si = min(max(int(sy // scale), 0), n - 1)
        sj = min(max(int(sx // scale), 0), n - 1)
        ni, nj, best = _match_center(a, feats, n, si, sj, radius)
        if best < same_cos:
            continue
        if prev_feats is not None:
            pi, pj, _ = _match_center(a, prev_feats, n, ci, cj, radius)
            pi, pj = pi + (si - ci), pj + (sj - cj)
        else:
            pi, pj = sy / scale - 0.5, sx / scale - 0.5
        cx = float(min(max(sx + (nj - pj) * scale, 0.0), 223.0))
        cy = float(min(max(sy + (ni - pi) * scale, 0.0), 223.0))
        nd = dict(d)
        nd["pos"] = (cx, cy)
        out.append(nd)
    return out


class Eyein(Box):
    BOX_ID = "eyein"

    def __init__(self, detect_every: int = DETECT_EVERY, detector: str = "floor") -> None:
        super().__init__()
        self.va = None
        self.detector = detector          # "floor"（段階3 作り直し）／"sam"（段階2までの MobileSAM）
        self.detect_every = int(detect_every)
        self._model = None
        self._sal = None
        self.device = "cpu"
        self.last_dets: list[dict] = []   # 記録用（その回の切り出し結果）
        self.reset()

    def setup(self, config: dict | None = None, answer_check: bool = False) -> None:
        """DINOv2 と目立ちの地図を作る。detector="sam" か answer_check=True のときだけ、旧太郎の
        VisualAttention（MobileSAM を読む）も作る。"""
        core_src = os.path.join(_REPO, "taro_core", "src")
        if core_src not in sys.path:
            sys.path.insert(0, core_src)
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(config or {})
        self.cohesion_gap_px = cfg.get("cohesion_gap_px")
        if self.detector == "sam" or answer_check:
            from brain.cerebral_cortex.parietal_lobe.visual_attention import VisualAttention
            self.va = VisualAttention(config=cfg)
            self._model, self._sal, self.device = self.va._model, self.va._sal, self.va.device
        else:
            from brain.midbrain.salience_map import SalienceMap
            att = cfg["attention"]
            self._sal = SalienceMap(cell=int(att.get("cell", GRID)),
                                    weights=tuple(att.get("weights", (1.0, 1.0, 1.0, 1.0))),
                                    dog_iters=int(att.get("dog_iters", 10)))
            self.device = os.environ.get("TARO_VISION_DEVICE") or (
                "cuda" if torch.cuda.is_available() else "cpu")
            self._model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                                         verbose=False).eval().to(self.device)
        self._ready = True

    def reset(self) -> None:
        if self._sal is not None:
            # 目立ちの地図は前のコマとの差（動き）を持つので、試行の頭で前のコマを捨てる
            self._sal._prev_gray = None
        self.last_dets = []
        self._since_detect = 0
        self._prev_feats = None
        self._prev_gray = None
        self.last_shift_px = (0.0, 0.0)   # 記録用：前の回から画面全体がずれた分（画素）
        self.last_mode = ""       # 記録用："floor"／"detect"（MobileSAM で全体を切り出した）／"track"

    def see(self, img) -> tuple[list, torch.Tensor]:
        img224 = np.asarray(img)
        if img224.shape[:2] != (224, 224):
            from PIL import Image
            img224 = np.array(Image.fromarray(np.clip(img224, 0, 255).astype(np.uint8))
                              .resize((224, 224)))
        img224 = np.clip(img224, 0, 255).astype(np.uint8)
        sal = self._sal.update(img224, (0.0, 0.0), False)["salience"]
        sal = np.clip(np.nan_to_num(np.asarray(sal, dtype=np.float32)), 0.0, 1.0)
        gray = img224.astype(np.float32).mean(axis=2)
        dx, dy = 0.0, 0.0
        if self._prev_gray is not None:
            dx, dy, _peak = image_shift(self._prev_gray, gray)
        self._prev_gray = gray
        self.last_shift_px = (dx, dy)
        if self.detector == "floor":
            from .. import floor_objects
            dets = floor_objects.detect(self._model, self.device, img224)
            self.last_mode = "floor"
        else:
            dets = self._see_sam(img224, dx, dy)
        step = 224.0 / GRID
        for d in dets:
            c = int(min(max(d["pos"][0] // step, 0), GRID - 1))
            r = int(min(max(d["pos"][1] // step, 0), GRID - 1))
            d["sal"] = float(sal[r, c])
        dets = sorted(dets, key=lambda d: -d["sal"])[:MAX_OBJECTS]
        self.last_dets = dets
        objs = [(noisy(appearance_tag(d["appearance"])), bump(*d["pos"])) for d in dets]
        return objs, torch.as_tensor(sal, dtype=DTYPE)

    def _see_sam(self, img224, dx, dy) -> list[dict]:
        """段階2までの切り出し（MobileSAM。DETECT_EVERY 回に1回、間は前の物を見た目で追う）。"""
        va = self.va
        p, n = va._patch_features(va._model, img224, va.device)
        feats = normalize_rows(np.asarray(p, dtype=np.float32))
        prev_feats = self._prev_feats
        jumped = (prev_feats is not None
                  and (np.hypot(dx, dy) > JUMP_PX
                       or scene_similarity(feats, prev_feats) < SCENE_SAME_COS))
        self._prev_feats = feats
        if (not self.last_dets or jumped or self.detect_every <= 1
                or self._since_detect + 1 >= self.detect_every):
            dets = va._detect(p, n, img=img224, mask_generator=va._mgen,
                              cohesion_gap_px=self.cohesion_gap_px)
            self._since_detect = 0
            self.last_mode = "detect"
        else:
            dets = track(self.last_dets, feats, n, prev_feats, shift_px=(dx, dy))
            self._since_detect += 1
            self.last_mode = "track"
        return dets

    def step(self, inbox):
        out = self.passthrough()
        if not getattr(self, "_ready", False) or inbox.get(ENV) is None:
            return out
        objs, sal = self.see(inbox[ENV])
        out["gaze"] = objs
        # 段階4：物は物の名簿へ。最後に目の動きの地図（荷札ゼロ）を足す。段階3で意味の辞書・目を向ける
        # 反射へ直接送っていた物の列は、名簿が引き受けたので送らない（矢印は構造図どおり残し、ゼロ）
        out["objfile"] = objs[:MAX_OBJECTS - 1] + [(torch.zeros(TAG_DIM, dtype=DTYPE),
                                                   ego_map(*self.last_shift_px))]
        out["prio"] = sal
        return out
