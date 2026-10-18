"""物の音の入口（入口（感覚）。段階9、2026-10-18、ユーザー承認の仕様書 `2026-10-18_段階9_仕様書_名詞を厚く.md` §3・付録）。

出来事の音（親が物を動かしたときの吠える声・車のうなり・すぷうんで皿をたたく音など。環境の info["object_sound"]、
16kHz の波形）を、見た目の荷札と同じ作り方で 64 の荷札にする。耳（語の入口 earin→listen）とは別の入口［Tier3］。
* 窓 25ms・ずらし 10ms のハン窓で、50〜8000Hz を対数メルの帯 N_BANDS（32）本に分け、時間で平均する（仕様書どおり）
* 帯の平均から、その音の帯の平均（全体の大きさ）を引く（音の大きさではなく、周波数の形だけを残す。ケプストラムの平均を
  引くのと同じ考え。引かないと、どの音の荷札も大きさの向きに揃い、別の種類の音どうしの似かたが平均 0.62、引くと 0.30。
  2026-10-18、world_eye/sounds.py の7種類×4個体で測った）
* 決まった乱数の行列（32→64、種 PROJECTION_SEED。見た目の行列とは別の種）で 64 にして長さ1にそろえる。学習しない（生まれつき）
* 音が続く間（波形の長さを 0.1 秒の回に切り上げた回数）、同じ荷札を意味の辞書（semantic）と種類の層（kind）へ出す。
  強さは荷札の長さで表さず（TAG は長さ1の決まり）、意味の辞書は荷札が届いている回を強さ1として足す（仕様書の v＝1）
* **時間の形を残す（段階9、2026-10-18、ユーザー決定、仕様書 `2026-10-18_段階9_仕様書_意味の辞書まばら化.md`）**：既定（shape="time"）は、
  帯の時間平均の代わりに、時間で N_SLICES（4）つに切った帯の平均（4×32、音の全体の大きさを引く）と、N_GROUPS（8）つの帯の組ごとの
  「変わる速さ」（10ms 先・40ms 先との差の大きさの平均、2×8、平均を引く）の2つの塊にし、塊ごとに長さ1にそろえて並べ、決まった乱数で 64 へ
  （生まれつきの耳の前段。Chi・Ru・Shamma 2005 を小さくしたもの）。帯の時間平均だけだと あひる・犬 0.98、猫・電車 0.96 だったのが
  0.84・0.70 になる（7種類×4個体で測った）。shape="mean" で前の荷札（1ビットも同じ）に戻せる（比べる用）
numpy の FFT だけを使う（scipy は使わない）。tagging.py は段1の指紋に入っているので触らず、行列はここで作る。
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ..formats import DTYPE, TAG_DIM, make_tag
from .base import ENV, Box

SR = 16000
N_BANDS = 32
WIN_S, HOP_S = 0.025, 0.010
F_LO, F_HI = 50.0, 8000.0
NFFT = 512
PROJECTION_SEED = 20261018
TICK_S = 0.1
SHAPE = "time"           # 既定。"mean"＝前の帯の時間平均（比べる用）
N_SLICES = 4             # 音を時間で切る数
N_GROUPS = 8             # 変わる速さを測る帯の組
LAGS = (1, 4)            # 10ms 先・40ms 先
TIME_PROJ_SEED = 20261019

_proj = None
_fb = None
_tproj = None


def _mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def _imel(m):
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def filterbank() -> np.ndarray:
    global _fb
    if _fb is None:
        pts = _imel(np.linspace(_mel(F_LO), _mel(F_HI), N_BANDS + 2))
        fq = np.fft.rfftfreq(NFFT, 1.0 / SR)
        fb = np.zeros((N_BANDS, len(fq)))
        for b in range(N_BANDS):
            lo, c, hi = pts[b], pts[b + 1], pts[b + 2]
            fb[b] = np.clip(np.minimum((fq - lo) / (c - lo), (hi - fq) / (hi - c)), 0.0, None)
        _fb = fb
    return _fb


def projection() -> np.ndarray:
    global _proj
    if _proj is None:
        _proj = np.random.default_rng(PROJECTION_SEED).standard_normal((N_BANDS, TAG_DIM)) / math.sqrt(TAG_DIM)
    return _proj


def band_mean(wave) -> np.ndarray:
    """波形 → 対数メルの帯32本の時間平均（全体の大きさを引いたもの）。"""
    y = np.asarray(wave, dtype=np.float64).reshape(-1)
    win, hop = int(WIN_S * SR), int(HOP_S * SR)
    if len(y) < win:
        y = np.pad(y, (0, win - len(y)))
    starts = range(0, len(y) - win + 1, hop)
    frames = np.stack([y[s:s + win] for s in starts]) * np.hanning(win)
    p = np.abs(np.fft.rfft(frames, NFFT)) ** 2
    v = np.log(p @ filterbank().T + 1e-10).mean(axis=0)
    return v - v.mean()


def sound_tag(wave) -> torch.Tensor:
    return make_tag((band_mean(wave) @ projection()).astype(np.float32))


def _spec(wave) -> np.ndarray:
    y = np.asarray(wave, dtype=np.float64).reshape(-1)
    win, hop = int(WIN_S * SR), int(HOP_S * SR)
    if len(y) < win + 4 * hop * N_SLICES:
        y = np.pad(y, (0, win + 4 * hop * N_SLICES - len(y)))
    frames = np.stack([y[s:s + win] for s in range(0, len(y) - win + 1, hop)]) * np.hanning(win)
    p = np.abs(np.fft.rfft(frames, NFFT)) ** 2
    return np.log(p @ filterbank().T + 1e-10)        # 時間 × 32帯


def time_features(wave) -> np.ndarray:
    """波形 → [時間で4つに切った帯の平均（4×32） | 帯の組ごとの変わる速さ（2×8）]、塊ごとに長さ1。"""
    S = _spec(wave)
    sl = np.stack([c.mean(0) for c in np.array_split(S, N_SLICES)]) - S.mean()
    a = sl.reshape(-1)
    G = S.reshape(len(S), N_GROUPS, -1).mean(2)
    b = np.concatenate([np.abs(G[lag:] - G[:-lag]).mean(0) for lag in LAGS])
    b = b - b.mean()
    unit = lambda v: v / (np.linalg.norm(v) + 1e-12)   # noqa: E731
    return np.concatenate([unit(a), unit(b)])


def sound_tag_time(wave) -> torch.Tensor:
    global _tproj
    f = time_features(wave)
    if _tproj is None:
        _tproj = np.random.default_rng(TIME_PROJ_SEED).standard_normal((len(f), TAG_DIM)) / math.sqrt(TAG_DIM)
    return make_tag((f @ _tproj).astype(np.float32))


def make_sound_tag(wave, shape: str | None = None) -> torch.Tensor:
    """入口の荷札の作り方（shape：time＝時間の形を残す（既定）、mean＝帯の時間平均）。"""
    return sound_tag_time(wave) if (shape or SHAPE) == "time" else sound_tag(wave)


class Soundin(Box):
    BOX_ID = "soundin"

    def __init__(self, shape: str | None = None) -> None:
        super().__init__()
        self.shape = shape or SHAPE
        if self.shape not in ("time", "mean"):
            raise ValueError(f"shape は time か mean：{shape!r}")
        self.reset()

    def reset(self) -> None:
        self.tag = None
        self.left = 0          # あと何回、同じ荷札を出すか
        self.n_heard = 0       # 記録用

    def step(self, inbox):
        out = self.passthrough()
        w = inbox.get(ENV)
        if w is not None and len(np.asarray(w).reshape(-1)) > 0:
            self.tag = make_sound_tag(w, self.shape)
            self.left = max(1, math.ceil(len(np.asarray(w).reshape(-1)) / SR / TICK_S - 1e-9))
            self.n_heard += 1
        if self.left > 0 and self.tag is not None:
            for k in out:
                out[k] = self.tag
            self.left -= 1
        return out


