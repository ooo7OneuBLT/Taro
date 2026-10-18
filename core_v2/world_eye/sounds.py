"""段階9：物の音を作る（仕様書 付録「音を作る」）。MuJoCo に音は無いので、出来事（親が物を動かす・すぷうんで皿をたたく）に
合わせて短い波形（16kHz、0.3〜0.8秒）を作り、物の音の入口（boxes/soundin.py）へ渡す。

種類ごとに作り方を1つ（仕様書のとおり）。個体ごとに高さ ±15%・長さ ±20%（決まった種）：
* 犬（わんわん）    … 雑音の短い連打2回（500〜2000Hz を残す）
* 猫（にゃんにゃん）… 上がって下がる倍音（基本 400→700→450Hz、倍音5つ）
* 車（ぶうぶう）    … 80〜150Hz のうなり（のこぎり波ふう、倍音8つ、ゆっくり揺れる）
* ライオン（がおお）… 低い雑音（400Hz より下）、ふくらんで消える
* アヒル（があがあ）… 鼻にかかった2音（基本 300Hz、3〜6倍音を強く）
* 電車（でんしゃ）  … 4回/秒の雑音の打ち＋最後に笛（1500Hz）
* すぷうん          … 3〜5kHz の減る音（4つの音を重ねる。皿をたたく「チン」）
音の形は人が聞いて「それらしい」程度の単純なもの［Tier3・モデル上の選択］。.wav でも残せる（save_wav）。
numpy だけで作る（scipy は使わない）。
"""

from __future__ import annotations

import hashlib
import wave

import numpy as np

SR = 16000
PITCH_JITTER = 0.15
LEN_JITTER = 0.2
BASE_LEN = {"dog": 0.5, "cat": 0.6, "car": 0.7, "lion": 0.7, "duck": 0.45, "train": 0.75, "spoon": 0.4}
SOUND_OF_WORD = {"わんわん": "dog", "にゃんにゃん": "cat", "ぶうぶう": "car", "がおお": "lion", "があがあ": "duck",
                 "でんしゃ": "train", "すぷうん": "spoon"}


def _rng(*parts) -> np.random.Generator:
    return np.random.default_rng(int.from_bytes(hashlib.sha256("/".join(map(str, parts)).encode()).digest()[:8], "little"))


def _band_noise(g, n: int, lo: float, hi: float) -> np.ndarray:
    x = g.standard_normal(n)
    f = np.fft.rfft(x)
    fr = np.fft.rfftfreq(n, 1.0 / SR)
    f[(fr < lo) | (fr > hi)] = 0.0
    y = np.fft.irfft(f, n)
    return y / (np.abs(y).max() + 1e-9)


def _env(n: int, attack: float = 0.05, release: float = 0.3) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    return np.minimum(1.0, t / max(attack, 1e-3)) * np.minimum(1.0, (1.0 - t) / max(release, 1e-3))


def _harm(f0: np.ndarray, amps) -> np.ndarray:
    ph = 2 * np.pi * np.cumsum(f0) / SR
    return sum(a * np.sin((h + 1) * ph) for h, a in enumerate(amps))


def instance_params(kind: str, k: int, event: int = 0) -> tuple[float, float]:
    """個体 k の高さの倍率・長さの倍率（決まった種。event は同じ個体の何回目の音か：わずかに揺らす分は無い＝同じ音）。"""
    g = _rng("sound", kind, k)
    return 1.0 + PITCH_JITTER * (2 * g.random() - 1), 1.0 + LEN_JITTER * (2 * g.random() - 1)


def synth(kind: str, k: int = 0, seed: int = 0) -> np.ndarray:
    """種類 kind・個体 k の音（float32、−1〜1）。seed は雑音の種（同じ個体でも鳴るたびに雑音は少し違う）。"""
    p, lf = instance_params(kind, k)
    n = int(SR * BASE_LEN[kind] * lf)
    g = _rng("noise", kind, k, seed)
    t = np.arange(n) / SR
    if kind == "dog":
        y = np.zeros(n)
        bl = int(n * 0.3)
        for s in (0, int(n * 0.5)):
            seg = _band_noise(g, bl, 500 * p, 2000 * p) * _env(bl, 0.05, 0.6)
            seg += 0.5 * np.sin(2 * np.pi * 450 * p * np.arange(bl) / SR) * _env(bl, 0.05, 0.6)
            y[s:s + bl] += seg[:max(0, min(bl, n - s))]
    elif kind == "cat":
        u = t / t[-1]
        f0 = p * np.where(u < 0.5, 400 + 600 * u, 700 - 500 * (u - 0.5))
        y = _harm(f0, [1.0, 0.6, 0.4, 0.25, 0.15]) * _env(n, 0.1, 0.4)
    elif kind == "car":
        f0 = p * (115 + 35 * np.sin(2 * np.pi * 1.5 * t))
        y = _harm(f0, [1.0 / (h + 1) for h in range(8)]) * _env(n, 0.15, 0.3)
    elif kind == "lion":
        y = _band_noise(g, n, 60 * p, 400 * p) * _env(n, 0.3, 0.5)
        y += 0.4 * _harm(np.full(n, 90 * p), [1.0, 0.5, 0.3]) * _env(n, 0.3, 0.5)
    elif kind == "duck":
        y = np.zeros(n)
        bl = int(n * 0.42)
        for s in (0, int(n * 0.55)):
            seg = _harm(np.full(bl, 300 * p), [0.3, 0.4, 1.0, 1.0, 0.8, 0.6, 0.2]) * _env(bl, 0.08, 0.3)
            y[s:s + bl] += seg[:max(0, min(bl, n - s))]
    elif kind == "train":
        chug = (np.sin(2 * np.pi * 4.0 * t) > 0.3).astype(float)
        y = _band_noise(g, n, 200 * p, 3000 * p) * chug * 0.7
        w0 = int(n * 0.6)
        y[w0:] += 0.8 * np.sin(2 * np.pi * 1500 * p * t[w0:]) * _env(n - w0, 0.1, 0.3)
    elif kind == "spoon":
        y = sum(a * np.sin(2 * np.pi * f * p * t) for f, a in ((3000, 1.0), (3700, 0.7), (4300, 0.5), (5000, 0.3)))
        y = y * np.exp(-t / 0.08)
    else:
        raise ValueError(f"音の無い種類: {kind}")
    y = y / (np.abs(y).max() + 1e-9) * 0.8
    return y.astype(np.float32)


def save_wav(path: str, y: np.ndarray) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())
