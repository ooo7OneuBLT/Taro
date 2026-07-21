"""色付きノイズ（1/f^β）生成器。

【背景】太郎の運動性喃語(D/scripts/d_c5_motor_quality.py::make_policy の babble)は
毎tick独立なガウスノイズ＝白色雑音(β=0)で、実測した自己相関(lag-1)は0.034（ほぼ0）
だった（E/scripts/e_efference_check.py、E/docs/研究日誌.md 2026-07-21）。
実際の乳児の運動探索ノイズは生後8週相当でβ≈0.7、30週相当でβ≈0.9まで滑らかになる
（López et al. 2026, "Baby Noise" — MIMoを使ったRL探索ノイズ研究）。

【やること】Timmer & König (1995) のスペクトル整形法：周波数領域で振幅を f^(-β/2) に
整形したガウス雑音を作り、逆FFTで時間領域に戻す。βが大きいほど低周波成分が強い＝
滑らかで持続的な動きになる。分散は呼び出し側でstdを掛けられるよう単位分散に正規化する。

【使い方】次元ごとに独立な過程をブロック単位(既定2000tick分)で作り、尽きたら次のブロックを
その時点のβで作り直す＝βを毎tick変えても（発達に応じたスケジュールでも）動く。
"""
import numpy as np


class ColoredNoiseGenerator:
    """1/f^β ノイズを次元ごとに独立生成する。sample(beta) を毎tick呼ぶと (n_dim,) を返す。"""

    def __init__(self, n_dim, block_len=2000, seed=None):
        self.n_dim = n_dim
        self.block_len = block_len
        self.rng = np.random.default_rng(seed)
        self._buf = None
        self._pos = 0

    def _make_block(self, beta):
        L = self.block_len
        freqs = np.fft.rfftfreq(L).copy()
        freqs[0] = freqs[1] if L > 1 else 1.0  # DC成分の0除算を避ける
        amp = freqs ** (-beta / 2.0)
        block = np.empty((L, self.n_dim), dtype=np.float32)
        for d in range(self.n_dim):
            re = self.rng.normal(size=amp.shape) * amp
            im = self.rng.normal(size=amp.shape) * amp
            im[0] = 0.0
            if L % 2 == 0:
                im[-1] = 0.0
            x = np.fft.irfft(re + 1j * im, n=L)
            x = x - x.mean()
            x = x / (x.std() + 1e-9)
            block[:, d] = x
        self._buf = block
        self._pos = 0

    def sample(self, beta):
        if self._buf is None or self._pos >= self.block_len:
            self._make_block(beta)
        v = self._buf[self._pos]
        self._pos += 1
        return v
