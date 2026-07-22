"""脊髄・中枢パターン発生器（CPG）相当のモジュール。

【解剖学的位置づけ、2026-07-23】新生児のwrithing GM（運動性喃語）は、皮質からの精密制御を
ほぼ受けず、脳幹・脊髄の自律的なリズム生成回路（central pattern generator, CPG）が主体と
考えられている（Hadders-Algra 2018ほか）。太郎のB-min（`d_c5_motor_quality.py`のE_WMEAN=0
経路）はこれに対応し、以下2つの本能を実装する：
  ①色付きノイズ（1/f^β）＝López et al. 2026 の実測βに基づく（旧colored_noise.py、ここに統合）
  ②粗いシナジー（脚・腕）＝Dominici 2011（脚, Tier1）・Physiopedia他（腕, Tier2）
運動野（motor_head/pc_latent/motor_gru、taro_brain_motor.py）や皮質脊髄路（w_mean混合、
`corticospinal.py`）とは別の、より下位の自律回路として位置づける。
"""
import numpy as np


class ColoredNoiseGenerator:
    """1/f^β ノイズを次元ごとに独立生成する。sample(beta) を毎tick呼ぶと (n_dim,) を返す。

    【使い方】次元ごとに独立な過程をブロック単位（既定2000tick分）で作り、尽きたら次の
    ブロックをその時点のβで作り直す＝βを毎tick変えても（発達に応じたスケジュールでも）動く。
    Timmer & König (1995) のスペクトル整形法：周波数領域で振幅を f^(-β/2) に整形したガウス
    雑音を作り、逆FFTで時間領域に戻す。βが大きいほど低周波成分が強い＝滑らかで持続的な動き。
    """

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


class CPG:
    """複数関節をまとめて駆動するCPG本体。独立ノイズ＋粗いシナジー（脚・腕）を合成して
    n_act次元の babbling 出力を1回のsample()で返す（呼び出し側は低頻度化キャッシュだけ
    考えればよい構成、2026-07-23の整理でd_c5_motor_quality.pyから抽出）。

    synergy=False（既定）なら各関節は完全独立＝従来のColoredNoiseGeneratorと1バイト差なし。
    """

    def __init__(self, n_act, leg_r=(), leg_l=(), arm_r=(), arm_l=(), seed=None):
        self.gen = ColoredNoiseGenerator(n_act, seed=seed)
        s = (lambda k: seed + k) if seed is not None else (lambda k: None)
        self.syn_leg = ColoredNoiseGenerator(1, seed=s(1))
        self.syn_arm_r = ColoredNoiseGenerator(1, seed=s(2))
        self.syn_arm_l = ColoredNoiseGenerator(1, seed=s(3))
        self._leg_r, self._leg_l = list(leg_r), list(leg_l)
        self._arm_r, self._arm_l = list(arm_r), list(arm_l)

    def sample(self, beta, synergy=False, syn_w=0.6):
        gen_np = self.gen.sample(beta)
        if not synergy:
            return gen_np
        gen_np = gen_np.copy()
        leg_s = float(self.syn_leg.sample(beta)[0])
        for i in self._leg_r:
            gen_np[i] = (1 - syn_w) * gen_np[i] + syn_w * leg_s
        for i in self._leg_l:   # 脚は左右逆位相（粗い交互パターン、Dominici 2011）
            gen_np[i] = (1 - syn_w) * gen_np[i] + syn_w * (-leg_s)
        arm_r_s = float(self.syn_arm_r.sample(beta)[0])
        for i in self._arm_r:
            gen_np[i] = (1 - syn_w) * gen_np[i] + syn_w * arm_r_s
        arm_l_s = float(self.syn_arm_l.sample(beta)[0])
        for i in self._arm_l:
            gen_np[i] = (1 - syn_w) * gen_np[i] + syn_w * arm_l_s
        return gen_np
