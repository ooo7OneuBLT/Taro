"""目の動き方（目だけの場面）。今の目を向ける反射の決まりを、筋肉なしの「動き方の式」で書いたもの。

MIMo の場面で目を動かしていた仕組み（taro_core/src/brain/midbrain/orienting.py の OrientingReflexV2、
地図を読んで撃つ道）から、core_v2 が使っている決まりだけを移した：
* set_map_target(h, v)：方向（[-1, 1]、右・上が正）を控える。撃つまで残る（MIMo と同じく、reset は消さない
  ので、run_v2.run_trial が試行の頭で `_map_target = None` を書く）
* 撃てる間隔：前に撃ってから MAP_LATENCY_S（0.25秒）たっていれば撃つ（orienting.py の map_latency）。
  1発の最中は撃たない
* 1発で動く角度：今の向き ＋ 方向×視野の半分（30度）×SACCADE_FRAC（1.0）（orienting.py の SACCADE_FRAC・
  VISION_FOVY_DEG/2）。右・上が正
* 目の可動域を超える狙いは、可動域の端で止まる（MIMo では目が機械的に止まる。orienting.py の blocked_target）

MIMo と違うところ（はっきり書いておく）：
* 速さ：MIMo は目の筋肉の力で動き、着くまでの時間は筋肉の強さで決まっていた（実測の跳びは指令の 0.55〜0.95倍、
  段階3の作り直しの仕様書）。ここでは人間の大人の主系列（Carpenter 1988：かかる時間 ≈ 21ミリ秒＋2.2ミリ秒×振幅（度））
  で、狙った所へちょうど着く。途中の形は最小躍度（なめらかな S 字）［Tier2・速さの式は大人の値。乳児は少し遅い（記憶ベース・要確認）］
* 撃っていない間、目は動かない（MIMo では首や体の揺れ・前庭動眼反射・目を真ん中へ戻すばねで、少し漂った）
* 首は無い（MIMo の場面も首は留めてあった：neck_fix）
"""

from __future__ import annotations

import math

MAP_LATENCY_S = 0.25        # orienting.py の self.map_latency（地図を読んで撃つ道の間隔）
HALF_FOV_DEG = 30.0         # orienting.py の VISION_FOVY_DEG / 2
SACCADE_FRAC = 1.0          # orienting.py の SACCADE_FRAC（既定 1.0）
PAN_LIMIT_DEG = 45.0        # MIMo の目の左右の可動域 90度（逸脱リスト「眼球の可動域 90°」）の半分
TILT_LIMIT_DEG = 33.0       # MIMo の目の上の端 +33度（逸脱リストの VOR の記録）。下も同じとみなす［Tier3］
MS_BASE_S = 0.021           # 主系列：かかる時間 = MS_BASE_S + MS_PER_DEG_S × 振幅（Carpenter 1988）
MS_PER_DEG_S = 0.0022


def saccade_duration(amp_deg: float) -> float:
    return MS_BASE_S + MS_PER_DEG_S * abs(float(amp_deg))


def min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


class EyeKinematics:
    """目の向き（pan＝右が正、tilt＝上が正、度）を持ち、step(dt) で時間を進める。
    env.unwrapped._orienting としてそのまま使える（run_v2 が使うのは set_map_target と _map_target）。"""

    def __init__(self) -> None:
        self.map_latency = MAP_LATENCY_S
        self.reset()

    def reset(self) -> None:
        self.pan = 0.0
        self.tilt = 0.0
        self._t = 0.0
        self._map_target = None
        self._last_saccade_t = -1e9
        self._sacc = None              # (開始時刻, かかる時間, 始めの向き, 狙いの向き)
        self.n_saccades = 0
        self.map_fired = 0
        self.blocked_target_n = 0
        self.fire_log: list[tuple] = []   # (撃った時刻, h, v, 振幅（度）, かかる時間)

    # ---- run_v2・道具から ----
    def set_map_target(self, h_dir: float, v_dir: float, t=None, latency_s=None) -> None:
        self._map_target = (max(-1.0, min(1.0, float(h_dir))), max(-1.0, min(1.0, float(v_dir))))
        if latency_s is not None:
            self.map_latency = float(latency_s)

    def is_saccading(self) -> bool:
        return self._sacc is not None

    def set_angles(self, pan: float, tilt: float) -> None:
        """道具用：目の向きを直接置く（跳びではない。記録に残すのは env の側）。"""
        self.pan = max(-PAN_LIMIT_DEG, min(PAN_LIMIT_DEG, float(pan)))
        self.tilt = max(-TILT_LIMIT_DEG, min(TILT_LIMIT_DEG, float(tilt)))
        self._sacc = None

    # ---- 環境の1歩ごと ----
    def step(self, dt: float) -> None:
        self._t += float(dt)
        if self._sacc is None and self._map_target is not None \
                and (self._t - self._last_saccade_t) >= self.map_latency - 1e-9:
            h, v = self._map_target
            self._map_target = None
            want_p = self.pan + SACCADE_FRAC * h * HALF_FOV_DEG
            want_t = self.tilt + SACCADE_FRAC * v * HALF_FOV_DEG
            tgt_p = max(-PAN_LIMIT_DEG, min(PAN_LIMIT_DEG, want_p))
            tgt_t = max(-TILT_LIMIT_DEG, min(TILT_LIMIT_DEG, want_t))
            if (tgt_p, tgt_t) != (want_p, want_t):
                self.blocked_target_n += 1
            amp = math.hypot(tgt_p - self.pan, tgt_t - self.tilt)
            dur = saccade_duration(amp)
            # 撃った時刻は「この歩の始め」（MIMo も撃った歩から筋へ指令が入る）
            self._sacc = (self._t - dt, dur, (self.pan, self.tilt), (tgt_p, tgt_t))
            self._last_saccade_t = self._t
            self.n_saccades += 1
            self.map_fired += 1
            if len(self.fire_log) < 4000:
                self.fire_log.append((round(self._t, 3), h, v, round(amp, 3), round(dur, 4)))
        if self._sacc is not None:
            t0, dur, (p0, q0), (p1, q1) = self._sacc
            s = min_jerk((self._t - t0) / dur)
            self.pan = p0 + (p1 - p0) * s
            self.tilt = q0 + (q1 - q0) * s
            if self._t - t0 >= dur - 1e-12:
                self.pan, self.tilt = p1, q1
                self._sacc = None
