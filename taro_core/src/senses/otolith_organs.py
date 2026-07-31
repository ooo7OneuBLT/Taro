"""耳石器（前庭迷路の一部）— 卵形嚢・球形嚢。直線加速度と、重力に対する頭の傾きを感知する。

【対応】MIMoの前庭センサーは「加速度3＋角速度3」の6次元(`vestibular_acc`+`vestibular_gyro`、
`MIMo/mimoVestibular/vestibular.py`)。このうち**加速度3軸**が耳石器に相当する
（耳石(otoconia)がゼラチン膜の上で慣性によりずれ、直線加速度・重力方向を感知することに対応）。

【2026-07-26】重力方向の推定と、傾きに対する眼球反応（ocular counter-roll）を実装した。
それまでは瞬間の加速度をそのまま返すだけで [ARBITRARY・未実装] とラベルしていた。

【なぜ実装したか】三半規管（動的）だけでは「頭が傾いたまま静止したとき、眼が最終的に
どこで落ち着くか」が決まらない。実際 VOR が眼を回し続けて可動域の限界に張り付いた
（研究日誌 2026-07-26 続き6）。人間では**この着地点を決めるのが耳石器**。

【重力と運動加速度は区別できない】
耳石器（と加速度センサー）が感じるのは「比力」＝
    f = a_linear - g        （g=重力加速度ベクトル、a_linear=並進運動の加速度）
つまり「重力で下に引かれている」のか「上に加速している」のかを**原理的に区別できない**。
これは tilt-translation ambiguity（重力慣性のあいまいさ）と呼ばれ、
人間の脳も同じ制約を抱えている（モーションシックネスの一因）。

【どう対処するか・3案あるうちの①を採用】
  ①比力を低域通過フィルタにかけて重力方向を推定する ← 採用
     運動加速度は短時間で向きが変わるので平均すると消え、重力は残る。
     ロボティクスの姿勢推定（AHRS）で使う相補フィルタと同じ発想。
     **人間が持っている情報だけで解く**＝逸脱が最小。
  ②MuJoCo が持つ頭の姿勢から重力方向を直接計算する
     注意：人間が持っていない情報を使う[ARBITRARY]。切り分け用の簡易版としてのみ可。
  ③Merfeld の observer model / カルマンフィルタで内部モデルを持つ
     理論的に最も正しいが実装コストが高い。将来の候補。

【ocular counter-roll（OCR）の利得】
  静的OCRの利得は **約0.15**（30度の頭部傾斜に対し眼球回旋は約4.5度）
  PubMed 16603921 "A clinical test of otolith function: static ocular counterroll
  with passive head tilt"。他の報告でも 0.1〜0.2 で一致（「代償率は約10%」の記述も）。
  つまり人間は傾きをほとんど補正していない。頭が40度傾いても眼は約6度しか回らない。

【ロールとピッチ】
  ロール（左右の傾き）のみ実装する。
  "ocular counter-roll" という用語自体がロール軸専用の現象で、
  ピッチ（前後の傾き）に対応する垂直眼球偏位は**標準値と呼べる利得が見つからない**
  （2026-07-26 の調査で確認）。ピッチ軸は当面、半規管の経路だけに任せる。

【動的OCRを別に実装しない理由】
  「傾けている最中の counter-roll は静的より大きい」という報告（Bockisch & Haslwanter 2003）は、
  傾けている最中は**半規管も同時に興奮している**ため両者が足し合わさって観測される、と
  説明される。半規管の経路と耳石の経路を単純に足せば自動的に再現されるので、
  第3の項は要らない。
"""
import numpy as np

# 重力加速度の大きさ [m/s^2]
GRAVITY = 9.81

# ocular counter-roll の利得。[Tier1] PubMed 16603921（30度→4.5度＝0.15）
OCR_GAIN = 0.15

# 重力方向を推定する低域通過フィルタの時定数 [秒]。
# 注意：[Tier3・ARBITRARY] 文献に直接の値がない。
#   「運動加速度は短時間で向きが変わり、重力は変わらない」を分ける長さとして選ぶ。
#   新生児の頭部運動は数Hz以下なので、数秒あれば十分に平均できる。
GRAVITY_ESTIMATE_TAU = 2.0


def read(vestibular_obs):
    """前庭ベクトル(6次元)から、耳石器に相当する直線加速度3軸を取り出す（生の値）。

    注意：動特性を通していない生の比力。傾きを知りたいときは `OtolithOrgans` を使うこと。
    """
    return np.asarray(vestibular_obs)[0:3]


class OtolithOrgans:
    """重力方向を推定し、傾きに対する眼球反応（ocular counter-roll）を出す耳石器。

    使い方:
        oto = OtolithOrgans()
        oto.update(accel, dt)          # accel は比力3軸[m/s^2]（頭部座標系）
        roll = oto.roll_angle()        # 推定した左右の傾き[rad]
        target = oto.counter_roll()    # 眼球の目標回旋角[rad]（＝ -0.15 × roll）

    注意：入力は**頭部座標系**の比力であること（MIMo の vestibular_acc がそれに当たる）。
      頭が傾くと、この座標系から見た重力の向きが変わる＝それが傾きの手がかりになる。
    """

    def __init__(self, tau=GRAVITY_ESTIMATE_TAU, gain=OCR_GAIN, upright=None):
        self.tau = float(tau)
        self.gain = float(gain)
        # 推定中の重力方向（頭部座標系）。初期値は「まだ分からない」ので0から始め、
        # 最初の数秒で実測に収束させる。
        self.gravity_est = np.zeros(3, dtype=float)
        self._warm = 0
        # 「直立（傾き0）」のときに重力がどちらを向いて見えるか。
        # 注意：仰向けの太郎では重力は体の背中方向から来るので、単純に -z とは限らない。
        #   None なら最初の観測を基準にする（＝リセット直後の姿勢を傾き0とみなす）。
        self.upright = None if upright is None else np.asarray(upright, dtype=float)

    def reset(self):
        self.gravity_est[:] = 0.0
        self._warm = 0

    def update(self, accel, dt):
        """比力 accel[m/s^2] を1ステップ入れて、推定した重力方向を返す。

        低域通過で平均する＝運動の加速度（短時間で向きが変わる）は消え、
        重力（ずっと同じ向き）が残る。
        """
        f = np.asarray(accel, dtype=float).reshape(-1)[:3]
        alpha = 1.0 - float(np.exp(-float(dt) / max(self.tau, 1e-9)))
        if self._warm == 0:
            self.gravity_est[:] = f      # 初回は観測をそのまま入れる（収束を速める）
        else:
            self.gravity_est += alpha * (f - self.gravity_est)
        self._warm += 1
        if self.upright is None and self._warm >= 1:
            # リセット直後の重力方向を「傾き0」の基準にする
            n = np.linalg.norm(self.gravity_est)
            if n > 1e-6:
                self.upright = self.gravity_est / n
        return self.gravity_est.copy()

    def roll_angle(self):
        """推定した左右の傾き[rad]。基準方向からのずれのうち、ロール成分を返す。

        注意：ロールだけを取り出す（ピッチは実装しない）。基準方向 upright と
        現在の推定重力方向のなす角のうち、頭部座標のy軸（左右）まわりの成分。
        """
        if self.upright is None:
            return 0.0
        g = self.gravity_est
        n = float(np.linalg.norm(g))
        if n < 1e-6:
            return 0.0
        gh = g / n
        # 基準からのずれを回転として見て、そのロール成分（x-z平面内の回転）を取る
        # 注意：簡略化：x成分（頭部の左右方向）のずれをロールとみなす
        return float(np.arcsin(np.clip(gh[0] - self.upright[0], -1.0, 1.0)))

    def counter_roll(self):
        """眼球の目標回旋角[rad]。傾きと逆向きに、利得 0.15 だけ回す。

        人間は傾きをほとんど補正しない：40度傾いても眼は約6度しか回らない。
        """
        return -self.gain * self.roll_angle()
