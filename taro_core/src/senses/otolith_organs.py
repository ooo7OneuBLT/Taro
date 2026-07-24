"""耳石器（前庭迷路の一部）— 卵形嚢・球形嚢。直線加速度と、重力に対する頭の傾きを感知する。

【対応】MIMoの前庭センサーは「加速度3＋角速度3」の6次元(`vestibular_acc`+`vestibular_gyro`、
`MIMo/mimoVestibular/vestibular.py`)。このうち**加速度3軸**が耳石器に相当する
（耳石(otoconia)がゼラチン膜の上で慣性によりずれ、直線加速度・重力方向を感知することに対応）。

【簡略化・ラベリング】実際の耳石器も慣性による動特性（適応）を持つが、ここではMuJoCoが
計算した**瞬間の加速度**をそのまま使い、この動特性は再現していない[ARBITRARY・未実装]。
"""
import numpy as np


def read(vestibular_obs):
    """前庭ベクトル(6次元)から、耳石器に相当する直線加速度3軸を取り出す。"""
    return np.asarray(vestibular_obs)[0:3]
