"""三半規管（前庭迷路の一部）— 頭の回転（角速度）を感知する。

【対応】MIMoの前庭センサーは「加速度3＋角速度3」の6次元(`vestibular_acc`+`vestibular_gyro`、
`MIMo/mimoVestibular/vestibular.py`)。このうち**角速度3軸**が三半規管に相当する
（3本の半規管がそれぞれ別平面の回転を感知することに対応）。

【簡略化・ラベリング】実際の三半規管はリンパ液の慣性で動く動的なセンサーで、
時定数（回転し続けると感覚が薄れる適応）を持つ。新生児はこの時定数が成人の約半分と
報告されている（`E/scripts/e_vor.py`参照）。ここではMuJoCoが計算した**瞬間の角速度**を
そのまま使い、この動特性（適応・時定数）は再現していない[ARBITRARY・未実装]。
"""
import numpy as np


def read(vestibular_obs):
    """前庭ベクトル(6次元)から、三半規管に相当する角速度3軸を取り出す。"""
    return np.asarray(vestibular_obs)[3:6]
