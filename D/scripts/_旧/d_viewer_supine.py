"""taro-C5用：仰向け環境(SupineMimoEnv)をフルモードのビューアで見る（視覚・ベータなし）。

スペース＝一時停止・再生。一時停止中は左パネルの「History」で巻き戻せる。
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "taro_core"))
import paths
paths.setup_brain_path()
sys.path.insert(0, paths.MIMO_DIR)

import mujoco
import mujoco.viewer
from d_supine_env import SupineMimoEnv

env = SupineMimoEnv(vision_params=None)
env.reset(seed=0)
m, d = env.unwrapped.model, env.unwrapped.data

print("フルモードのビューアを起動します（閉じるまで待機します）。")
print("仰向けの太郎（視覚・ベータなし）＝taro-C5の測定に使う予定の環境です。")
mujoco.viewer.launch(m, d)
