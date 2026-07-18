"""MuJoCo純正の「フルモード」ビューア。一時停止(スペースキー)・巻き戻し(履歴スクラブ)が使える。

launch_passive（d_viewer_quicktest.py）と違い、物理演算はMuJoCo自身が別スレッドで進める
（Python側は毎ステップ制御できない＝静止した確認用途向け）。ベータの目標位置は起動前に
一度だけ設定し、あとはMuJoCo純正のサーボ制御に任せる。

操作: スペース＝一時停止／再生／一時停止中は画面下のタイムラインバーで巻き戻し可能
      左ドラッグ=視点回転／右ドラッグ=平行移動／スクロール=ズーム
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
from d_beta_sitting_env import BetaSittingEnv

env = BetaSittingEnv()
env.reset(seed=0)
m, d = env.unwrapped.model, env.unwrapped.data
env.unwrapped.set_beta_target(env.unwrapped.BETA_HOME)  # 目標を一度だけ設定（あとはMuJoCo任せ）

print("フルモードのビューアを起動します（閉じるまで待機します）。")
print("スペース＝一時停止・再生。一時停止中は画面下のタイムラインで巻き戻せます。")
mujoco.viewer.launch(m, d)
