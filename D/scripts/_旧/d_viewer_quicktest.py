"""3D操作画面(MuJoCo純正ビューア)で、位置・向きを直した後のベータ配置を確認する。

【確認ポイント】
・ベータがアルファの正面・目の高さ付近・1.25m先にいるか（距離感）
・ベータがアルファの方を向いているか（背中を向けていないか）

操作: 左ドラッグ=視点回転／右ドラッグ=平行移動／スクロール=ズーム
     （位置決めは完了したので、今回はドラッグでのベータ移動は無効化していない=
      試したければ従来通りダブルクリック→Ctrl+右ドラッグで動かせる。ただし
      サーボが効いているので離すと元の位置に戻る＝これが「正しいホーム位置」）
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
env.unwrapped.set_beta_target(env.unwrapped.BETA_HOME)  # ホーム位置を維持するサーボ目標

print("ビューアを起動します。ベータがアルファの正面・目の高さ付近にいて、")
print("アルファの方を向いているか（背中でなく）を確認してください。")
with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        env.unwrapped.set_beta_target(env.unwrapped.BETA_HOME)
        mujoco.mj_step(m, d)
        viewer.sync()
