"""筋肉モデル(MuscleModel)の最小デモ：新生児体×色付き筋活性化で太郎を動かし、動きの
滑らかさ(jerk)を測り録画する。学習なし＝純粋に探索ノイズでのもがき。

【なぜ】太郎の関節駆動は今トルク直接制御(SpringDamperModel、1関節1双方向モーター、瞬時
トルク)で、人間の筋肉(引くだけ・拮抗筋2本・力がじわっと立ち上がる・共収縮で剛性調節)とは
根本的に違う。MuscleModelに切り替えると action は 90→180次元・[0,1](筋活性化)になり、
学習済みモデルは非互換=白紙から学習し直し。本格学習の前に、まず「筋肉モデルの体が探索
ノイズでどう動くか(立ち上がり遅延で滑らかになるか)」を学習なしで掴む。

【この版の限界】各筋を独立に色付きノイズで駆動する最小版＝拮抗筋の共収縮(スティフネス
調節)は入れていない。共収縮は拮抗筋の本領なので次段階。ここで見えるのは主に「筋活性化
ダイナミクス(力の立ち上がりの遅れ)」の効果。
"""

# ⚠️★古い方式（2026-07-30 に整理）。新しい実験は `run/main.py` を通す。
#   【経緯】目標Eの実験スクリプトが118本あり、うち66本が**独立に環境を組み立てていた**。
#     そのため「学習は関節モード（90関節を独立に駆動＝逸脱リスト 逸脱5）、
#     測定とViewerは筋肉モード（拮抗筋2本/関節）」という**別の体で動く**事故が起きた
#     （ユーザーの目視「視線誘導反射の実験の時とは動きが全然違う」で発覚。
#      実測で動きが人間の新生児の約3.3倍速かった）。
#   【設計と移行計画】`E/docs/実行基盤_設計.md`
#   ⚠️このファイルは**記録として残す**（削除しない方針）。
#     中の測り方は再利用できるので、プラグインへ移すときの元にする。
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "D", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "taro_core"))
import paths; paths.setup_brain_path()
sys.path.insert(0, paths.MIMO_DIR)
sys.path.insert(0, os.path.join(paths.SRC, "brain"))

import numpy as np
import mujoco
from d_supine_env import SupineMimoEnv
from mimoActuation.muscle import MuscleModel
from spinal_cord.cpg import ColoredNoiseGenerator

AGE = 0.0
BETA = float(os.environ.get("E_BETA", "0.7"))
N_TICK = int(os.environ.get("N_TICK", "60"))
K = 100
ACT_CENTER = 0.3   # 筋活性化の中心(脱力気味の新生児)[ARBITRARY]
ACT_AMP = 0.3      # 色付きゆらぎの振幅[ARBITRARY]

env = SupineMimoEnv(actuation_model=MuscleModel, vision_params=None, age=AGE)
m, d = env.unwrapped.model, env.unwrapped.data
n_act = env.action_space.shape[0]
print(f"筋肉モデル: action={n_act}次元 [{env.action_space.low[0]},{env.action_space.high[0]}]  age={AGE}")

gen = ColoredNoiseGenerator(n_act, seed=0)
dt_env = m.opt.timestep * env.unwrapped.frame_skip
dofs = [m.jnt_dofadr[i] for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]

# 録画
renderer = mujoco.Renderer(m, height=480, width=480)
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance = 1.2; cam.azimuth = 90; cam.elevation = -35

obs, _ = env.reset(seed=0)
frames = []; jerks = []
prev_qacc = None
for tick in range(N_TICK):
    noise = gen.sample(BETA)                      # 色付き(1/f^β)、周辺N(0,1)
    act = np.clip(ACT_CENTER + ACT_AMP * noise, 0.0, 1.0).astype(np.float32)
    for k in range(K):
        obs, r, te, tr, info = env.step(act)
        qacc = d.qacc[dofs].copy()
        if prev_qacc is not None:
            jerks.append(np.abs((qacc - prev_qacc) / dt_env).mean())
        prev_qacc = qacc
        if k % 4 == 0:
            cam.lookat = d.body("upper_body").xpos.copy()
            renderer.update_scene(d, camera=cam)
            frames.append(renderer.render().copy())
        if te or tr:
            obs, _ = env.reset(); prev_qacc = None
            break

print(f"mean|jerk| = {np.mean(jerks):.1f}  (トルクモデルの色付き=2224と比較)")

out_mp4 = os.path.join(paths.CORE_ROOT, os.pardir, "E", "logs", "E", "growth_curriculum",
                       "muscle_demo_age0_colored.mp4")
os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
try:
    import cv2
    fps = (1.0 / (m.opt.timestep * env.unwrapped.frame_skip)) / 4
    hh, ww, _ = frames[0].shape
    vw = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ww, hh))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"RECORDED {out_mp4} ({len(frames)}フレーム, {fps:.0f}fps)")
except Exception as e:
    print("録画失敗", type(e).__name__, e)
env.close()
