"""
視覚ONの実測速度チェック（0コスト診断）。

「egomotion割引を本能で学ばせるとして、視覚ONだと1秒に何サンプル作れるか」を
実測する。設計を作り込む前に、まず数字で「本当に時間がかかりすぎるか」を確認する。

計測対象：
  A) 物理のみ（視覚OFF）: env.step() だけ
  B) 視覚あり: env.step() + get_vision_obs()（実際にレンダリングする）

使い方: python bench_vision_speed.py [n_steps]
"""
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "taro_core"))
import paths
paths.setup_brain_path()
sys.path.insert(0, paths.MIMO_DIR)

from d1_carer_vision_env import CarerVisionEnv, lean_vision_params


def bench(vision_on, n_steps, res=64):
    vp = lean_vision_params(size=res) if vision_on else None
    env = CarerVisionEnv(vision_params=vp)
    env.reset()
    na = env.action_space.shape[0]
    rng = np.random.default_rng(0)

    # ウォームアップ（レンダラ初期化・JIT等の初回コストを除外）
    for _ in range(5):
        env.step(rng.uniform(-1, 1, na).astype(np.float32))
        if vision_on:
            env.get_vision_obs()

    t0 = time.perf_counter()
    for _ in range(n_steps):
        env.step(rng.uniform(-1, 1, na).astype(np.float32))
        if vision_on:
            imgs = env.get_vision_obs()
    elapsed = time.perf_counter() - t0

    sps = n_steps / elapsed
    label = f"視覚ON(res={res})" if vision_on else "視覚OFF(物理のみ)"
    print(f"{label}: {n_steps}ステップ / {elapsed:.2f}秒 = {sps:.1f} サンプル/秒"
          f"（1サンプル {1000/sps:.1f}ms）")
    return sps


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200

    print("=== 実測：視覚ON/OFFでの生成速度 ===")
    sps_off = bench(vision_on=False, n_steps=n)
    sps_on = bench(vision_on=True, n_steps=n, res=64)

    print()
    print(f"視覚ONは視覚OFFの {sps_on/sps_off*100:.1f}% の速度（{sps_off/sps_on:.1f}倍遅い）")
    print()
    print("=== 目安：門番が天井に達したn=400サンプルを集めるのにかかる時間 ===")
    print(f"  視覚OFF: {400/sps_off:.1f}秒")
    print(f"  視覚ON : {400/sps_on:.1f}秒")
