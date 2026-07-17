"""
HybridEnv の動作確認（Phase 4 手順②③のテスト）。

2部構成：
  A. 統合テスト：実際にMIMoを通してラッパーをステップし、クラッシュせず・観測に
     内受容感覚が入り・報酬が数値で返ることを確認。
  B. 身体ダイナミクステスト：内臓は「1秒＝100 MIMoステップ」と非常にゆっくり動くため
     （人間の空腹は数時間スケール）、MIMoを何百万ステップも回すのは非現実的。そこで
     内臓の1秒処理（advance_body_one_second）を直接まわし、空腹が上下し・授乳で下がり・
     恒常性報酬が出ることを高速に確認する。

脳はまだ繋がない。行動はランダム（脳の代わり）。
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import gymnasium as gym
import mimoEnv  # 環境登録のため

_BRIDGE = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_BRIDGE, "src", "wrapper"))
from hybrid_env import HybridEnv


def test_A_integration(n_steps=300):
    print("=" * 60)
    print("A. 統合テスト（MIMoを通してラッパーをステップ）")
    print("=" * 60)
    base = gym.make("MIMoBenchV2-v0")
    env = HybridEnv(base)

    obs, info = env.reset()
    assert "interoception" in obs, "観測に interoception が無い"
    assert obs["interoception"].shape == (4,), "interoception の形が違う"
    print("reset OK。観測キー:", sorted(obs.keys()))
    print("interoception 初期値 [hunger, sleepiness, discomfort, arousal]:",
          np.round(obs["interoception"], 4))

    total_reward = 0.0
    for i in range(n_steps):
        action = env.action_space.sample() if hasattr(env.action_space, "sample") else base.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        assert "interoception" in obs
        hunger = obs["interoception"][0]
        assert 0.0 <= hunger <= 1.0, f"hunger が範囲外: {hunger}"

    print(f"{n_steps}ステップ完走（クラッシュなし）")
    print(f"内臓が進んだ秒数: {n_steps // env.steps_per_body_second} 秒")
    print(f"最終 hunger: {obs['interoception'][0]:.4f}, 累積報酬: {total_reward:.4f}")
    env.close()
    print("A: PASS\n")


def test_B_body_dynamics(n_seconds=30000):
    print("=" * 60)
    print("B. 身体ダイナミクステスト（内臓を直接まわす）")
    print("=" * 60)
    base = gym.make("MIMoBenchV2-v0")
    env = HybridEnv(base)
    env.reset()

    hunger_trace = []
    reward_trace = []
    feeding_seconds = 0
    positive_reward_events = 0

    for sec in range(n_seconds):
        r = env.advance_body_one_second()
        hunger_trace.append(env.internal_state.hunger)
        reward_trace.append(r)
        if env.stomach.is_feeding():
            feeding_seconds += 1
        if r > 1e-6:
            positive_reward_events += 1

    hunger_trace = np.asarray(hunger_trace)
    print(f"{n_seconds} 身体秒を実行")
    print(f"hunger  最小 {hunger_trace.min():.3f} / 最大 {hunger_trace.max():.3f} / 平均 {hunger_trace.mean():.3f}")
    print(f"授乳していた秒数: {feeding_seconds}")
    print(f"恒常性報酬が正だったイベント数（つらさが下がった瞬間）: {positive_reward_events}")

    # 数千秒ごとの hunger を表示（上下しているのを目で見る）
    print("\n  時刻(時)   hunger")
    for sec in range(0, n_seconds, n_seconds // 10):
        print(f"   {sec/3600:5.1f}h   {hunger_trace[sec]:.3f}")

    # 判定：空腹が意味のある幅で動いた／授乳が起きた／報酬が出た
    assert hunger_trace.max() > hunger_trace[0] + 0.1, "空腹が上がっていない"
    assert feeding_seconds > 0, "一度も授乳していない"
    assert positive_reward_events > 0, "恒常性報酬が一度も正になっていない"
    print("\nB: PASS（空腹が上下し・授乳が起き・恒常性報酬が出た）\n")
    env.close()


if __name__ == "__main__":
    test_A_integration()
    test_B_body_dynamics()
    print("=" * 60)
    print("全テスト PASS：ハイブリッド環境（MIMo＋太郎内臓）が1つの身体として動作")
    print("=" * 60)
