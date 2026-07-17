"""
Phase 7：結合テスト — 環境（MIMo+太郎内臓）とエージェント（試作の感覚運動の脳）を
結合し、クラッシュせず回ることを確認する。

流れ（毎ステップ）：
  1. HybridEnvの観測（5感覚）を、それぞれの通訳層で64次元に変換し、束ねる（320次元）
  2. SensorimotorBrain（試作の脳）に渡し、次の感覚の予測と、運動命令を得る
  3. 運動命令をMIMoの実際の関節可動域に合わせて変換し、HybridEnvに渡して1歩進める
  4. 新しい観測を同じく変換し、予測とのズレ（感覚運動予測誤差）を計算する

このテストは「配線が正しく繋がり、クラッシュせず・NaNも出ずに回るか」の確認が目的。
学習（重みを更新して賢くする）はまだ行わない（それは次の段階）。
ただし、勾配がちゃんと流れる設計になっているか（微分可能か）は、短い区間で
1回だけbackward()を試して確認する。
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import gymnasium as gym
import mimoEnv

_BRIDGE = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_BRIDGE, "src", "wrapper"))
sys.path.insert(0, os.path.join(_BRIDGE, "src", "senses"))
sys.path.insert(0, os.path.join(_BRIDGE, "src", "brain"))
from hybrid_env import HybridEnv
from sensory_encoders import ProprioceptionEncoder, VestibularEncoder, TouchEncoder
from vision_encoder import VisionEncoder
from sensorimotor_brain import SensorimotorBrain
from insula import Insula


def to_tensor(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


class SensoryFusion:
    """5つの通訳層をまとめて持ち、観測dictを320次元の1本のベクトルに変換する。"""

    def __init__(self):
        self.insula = Insula(state_dim=4, embedding_dim=64)          # 内受容感覚（太郎から流用）
        self.proprio = ProprioceptionEncoder(input_dim=621)
        self.vestibular = VestibularEncoder(input_dim=6)
        self.touch = TouchEncoder(input_dim=12822)
        self.vision = VisionEncoder(embedding_dim=64)

    def parameters(self):
        import itertools
        return itertools.chain(
            self.insula.parameters(), self.proprio.parameters(),
            self.vestibular.parameters(), self.touch.parameters(),
            self.vision.parameters(),
        )

    def encode(self, obs):
        intero_vec = self.insula(to_tensor(obs["interoception"]))
        proprio_vec = self.proprio(to_tensor(obs["observation"]))
        vestibular_vec = self.vestibular(to_tensor(obs["vestibular"]))
        touch_vec = self.touch(to_tensor(obs["touch"]))
        vision_vec = self.vision(obs["eye_left"], obs["eye_right"])
        return torch.cat([intero_vec, proprio_vec, vestibular_vec, touch_vec, vision_vec], dim=-1)


def rescale_action(raw_action, action_space):
    """tanh出力(-1〜1)を、MIMoの実際の関節可動域[low, high]に変換する。"""
    low = torch.as_tensor(action_space.low, dtype=torch.float32)
    high = torch.as_tensor(action_space.high, dtype=torch.float32)
    return (low + (raw_action + 1.0) / 2.0 * (high - low)).detach().numpy()


def main():
    print("=" * 60)
    print("Phase 7 結合テスト：環境（MIMo+内臓）とエージェント（試作の脳）")
    print("=" * 60)

    base = gym.make("MIMoBenchV2-v0")
    env = HybridEnv(base)
    fusion = SensoryFusion()
    brain = SensorimotorBrain(n_actuators=env.action_space.shape[0])

    obs, info = env.reset()
    hidden = brain.init_hidden()

    n_steps = 200
    pred_errors = []
    rewards = []
    nan_found = False

    for i in range(n_steps):
        sensory_vec = fusion.encode(obs)
        hidden_new, predicted_next, motor_raw = brain.step(sensory_vec, hidden.detach())
        action = rescale_action(motor_raw, env.action_space)

        obs, reward, terminated, truncated, info = env.step(action)

        next_sensory_vec = fusion.encode(obs)
        pred_err = brain.prediction_error(predicted_next, next_sensory_vec.detach())

        if torch.isnan(pred_err) or not np.isfinite(reward):
            nan_found = True
            print(f"  [{i}] NaN/Inf検出！ pred_err={pred_err.item()}, reward={reward}")
            break

        pred_errors.append(pred_err.item())
        rewards.append(reward)
        hidden = hidden_new

        if terminated or truncated:
            obs, info = env.reset()
            hidden = brain.init_hidden()

    print(f"{len(pred_errors)}/{n_steps} ステップ完走（クラッシュ・NaNなし: {not nan_found}）")
    print(f"感覚運動予測誤差: 最初={pred_errors[0]:.4f} 最後={pred_errors[-1]:.4f} 平均={np.mean(pred_errors):.4f}")
    print(f"累積報酬（恒常性+内臓）: {sum(rewards):.4f}")

    # --- 微分可能性の確認（学習ではなく、勾配が流れる設計かどうかの検証） ---
    print("\n--- 勾配が流れるかの確認（3ステップ分をbackward） ---")
    obs, info = env.reset()
    hidden = brain.init_hidden()
    total_loss = torch.tensor(0.0)
    for i in range(3):
        sensory_vec = fusion.encode(obs)
        hidden, predicted_next, motor_raw = brain.step(sensory_vec, hidden)
        action = rescale_action(motor_raw, env.action_space)
        obs, reward, terminated, truncated, info = env.step(action)
        next_sensory_vec = fusion.encode(obs)
        pred_err = brain.prediction_error(predicted_next, next_sensory_vec.detach())
        total_loss = total_loss + pred_err

    total_loss.backward()

    # motor_headだけは意図的に除外する：運動命令はMIMo(微分できない物理シミュレータ)を
    # 経由するため、感覚運動予測誤差からのbackpropでは学習できない（太郎のbasal_ganglia.py
    # が知覚=backprop／運動選択=方策勾配(REINFORCE)を使い分けているのと同じ理由）。
    # motor_headの学習は将来、方策勾配を別途実装する必要がある（今回はまだ未実装）。
    sensory_params = [(n, p) for n, p in
                       list(fusion.insula.named_parameters(prefix="insula"))
                       + list(fusion.proprio.named_parameters(prefix="proprio"))
                       + list(fusion.vestibular.named_parameters(prefix="vestibular"))
                       + list(fusion.touch.named_parameters(prefix="touch"))
                       + list(fusion.vision.named_parameters(prefix="vision"))
                       + list(brain.gru.named_parameters(prefix="gru"))
                       + list(brain.prediction_head.named_parameters(prefix="prediction_head"))]
    grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in sensory_params)
    motor_no_grad = all(p.grad is None for p in brain.motor_head.parameters())

    print(f"感覚エンコーダ＋予測head：全パラメータに有限な勾配が流れた: {grad_ok}")
    print(f"運動head：勾配が流れない（想定通り。運動学習には別途、方策勾配が必要）: {motor_no_grad}")

    env.close()
    print("\n" + "=" * 60)
    print(f"Phase 7 結合テスト: {'PASS' if (not nan_found and grad_ok) else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
