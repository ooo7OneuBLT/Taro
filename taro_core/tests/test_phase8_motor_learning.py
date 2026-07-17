"""
Phase 8：運動性喃語の学習ループ（方策勾配）。

ここまでの位置づけ：
  Phase 7で「配線が繋がり、勾配が正しく流れる設計か」だけを確認した。
  今回は実際に重みを更新し、感覚運動予測誤差が下がっていくかを見る。

流れ（毎ステップ）：
  1. 5感覚を融合ベクトル(320次元)に変換する
  2. TaroBrainWithMotor.step_motor()で、次の感覚の予測と、運動命令を得る。
     運動命令は確定的な1点ではなく正規分布からサンプリングし、選んだ行動の
     対数確率(log_prob)も受け取る（方策勾配で学習するために必要）。
     ゆらぎの大きさ（探索の強さ）は太郎の既存の探索本能＝NE（青斑核）で決める。
  3. 運動命令をMIMoに渡して1歩進め、実際の次の感覚を得る
  4. 予測とのズレ（感覚運動予測誤差）を計算する：
       - 知覚学習（予測head・感覚エンコーダ）はこの誤差をそのままbackpropで
         最小化する（誤差の"正解"が実際の次の感覚として存在するため）
       - 運動学習（motor_head）はMIMoの物理（微分不可能）を経由するため
         backpropできない。誤差を報酬に変換し（小さいほど高報酬）、
         ドーパミン（Dopamine）で報酬予測誤差δを求め、大脳基底核
         （TaroLearner.learn_action、太郎の既存コードをそのまま流用）で
         方策勾配として学習する
  5. NE（LocusCoeruleus、太郎の既存の探索本能をそのまま流用）が今回の報酬を
     観測し、次ステップの探索の強さを更新する

【2026-07-10追加】予測・再構成の"正解"は、学習中のfusionではなく、独立した
固定のtarget_fusionから作る。同じ1つのエンコーダを予測にも正解作りにも
使うと、エンコーダが「入力を無視して同じ値を返す」よう変化するだけで両方
"正解"してしまう馴れ合いが起き、実測でも8シード中6シードが崩壊した
（docs/人間模倣からの逸脱リスト.md B4）。

このテストの目的：クラッシュ・NaNなく回り、感覚運動予測誤差が学習によって
下がる傾向を示すこと、および違う入力を区別できなくなる崩壊が起きないことの確認。
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
from taro_brain_motor import TaroBrainWithMotor
from basal_ganglia import TaroLearner
from dopamine import Dopamine
from locus_coeruleus import LocusCoeruleus
from homeostatic_scaling import HomeostaticScaling
from insula import Insula


def to_tensor(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


class SensoryFusion:
    """5つの通訳層をまとめて持ち、観測dictを320次元の1本のベクトルに変換する。"""

    def __init__(self):
        self.insula = Insula(state_dim=4, embedding_dim=64)
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
        fused = torch.cat([intero_vec, proprio_vec, vestibular_vec, touch_vec, vision_vec], dim=-1)

        # 感覚エンコーダをbrainと一緒に学習させる場合、「予測する対象」自体
        # （このencodeの出力）も毎ステップ動く。学習係数×次元数の噴出で
        # スケールが際限なく膨張しうる（実測：エンコーダを凍結せず素朴に
        # 学習させたところ、予測誤差が数万→9600万へ発散した）。
        # LayerNorm（学習パラメータなし＝恣意的な定数を持ち込まない、
        # 平均0・分散1への正規化のみ）を融合直後にかけ、スケールを毎回
        # 立て直すことで、この暴走を防ぐ。
        fused = torch.nn.functional.layer_norm(fused, fused.shape)
        return fused

    def freeze(self):
        """
        正解を作る専用の"目標エンコーダ"として使うために、学習させず固定する。

        【既存AI研究】Random Network Distillation（Burda et al., 2018）と同じ
        発想：固定されたランダムな変換を予測対象にすると、予測する側と正解を
        作る側が馴れ合って"どんな入力でも同じ値"に退化することがない
        （実測：オンラインの1つのエンコーダを両方に使うと、8シード中6シードで
        崩壊した。詳細はdocs/人間模倣からの逸脱リスト.md B4）。
        """
        for p in self.parameters():
            p.requires_grad_(False)
        return self


class CombinedParams:
    """
    TaroLearnerは`brain.parameters()`しか最適化しない（本来、音声のみで
    完結していた設計のため）。しかしMIMo統合では触覚・視覚などの感覚
    エンコーダ（fusion）も「感覚を解釈する層」であり、脳と同様に経験から
    鍛えられるべき（Fristonの予測処理はエンコーダ側にも及ぶ）。
    TaroLearner自体（太郎の既存コード）は無編集のまま、脳＋感覚エンコーダの
    パラメータをまとめて渡すための薄いラッパー。
    """

    def __init__(self, *modules):
        self.modules = modules

    def parameters(self):
        import itertools
        return itertools.chain(*(m.parameters() for m in self.modules))


def rescale_action(raw_action, action_space):
    """tanh/clamp出力(-1〜1)を、MIMoの実際の関節可動域[low, high]に変換する。"""
    low = torch.as_tensor(action_space.low, dtype=torch.float32)
    high = torch.as_tensor(action_space.high, dtype=torch.float32)
    return (low + (raw_action + 1.0) / 2.0 * (high - low)).detach().numpy()


def main():
    print("=" * 60)
    print("Phase 8 運動性喃語の学習ループ（方策勾配）")
    print("=" * 60)

    base = gym.make("MIMoBenchV2-v0")
    env = HybridEnv(base)
    fusion = SensoryFusion()
    # 正解（次の感覚・今の感覚）を作る専用の、独立した・学習させないエンコーダ。
    # fusion（学習中）と重みを共有しないため、予測する側と正解を作る側の馴れ合いが起きない。
    target_fusion = SensoryFusion().freeze()
    brain = TaroBrainWithMotor(vocab_size=3, n_actuators=env.action_space.shape[0])

    # 学習・報酬・探索：すべて太郎の既存の本能をそのまま流用する
    learner = TaroLearner(CombinedParams(brain, fusion), lr=0.005)
    dopamine = Dopamine()
    ne = LocusCoeruleus()
    homeostat = HomeostaticScaling(dim=320)

    obs, info = env.reset()
    hidden = brain.init_motor_hidden()

    n_steps = 300
    pred_errors = []
    rewards = []
    ne_levels = []
    kl_losses = []
    nan_found = False

    for i in range(n_steps):
        sensory_vec = fusion.encode(obs)
        # 再構成損失(recon_loss)の"正解"は、学習中のfusionではなく独立したtarget_fusionから
        current_sensory_target = target_fusion.encode(obs).detach()
        ne_level = ne.get_ne_level()
        hidden_new, predicted_next, motor_raw, log_prob, kl_loss, recon_loss = brain.step_motor(
            sensory_vec, hidden.detach(), current_sensory_target=current_sensory_target, ne_level=ne_level)
        action = rescale_action(motor_raw, env.action_space)

        obs, env_reward, terminated, truncated, info = env.step(action)

        # 感覚運動予測誤差の"正解"も、同じ理由でtarget_fusionから作る
        next_sensory_vec_target = target_fusion.encode(obs).detach()
        pred_err = brain.prediction_error(predicted_next, next_sensory_vec_target)

        if torch.isnan(pred_err) or not np.isfinite(env_reward):
            nan_found = True
            print(f"  [{i}] NaN/Inf検出！ pred_err={pred_err.item()}, reward={env_reward}")
            break

        reward = brain.sensorimotor_reward(pred_err.item())
        delta = dopamine.compute_rpe(reward)
        policy_loss = learner.learn_action([log_prob], delta)

        # 恒常的シナプススケーリング：感覚エンコーダが「どんな入力でも同じ値」に
        # 潰れないよう、時間をまたいだ活動の乏しさにペナルティを足す
        # (PV-RNN的な仕組み(kl_loss)を追加した後も、安全装置として残す)
        homeostatic_loss = homeostat.homeostatic_loss(sensory_vec)
        homeostat.observe(sensory_vec)

        perception_loss = pred_err + homeostatic_loss + kl_loss + recon_loss
        p_val, a_val = learner.update(perception_loss, policy_loss)

        ne.observe_reward(reward)
        ne.release_ne()

        pred_errors.append(pred_err.item())
        rewards.append(reward)
        ne_levels.append(ne_level)
        kl_losses.append(kl_loss.item())
        hidden = hidden_new.detach()

        if terminated or truncated:
            obs, info = env.reset()
            hidden = brain.init_motor_hidden()

    n_done = len(pred_errors)
    print(f"{n_done}/{n_steps} ステップ完走（クラッシュ・NaNなし: {not nan_found}）")

    if n_done >= 20:
        first10 = np.mean(pred_errors[:10])
        last10 = np.mean(pred_errors[-10:])
        print(f"感覚運動予測誤差: 最初10平均={first10:.4f}  最後10平均={last10:.4f}")
        print(f"報酬: 最初10平均={np.mean(rewards[:10]):.4f}  最後10平均={np.mean(rewards[-10:]):.4f}")
        print(f"NE水準: 最初10平均={np.mean(ne_levels[:10]):.4f}  最後10平均={np.mean(ne_levels[-10:]):.4f}")
        print(f"KL損失: 最初10平均={np.mean(kl_losses[:10]):.4f}  最後10平均={np.mean(kl_losses[-10:]):.4f}")
        improved = last10 < first10
    else:
        improved = False

    # 崩壊チェック：全く違う2つの身体状態で、融合ベクトルがどれだけ区別できるか
    obs1, _ = env.reset()
    vec1 = fusion.encode(obs1).detach()
    for _ in range(5):
        obs2, r2, term2, trunc2, info2 = env.step(env.action_space.sample())
    vec2 = fusion.encode(obs2).detach()
    diff_ratio = (vec1 - vec2).norm().item() / vec1.norm().item() * 100
    print(f"崩壊チェック（違う入力の見分けやすさ）: {diff_ratio:.2f}%")

    env.close()
    print("\n" + "=" * 60)
    ok = (not nan_found) and n_done == n_steps
    print(f"Phase 8 学習ループ: {'PASS' if ok else 'FAIL'}（誤差が減少傾向: {improved}）")
    print("=" * 60)


if __name__ == "__main__":
    main()
