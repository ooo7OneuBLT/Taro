"""
太郎の学習メカニズム

2つの学習を行う：
  1. 知覚の学習 — 次トークン予測の誤差を減らす（予測処理）【人間模倣】
  2. 行動の学習 — 高報酬だった発話を出やすくする（方策勾配 REINFORCE）【人間模倣＝既存AI研究】

学習信号はドーパミン（報酬予測誤差 δ = R - baseline）。

⚠️ 逸脱の明記：重み更新に誤差逆伝播（backprop）を使用する。
脳が厳密にはやっていないとされる手法【既存AI研究・⚠️逸脱】。
「報酬で重みを調整する」大枠は脳と同じ。
"""

import torch
import torch.nn.functional as F


class TaroLearner:
    """
    太郎の学習を管理する。

    知覚学習と行動学習を1つのオプティマイザで更新する。
    """

    def __init__(self, brain, lr=0.005, grad_clip=1.0):
        self.brain = brain
        self.optimizer = torch.optim.Adam(brain.parameters(), lr=lr)
        self.grad_clip = grad_clip

    def learn_perception(self, input_tokens):
        """
        知覚の学習：入力トークン列の次トークン予測誤差を最小化する。

        【人間模倣】脳は次の入力を予測し、予測誤差を減らそうとする
        （予測処理 / Friston）。

        input_tokens: list of int（親の発話のトークン列）
        戻り値: prediction_loss (float), prediction_probs (list of tensors)
        """
        if len(input_tokens) < 2:
            return 0.0, []

        device = self.brain._device()
        x = torch.tensor([input_tokens[:-1]], device=device)
        target = torch.tensor([input_tokens[1:]], device=device)

        logits, _ = self.brain(x)
        loss = F.cross_entropy(logits[0], target[0])

        probs_list = []
        with torch.no_grad():
            probs_all = F.softmax(logits[0] / max(self.brain.temperature, 1e-8), dim=-1)
            for i in range(probs_all.size(0)):
                probs_list.append(probs_all[i])

        return loss, probs_list

    def learn_action(self, log_probs, delta):
        """
        行動の学習：方策勾配（REINFORCE）。

        【人間模倣＝既存AI研究】
        「高報酬（δ>0）だった発話を出やすくする」更新。
        δはドーパミン（報酬予測誤差）。

        loss = -(δ) * Σ log_prob(生成した各トークン)

        log_probs: list of scalar tensors（生成時の各トークンのlog確率）
        delta: float（ドーパミン＝報酬予測誤差）
        戻り値: policy_loss (float)
        """
        if len(log_probs) == 0:
            return 0.0

        log_prob_sum = torch.stack(log_probs).sum()
        policy_loss = -delta * log_prob_sum

        return policy_loss

    def update(self, perception_loss, policy_loss):
        """
        知覚と行動の損失を合算し、重みを1回更新する。

        ⚠️ ここで誤差逆伝播（backprop）を使う【⚠️逸脱】。
        """
        total_loss = perception_loss + policy_loss

        self.optimizer.zero_grad()
        if isinstance(total_loss, torch.Tensor) and total_loss.requires_grad:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.brain.parameters(), self.grad_clip)
            self.optimizer.step()

        p_val = perception_loss.item() if isinstance(perception_loss, torch.Tensor) else perception_loss
        a_val = policy_loss.item() if isinstance(policy_loss, torch.Tensor) else policy_loss
        return p_val, a_val
