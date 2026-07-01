"""
大脳基底核（Basal Ganglia） — 行動選択と強化学習

【人間模倣】大脳基底核はドーパミンを使って行動を強化・抑制する。
「うまくいった行動を増やし、うまくいかなかった行動を減らす」。

A2-10：旧learning.pyから改名。人間の脳の部品名に合わせた。

⚠️ 逸脱：重み更新にbackpropを使用【既存AI研究・⚠️逸脱】。
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

    def learn_perception(self, input_tokens, body_state=None):
        """
        知覚の学習：入力トークン列の次トークン予測誤差を最小化する。

        【人間模倣】脳は次の入力を予測し、予測誤差を減らそうとする
        （予測処理 / Friston）。

        input_tokens: list of int（親の発話のトークン列）
        body_state: 内部状態テンソル（モデルB用。NoneならモデルA互換）
        戻り値: prediction_loss (float), prediction_probs (list of tensors)
        """
        if len(input_tokens) < 2:
            return 0.0, []

        device = self.brain._device()
        x = torch.tensor([input_tokens[:-1]], device=device)
        target = torch.tensor([input_tokens[1:]], device=device)

        # GRUを1回だけ流して知覚損失を計算する
        out, _ = self.brain.forward_hidden(x, body_state=body_state)
        logits = self.brain.perception_head(out)
        loss = F.cross_entropy(logits[0], target[0])

        probs_list = []
        with torch.no_grad():
            probs_all = F.softmax(logits[0] / max(self.brain.temperature, 1e-8), dim=-1)
            for i in range(probs_all.size(0)):
                probs_list.append(probs_all[i])

        return loss, probs_list

    def compute_value_loss(self, value_pred, reward):
        """
        クリティック（状態価値関数）の学習：TD誤差の二乗を最小化する。

        【人間模倣＝既存AI研究】Actor-Critic法のcritic更新。
        Dopamineの単純な移動平均baselineを、body_state依存の
        価値予測に置き換えるための学習（B-11）。

        value_pred: brain.critic(body_state) の出力（勾配あり）
        reward: 実際に得られた報酬（スカラー）
        """
        target = torch.tensor(reward, device=value_pred.device, dtype=value_pred.dtype)
        return F.mse_loss(value_pred, target)

    def learn_action(self, log_probs, delta, credits=None):
        """
        行動の学習：方策勾配（REINFORCE）。

        【人間模倣＝既存AI研究】
        「高報酬（δ>0）だった発話を出やすくする」更新。
        δはドーパミン（報酬予測誤差）。

        B2-2：従来は発話全体に単一のδしか与えず、系列内のどの文字が
        良かったかを区別できなかった（クレジット割り当て問題）。
        creditsを渡すと、文字ごとの一致度（compute_alignment_credit、
        ⚠️逸脱：詳細はimitation.py参照）でδを重み付けし、良かった文字は
        より強く強化し、目標語にない余分な文字は抑制する。

        loss = -Σ (δ * credit_i) * log_prob(生成したi番目のトークン)

        log_probs: list of scalar tensors（生成時の各トークンのlog確率）
        delta: float（ドーパミン＝報酬予測誤差）
        credits: log_probsと同じ長さのlist（各文字の目標語との一致度、[-1,1]程度）。
            Noneなら従来通り全文字に同じδを適用する
        戻り値: policy_loss (float)
        """
        if len(log_probs) == 0:
            return 0.0

        if credits is not None and len(credits) == len(log_probs):
            weighted = torch.stack([lp * c for lp, c in zip(log_probs, credits)]).sum()
            policy_loss = -delta * weighted
        else:
            log_prob_sum = torch.stack(log_probs).sum()
            policy_loss = -delta * log_prob_sum

        return policy_loss

    def update(self, perception_loss, policy_loss, value_loss=None):
        """
        知覚・行動・価値の損失を合算し、重みを1回更新する。

        ⚠️ ここで誤差逆伝播（backprop）を使う【⚠️逸脱】。
        """
        total_loss = perception_loss + policy_loss
        if value_loss is not None:
            total_loss = total_loss + value_loss

        self.optimizer.zero_grad()
        if isinstance(total_loss, torch.Tensor) and total_loss.requires_grad:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.brain.parameters(), self.grad_clip)
            self.optimizer.step()

        p_val = perception_loss.item() if isinstance(perception_loss, torch.Tensor) else perception_loss
        a_val = policy_loss.item() if isinstance(policy_loss, torch.Tensor) else policy_loss
        return p_val, a_val
