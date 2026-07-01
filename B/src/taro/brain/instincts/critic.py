"""
価値関数（Critic） — 内的状態に応じた報酬の見込みを学習する

【人間模倣＝既存AI研究】
Dopamineクラスの移動平均baselineは、内的状態（空腹等）に関わらず
単一のスカラー値だった。しかし「空腹時にまんまと言う」ことへの
期待報酬と「機嫌がいい時にあやされる」ことへの期待報酬は本来異なる。
Actor-Criticのように、body_stateから価値V(s)を予測する小さな
ネットワークに置き換えることで、状態依存のbaselineにする
（複数エージェントレビューでの指摘：B-11）。
"""

import torch.nn as nn


class Critic(nn.Module):
    """
    内的状態ベクトル [hunger, sleepiness, discomfort, arousal] から
    その状態で見込める報酬（価値V(s)）を予測する。
    """

    def __init__(self, state_dim=4, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_vector):
        return self.net(state_vector).squeeze(-1)
