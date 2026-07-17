"""
感覚の通訳層（Phase 5・接続層） — MIMoの各感覚を、太郎の脳が読める形に変換する。

insula.py（内受容感覚の通訳、太郎から流用）と同じ考え方：
  「生の数字の塊」→ Linear層1枚 → embedding_dim次元のベクトル

対象は固有感覚・前庭覚・触覚の3つ（いずれも空間的な画像ではなく、ただの数値の並びなので
insulaと同じ単純な変換で十分）。視覚（画像）だけは別ファイル vision_encoder.py でCNNを使う。

根拠ラベル：【既存AI研究】（Linear層による特徴圧縮。生物の感覚神経系そのものの模倣ではないが、
「多数の受容器からの信号を少数の特徴に集約する」という機能的役割はinsulaと同じ扱いとする）
"""

import torch
import torch.nn as nn


class ProprioceptionEncoder(nn.Module):
    """
    固有感覚の通訳。MIMoの `observation`（関節角度・速度・トルク・可動域、621次元）を
    脳が読める embedding_dim 次元のベクトルに変換する。

    入力：MIMoの `observation` （次元数はMIMoの設定に依存。既定621）
    出力：embedding_dim次元のベクトル
    """

    def __init__(self, input_dim=621, embedding_dim=64):
        super().__init__()
        self.encoder = nn.Linear(input_dim, embedding_dim)

    def forward(self, observation):
        return self.encoder(observation)


class VestibularEncoder(nn.Module):
    """
    前庭覚の通訳。MIMoの `vestibular`（加速度3＋角速度3、6次元）を
    脳が読める embedding_dim 次元のベクトルに変換する。
    """

    def __init__(self, input_dim=6, embedding_dim=64):
        super().__init__()
        self.encoder = nn.Linear(input_dim, embedding_dim)

    def forward(self, vestibular):
        return self.encoder(vestibular)


class TouchEncoder(nn.Module):
    """
    触覚の通訳。MIMoの `touch`（全身4,274点×力ベクトル3、12,822次元）を
    脳が読める embedding_dim 次元のベクトルに変換する。

    次元がかなり大きい（12,822）ので、Linear1枚でも動くが、いったん中間層を
    挟んで段階的に圧縮する（12822 → hidden_dim → embedding_dim）。
    """

    def __init__(self, input_dim=12822, hidden_dim=256, embedding_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, touch):
        return self.encoder(touch)
