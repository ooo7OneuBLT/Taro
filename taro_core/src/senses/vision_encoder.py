"""
視覚の通訳層（Phase 5・接続層） — MIMoの画像を、太郎の脳が読める形に変換する。

【根拠ラベル：既存AI研究・⚠️逸脱】
MIMoの視覚は既に年齢依存のacuityぼかし・中心窩フォビエーションを経ているが、
それでも256×256×3＝約20万個のピクセル値のまま。これをCNN（畳み込みニューラル
ネットワーク）で段階的に圧縮する。CNNは人間の視覚野の計算を模したものではなく、
AI研究で確立された画像圧縮の手法。詳細な設計判断は開発計画.mdを参照。

左目・右目は同じCNN（重み共有）で処理する（＝両目とも同じ視覚野を使う、という
生物学的に妥当な仮定）。
"""

import numpy as np
import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    """
    視覚の通訳。MIMoのステレオ視覚（eye_left, eye_right、各256×256×3）を
    脳が読める embedding_dim 次元のベクトルに変換する。

    入力：eye_left, eye_right （それぞれ (H, W, 3) の画像。0〜255のuint8想定）
    出力：embedding_dim次元のベクトル（左右を統合した1本）
    """

    def __init__(self, embedding_dim=64, image_size=256):
        super().__init__()
        # 3段階の畳み込みで空間解像度を1/8に圧縮しつつチャンネル数を増やす。
        # 256 -> 128 -> 64 -> 32 （strideによる圧縮）
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),  # 256->128
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),  # 128->64
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # 64->32
            nn.ReLU(),
        )
        flat_size = 32 * (image_size // 8) * (image_size // 8)
        self.per_eye_fc = nn.Linear(flat_size, embedding_dim)
        # 左右2つ分（embedding_dim*2）を1本のembedding_dimに統合
        self.fuse = nn.Linear(embedding_dim * 2, embedding_dim)

    def _encode_one_eye(self, image):
        """
        image: (H, W, 3) の numpy配列 または tensor。0〜255想定。
        戻り値: (embedding_dim,) のベクトル。
        """
        if not torch.is_tensor(image):
            # MIMo(OpenGL)の画像は上下反転ビュー由来で負のstrideを持つことがあり、
            # そのままだとtorch.as_tensorが失敗する。copy()でメモリを実体化してから渡す。
            image = torch.as_tensor(np.asarray(image).copy(), dtype=torch.float32)
        else:
            image = image.float()
        image = image / 255.0
        # (H, W, 3) -> (1, 3, H, W)  （CNNはチャンネルを先頭に置く形式を使う）
        x = image.permute(2, 0, 1).unsqueeze(0)
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        return self.per_eye_fc(x).squeeze(0)

    def forward(self, eye_left, eye_right):
        left_vec = self._encode_one_eye(eye_left)
        right_vec = self._encode_one_eye(eye_right)
        combined = torch.cat([left_vec, right_vec], dim=-1)
        return self.fuse(combined)
