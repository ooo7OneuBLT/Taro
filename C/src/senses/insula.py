# --- コピー元: Taro (github.com/ooo7OneuBLT/Taro) commit 3b976fc ---
# --- 元パス: B/src/taro/... （unificationMIMoでは無編集のまま流用） ---

"""
島皮質（Insula） — 体の感覚を大脳皮質に伝える経路

【人間模倣】
島皮質は内受容感覚（空腹・痛み・体温など体内の状態）を
大脳皮質に届ける中枢。Craig (2002) "How do you feel?"

太郎では、internal_stateの数値をベクトルに変換し、
大脳皮質のGRU入力に合流させる。
耳（言葉）と並ぶもう1本の入力線。
"""

import torch
import torch.nn as nn


class Insula(nn.Module):
    """
    島皮質。体の感覚を脳が処理できるベクトルに変換する。

    入力：内部状態ベクトル [hunger, sleepiness, discomfort, arousal]
    出力：embedding_dim次元のベクトル（大脳皮質のGRU入力と同じサイズ）
    """

    def __init__(self, state_dim=4, embedding_dim=64):
        """
        state_dim: 内部状態の次元数（hunger, sleepiness, discomfort, arousal）
        embedding_dim: 大脳皮質の埋め込み次元と合わせる
        """
        super().__init__()
        self.encoder = nn.Linear(state_dim, embedding_dim)

    def forward(self, state_vector):
        """
        内部状態ベクトル → 脳が処理できるベクトル。

        state_vector: [hunger, sleepiness, discomfort, arousal] のテンソル
        戻り値: embedding_dim次元のベクトル
        """
        return self.encoder(state_vector)
