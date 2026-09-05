# -*- coding: utf-8 -*-
"""視覚投射（VisualProjection） — 視覚野から連合野への「軸索の束」。

【設計】F/docs/設計_視覚とGRUの統合（表の卒業）.md V1（2026-08-31）。
DINOv2の視覚情報（384個の数字）を、言語GRUのトークンと同じ64次元へ翻訳し、
「視覚トークン」として発話の頭に添えるための変換層。

【人間対応】視覚野→連合野の投射。エリア間を繋ぐ軸索の束は情報を圧縮・変換して
届ける（網膜1億→視神経100万の圧縮と同型）。この層は白紙から始まり、
「次の音の予測に役立つ視覚の側面」だけが誤差逆伝搬で残る＝この経路に流れる
視覚情報が言語によって形作られる。

【工学対応】画像つきLLM（LLaVA等）の projection layer と同じ役割・同じ規模感。
"""
import torch.nn as nn


class VisualProjection(nn.Module):
    """384次元の視覚ベクトル → embedding次元の「視覚トークン」1個。"""

    def __init__(self, in_dim=384, out_dim=64):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        # LayerNorm：視覚トークンの大きさを語のトークンの埋め込みと同じ土俵に
        # 揃える（視覚だけ信号が大きい/小さいと、GRUが視覚を無視or偏重して育つ）
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, vec):
        """vec: tensor (in_dim,) → (out_dim,)"""
        return self.norm(self.proj(vec))
