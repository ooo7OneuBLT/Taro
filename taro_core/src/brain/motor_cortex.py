"""運動野（motor cortex）相当のモジュール。

【解剖学的位置づけ、2026-07-23】感覚フィードバックを見て精密な運動指令を計算する部分。
新生児期は皮質脊髄路が未髄鞘化でこの精密制御がほぼ機能しないため、太郎のB-min
（`d_c5_motor_quality.py`のE_WMEAN=0経路）ではここの出力を退け、脊髄CPG
（spinal_cord/cpg.py）を主役にする。生後2-5ヶ月にかけて皮質脊髄路が成熟し、
そわそわ運動への移行とともにここの寄与が戻っていく想定（研究日誌続き16）。

内訳：
  motor_gru   : 運動専用のGRU（音声用self.gruとは別、重み共有なし）
  pc_latent   : 確率的な潜在変数の推論（PredictiveCodingLatent、PV-RNNに着想）
  motor_head  : 潜在変数zから関節指令を計算する最終層
"""
import torch.nn as nn
from predictive_coding_latent import PredictiveCodingLatent


class MotorCortex(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, latent_dim, sensory_dim, n_actuators):
        super().__init__()
        self.motor_gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.pc_latent = PredictiveCodingLatent(hidden_dim, sensory_dim, latent_dim=latent_dim)
        self.motor_head = nn.Linear(latent_dim, n_actuators)
