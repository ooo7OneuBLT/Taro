"""感覚融合（fusion）— 各感覚エンコーダの出力を1本のベクトルに統合する、太郎の"本物の融合層"。

従来 `C/scripts/run_c_metrics_ac_lr.py` に `MinimalFusion` として埋まっていたものを taro_core へ
抽出（移行記録：doc/移行記録_taro_core化_2026-07-17.md の残課題「MinimalFusion抽出」の解消）。

各感覚を64次元に通訳し（insula=内受容 / proprio=固有感覚 / vestibular=前庭覚 / touch=触覚 /
vision=視覚）、連結して layer_norm する。触覚・視覚は**フラグで後付け**でき、両方OFFなら従来の
C（内受容+固有感覚+前庭覚）と1バイトも変わらない（後方互換）。

視覚込みの融合ロジックは Phase 8 の `SensoryFusion`（tests/test_phase8_motor_learning.py）で
実績のある「各64次元→連結→layer_norm」をそのまま踏襲する（＝車輪の再発明をしない）。
layer_norm は「感覚エンコーダを脳と一緒に学習させると予測対象のスケールが発散する」問題への
歯止め（学習パラメータなし＝恣意的な定数を持ち込まない。実測：凍結せず学習で誤差が数万→9600万へ発散）。

根拠ラベル：【既存AI研究】（Linear/CNNによる特徴圧縮＋LayerNorm）。生物の感覚神経系そのものの
模倣ではないが、「多数の受容器→少数の特徴に集約」という機能的役割で insula と同じ扱いとする。
"""
import itertools

import numpy as np
import torch

from insula import Insula
from sensory_encoders import ProprioceptionEncoder, VestibularEncoder, TouchEncoder
from vision_encoder import VisionEncoder


def to_tensor(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


class MinimalFusion:
    """内受容＋固有感覚＋前庭覚を基本に、触覚・視覚をフラグで足せる融合層。

    touch_dim=0 かつ vision_res=0 なら従来のCと完全に同一（内受容+固有感覚+前庭覚の3つ）。
    - touch_dim>0  : 触覚を足す（次元数はMIMoの触覚センサ数×3）。
    - vision_res>0 : 視覚を足す（vision_res＝眼球カメラの一辺の画素数＝低視力なら小さく）。
    """

    def __init__(self, touch_dim=0, vision_res=0):
        self.insula = Insula(state_dim=4, embedding_dim=64)
        self.proprio = ProprioceptionEncoder(input_dim=621)
        self.vestibular = VestibularEncoder(input_dim=6)
        self.touch = TouchEncoder(input_dim=touch_dim, hidden_dim=256, embedding_dim=64) if touch_dim else None
        # 視覚は vision_res>0 のときだけ有効。VisionEncoder は画像サイズに依存するので、
        # 低視力（低解像度）なら小さい image_size を渡す＝CNNの計算も軽くなる。
        self.vision = VisionEncoder(embedding_dim=64, image_size=vision_res) if vision_res else None

    def parameters(self):
        ms = [self.insula.parameters(), self.proprio.parameters(), self.vestibular.parameters()]
        if self.touch is not None:
            ms.append(self.touch.parameters())
        if self.vision is not None:
            ms.append(self.vision.parameters())
        return itertools.chain(*ms)

    def encode(self, obs):
        parts = [self.insula(to_tensor(obs["interoception"])),
                 self.proprio(to_tensor(obs["observation"])),
                 self.vestibular(to_tensor(obs["vestibular"]))]
        if self.touch is not None:
            parts.append(self.touch(to_tensor(obs["touch"])))
        if self.vision is not None:
            parts.append(self.vision(obs["eye_left"], obs["eye_right"]))
        f = torch.cat(parts, dim=-1)
        return torch.nn.functional.layer_norm(f, f.shape)

    def freeze(self):
        """正解ターゲット用に凍結（RND式＝予測側と正解側の馴れ合い崩壊を防ぐ）。"""
        for p in self.parameters():
            p.requires_grad_(False)
        return self
