"""
感覚運動の脳（試作） — Phase 6・運動出力＋感覚運動予測ループ

【重要な設計判断・注意】
太郎の本体の脳（Taro/B/src/taro/brain/cortex.py の TaroBrain）は、音声・発話計画・
語彙と密結合した複雑な設計になっている。それを直接拡張するのはリスクが大きいため、
ここでは Phase 6/7 の「配線が正しく動くか」を検証するための、別の・小さな試作の脳を
新規に作る。TaroBrainの実物には一切触れていない。

本物のTaroBrainと統合する（音声・発話・満腹予期などと同じ脳の中で、身体運動も
扱えるようにする）のは、この試作で配線を検証した後の、より大きな後続作業とする。

【設計】
GRU（TaroBrainと同じ発想：1層の再帰ネットで隠れ状態を更新）を使い、毎ステップ：
  1. 5つの感覚（内受容・固有感覚・前庭覚・触覚・視覚、それぞれ64次元＝計320次元）を
     受け取り、GRUの隠れ状態を更新する
  2. 隠れ状態から「次に来る感覚（5感覚を束ねた320次元ベクトル）」を予測する
     → 感覚運動予測ループの中核。DIVA的に「自分が動いたらこう感じるはず」を予測する
  3. 隠れ状態から「次の運動」（MIMoのアクチュエータへの命令）を出す

【根拠ラベル】
GRU自体はTaroBrainと同じく【既存AI研究】（役割の代替、神経回路そのものの模倣ではない）。
「次の感覚を予測し、外れたら誤差を修正する」という原理そのものは【人間模倣】（予測処理、
Friston / DIVA, Guenther）。具体的な実装（320次元ベクトルのMSE予測）は簡略化した近似で
あり、⚠️今後の検証・改善の余地がある試作段階。
"""

import torch
import torch.nn as nn


class SensorimotorBrain(nn.Module):
    """
    5感覚を統合し、次の感覚を予測しながら、運動を出力する試作の脳。

    入力（毎ステップ）：5感覚それぞれ64次元のベクトル（計320次元）
    出力：
      - hidden: GRUの隠れ状態（次のステップに引き継ぐ）
      - predicted_next_sensory: 次のステップで来るはずの感覚（320次元、予測）
      - motor_action: MIMoのアクチュエータへの命令（n_actuators次元、-1〜1）
    """

    SENSE_DIM = 64
    NUM_SENSES = 5  # interoception, proprioception, vestibular, touch, vision
    FUSED_DIM = SENSE_DIM * NUM_SENSES  # 320

    def __init__(self, n_actuators, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_actuators = n_actuators

        self.gru = nn.GRUCell(self.FUSED_DIM, hidden_dim)

        # 感覚運動予測ループ：隠れ状態から「次に来る感覚」を予測する
        self.prediction_head = nn.Linear(hidden_dim, self.FUSED_DIM)

        # 運動出力：隠れ状態からMIMoの関節命令を出す
        self.motor_head = nn.Linear(hidden_dim, n_actuators)

    def init_hidden(self):
        return torch.zeros(self.hidden_dim)

    def step(self, sensory_vec, hidden):
        """
        1ステップ分の処理。

        sensory_vec: (FUSED_DIM,) 現在の5感覚を束ねたベクトル
        hidden: (hidden_dim,) 前回の隠れ状態

        戻り値: (new_hidden, predicted_next_sensory, motor_action)
        """
        new_hidden = self.gru(sensory_vec.unsqueeze(0), hidden.unsqueeze(0)).squeeze(0)
        predicted_next_sensory = self.prediction_head(new_hidden)
        motor_action = torch.tanh(self.motor_head(new_hidden))  # -1〜1に収める
        return new_hidden, predicted_next_sensory, motor_action

    @staticmethod
    def prediction_error(predicted_next_sensory, actual_next_sensory):
        """
        感覚運動予測誤差。予測と実際の感覚のズレ（MSE）。
        小さいほど「予測が当たった」＝良い。太郎のprediction.py（次の音の予測）と
        同じ発想を、身体感覚に拡張したもの。
        """
        return torch.nn.functional.mse_loss(predicted_next_sensory, actual_next_sensory)
