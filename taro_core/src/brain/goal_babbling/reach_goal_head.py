"""目標専用の順モデル（ReachGoalHead）— 手先位置の目標表現（案C）用。

【なぜ nat_head と分離するか】既存の `nat_head`（forward_model_head）は固有感覚
621/801次元まるごとを予測対象にしている。手先位置の目標 `g`（22次元、スケールも
別）はこれと次元もスケールも違う予測対象なので、同じヘッドに相乗りさせると
次元数の多い方に薄められる（落とし穴チェックリスト 項11「合計で測ると次元数の
多い方に薄められる」を3回踏んだ教訓）。`vision_forward_head` が視覚を専用ヘッドに
分離しているのと同じパターンをそのまま踏襲する。

【taro_brain_motor.py に置かない理由】このヘッドは
`cfg.goal_babbling and cfg.goal_space=="reach_self"` のときだけ構築される
（条件付き構築）。太郎の脳の中枢（`taro_brain_motor.py`）に置くと、既存の
`self.brain = TaroBrainWithMotor(...)` という1行の中で新しいパラメータが
初期化されることになり、既定値であっても乱数消費のタイミングが混ざる
リスクがある。独立モジュールにして `Taro.__init__` 側で条件付きに構築することで、
`taro_brain_motor.py` は1行も変更せずに済む（設計の統合判断「決定1」）。

【学習】独立した optimizer（`Taro.reach_opt`）で学習する。既存の自己モデル学習
（nat_head・pe）の loss には混ぜない（設計の統合判断「決定2」。運動小脳
`MotorCerebellum` と同じ「独立モジュール・独立optimizer」パターン）。
"""
import torch
import torch.nn as nn


class ReachGoalHead(nn.Module):
    """[z（内部表現）, 行動] → 目標ベクトルgの変化量Δg、を予測する順モデル。

    forward_model_head（nat_head）・vision_forward_head と全く同じ形
    （Linear→SiLU→LayerNorm→Linear）。予測対象（g）が違うだけ。
    """

    def __init__(self, latent_dim, n_actuators, goal_dim, hidden_dim=128):
        super().__init__()
        self.goal_dim = goal_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + n_actuators, hidden_dim), nn.SiLU(),
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, goal_dim))

    def forward(self, z, action):
        """Δg（残差）を返す。呼び出し側が `current_g + Δg` にして使う。"""
        return self.net(torch.cat([z, action], dim=-1))

    def predict(self, z, action, current_g):
        """残差予測：現在の目標g + Δ(z, 行動) = 次のgの予測。predict_proprioと同じ形。"""
        return current_g + self.forward(z, action)
