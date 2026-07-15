"""
運動小脳（連続版, MotorCerebellum） — 運動の自動化（結晶化）

【人間模倣】反復練習した運動は自動化し、実行のばらつきが減って形が安定する
（VMS＝Vocal Motor Schemes, McCune & Vihman 2001／小脳のフィードフォワード制御, Miall & Wolpert 1996, Neural Networks 9:1265-1279）。
太郎本体（B/src/taro/brain/cerebellum.py）の離散版（口の動きの辞書＋ n/(n+K)）と
同じ原理を、MIMo の連続90関節へ移植したもの。

役割の分離（strawman回避 — 既存の2機構と被らせない）：
- NE（青斑核, locus_coeruleus）＝探索ノイズを"全体"で下げる（報酬ドリブン）。← 別物
- 順モデル（nat_head）＝行動→感覚の"予測"。                                   ← 別物
- 小脳＝方針(motor_head)が出す行動を状態(z)から真似る"フィードフォワード地図"。
  よく練習した（＝小脳が方針をよく再現できる）状態ほど自動化重み w を上げ、滑らかな
  小脳出力で行動を置き換え、探索ノイズも下げる＝"状態ごと"の結晶化。NEの全体調整とは別。

【工学近似 ⚠️】自動化重み w の形（自己正規化した馴染み度の指数）と上限 w_max は暫定。
恣意的な絶対スケール定数を避けるため、馴染み度は誤差の走行平均(err_ema)で自己正規化する
（Bの当てずっぽう K=2000 より一歩改善）。

【注意すべき機能（逸脱ではない）】これは人間模倣に忠実な機能だが、既定ONで常時稼働し、
現時点ではCの指標をほとんど変えない（良性・不変）。＝壁に当たったとき「小脳が黙って
効いている」可能性を見落としやすい。デバッグ時はON/OFF比較を候補に入れること。
未実装＝部位ごとのw（somatotopy。今は全身1つのw）。詳細は `doc/注意すべき機能リスト.md`。
"""
import torch
import torch.nn as nn


class MotorCerebellum(nn.Module):
    def __init__(self, latent_dim, n_actuators, hidden=128, w_max=0.7, ema_beta=0.99):
        super().__init__()
        # z（潜在状態）→ 行動 のフィードフォワード地図（方針を真似る）。
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.SiLU(),
            nn.LayerNorm(hidden), nn.Linear(hidden, n_actuators),
        )
        self.w_max = w_max          # ⚠️自動化の上限（この割合まで小脳に任せる）
        self.ema_beta = ema_beta    # 馴染み度の自己正規化に使う走行平均の時定数
        self.register_buffer("err_ema", torch.tensor(1.0))

    def predict(self, z):
        """小脳のフィードフォワード行動（[-1,1]）。"""
        return torch.tanh(self.net(z))

    def gate(self, z, policy_mean):
        """馴染み度 → 自動化重み w（状態特異）。小脳が方針をよく再現できる状態ほど w 大。
        自己正規化（err_ema）で絶対スケール定数を持たない。w・cere は detach して返す
        （小脳の勾配が方針・順モデルへ漏れないように）。"""
        cere = self.predict(z)
        e = (cere - policy_mean).abs().mean()
        w = self.w_max * torch.exp(-e / (self.err_ema + 1e-6))
        return w.detach(), cere.detach(), e.detach()

    def observe(self, e):
        """馴染み度の基準（誤差の走行平均）を更新。覚醒中の学習1回につき1度だけ呼ぶ。"""
        self.err_ema = self.ema_beta * self.err_ema + (1.0 - self.ema_beta) * e

    def imitation_loss(self, z, action):
        """実際に行った運動を状態から真似る（反復で滑らかな平均＝結晶化パターンを獲得）。"""
        return ((self.predict(z) - action) ** 2).mean()
