"""
確率的な潜在変数＋推論時のその場調整（PV-RNNに着想）

【人間模倣】予測符号化の代表的な計算モデルPV-RNN（Ahmadi & Tani, 2019, Neural
Computation）に着想。知覚は「重みの学習」だけでなく「その場（推論時）の
状態調整」でも作られる。GRUの状態から「事前の予想(prior)」を、実際の
感覚も見た後に「事後の判断(posterior)」を、それぞれ確率分布（平均・
ばらつき）として持つ。posteriorは、重みではなく posterior自身の
パラメータだけを、数ステップその場で勾配降下（誤差回帰）して調整してから
使う。事前と事後のズレ（KLダイバージェンス）を訓練の損失に加えることで、
「予想からあまり離れすぎない」という歯止めも学習に反映される。

⚠️簡略化（本物のPV-RNNとの違い）：
- 本物は複数時刻の窓をまたいで誤差回帰するが、ここでは1ステップだけの
  簡略版（今回の感覚をzがどれだけ再構成できるかで、その場の調整を行う）
- 潜在変数zはGRUの再帰そのものには組み込まず、GRUの出力から作る
  下流の追加要素として扱う（土台のGRUは変更しない）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictiveCodingLatent(nn.Module):
    def __init__(self, hidden_dim, sensory_dim, latent_dim=32,
                 n_regression_steps=3, regression_lr=0.1, kl_weight=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_regression_steps = n_regression_steps
        self.regression_lr = regression_lr
        self.kl_weight = kl_weight

        # 事前の予想：GRUの状態(今回の感覚を見る"前")から作る
        self.prior_net = nn.Linear(hidden_dim, latent_dim * 2)
        # 事後の判断の初期値：GRUの状態(今回の感覚を処理した"後")から作る
        self.posterior_net = nn.Linear(hidden_dim, latent_dim * 2)
        # zが「今回の感覚」をどれだけ説明できるかを見るための再構成デコーダ
        self.decoder = nn.Linear(latent_dim, sensory_dim)

    def infer(self, hidden_before, hidden_after, current_sensory):
        """
        hidden_before: GRUに今回の感覚を通す"前"の状態 (prior用)
        hidden_after : GRUが今回の感覚を処理した"後"の状態 (posterior用)
        current_sensory: 今回の融合ベクトル（zの再構成目標、detach済み想定）

        戻り値: (z, kl_loss)
          z: その場で調整した後の潜在変数（勾配は事後netの重みに繋がる）
          kl_loss: 事前・事後のズレ（訓練の損失に加える）
        """
        prior_mean, prior_logvar = self.prior_net(hidden_before).chunk(2, dim=-1)
        post_mean, post_logvar = self.posterior_net(hidden_after).chunk(2, dim=-1)

        # 複数エージェントによるレビューで判明したバグの修正：
        # logvarをクランプせずexp(logvar)を_kl()の分母に使うと、prior_netが
        # 未学習のランダム初期値のままの間、シードの巡り合わせだけでexp(logvar_p)
        # が極端に小さくなり、KL項が暴走する（8シード中一部だけ崩壊する原因の
        # 有力候補として複数の独立した分析が一致して指摘）。安全域にクランプする。
        prior_logvar = torch.clamp(prior_logvar, min=-4.0, max=4.0)
        post_logvar = torch.clamp(post_logvar, min=-4.0, max=4.0)

        # --- その場の誤差回帰：post_mean/post_logvarの"値"だけを数ステップ調整 ---
        # (重みには一切触れない。posterior自身のパラメータのみの勾配降下)
        mean = post_mean.detach().clone().requires_grad_(True)
        logvar = post_logvar.detach().clone().requires_grad_(True)
        prior_mean_d = prior_mean.detach()
        prior_logvar_d = prior_logvar.detach()

        for _ in range(self.n_regression_steps):
            std = torch.exp(0.5 * logvar)
            z_sample = mean + torch.randn_like(std) * std
            recon = F.mse_loss(self.decoder(z_sample), current_sensory)
            kl = self._kl(mean, logvar, prior_mean_d, prior_logvar_d)
            free_energy = recon + self.kl_weight * kl
            g_mean, g_logvar = torch.autograd.grad(free_energy, [mean, logvar])
            mean = (mean - self.regression_lr * g_mean).detach().requires_grad_(True)
            logvar = (logvar - self.regression_lr * g_logvar).detach().requires_grad_(True)

        # straight-through：使う"値"は調整後のもの、勾配は元のpost_net(重み)へ流す
        mean_out = post_mean + (mean.detach() - post_mean.detach())
        logvar_out = post_logvar + (logvar.detach() - post_logvar.detach())

        std_out = torch.exp(0.5 * logvar_out)
        z = mean_out + torch.randn_like(std_out) * std_out

        # 以前はprior_mean/prior_logvarもdetachしており、prior_netが一度も
        # 学習されずランダム初期化のまま固定され続けていた（レビューで発覚した
        # バグ）。外側のKL損失だけはprior側もdetachせず、"事前の予想"自体が
        # 経験で更新されるようにする（本物のPV-RNNと同様、prior/posteriorの
        # 両方が学習対象）。内側のその場調整(上のprior_mean_d/prior_logvar_d)は、
        # 「その瞬間はpriorを固定して判断だけ動かす」という意図のままdetachを維持する。
        kl_loss = self._kl(post_mean, post_logvar, prior_mean, prior_logvar)

        # デコーダ(self.decoder)自体は内側のその場調整では重みが更新されない
        # （調整対象はmean/logvarの値のみのため）。ここで勾配ありの再構成損失を
        # 別途返し、外側の重み学習でdecoderも一緒に鍛える。
        recon_loss = F.mse_loss(self.decoder(z), current_sensory)

        return z, kl_loss, recon_loss

    @staticmethod
    def _kl(mean_q, logvar_q, mean_p, logvar_p):
        """事後(q)と事前(p)のKLダイバージェンス（"予想から離れすぎるな"という歯止め）"""
        return 0.5 * torch.mean(
            logvar_p - logvar_q
            + (torch.exp(logvar_q) + (mean_q - mean_p) ** 2) / torch.exp(logvar_p)
            - 1
        )
