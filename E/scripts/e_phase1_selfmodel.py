"""目標E フェーズ1：誰もいない環境で、視覚版の自己モデル(predict_vision)を確立する。

【位置づけ】egomotion割引（自分の動きと他者の動きの区別）の本実装に向けた最初の段階。
親などの外部エージェントは一切置かず、太郎ひとりの自己運動だけで「行動→視覚の結果」の
予測（`taro_brain_motor.py::predict_vision`）を確立する。フェーズ2で親を導入する前に、
まずこの土台を作る（`E/docs/研究日誌.md` 2026-07-21の設計議論を参照）。

【学習の対象＝案B（一緒に学習）】
z推論に関わる部分（emb_proj・motor_gru・pc_latent）と、両方の予測ヘッド
（forward_model_head＝固有感覚、vision_forward_head＝視覚）を一緒に学習する。
既存の insula/proprio/vestibular エンコーダ（目標Cで確立済み）は凍結のまま触らない。

【視覚エンコーダは今回はじめて学習させる】
`fusion.vision`（VisionEncoder, CNN）はこれまで一度もチェックポイントに保存・
読込されておらず、ランダムな重みのまま使われてきた（2026-07-21発覚）。今回が
初めての学習。

【⚠️採点方法（損失）＝[既存AI研究・工学的対処]。人間模倣ではない（ユーザーと確認済み）】
2026-07-21に、素朴なMSE（絶対値・自作の分散正規化）で2回失敗した：
  1. 絶対値そのものをMSE：次元ごとの"相場"を覚えるだけで低損失に見え、別seedでR²=-122。
  2. 変化分を移動平均の分散で正規化：分散が極小(1.9e-6)のため勾配が異常増幅し不安定。
これらは自己予測表現学習(SPR: Schwarzer et al. 2021)・JEPA系の知見と照合した結果、
「予測対象の作り方」自体が分野の標準とズレていたのが根因と判明。標準に合わせて作り直す：
  ①正解を作るターゲット視覚エンコーダを、**固定ランダム**でなく、学習側のEMA
    （指数移動平均＝ゆっくり追従するコピー）にする（SPR/BYOL/SimSiamの標準。固定ランダムな
    エンコーダを予測するのはRND=探索ボーナス用の別技術で、良い予測モデル学習には不向き）。
  ②損失を、生MSEでなく**コサイン類似度**（L2正規化した予測と正解の向きの一致）にする
    （SPRのlossそのもの。大きさの相場を覚えるズルを排除）。
  ③崩壊（どんな入力でも同じ埋め込みを出す）防止に、**分散正則化**（VICReg風に各次元の
    標準偏差を1以上に保つヒンジ項）を軽く足す。
これらは全て[既存AI研究]の工学テクニックで、人間模倣の根拠はない（脳に「EMAコピー」「コサイン
損失」に相当する仕組みは確認できていない）。視覚エンコーダの学習アルゴリズムは"脳内部の学習則"
に相当し、太郎の既存部品(pc_latent等)同様に工学的手法を使わざるを得ない領域という整理。

【2026-07-21 バグ修正（コード監査で発覚。検証の落とし穴チェックリスト項27）】
上記③のコサイン損失を"次の視覚そのもの(絶対値)"に張っていたのが誤り。predict_visionは
「現在+Δ」の残差予測なので、変化ゼロ(現在のまま)でもコサイン0.99997=満点になり、肝心のΔに
勾配が来なかった（絶対値採点で変化分が埋もれる罠を、手段を変えつつ3回踏んだ）。修正：
  バグ①：損失を**Δ(変化分)の向き**で採点＝cosine(delta_pred, delta_true)。
  バグ②：正解(next)と基準(current)を**同じターゲットθ**で符号化（毎tick現在フレームを符号化し直す）。
    持ち越すとエンコーダのEMAドリフトが変化分に混入し予測不能なノイズになっていた。
  バグ③(崩壊対策の共分散項)は、①②の効果を切り分けるため今回は入れず、崩壊が出たら追加する。

使い方: python e_phase1_selfmodel.py [n_tick]
環境変数: E_P1_WMEAN(既定0.3) E_P1_LR(既定1e-3) E_P1_SEED(既定0)
          E_P1_EMA(既定0.99, ターゲットEMAの追従の遅さ) E_P1_SAVE(保存先)
"""

# ⚠️★古い方式（2026-07-30 に整理）。新しい実験は `run/main.py` を通す。
#   【経緯】目標Eの実験スクリプトが118本あり、うち66本が**独立に環境を組み立てていた**。
#     そのため「学習は関節モード（90関節を独立に駆動＝逸脱リスト 逸脱5）、
#     測定とViewerは筋肉モード（拮抗筋2本/関節）」という**別の体で動く**事故が起きた
#     （ユーザーの目視「視線誘導反射の実験の時とは動きが全然違う」で発覚。
#      実測で動きが人間の新生児の約3.3倍速かった）。
#   【設計と移行計画】`E/docs/実行基盤_設計.md`
#   ⚠️このファイルは**記録として残す**（削除しない方針）。
#     中の測り方は再利用できるので、プラグインへ移すときの元にする。
import os
import sys
import copy
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "D", "scripts"))
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "C", "scripts"))
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "taro_core"))
import paths  # noqa: E402
paths.setup_brain_path()
sys.path.insert(0, paths.MIMO_DIR)

import mimoEnv  # noqa: F401,E402
import mujoco  # noqa: E402,F401
import d_c5_motor_quality as mq  # noqa: E402
from vision_encoder import VisionEncoder  # noqa: E402
from e_toy_env import VISION_RES  # noqa: E402

torch.set_num_threads(4)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

W_MEAN = float(os.environ.get("E_P1_WMEAN", "0.3"))   # 2026-07-21実測に基づく採用値
LR = float(os.environ.get("E_P1_LR", "1e-3"))
EMA = float(os.environ.get("E_P1_EMA", "0.99"))       # ターゲットEMAの追従の遅さ[ARBITRARY]
NE_LEVEL = 0.095  # 学習後期のNE水準（d_c5_motor_quality.pyと同じ値）
VAR_COEF = 1.0    # 分散正則化の重み[ARBITRARY]


def to_tensor(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


@torch.no_grad()
def ema_update(target, online, decay):
    """target ← decay*target + (1-decay)*online（BYOL/SPR標準のEMA更新）。"""
    for pt, po in zip(target.parameters(), online.parameters()):
        pt.data.mul_(decay).add_(po.data, alpha=1 - decay)


def encode_target_fusion(fusion, target_vision, obs):
    """pc_latentの再構成損失用の"正解"融合ベクトル。視覚だけEMAターゲットを使う（崩壊防止）。
    他の感覚は目標Cで確立済み・凍結のままなのでそのまま使う。勾配は流さない。"""
    with torch.no_grad():
        parts = [fusion.insula(to_tensor(obs["interoception"])),
                  fusion.proprio(to_tensor(obs["observation"])),
                  fusion.vestibular(to_tensor(obs["vestibular"]))]
        if fusion.touch is not None:
            parts.append(fusion.touch(to_tensor(obs["touch"])))
        parts.append(target_vision(obs["eye_left"], obs["eye_right"]))
        f = torch.cat(parts, dim=-1)
        return torch.nn.functional.layer_norm(f, f.shape)


def cosine_loss(pred, target):
    """L2正規化した予測と正解の負のコサイン類似度（SPRのloss）。0=完全一致方向、2=真逆。"""
    p = F.normalize(pred, dim=-1, eps=1e-6)
    t = F.normalize(target.detach(), dim=-1, eps=1e-6)
    return (1.0 - (p * t).sum(-1)).mean()


def main():
    n_tick = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    seed = int(os.environ.get("E_P1_SEED", "0"))
    save_path = os.environ.get(
        "E_P1_SAVE", os.path.join(_HERE, os.pardir, "models", f"phase1_selfmodel_seed{seed}.pt"))
    print(f"=== フェーズ1：自己モデル(predict_vision)確立（{n_tick}tick, seed={seed}, "
          f"w_mean={W_MEAN}, EMA={EMA}）===\n")

    env, brain, fusion, emb_proj, cereb, n_act = mq.build("off", age=0)

    # 正解を作るターゲット視覚エンコーダ＝学習側(fusion.vision)のEMAコピー（固定ランダムでない）
    target_vision = copy.deepcopy(fusion.vision)
    for p in target_vision.parameters():
        p.requires_grad_(False)
    target_vision.eval()

    params = (list(emb_proj.parameters()) + list(brain.motor_gru.parameters())
              + list(brain.pc_latent.parameters()) + list(brain.forward_model_head.parameters())
              + list(brain.vision_forward_head.parameters()) + list(fusion.vision.parameters()))
    opt = torch.optim.Adam(params, lr=LR, weight_decay=1e-4)
    print(f"  学習対象パラメータ数: {sum(p.numel() for p in params):,}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    obs, _ = env.reset(seed=seed)
    hidden = brain.init_motor_hidden()
    prev_a = torch.zeros(n_act)
    proprio_now = to_tensor(obs["observation"])
    # 【バグ②修正・2026-07-21】視覚の基準(vis_cur)は、前tickから持ち越さず**毎tick現在フレームを
    # 現時点のターゲットθで符号化し直す**。理由：持ち越すと、基準は θ_{t-1}・正解は θ_t で符号化され、
    # 変化分にエンコーダのEMAドリフト(=行動から予測不能なノイズ)が混入していた（コード監査で発覚）。
    # 毎tick current/next を同じ θ_t で符号化すれば、変化分は純粋なフレーム変化になる。

    vis_losses, proprio_losses, kl_losses, var_losses = [], [], [], []
    for t in range(n_tick):
        sv = fusion.encode(obs)                              # zの材料（視覚は学習中のエンコーダ）
        cf = encode_target_fusion(fusion, target_vision, obs)  # 再構成損失の正解（EMAで崩壊しない）

        emb = emb_proj(torch.cat([sv, prev_a], dim=-1)).unsqueeze(0).unsqueeze(0)
        out, hidden_new = brain.motor_gru(emb, hidden)
        z, kl_loss, recon_loss = brain.pc_latent.infer(hidden[-1, 0], out[0, -1], cf)

        # babble方策：w_mean=0.3・白色雑音（2026-07-21確定）
        # 【2026-07-25】太郎の motor_drive を呼ぶだけに変更（core へ一元化）。
        # 旧実装は同じ式を手書きしていた＝**数値は完全に同一**
        # （手書きの std=0.05+ne*0.45 は core の既定 min_std=0.05／max_std=0.5 と一致）。
        mean, std, _w_c, _ = brain.motor_drive(z, NE_LEVEL, cerebellum=cereb)
        noise = torch.randn_like(mean)
        a = torch.clamp(W_MEAN * mean + std * noise, -1, 1)

        pred_proprio = brain.predict_proprio(z, a, proprio_now)
        # 【バグ②修正】現在フレームを"今の"ターゲットθで符号化し直す（正解と同じθ＝ドリフト混入なし）
        with torch.no_grad():
            vis_cur = target_vision(obs["eye_left"], obs["eye_right"])
        pred_vision = brain.predict_vision(z, a, vis_cur)     # vis_cur + Δ(z,行動)。Δ=vision_forward_head出力

        ctrl = mq.rescale_action(a, env.action_space)
        for k in range(mq.K):
            next_obs, r, term, trunc, info = env.step(ctrl)
            if term or trunc:
                break

        actual_proprio = to_tensor(next_obs["observation"])
        online_next_vision = fusion.vision(next_obs["eye_left"], next_obs["eye_right"])  # 崩壊防止の分散用
        with torch.no_grad():
            actual_vision = target_vision(next_obs["eye_left"], next_obs["eye_right"])   # 正解（vis_curと同じθ）

        loss_proprio = nn.functional.mse_loss(pred_proprio, actual_proprio)
        # 【バグ①修正】損失は「変化分Δ」の向きで採点する（絶対値の"次の視覚"だと、変化ゼロ＝現在のまま
        #   でもコサイン0.99997＝満点になり、変化分に勾配が来なかった＝3回失敗の真因。コード監査で確定）。
        delta_pred = pred_vision - vis_cur     # = vision_forward_head([z,a])。勾配あり
        delta_true = actual_vision - vis_cur   # 純粋なフレーム変化（vis_cur/actualは同じθ・no_grad）
        loss_vision = cosine_loss(delta_pred, delta_true)
        # ③分散正則化（VICReg風）：学習側エンコーダの各次元が潰れない（同じ値を出さない）よう
        #   標準偏差を1以上に保つヒンジ。1サンプルでは分散が測れないので、直近の埋め込みを貯めて使う。
        #   【注意】バッファの過去分は前ステップの計算グラフに属し、opt.step()後は逆伝播できない
        #   （inplace変更エラー）。過去分はdetachして"文脈"として使い、勾配は今のサンプルにだけ通す。
        if len(var_losses_buf) >= 8:
            emb_batch = torch.stack(var_losses_buf + [online_next_vision])
            std_per_dim = torch.sqrt(emb_batch.var(dim=0) + 1e-6)
            loss_var = torch.clamp(1.0 - std_per_dim, min=0).mean()
        else:
            loss_var = torch.zeros(())
        var_losses_buf.append(online_next_vision.detach())
        if len(var_losses_buf) > VAR_WINDOW:
            var_losses_buf.pop(0)

        loss = loss_proprio + loss_vision + VAR_COEF * loss_var + kl_loss + recon_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()

        ema_update(target_vision, fusion.vision, EMA)   # 正解係をゆっくり学習側へ追従させる

        vis_losses.append(float(loss_vision.item()))
        proprio_losses.append(float(loss_proprio.item()))
        kl_losses.append(float(kl_loss.item()))
        var_losses.append(float(loss_var.item()) if torch.is_tensor(loss_var) else 0.0)

        obs = next_obs
        hidden = hidden_new.detach()
        prev_a = a.detach()
        proprio_now = actual_proprio.detach()
        # vis_curは次tick冒頭で obs から符号化し直すので持ち越さない（バグ②修正）

        if term or trunc:
            obs, _ = env.reset()
            hidden = brain.init_motor_hidden()
            prev_a = torch.zeros(n_act)
            proprio_now = to_tensor(obs["observation"])

        if (t + 1) % 200 == 0:
            w = 200
            # コサイン損失は0(完全一致方向)〜2(真逆)。1.0=無相関(向きバラバラ)＝実質予測できてない。
            print(f"  [{t+1:5d}/{n_tick}] 視覚コサイン損失={np.mean(vis_losses[-w:]):.4f}"
                  f"(0=完璧,1=無相関) 固有感覚損失={np.mean(proprio_losses[-w:]):.4f} "
                  f"分散罰={np.mean(var_losses[-w:]):.4f} KL={np.mean(kl_losses[-w:]):.4f}", flush=True)

    env.close()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    blob = {
        "brain": brain.state_dict(),
        "fusion_insula": fusion.insula.state_dict(),
        "fusion_proprio": fusion.proprio.state_dict(),
        "fusion_vestibular": fusion.vestibular.state_dict(),
        "fusion_vision": fusion.vision.state_dict(),   # 今回はじめて保存（今まで無かった）
        "target_vision": target_vision.state_dict(),   # 検証で正解係を再現するため
        "emb_proj": emb_proj.state_dict(),
        "cereb": cereb.state_dict(),
        "config": {"n_tick": n_tick, "seed": seed, "w_mean": W_MEAN, "lr": LR, "ema": EMA},
    }
    torch.save(blob, save_path)
    print(f"\n保存: {save_path}")
    print(f"最終200tickの平均 - 視覚コサイン損失: {np.mean(vis_losses[-200:]):.4f} "
          f"固有感覚損失: {np.mean(proprio_losses[-200:]):.4f}")
    np.savez(os.path.join(_HERE, os.pardir, "logs", f"phase1_losses_seed{seed}.npz"),
             vis=np.array(vis_losses), proprio=np.array(proprio_losses),
             kl=np.array(kl_losses), var=np.array(var_losses))


# 分散正則化用の直近埋め込みバッファ（VICReg風、1サンプルでは分散が測れないため）
var_losses_buf = []
VAR_WINDOW = 32


if __name__ == "__main__":
    main()
