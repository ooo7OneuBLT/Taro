"""
本番・追加学習：Cで学習済みの自己モデル（脳）を読み込み、脳をリセットせずに視覚を加えて
学習を継続する（＝人間は脳をリセットしない、A→Bと同じ方針のC→D版）。

【何を学ぶか】Stage 1a：視覚を融合の"入力"に加え、自己モデル（行動→次の固有感覚の予測）を
継続学習する。視覚が入っても脳が壊れず学び続けられるか＝ベースライン。
（視覚も"予測対象"にする reafference＝egomotion割引の本体は、次のStage 1bで足す。
 学習と測定は別フェーズ＝今回は「暴れず学習が回る」ことの確認と録画まで。）

【学習手法】すべてCで確立済み・新規なし：
  ①予測誤差最小化（教師なし回帰）②方策勾配REINFORCE（報酬=予測成功＝内発的動機）
  ③変分推論（pc_latent）④睡眠リプレイ（hippocampus/consolidate）
run_c_metrics_ac_lr.py の学習ティックを、2体・座位・視覚あり用に最小構成で移植。

【モデルの引き継ぎ】次元が変わる層(emb_proj・sensory_proj・pc_latent.decoder等)だけ
新しく初期化し、それ以外(GRU・順モデルヘッド・方策・固有感覚/前庭/内受容エンコーダ)は
Cの重みをそのまま読み込む（＝転移学習：形が合う層だけロード、変わった入出口だけ作り直し）。

使い方: python d_vision_train.py [n_train]   （既定200ティック＝デモ用に短め）
出力: D/logs/video/vision_train_*.png/.mp4（第三者視点＋一人称視点）
"""
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import mujoco
import cv2

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
torch.set_num_threads(4)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "taro_core"))
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "C", "scripts"))  # MinimalFusion再エクスポート等
import paths
paths.setup_brain_path()
sys.path.insert(0, paths.MIMO_DIR)

from hybrid_env import HybridEnv
from fusion import MinimalFusion, to_tensor
from taro_brain_motor import TaroBrainWithMotor
from basal_ganglia import TaroLearner
from dopamine import Dopamine
from locus_coeruleus import LocusCoeruleus
from homeostatic_scaling import HomeostaticScaling
from cerebellum_motor import MotorCerebellum
from test_phase8_motor_learning import CombinedParams, rescale_action
from d_beta_sitting_env import BetaSittingEnv
from d1_carer_vision_env import lean_vision_params

CKPT_PATH = os.path.join(_HERE, os.pardir, os.pardir, "C", "models", "c_pred_abs_seed0.pt")
OUT = os.path.abspath(os.path.join(_HERE, os.pardir, "logs", "video"))
mse = torch.nn.functional.mse_loss
K = 100
DT = 0.01
RES = 64


def ln_prop(obs):
    v = to_tensor(obs["observation"])
    return torch.nn.functional.layer_norm(v, v.shape).detach()


def load_matching(module, saved_sd, tag):
    """形が一致する重みだけロードし、次元が変わって読めなかった層名を返す（転移学習）。"""
    own = module.state_dict()
    matched = {k: v for k, v in saved_sd.items() if k in own and own[k].shape == v.shape}
    skipped = [k for k in own if k not in matched]
    module.load_state_dict(matched, strict=False)
    print(f"  [{tag}] ロード {len(matched)}層 / 作り直し {len(skipped)}層"
          + (f"（作り直し: {', '.join(sorted(set(k.split('.')[0] for k in skipped)))}）" if skipped else ""))
    return skipped


def main():
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    seed = 0
    torch.manual_seed(seed); np.random.seed(seed)

    print("=== 本番・追加学習（Cの自己モデル＋視覚を継続学習）===")
    # ★訂正：fovy拡大とベータ高さ修正を同時に変えてしまい、どちらが「視野がぐちゃぐちゃ」
    # の原因か切り分けられなくなった（1機構ずつ検証、の原則違反）。fovyは既定60に戻し、
    # 高さ修正だけを単独でテストする。
    env = HybridEnv(BetaSittingEnv(vision_params=lean_vision_params(RES, fovy=60)))
    obs, _ = env.reset(seed=seed)
    n_act = env.action_space.shape[0]
    fusion = MinimalFusion(vision_res=RES)
    target_fusion = MinimalFusion(vision_res=RES).freeze()
    sdim = fusion.encode(obs).shape[0]
    prop_dim = to_tensor(obs["observation"]).shape[0]
    print(f"融合次元 sdim={sdim}（視覚64込み）／固有感覚 prop_dim={prop_dim}／行動 n_act={n_act}")

    brain = TaroBrainWithMotor(vocab_size=3, sensory_dim=sdim, n_actuators=n_act)
    emb_dim = brain.sensory_proj.out_features
    emb_proj = nn.Linear(sdim + n_act, emb_dim)
    nat_head = nn.Sequential(nn.Linear(brain.latent_dim + n_act, 128), nn.SiLU(),
                             nn.LayerNorm(128), nn.Linear(128, prop_dim))

    # ── Cの学習済みモデルを引き継ぐ（脳をリセットしない）──
    print(f"\nCの自己モデルを読み込み: {os.path.basename(CKPT_PATH)}")
    blob = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    load_matching(brain, blob["brain"], "脳")
    fusion.insula.load_state_dict(blob["fusion_insula"])
    fusion.proprio.load_state_dict(blob["fusion_proprio"])
    fusion.vestibular.load_state_dict(blob["fusion_vestibular"])
    print("  [融合] 内受容・固有感覚・前庭覚エンコーダをロード（視覚エンコーダは新規＝これから学ぶ）")
    load_matching(emb_proj, blob["emb_proj"], "emb_proj")
    load_matching(nat_head, blob["nat_head"], "nat_head")
    # target_fusionは正解づくり用（凍結）。学習側と同じ初期値にそろえてからfreezeは既にfusionで
    # 済んでいるので、固有感覚系だけ合わせておく（視覚は両者とも新規ランダムで独立性を保つ）。

    learner = TaroLearner(CombinedParams(brain, fusion, emb_proj, nat_head), lr=0.005)
    dop = Dopamine(); ne = LocusCoeruleus(); homeo = HomeostaticScaling(dim=sdim)
    # 運動小脳：練習を重ねるほど動きが滑らかに自動化される（＝"首が座る"等の制御の上達に対応）。
    # Cから読み込む（重み形は同じなのでそのままロード＝脳と同じく引き継ぐ）。
    cereb = MotorCerebellum(brain.latent_dim, n_act)
    load_matching(cereb, blob["cereb"], "小脳")
    cere_opt = torch.optim.Adam(cereb.parameters(), lr=0.005)
    hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)
    buf = {k: [] for k in ("sv", "prev_a", "a", "cf", "clp", "nlp", "h")}

    def zc(sv, pa, cf, h):
        emb = emb_proj(torch.cat([sv, pa], dim=-1)).unsqueeze(0).unsqueeze(0)
        out, nh = brain.motor_gru(emb, h)
        z, kl, rc = brain.pc_latent.infer(h[-1, 0], out[0, -1], cf)
        return z, kl, rc, nh

    def step_k(a):
        o, term = obs, False
        for _ in range(K):
            o, r, te, tr, info = env.step(a)
            if te or tr:
                term = True; break
        return o, term

    def consolidate(n_batches=100, bs=128):
        N = len(buf["sv"])
        if N < bs:
            return
        SV = torch.stack(buf["sv"]); PA = torch.stack(buf["prev_a"]); AA = torch.stack(buf["a"])
        CF = torch.stack(buf["cf"]); CLP = torch.stack(buf["clp"]); NLP = torch.stack(buf["nlp"])
        H = torch.cat(buf["h"], dim=1)
        for _ in range(n_batches):
            idx = torch.randint(0, N, (bs,))
            hb = H[:, idx].contiguous()
            emb = emb_proj(torch.cat([SV[idx], PA[idx]], dim=-1)).unsqueeze(1)
            out, _ = brain.motor_gru(emb, hb)
            z, kl, rc = brain.pc_latent.infer(hb[-1], out[:, 0], CF[idx])
            pred = CLP[idx] + nat_head(torch.cat([z, AA[idx]], dim=-1))
            loss = mse(pred, NLP[idx]) + kl + rc
            learner.optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(learner.brain.parameters(), learner.grad_clip)
            learner.optimizer.step()
            # ★今回の新規（ユーザー提案）：小脳の練習も睡眠リプレイの対象にする。
            # 従来は小脳は"その場の1回"しか練習できず生の経験回数に律速されていた。
            # 貯めた(状態z, 実際の行動)を復習して自動化を多重に鍛える＝自己教師ありのまま多重化。
            closs = cereb.imitation_loss(z.detach(), AA[idx])
            cere_opt.zero_grad(); closs.backward(); cere_opt.step()

    print(f"\n学習開始（n_train={n_train}ティック・K={K}・視覚ON{RES}x{RES}）")
    t0 = time.time()
    # ★ユーザー指摘(2026-07-17)：録画した映像が「視野がぐちゃぐちゃ」に見えた。訂正：
    # record()の行動は探索ノイズ(std)を足さない方策の"平均値"そのもの＝喃語(ランダム性)は
    # 無関係で、方策自体(まだ十分学習していない読込直後の脳)の出力の"大きさ"が原因の疑い。
    # 90アクチュエータの多くが極端な値だと全身が毎秒(K=100=1秒)大きく振れ暴れて見える。
    # ⚠️経験的な定数＝感度確認が要る。人間の乳児の運動性喃語も全力で動くわけではない、
    # という考えに基づき、方策の出力に上限をかけて抑える。
    ACTION_SCALE = 0.3

    pe_hist = []
    for i in range(n_train):
        # ベータ（親）を左右にゆっくり往復させる（egomotion＝相手の動き）
        beta_z = env.unwrapped.BETA_HOME[2]   # ここも0.35のハードコード残りだった（見落とし）
        y = 0.3 * np.sin(2 * np.pi * i / 60.0)
        env.unwrapped.set_beta_target([0.3, y, beta_z])

        sv = fusion.encode(obs); cf = target_fusion.encode(obs).detach(); clp = ln_prop(obs)
        z, kl, rc, hn = zc(sv, prev_a, cf, hidden.detach())
        policy_m = torch.tanh(brain.motor_head(z.detach())) * ACTION_SCALE
        std = (0.05 + ne.get_ne_level() * 0.45) * ACTION_SCALE
        # 小脳の自動化ブレンド：馴染んだ状態ほど小脳の滑らかな出力で置換＋探索ノイズ減（結晶化）
        w_c, cere_a, e_c = cereb.gate(z.detach(), policy_m)
        mean = (1.0 - w_c) * policy_m + w_c * cere_a
        std = std * (1.0 - w_c)
        dist = torch.distributions.Normal(mean, std)
        raw = dist.sample(); a = torch.clamp(raw, -1.0, 1.0); lp = dist.log_prob(a).sum()
        pred = clp + nat_head(torch.cat([z, a.detach()], dim=-1))

        obs, term = step_k(rescale_action(a, env.action_space)); nlp = ln_prop(obs)
        buf["sv"].append(sv.detach()); buf["prev_a"].append(prev_a.detach())
        buf["a"].append(a.detach()); buf["cf"].append(cf.detach())
        buf["clp"].append(clp.detach()); buf["nlp"].append(nlp.detach())
        buf["h"].append(hidden.detach())

        pe = mse(pred, nlp)
        rew = brain.sensorimotor_reward(pe.item())
        pl = learner.learn_action([lp], dop.compute_rpe(rew))
        hl = homeo.homeostatic_loss(sv); homeo.observe(sv)
        learner.update(pe + hl + kl + rc, pl)
        ne.observe_reward(rew); ne.release_ne()
        # 小脳：覚醒中も「実際に行った運動」を状態から真似て自動化を鍛える＋馴染み度を更新
        closs = cereb.imitation_loss(z.detach(), a.detach())
        cere_opt.zero_grad(); closs.backward(); cere_opt.step()
        cereb.observe(e_c)
        hidden = hn.detach(); prev_a = a.detach()
        pe_hist.append(pe.item())

        if (i + 1) % 50 == 0:
            recent = np.mean(pe_hist[-50:])
            print(f"  tick {i+1:4d}/{n_train}  予測誤差(直近50平均)={recent:.4f}  "
                  f"経過={time.time()-t0:.0f}s", flush=True)
            consolidate()   # 睡眠リプレイ

    print(f"学習完了（{time.time()-t0:.0f}s）。前半50 vs 後半50の予測誤差: "
          f"{np.mean(pe_hist[:50]):.4f} → {np.mean(pe_hist[-50:]):.4f}")
    return env, brain, fusion, emb_proj, nat_head, zc, step_k, dict(hidden=hidden, prev_a=prev_a, cereb=cereb)


def record(env, brain, fusion, emb_proj, nat_head, zc, step_k, state, n=100):
    """学習後の太郎を、第三者視点＋一人称視点(眼球)で録画。"""
    print("\n録画中（第三者視点＋一人称視点）...")
    m = env.unwrapped.model
    m.vis.global_.offwidth = 640; m.vis.global_.offheight = 480
    ren = mujoco.Renderer(m, height=480, width=640)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 1.9; cam.elevation = -12; cam.azimuth = 90; cam.lookat = [0.2, 0.0, 0.25]

    hidden, prev_a = state["hidden"], state["prev_a"]
    beta_z = env.unwrapped.BETA_HOME[2]   # クラス側の較正済み高さ(0.45)を使う（ここでの再ハードコード禁止）
    third, eye = [], []
    obs, _ = env.reset()
    # ★滑らかな録画にする：1ティック(K=100物理ステップ=1秒)を一気に飛ばすと「コマ送り」に
    # なるので、K内をSUB刻みで描画する。行動は1ティックの間は固定(現Taroと同じ)なので、
    # 学習ロジックは変えず「途中経過も撮る」だけ。
    SUB = 10   # 1秒あたり8フレーム（K/SUB=約12ステップごとに1描画）
    for i in range(n):
        y = 0.3 * np.sin(2 * np.pi * i / 60.0)
        env.unwrapped.set_beta_target([0.3, y, beta_z])
        sv = fusion.encode(obs); cf = target_fusion_dummy(sv)
        z, kl, rc, hn = zc(sv, prev_a, cf, hidden.detach())
        policy_m = torch.tanh(brain.motor_head(z.detach())) * 0.3  # 学習側と同じACTION_SCALE
        # 学習した小脳の自動化を反映（ノイズなしの決定的行動＝方策と小脳のブレンド）
        w_c, cere_a, _ = state["cereb"].gate(z.detach(), policy_m)
        a = torch.clamp((1.0 - w_c) * policy_m + w_c * cere_a, -1, 1)
        ctrl = rescale_action(a, env.action_space)
        term = False
        for k in range(K):
            o, r, te, tr, info = env.step(ctrl)
            if (k % (K // SUB)) == 0:   # K内でSUB回描画＝滑らかに
                ren.update_scene(env.unwrapped.data, camera=cam)
                third.append(ren.render())
                eye.append(cv2.resize(np.asarray(o["eye_left"]).copy(), (240, 240),
                                      interpolation=cv2.INTER_NEAREST))
            if te or tr:
                term = True; break
        obs = o
        hidden = hn.detach(); prev_a = a.detach()

    os.makedirs(OUT, exist_ok=True)
    eye_mp4 = os.path.join(OUT, "vision_train_eye.mp4")
    vw = cv2.VideoWriter(eye_mp4, cv2.VideoWriter_fourcc(*"mp4v"), 10, (240, 240))
    for f in eye:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    mp4 = os.path.join(OUT, "vision_train_thirdperson.mp4")
    vw = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 480))
    for f in third:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    # シート：第三者4コマ＋一人称は多めに(12コマ)並べて「見えている割合」を目視しやすくする
    idxs4 = [0, n // 3, 2 * n // 3, n - 1]
    row1 = np.concatenate([cv2.resize(third[k], (240, 240)) for k in idxs4], axis=1)
    png = os.path.join(OUT, "vision_train_sheet.png")
    cv2.imwrite(png, cv2.cvtColor(row1, cv2.COLOR_RGB2BGR))

    idxs12 = np.linspace(0, n - 1, 12).astype(int)
    eye_rows = [np.concatenate([eye[k] for k in idxs12[j:j+4]], axis=1) for j in range(0, 12, 4)]
    eye_sheet = np.concatenate(eye_rows, axis=0)
    eye_png = os.path.join(OUT, "vision_train_eye_sheet.png")
    cv2.imwrite(eye_png, cv2.cvtColor(eye_sheet, cv2.COLOR_RGB2BGR))
    print("mp4(第三者):", mp4)
    print("mp4(一人称/眼球):", eye_mp4)
    print("png(第三者):", png)
    print("png(一人称12コマ):", eye_png)


def target_fusion_dummy(sv):
    return sv.detach()


if __name__ == "__main__":
    result = main()
    record(*result)
