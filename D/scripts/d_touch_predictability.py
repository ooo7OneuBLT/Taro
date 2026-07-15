"""
診断：触覚は原理的に予測できるのか。それとも予測すべき情報がそもそも無いのか。

【背景】2026-07-15
仰向け・触覚ありの3シードは 触覚 persist 97〜104%（＝「何も変わらない」予測と互角以下）で
**触覚が学べない**。一方 `d_supine_touch_truth.py` で測ると触覚は何もしない太郎の**22.4倍**動いて
おり、「学ぶ対象が無い」わけではない。＝**学習側の問題**か、**情報が足りない**かのどちらか。

【ユーザーの仮説】
「人間は質感や温度で、これが人なのか物なのかを判断している」
＝今の触覚は「どこかに何かが当たった力」しか伝えず、**何に当たったのか（床か・自分の腕か）を
区別する情報が無い**。だから触覚は太郎にとって原理的に予測不能で、学習の失敗は当然である。

【この診断のやり方】
太郎が実際にやっているのは「今の状態と行動から**0.5秒後**の触覚を当てる」という難しい課題。
そこで**もっと簡単な問題**を出す：「**今の**関節角度から、**今の**触覚を当てる」。時間予測も行動も
不要な、ただの同時刻の対応づけ。関節角度が全部分かれば体の形は幾何学的に決まるので、
自分の腕がどこに触れているかは原理的に決まっているはず。

  簡単な問題すら解けない → 触覚と体の状態が結びついていない＝**情報が無い**（仮説が正しい）
  簡単な問題は解ける     → 情報はある＝**学習側**の問題

【入力を3段階にして、情報がどこで失われるかまで特定する】
  ①固有感覚621のみ      … 太郎が予測に使っている情報
  ②固有感覚＋前庭6      … 太郎が実際に持っている全情報（体の向きを含む）
  ③物理の真の状態(qpos+qvel) … **オラクル**＝情報が原理的に存在するか
オラクルでも解けない → 情報が存在しない。オラクルなら解けるのに①②で解けない → **太郎の感覚に
情報が足りない**（＝仮説がピンポイントで当たる）。

【注意】proprioは関節空間のみ（位置・速度・トルク・可動限界・筋活性）で、**体が世界のどこに
どんな向きであるかは入っていない**（DEFAULT_PROPRIOCEPTION_PARAMS）。向きは前庭が別に持つ。
＝床との接触は①だけでは原理的に当てられない可能性がある。だから②③と比べる必要がある。

使い方: python d_touch_predictability.py [model_path] [n_sample]
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("C_TOUCH", "1")
os.environ.setdefault("C_SUPINE", "1")
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(4)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "C", "scripts"))
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "C"))
import paths
paths.setup_brain_path()
sys.path.insert(0, paths.MIMO_DIR)

import gymnasium as gym
import mimoEnv  # noqa
from hybrid_env import HybridEnv
from taro_brain_motor import TaroBrainWithMotor
from cerebellum_motor import MotorCerebellum
from test_phase8_motor_learning import rescale_action
from run_c_metrics_ac_lr import MinimalFusion, _ENV_ID, _touch_params

K = 100
SUB = 10   # K=100の中で10stepごとに標本を取る（隣接stepは相関が強すぎて水増しになる）

# 【測定器の作り直し・2026-07-15】初版は健康診断に2つとも落ちた：
#   負例（入力=乱数）が学習R²=+90.2% → 3606次元の出力を3200標本で当てさせており、
#     ネットが入力を無視して**答えを丸暗記**できてしまっていた（学習R²は全部ただの暗記）。
#   正例（入力=触覚そのもの＝恒等写像）が検証R²=+60.3% → **恒等写像すら般化しない**＝
#     時系列で後ろ20%を切ったせいで、太郎が転がって別姿勢になった区間を検証にしていた。
# → 3つの負のR²は「情報が無い」ではなく「測れていない」を意味していた。
#    自動判定は"ユーザーの仮説を支持する"と出したが、**壊れた測定器が欲しい答えを返しただけ**。
# 対策：①出力を部位ごとの触覚合計(約59次元)に落とす ②多数のエピソードから集める
#       ③エピソード単位で分割する（外挿ではなく般化を測る）


def touch_by_body(env):
    """部位ごとの触覚の合計を返す。＝「どの部位が何かに触れているか」

    3606次元の生の力ベクトルをそのまま当てさせると、標本数に対して出力次元が多すぎて
    ネットが丸暗記する（初版の失敗）。部位ごとに畳むと約59次元になり、物理的にも
    「どこが触れているか」という意味のある量になる。
    """
    t = env.unwrapped.touch
    out, names = [], []
    for bid in sorted(t.sensor_outputs.keys()):
        out.append(float(np.abs(np.asarray(t.sensor_outputs[bid])).sum()))
        names.append(env.unwrapped.model.body(bid).name)
    return np.asarray(out, dtype=np.float32), names


def collect(mp, n_dec):
    """学習済みの太郎を動かし、(固有感覚, 前庭, 真の状態, 触覚) を同時刻で集める。"""
    ck = torch.load(mp, weights_only=False); cfg = ck["config"]
    T = cfg["touch_dim"]
    env = HybridEnv(gym.make(_ENV_ID, vision_params=None, touch_params=_touch_params()))
    fusion = MinimalFusion(T); tfusion = MinimalFusion(T).freeze()
    n_act = env.action_space.shape[0]
    obs, _ = env.reset(seed=cfg["seed"])
    d = env.unwrapped.data
    brain = TaroBrainWithMotor(vocab_size=3, sensory_dim=cfg["sdim"], n_actuators=n_act)
    emb_proj = nn.Linear(cfg["sdim"] + n_act, brain.sensory_proj.out_features)
    cereb = MotorCerebellum(brain.latent_dim, n_act)
    brain.load_state_dict(ck["brain"])
    fusion.insula.load_state_dict(ck["fusion_insula"]); fusion.proprio.load_state_dict(ck["fusion_proprio"])
    fusion.vestibular.load_state_dict(ck["fusion_vestibular"]); fusion.touch.load_state_dict(ck["fusion_touch"])
    emb_proj.load_state_dict(ck["emb_proj"]); cereb.load_state_dict(ck["cereb"])
    for m in (brain, emb_proj, cereb):
        for p in m.parameters():
            p.requires_grad_(False)
    fusion.freeze()

    # エピソードを分けて集める。1本の軌跡から取ると標本が相関しきっていて、
    # 「学習用と検証用が実質同じデータ」か「検証用が完全な外挿」のどちらかになる（初版の失敗）。
    n_ep = 40
    per_ep = max(1, n_dec // n_ep)
    P, V, S, Y, EP = [], [], [], [], []
    names = None
    for ep in range(n_ep):
        obs, _ = env.reset(seed=1000 + ep)      # 毎回ちがう初期姿勢（jitterで揺らぐ）
        h = brain.init_motor_hidden(); pa = torch.zeros(n_act)
        for i in range(per_ep):
            sv = fusion.encode(obs); cf = tfusion.encode(obs).detach()
            emb = emb_proj(torch.cat([sv, pa], dim=-1)).unsqueeze(0).unsqueeze(0)
            out, hn = brain.motor_gru(emb, h)
            z, _, _ = brain.pc_latent.infer(h[-1, 0], out[0, -1], cf); z = z.detach()
            pm = torch.tanh(brain.motor_head(z)); w, ca, _ = cereb.gate(z, pm)
            a = torch.clamp((1 - w) * pm + w * ca, -1, 1).detach()
            # 半分は探索的な行動にして姿勢の多様性を稼ぐ（学習した太郎だけだと軌跡が偏る）
            if ep % 2 == 1:
                a = torch.clamp(a + torch.randn(n_act) * 0.5, -1, 1)
            ctrl = rescale_action(a, env.action_space)
            te = tr = False
            for k in range(K):
                obs, r, te, tr, info = env.step(ctrl)
                if k % SUB == 0:
                    tb, names = touch_by_body(env)
                    P.append(np.asarray(obs["observation"], dtype=np.float32))
                    V.append(np.asarray(obs["vestibular"], dtype=np.float32))
                    S.append(np.concatenate([d.qpos, d.qvel]).astype(np.float32))
                    Y.append(tb)
                    EP.append(ep)
                if te or tr:
                    break
            h = hn.detach(); pa = a
            if te or tr:
                break
        if (ep + 1) % 10 == 0:
            print(f"  収集中 {ep+1}/{n_ep}エピソード  標本{len(P)}", flush=True)
    env.close()
    return (np.stack(P), np.stack(V), np.stack(S), np.stack(Y), np.asarray(EP), names)


EP_GLOBAL = None


def fit_and_score(X, Y, name, steps=4000):
    """Xから同時刻のYを当てられるかを測る。時系列なので前80%で学習・後20%で検証。

    ランダム分割にすると、隣接して相関の強い標本が学習側と検証側の両方に入り、
    **カンニング（情報漏れ）で「予測できた」ことになってしまう**ので時系列分割にする。

    **学習側のR²も必ず併記する**。検証R²だけ見ると「学習不足で当てられない」と
    「情報が無くて当てられない」が区別できない（2026-07-15に実際に取り違えかけた
    ＝300stepしか回さず、オラクルまでR²が負になった）。
      学習R²も低い → 学習不足 or 情報が無い（測定器の健康診断＝下の正例で切り分ける）
      学習R²は高いのに検証R²が低い → 過学習 or 検証区間が外挿
    """
    # **エピソード単位で分ける**。同じエピソードの標本が学習側と検証側に跨ると、
    # 隣接コマ同士でカンニングになり「予測できた」が出てしまう。
    te_mask = EP_GLOBAL >= (EP_GLOBAL.max() * 0.8)
    tr_mask = ~te_mask
    mu, sd = X[tr_mask].mean(0), X[tr_mask].std(0) + 1e-6
    Xtr = torch.tensor((X[tr_mask] - mu) / sd); Xte = torch.tensor((X[te_mask] - mu) / sd)
    ym, ys = Y[tr_mask].mean(0), Y[tr_mask].std(0) + 1e-6
    Ytr = torch.tensor((Y[tr_mask] - ym) / ys); Yte = torch.tensor((Y[te_mask] - ym) / ys)
    ntr = len(Xtr)

    # 基準：学習データの平均をいつも答える（＝何も学ばない）
    base_te = ((Yte - Ytr.mean(0)) ** 2).mean().item()
    base_tr = ((Ytr - Ytr.mean(0)) ** 2).mean().item()

    # 容量を絞る（初版は512x512で、入力が乱数でも学習R²+90%＝**丸暗記**できてしまった）。
    net = nn.Sequential(nn.Linear(X.shape[1], 128), nn.SiLU(),
                        nn.LayerNorm(128), nn.Linear(128, Y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(steps):
        idx = torch.randperm(ntr)[:256]
        loss = ((net(Xtr[idx]) - Ytr[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        te = ((net(Xte) - Yte) ** 2).mean().item()
        tr = ((net(Xtr) - Ytr) ** 2).mean().item()
    r2te = 1 - te / max(base_te, 1e-12)
    r2tr = 1 - tr / max(base_tr, 1e-12)
    print(f"[{name:26s}] 入力{X.shape[1]:5d}次元 | 学習R²={r2tr*100:+6.1f}%  検証R²={r2te*100:+6.1f}%")
    return r2te, r2tr


def main():
    mp = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, os.pardir, "models", "supine_touch1_seed1.pt")
    n_dec = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    print(f"=== 触覚は原理的に予測できるか（同時刻の対応づけ）===")
    print(f"model={os.path.basename(mp)}  {n_dec}判断ぶん収集（{SUB}stepごとに標本）\n")
    P, V, S, Y, EP, names = collect(mp, n_dec)
    global EP_GLOBAL
    EP_GLOBAL = EP
    ntr = int((EP < EP.max() * 0.8).sum()); nte = len(EP) - ntr
    print(f"\n標本数={len(P)}（学習{ntr} / 検証{nte}・**エピソード単位で分割**）")
    print(f"固有感覚={P.shape[1]}  前庭={V.shape[1]}  真の状態={S.shape[1]}  触覚(部位ごと)={Y.shape[1]}")
    print(f"触覚が立っている部位の割合={float((Y>1e-6).mean())*100:.1f}%  "
          f"部位ごとの変動係数の中央値={float(np.median(Y.std(0)/(Y.mean(0)+1e-9)))*100:.0f}%\n")
    print("R²＝『平均を答えるだけ』と比べてどれだけ誤差を減らせたか。0%=何も説明できていない、100%=完璧")
    print("\n--- 測定器の健康診断（落とし穴チェック項6）---")
    print("正例＝入力に触覚そのものを与える。ここで高いR²が出なければ、学習の手続きが壊れており")
    print("下の結果は全て無意味（『情報が無い』ではなく『測れていない』を意味する）。")
    ctrl_te, ctrl_tr = fit_and_score(Y, Y, "正例：触覚→触覚(恒等)")
    # 負例＝入力を乱数にする。ここでR²が0近辺でなければ、カンニング経路がある。
    rng = np.random.default_rng(0)
    fit_and_score(rng.standard_normal((len(Y), 64), dtype=np.float32), Y, "負例：乱数→触覚")
    if ctrl_tr < 0.5:
        print(f"\n[中止] 正例の学習R²が{ctrl_tr*100:.0f}%しかない＝**測定器が壊れている**。")
        print("       触覚を入力に与えてすら当てられないなら、他の入力で当てられなくて当然。")
        print("       学習手続き（step数・容量・正規化）を直してから測り直すこと。")
        env_msg = True
    else:
        env_msg = False
    # ★太郎の課題における正例（2026-07-15追加・これが最も効く対照）
    # 上の「触覚→触覚(恒等)」は学習手続きが動くかしか見ていない。ここでは
    # **同じ測定器・同じデータ・同じ分割で、予測対象を固有感覚に差し替える**。
    # 太郎は固有感覚なら学べることが分かっている（margin+45 / persist73）ので、
    # 固有感覚は「太郎の課題における正解が分かっている対象」＝真の正例になる。
    #   固有感覚は高R²で触覚だけ負 → **触覚という対象に固有の問題**
    #   固有感覚も負               → **この測定器は太郎の課題を映していない**＝設計し直す
    print("\n--- ★太郎の課題における正例：対象を固有感覚に差し替える ---")
    print("太郎は固有感覚なら学べている(margin+45/persist73)。同じ測定器でこれが負なら、")
    print("測定器が太郎の課題を映していない＝下の触覚の結果ごと無効。")
    P_small = P[:, :55]   # 触覚(55次元)と同じ出力次元にして条件を揃える
    (ctrl2_te, ctrl2_tr) = fit_and_score(S, P_small, "正例2：真の状態→固有感覚")
    fit_and_score(np.concatenate([P, V], 1), P_small, "参考：固有感覚→固有感覚(一部)")

    print("\n--- 本題 ---")
    (r1, r1tr) = fit_and_score(P, Y, "①固有感覚のみ")
    (r2, r2tr) = fit_and_score(np.concatenate([P, V], 1), Y, "②固有感覚＋前庭")
    (r3, r3tr) = fit_and_score(S, Y, "③真の状態（オラクル）")

    print("\n=== 判定 ===")
    if env_msg:
        print("→ 測定器が壊れているので判定しない。")
        return
    # 【重要・2026-07-15】「オラクルが解けない＝情報が無い」と判定してはいけない。
    # **物理が情報の存在を保証している**：関節角度が決まれば体の形は幾何学的に確定し、
    # MuJoCoは決定論的なので (qpos, qvel) から接触力は数学的に一意に決まる。
    # よってオラクルのR²が負でも、意味するのは「情報が無い」ではなく
    # **「決まってはいるが実質的に学習不能なほど暴れる関数である」**。
    # 判定は必ず**正例2（対象を固有感覚に差し替え）との対比**で行う。同じ入力・同じ分割・
    # 同じ出力次元なので、データ量・容量・分割の厳しさは両者に等しく効く＝それらを交絡から外せる。
    gap = ctrl2_te - r3
    print(f"対比：同じ入力(真の状態)・同じ分割・同じ出力次元で、")
    print(f"      対象=固有感覚 → 検証R² {ctrl2_te*100:+.1f}%")
    print(f"      対象=触覚     → 検証R² {r3*100:+.1f}%   （差 {gap*100:.0f}ポイント）")
    if gap > 0.3:
        print(f"\n→ **触覚という対象に固有の困難**。データ量・ネット容量・分割の厳しさは")
        print(f"   固有感覚側にも等しく効いているのに、固有感覚は正・触覚は負。")
        print(f"   物理は情報の存在を保証しているので、これは『情報が無い』ではなく")
        print(f"   **『接触力は不連続でカオス的＝決まっているが学習不能』**を意味する。")
        print(f"   ＝予測対象を「暴れる力」から「滑らかな量」に変えるべき。")
        print(f"   （ユーザーの仮説『力だけでは足りない』は、理由が変わるが生きる）")
    elif ctrl2_te < 0.1:
        print(f"\n→ **正例2（固有感覚）すら般化しない(R²={ctrl2_te*100:.0f}%)**＝この測定器は")
        print(f"   太郎の課題を映していない（太郎は固有感覚なら学べている）。触覚の結果ごと無効。")
        print(f"   測定の設計を作り直すこと。")
    else:
        print(f"\n→ 触覚と固有感覚の差が小さい（{gap*100:.0f}ポイント）＝触覚固有の困難とは言えない。")
        print(f"   本文の数字を見て考える。")


if __name__ == "__main__":
    main()
