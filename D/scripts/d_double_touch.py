"""
★二重接触（double touch）で、触覚を「自己」と「外部」に分けて測る。

【二重接触とは】2026-07-16
自分で自分を触ると、**触覚が2箇所同時に立つ**（触る手と、触られる胸）。
他者や床に触られると、**1箇所しか立たない**。この非対称は太郎の環境に**既に存在していた**。

  実測（580接触・`d1_carer_env` で床・自己・養育者の3つが同居する場面）：
    | 発信元     | 二重接触(両側にセンサ) | 片側だけ |
    | 自分の体   |        396           |    0    |
    | 床         |          0           |  152    |
    | 養育者の手 |          0           |   32    |
  ＝**100%分離**。統計的傾向ではなく**構造的必然**（MIMoは太郎の体にしかセンサを置かないので、
  太郎の体同士が触れれば必ず両方が感じ、床や養育者が触れれば片方しか感じない）。
  ただし**床と養育者は区別できない**（どちらも片側）。それは別の問題として残る。

【この実験が問うこと】
3日間、触覚の予測は一貫して失敗してきた。直近の測定（`d_touch_classify.py`）では、
**接触の有無**の分類にしても AUC 0.63〜0.69 と弱いままだった。

**仮説**：あの中途半端な数字は、**性質の違う2つを混ぜた平均**だったのではないか。
  ・**自己接触は、姿勢から幾何学的に完全に決まる**（自分の手と自分の胸の位置は関節角度で確定）
    → **予測できるはず**
  ・**外部接触（床）は、世界次第**（体が世界のどこにあるか。固有感覚は関節空間のみで
    世界での位置を持たない＝原理的に不利）
    → **予測できないし、すべきでもない**（それは自分の情報ではない）

＝**同じ入力・同じデータ・同じ分割で、対象を「全部混ぜ」「自己のみ」「外部のみ」に分けて測る。**
変えるのは**対象の切り分け方だけ**。

  自己が高く外部が低い → **混ぜていたのが失敗の原因**。分ければ自己の内部モデルは学べる
  どちらも低い         → 分離は効かない。仮説は棄却

【★反則をしていないか（落とし穴チェック項14・項6）】
二重接触の判定には「このジオメトリがどの体に属するか」という**太郎が持っていない情報**を使う。
これは**対象（正解ラベル）の定義に使うだけ**なので反則ではない（ラベルとはそういうもの）。
**入力には一切使っていない**（入力は固有感覚／真の状態のみ）。

ただし本実験は「**分離できたら学べるか**」しか答えない。「**太郎が触覚ベクトルだけから
二重接触を取り出せるか**」は別問題で、ここでは測っていない（寝ているだけで1202点中589点が
既に立っているので、「2つ立っている」だけでは足りず「どの2つが互いに触れ合っているか」を
見つける必要がある）。＝本実験が陽性でも、それは**前提が成立するかの門**を通っただけ。

使い方: python d_double_touch.py [model_path] [n_dec]
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("C_TOUCH", "1")
os.environ.setdefault("C_SUPINE", "1")
import numpy as np
import torch
import torch.nn as nn
import mujoco

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
from d_touch_classify import auc, fit_classify, BAL_LO, BAL_HI

K = 100
SUB = 10


def contact_by_source(env, bodies, idx):
    """部位ごとに「自己接触(二重)」と「外部接触(片側)」を別々に立てる。

    MIMo公式 `catch.py` に倣い、**実際に力がかかっている接触だけ**を数える
    （力ゼロの接触はMuJoCoでは非活性＝触れていないのと同じ）。
    """
    m, d = env.unwrapped.model, env.unwrapped.data
    T = env.unwrapped.touch
    self_c = np.zeros(len(bodies), dtype=np.float32)
    ext_c = np.zeros(len(bodies), dtype=np.float32)
    f = np.zeros(6, dtype=np.float64)
    for c in range(d.ncon):
        mujoco.mj_contactForce(m, d, c, f)
        if abs(f[0]) < 1e-9:
            continue
        b1 = int(m.geom_bodyid[d.contact[c].geom1])
        b2 = int(m.geom_bodyid[d.contact[c].geom2])
        s1, s2 = T.has_sensors(b1), T.has_sensors(b2)
        if s1 and s2:                       # 両側にセンサ＝二重接触＝自分の体同士
            self_c[idx[b1]] = 1.0
            self_c[idx[b2]] = 1.0
        elif s1:                            # 片側だけ＝外部（床）
            ext_c[idx[b1]] = 1.0
        elif s2:
            ext_c[idx[b2]] = 1.0
    return self_c, ext_c


def collect(mp, n_dec):
    """学習済みの太郎を動かし、(固有感覚, 前庭, 真の状態, 自己接触, 外部接触) を同時刻で集める。

    `d_touch_predictability.collect` と**同じ手続き**（同じシード・同じエピソード数・
    同じ間引き）にして、既存の結果と直接比べられるようにする。
    """
    ck = torch.load(mp, weights_only=False)
    cfg = ck["config"]
    T_dim = cfg["touch_dim"]
    env = HybridEnv(gym.make(_ENV_ID, vision_params=None, touch_params=_touch_params()))
    fusion = MinimalFusion(T_dim); tfusion = MinimalFusion(T_dim).freeze()
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
    for m_ in (brain, emb_proj, cereb):
        for p in m_.parameters():
            p.requires_grad_(False)
    fusion.freeze()

    bodies = sorted(env.unwrapped.touch.sensor_outputs.keys())
    idx = {b: i for i, b in enumerate(bodies)}
    names = [env.unwrapped.model.body(b).name for b in bodies]

    n_ep = 40
    per_ep = max(1, n_dec // n_ep)
    P, V, S, YS, YE, EP = [], [], [], [], [], []
    for ep in range(n_ep):
        obs, _ = env.reset(seed=1000 + ep)
        h = brain.init_motor_hidden(); pa = torch.zeros(n_act)
        for i in range(per_ep):
            sv = fusion.encode(obs); cf = tfusion.encode(obs).detach()
            emb = emb_proj(torch.cat([sv, pa], dim=-1)).unsqueeze(0).unsqueeze(0)
            out, hn = brain.motor_gru(emb, h)
            z, _, _ = brain.pc_latent.infer(h[-1, 0], out[0, -1], cf); z = z.detach()
            pm = torch.tanh(brain.motor_head(z)); w, ca, _ = cereb.gate(z, pm)
            a = torch.clamp((1 - w) * pm + w * ca, -1, 1).detach()
            if ep % 2 == 1:                       # 半分は探索的に＝姿勢の多様性を稼ぐ
                a = torch.clamp(a + torch.randn(n_act) * 0.5, -1, 1)
            ctrl = rescale_action(a, env.action_space)
            te = tr = False
            for k in range(K):
                obs, r, te, tr, info = env.step(ctrl)
                if k % SUB == 0:
                    sc, ec = contact_by_source(env, bodies, idx)
                    P.append(np.asarray(obs["observation"], dtype=np.float32))
                    V.append(np.asarray(obs["vestibular"], dtype=np.float32))
                    S.append(np.concatenate([d.qpos, d.qvel]).astype(np.float32))
                    YS.append(sc); YE.append(ec); EP.append(ep)
                if te or tr:
                    break
            h = hn.detach(); pa = a
            if te or tr:
                break
        if (ep + 1) % 10 == 0:
            print(f"  収集中 {ep+1}/{n_ep}エピソード  標本{len(P)}", flush=True)
    env.close()
    return (np.stack(P), np.stack(V), np.stack(S), np.stack(YS), np.stack(YE),
            np.asarray(EP), names)


_FINGER = ("ffdistal", "mfdistal", "rfdistal", "lfdistal", "thdistal", "ffmiddle", "mfmiddle",
           "rfmiddle", "lfmiddle", "thhub", "ffknuckle", "mfknuckle", "rfknuckle", "lfknuckle",
           "thbase", "lfmetacarpal", "big_toe", "toes")


def is_finger(name):
    """指・つま先の節か（＝ミリ単位の小さな部位）。

    MIMo v2 の触覚は55部位あるが、**うち32個が指の節**（片手16×2）。
    指は接触面がミリ単位なので、姿勢がほんの少し違うだけで接触がON/OFFする＝最も不連続。
    平均を取ると**指が多数派なので平均が指に支配される**＝落とし穴11（希釈）が起きる。
    だから「大きい部位」と「指」を分けて報告する。
    """
    return any(k in name for k in _FINGER)


def report_groups(names, ok_pairs, rate_te, title):
    """AUCを「大きい部位」と「指の節」に分けて報告する（希釈の切り分け）。"""
    big = [(names[j], a) for j, a in ok_pairs if not is_finger(names[j])]
    fin = [(names[j], a) for j, a in ok_pairs if is_finger(names[j])]
    print(f"  【{title}】")
    for lab, g in (("大きい部位（腕・胴・脚・頭）", big), ("指・つま先の節", fin)):
        if g:
            m = float(np.mean([a for _, a in g]))
            print(f"    {lab:24s} {len(g):2d}部位  平均AUC={m:.3f}")
    return (float(np.mean([a for _, a in big])) if big else np.nan,
            float(np.mean([a for _, a in fin])) if fin else np.nan)


def fit_detail(X, Yb, tr_mask, te_mask, hidden=128, steps=4000, depth=1):
    """fit_classify と同じだが、**部位ごとのAUCを返す**＋容量を振れるようにした版。

    容量を振る理由（落とし穴チェック項6）：128ユニットという容量は、初版で
    3606次元の回帰を**丸暗記**されたのを防ぐために絞った値。今回の対象は55次元の二値
    なので、**同じ容量が今度は小さすぎる**可能性がある。容量を上げてAUCが動くなら
    「関数が学習不能」ではなく「ネットが足りなかった」＝測定の欠陥になる。
    """
    mu, sd = X[tr_mask].mean(0), X[tr_mask].std(0) + 1e-6
    Xtr = torch.tensor((X[tr_mask] - mu) / sd, dtype=torch.float32)
    Xte = torch.tensor((X[te_mask] - mu) / sd, dtype=torch.float32)
    Ytr = torch.tensor(Yb[tr_mask], dtype=torch.float32)
    layers = [nn.Linear(X.shape[1], hidden), nn.SiLU(), nn.LayerNorm(hidden)]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden)]
    layers += [nn.Linear(hidden, Yb.shape[1])]
    net = nn.Sequential(*layers)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(steps):
        idx = torch.randperm(len(Xtr))[:256]
        loss = lossf(net(Xtr[idx]), Ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        ste = net(Xte).numpy()
        str_ = net(Xtr).numpy()
    rate_te = Yb[te_mask].mean(0)
    ok = []
    for j in range(Yb.shape[1]):
        a, b = Yb[tr_mask, j], Yb[te_mask, j]
        if (min(a.sum(), (1 - a).sum()) >= 3 and min(b.sum(), (1 - b).sum()) >= 3
                and BAL_LO <= rate_te[j] <= BAL_HI):
            v = auc(b, ste[:, j])
            if not np.isnan(v):
                ok.append((j, v))
    # 学習側のAUCも返す（低ければ学習不足、高くて検証が低ければ過学習＝落とし穴の切り分け）
    rate_tr = Yb[tr_mask].mean(0)
    ok_tr = [auc(Yb[tr_mask, j], str_[:, j]) for j, _ in ok]
    ok_tr = [v for v in ok_tr if not np.isnan(v)]
    return ok, (float(np.mean(ok_tr)) if ok_tr else np.nan), rate_te


def main():
    mp = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _HERE, os.pardir, "models", "supine_touch1_seed1.pt")
    n_dec = int(sys.argv[2]) if len(sys.argv) > 2 else 400

    print("=== ★二重接触：触覚を『自己』と『外部』に分けたら、自己は学べるのか ===")
    print(f"model={os.path.basename(mp)}  {n_dec}判断ぶん収集")
    print("既知（同じ入力・同じデータ・同じ分割、**全部混ぜ**）: 接触の分類 AUC 0.63〜0.69\n")

    P, V, S, YS, YE, EP, names = collect(mp, n_dec)
    te_mask = EP >= (EP.max() * 0.8)
    tr_mask = ~te_mask
    YA = np.maximum(YS, YE).astype(np.float32)     # 全部混ぜ＝従来の測り方の再現

    print(f"\n標本数={len(P)}（学習{int(tr_mask.sum())} / 検証{int(te_mask.sum())}・エピソード単位で分割）")
    for lab, Y in (("自己接触(二重)", YS), ("外部接触(片側=床)", YE), ("全部混ぜ", YA)):
        r = Y.mean(0)
        n_bal = int(((r >= BAL_LO) & (r <= BAL_HI)).sum())
        print(f"  {lab:18s} 平均接触率={float(Y.mean())*100:5.1f}%  "
              f"変動する部位(10-90%)={n_bal:2d}/{Y.shape[1]}")
    print("→ 採点は**接触率10-90%の部位のみ**（常時接触/常時非接触は当てる中身が無い＝希釈の回避）")

    print("\n--- 測定器の健康診断 ---")
    fit_classify(YS, YS, "正例：自己接触→自己接触(恒等)", tr_mask, te_mask)
    rng = np.random.default_rng(0)
    fit_classify(rng.standard_normal((len(YS), 64), dtype=np.float32), YS,
                 "負例：乱数→自己接触", tr_mask, te_mask)

    print("\n--- ★本題：同じ入力・同じデータ・同じ分割で、対象の切り分け方だけを変える ---")
    print("  【予測】自己接触は姿勢から幾何学的に決まる → 高い")
    print("          外部接触(床)は世界次第・固有感覚は世界での位置を持たない → 低い")
    res = {}
    for lab, Y in (("自己接触(二重)", YS), ("外部接触(片側)", YE), ("全部混ぜ(従来)", YA)):
        a_p, _ = fit_classify(P, Y, f"固有感覚 → {lab}", tr_mask, te_mask)
        a_s, _ = fit_classify(S, Y, f"真の状態 → {lab}", tr_mask, te_mask,
                              names=names, verbose=(lab == "自己接触(二重)"))
        res[lab] = (a_p, a_s)
        print()

    print("=== 判定 ===")
    print(f"{'対象':20s} {'固有感覚から':>12s} {'真の状態から':>12s}")
    for lab, (a_p, a_s) in res.items():
        print(f"{lab:20s} {a_p:12.3f} {a_s:12.3f}")
    sp, se = res["自己接触(二重)"][0], res["外部接触(片側)"][0]

    # ★切り分け1（落とし穴11＝希釈）：55部位のうち**32個が指の節**。指は接触面がミリ単位で
    # 最も不連続なので、平均が指に引きずられている可能性がある。大きい部位と分けて見る。
    print("\n--- ★切り分け1：平均は『指』に希釈されていないか（55部位中32個が指の節）---")
    ok_self, _, rate_self = fit_detail(P, YS, tr_mask, te_mask)
    big_s, fin_s = report_groups(names, ok_self, rate_self, "固有感覚 → 自己接触")
    ok_ext, _, rate_ext = fit_detail(P, YE, tr_mask, te_mask)
    report_groups(names, ok_ext, rate_ext, "固有感覚 → 外部接触")

    # ★切り分け2（落とし穴チェック項6＝測定器の健康診断）：容量128は、初版で3606次元の
    # 回帰を丸暗記されたのを防ぐために絞った値。55次元の二値には**小さすぎる**かもしれない。
    # 上げてAUCが動くなら「学習不能」ではなく「ネットが足りなかった」＝測定の欠陥。
    print("\n--- ★切り分け2：ネットの容量が足りているか（自己接触・固有感覚から）---")
    print(f"  {'容量':22s} {'学習AUC':>9s} {'検証AUC':>9s}")
    for hid, dep, st in ((128, 1, 4000), (512, 2, 8000), (1024, 3, 12000)):
        ok_c, tr_auc, _ = fit_detail(P, YS, tr_mask, te_mask, hidden=hid, depth=dep, steps=st)
        m = float(np.mean([a for _, a in ok_c])) if ok_c else np.nan
        print(f"  {f'{hid}ユニット × {dep}層':22s} {tr_auc:9.3f} {m:9.3f}")

    print()
    if sp > 0.80 and sp - se > 0.10:
        print("→ ★**混ぜていたのが失敗の原因だった**。自己接触は姿勢から学べる。")
        print("   外部接触は学べない（そして**学べなくて正しい**＝それは自分の情報ではない）。")
        print("   3日間の『触覚は学べない』は、**性質の違う2つを平均していた**からだった。")
    elif sp - se > 0.10:
        print(f"→ **分離は効いている**（自己{sp:.3f} > 外部{se:.3f}）が、自己接触も0.80に届かない。")
        print("   方向は正しいが、これだけでは内部モデルには足りない。")
    else:
        print(f"→ **分離しても自己接触は学べない**（自己{sp:.3f} vs 外部{se:.3f}）＝仮説は棄却。")
        print("   混ぜていたことが原因ではなかった。別の要因を疑う。")

    print("\n【この実験が言えないこと】")
    print("・二重接触の判定には『どの体か』という**太郎が持たない情報**を使っている。")
    print("  対象(正解ラベル)の定義に使うだけなので反則ではないが、")
    print("  **太郎が触覚ベクトルだけから二重接触を取り出せるか**は別問題で、未測定。")
    print("・床と養育者の手はどちらも『片側だけ』＝**この方法では区別できない**。")


if __name__ == "__main__":
    main()
