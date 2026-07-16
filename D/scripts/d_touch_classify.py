"""
★決定実験：触覚は「力の回帰」ではなく「接触の分類」なら学べるのか。

【なぜこれを測るか・2026-07-16】
これまで触覚の予測対象は**一貫して「力ベクトルの回帰」**だった（3606次元→部位ごと55次元）。
その結果は一貫して失敗：
    同じ入力(真の状態225次元)・同じデータ・同じ分割・同じ出力次元(55)で対象だけ変えると
        固有感覚（学べると分かっている） 検証R² **+26.5%**
        温度                             検証R² **−53.1%**
        接触力                           検証R² **−54.4%**

【文献調査が示したこと（2026-07-16・deep-research・25主張を3票で敵対的検証）】
人間側で**実際に測定されてきた**触覚予測の形式は、**一貫して離散（接触イベント／部位）**であり、
力の大きさ・方向を連続予測した一次研究は検証済み文献に**一件も無い**：
  ・Shen et al. 2021/22 (Infancy 27:97-114) … 全文検索で force/newton/pressure/intensity/
      load cell/regression の出現が**すべて0回**。唯一の刺激変数は「どちらの手か」の二値
  ・Shen et al. 2018 (Int J Psychophysiol 134:144-150, 乳児sMMNの初実証, n=31)
      … 逸脱を**身体部位カテゴリ**で定義し、強度は一定(60psi)に固定
  ・Myowa-Yamakoshi & Takeshita 2006 … 測定変数は「口の開きが接触に先行するか否か」の二値

【★この実験の限界を先に書く（過大解釈を防ぐため）】
上の文献は「**分類形式には人間側の裏づけがあるが、力回帰には無い**」という**証拠の非対称**で
あって、「力回帰が誤り」の積極的な反証ではない（文献調査レポート自身の警告）。よって本実験は
**人間の発達からの演繹ではなく、工学的な仮説検証**として扱う。

さらにレポートは別の釘も刺している：「**他者が触覚の3.9%しかない**という信号量の問題は予測形式
とは独立であり、**形式変更だけで解決するとは言えない**」。＝本実験が陽性でも、それは
「形式」の問題が解けただけで、「信号量」の問題（床が82.5%）は別途残る。

【設計＝対象だけを差し替える】
`d_touch_predictability.collect()` をそのまま流用する（同じ入力・同じデータ・同じエピソード
単位の分割）。**変えるのは予測対象の形式だけ**：
    従来: 部位ごとの力の合計（連続値）を回帰   → R²
    今回: 部位ごとに「触れているか否か」(二値) を分類 → AUC

【★指標にAUCを使う理由（落とし穴11＝希釈、の回避）】
正答率は使えない。**寝ているだけで太郎の1202センサ点のうち589点(49%)が既に反応している**
（床＋自己接触）ので、「常に触れている」と答えるだけで高い正答率が出てしまう部位が多数ある
＝**暗い部屋問題の分類版**。AUCは基準率に依存しない（0.5=でたらめ、1.0=完璧）。
さらに**学習側・検証側の両方で2クラスが揃う部位だけ**を採点する（常時接触/常時非接触の部位は
予測する中身が無く、混ぜると平均を水増しするだけ）。

【測定器の健康診断（落とし穴チェック項6・項13）】
  正例1: 触覚そのもの→接触   … 恒等写像。AUC≈1.0 でなければ手続きが壊れている
  負例 : 乱数→接触           … AUC≈0.5 でなければカンニング経路がある
  ★正例2（最も効く）: **同じ測定器で、床との接触だけを対象にする**。床接触は姿勢から
      幾何学的にほぼ自明（体が床に着いていれば触れる）なので、**この課題で学べると
      分かっている対象**にあたる。ここが低ければ測定器ごと無効。

使い方: python d_touch_classify.py [model_path] [n_dec]
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

torch.set_num_threads(4)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "C", "scripts"))
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "C"))
import paths
paths.setup_brain_path()
sys.path.insert(0, paths.MIMO_DIR)

import d_touch_predictability as dtp

THRESH = 1e-6      # これを超える力がかかっていれば「触れている」
MIN_MINORITY = 3   # 学習・検証それぞれで、少数派クラスがこの数以上ある部位だけ採点
# 【★2026-07-16・初版の欠陥】MIN_MINORITY=3 だけでは**フィルタが甘すぎる**。
# 800標本中797回触れている部位は「少数派3」で条件を通るが、実質**常時接触**であり
# 予測する中身が無い（＝落とし穴11の希釈が、分類版でそのまま再発する）。
# そこで**接触率が偏りすぎている部位を除いた平均**を併記する。
BAL_LO, BAL_HI = 0.10, 0.90   # 接触率がこの範囲＝本当に変動している部位


def auc(y_true, score):
    """ROC-AUC（順位法）。0.5=でたらめ、1.0=完璧。基準率に依存しない。

    正例と負例のペアを全部見て「正例のほうが高いスコアだった割合」＝順位和から計算する。
    """
    y_true = np.asarray(y_true).astype(bool)
    n_pos, n_neg = int(y_true.sum()), int((~y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    # 同点は平均順位に（同点だらけだとAUCが不当に上下するため）
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return (ranks[y_true].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def fit_classify(X, Yb, name, tr_mask, te_mask, steps=4000, names=None, verbose=False):
    """Xから同時刻の「接触の有無」(部位ごとの二値)を分類できるかを測る。

    回帰版(`d_touch_predictability.fit_and_score`)と**同じ容量・同じ最適化・同じ分割**にして、
    変わるのが「対象の形式」だけになるようにする（損失だけMSE→BCEに変える）。
    """
    mu, sd = X[tr_mask].mean(0), X[tr_mask].std(0) + 1e-6
    Xtr = torch.tensor((X[tr_mask] - mu) / sd, dtype=torch.float32)
    Xte = torch.tensor((X[te_mask] - mu) / sd, dtype=torch.float32)
    Ytr = torch.tensor(Yb[tr_mask], dtype=torch.float32)
    ntr = len(Xtr)

    # 学習側・検証側の**両方で2クラスが揃う部位だけ**を採点対象にする。
    # 常時接触（床に着いている腰・背中など）は予測する中身が無く、混ぜると平均を水増しする。
    ok = []
    for j in range(Yb.shape[1]):
        a, b = Yb[tr_mask, j], Yb[te_mask, j]
        if (min(a.sum(), (1 - a).sum()) >= MIN_MINORITY
                and min(b.sum(), (1 - b).sum()) >= MIN_MINORITY):
            ok.append(j)
    if not ok:
        print(f"[{name:30s}] 採点できる部位が無い（全部位が常時接触または常時非接触）")
        return np.nan, []

    net = nn.Sequential(nn.Linear(X.shape[1], 128), nn.SiLU(),
                        nn.LayerNorm(128), nn.Linear(128, Yb.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(steps):
        idx = torch.randperm(ntr)[:256]
        loss = lossf(net(Xtr[idx]), Ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        ste = net(Xte).numpy()

    pairs = [(j, auc(Yb[te_mask, j], ste[:, j])) for j in ok]
    pairs = [(j, a) for j, a in pairs if not np.isnan(a)]
    aucs = [a for _, a in pairs]
    m = float(np.mean(aucs)) if aucs else np.nan
    # ★偏りすぎた部位を除いた平均（希釈の回避）。接触率が0.9超/0.1未満の部位は
    # 「ほぼ常に触れている／触れていない」＝当てる中身が無いので、平均を水増しする。
    rate_te = Yb[te_mask].mean(0)
    bal = [(j, a) for j, a in pairs if BAL_LO <= rate_te[j] <= BAL_HI]
    mb = float(np.mean([a for _, a in bal])) if bal else np.nan
    print(f"[{name:30s}] 入力{X.shape[1]:5d}次元 | 全{len(aucs):3d}部位 AUC={m:.3f} "
          f"| **変動する{len(bal):2d}部位のみ AUC={mb:.3f}**")
    if verbose and names:
        per = sorted(bal if bal else pairs, key=lambda x: -x[1])
        for j, a in per[:6]:
            print(f"       {names[j]:24s} AUC={a:.3f}  (接触率{rate_te[j]*100:.0f}%)")
        if len(per) > 6:
            j, a = per[-1]
            print(f"       ... 最下位 {names[j]:18s} AUC={a:.3f}  (接触率{rate_te[j]*100:.0f}%)")
    return mb, aucs


def proprio_layout():
    """固有感覚621次元の内訳（どの区間が何か）を実測で返す。

    MIMoの `SimpleProprioception.get_proprioception_obs` は
    `np.concatenate([... for key in sorted(keys)])` で連結する＝**辞書キーのアルファベット順**：
        actuation（筋の活性）, limits（可動限界）, qpos（関節角度）, qvel（関節速度）, torques（トルク）
    ここが重要な理由：**固有感覚には torque（関節にかかる力）と actuation（筋の活性）が入っている**
    ＝接触したときの「手ごたえ」を既に含んでいる。人間で言えば腱器官・筋紡錘にあたる。
    """
    import gymnasium as gym
    from hybrid_env import HybridEnv
    from run_c_metrics_ac_lr import _ENV_ID, _touch_params
    env = HybridEnv(gym.make(_ENV_ID, vision_params=None, touch_params=_touch_params()))
    env.reset(seed=0)
    so = env.unwrapped.proprioception.sensor_outputs
    out, off = {}, 0
    for k in sorted(so.keys()):
        n = int(np.asarray(so[k]).flatten().shape[0])
        out[k] = (off, off + n)
        off += n
    env.close()
    return out, off


def main():
    mp = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _HERE, os.pardir, "models", "supine_touch1_seed1.pt")
    n_dec = int(sys.argv[2]) if len(sys.argv) > 2 else 400

    print("=== ★決定実験：触覚は『力の回帰』ではなく『接触の分類』なら学べるのか ===")
    print(f"model={os.path.basename(mp)}  {n_dec}判断ぶん収集")
    print("既知（同じ入力・同じデータ・同じ分割・回帰）: 固有感覚 R2=+26.5% / 接触力 R2=-54.4%\n")

    P, V, S, Y, TH, EP, names = dtp.collect(mp, n_dec)
    te_mask = EP >= (EP.max() * 0.8)
    tr_mask = ~te_mask

    Yb = (Y > THRESH).astype(np.float32)     # ★対象を二値化（これだけが変更点）
    rate = Yb.mean(0)
    varying = [j for j in range(Yb.shape[1])
               if min(Yb[tr_mask, j].sum(), (1 - Yb[tr_mask, j]).sum()) >= MIN_MINORITY
               and min(Yb[te_mask, j].sum(), (1 - Yb[te_mask, j]).sum()) >= MIN_MINORITY]

    print(f"\n標本数={len(P)}（学習{int(tr_mask.sum())} / 検証{int(te_mask.sum())}・エピソード単位で分割）")
    print(f"部位数={Yb.shape[1]}  2クラス揃う={len(varying)}")
    print("接触率の分布（＝どれだけ「変動」しているか。0%や100%に貼り付いた部位は当てる中身が無い）：")
    for lo, hi, lab in ((0.99, 1.01, "  99%超（ほぼ常時接触）"), (0.90, 0.99, "  90-99%"),
                        (0.10, 0.90, "**10-90%（本当に変動）**"), (0.01, 0.10, "  1-10%"),
                        (-0.01, 0.01, "  1%未満（ほぼ常時非接触）")):
        n = int(((rate >= lo) & (rate < hi)).sum())
        print(f"    {lab:26s} {n:3d}部位")
    print("→ **10-90%の部位だけの平均**を主指標にする（落とし穴11＝希釈の回避）")

    print("\n--- 測定器の健康診断 ---")
    fit_classify(Y, Yb, "正例1：触覚→接触(恒等)", tr_mask, te_mask)
    rng = np.random.default_rng(0)
    fit_classify(rng.standard_normal((len(Yb), 64), dtype=np.float32), Yb,
                 "負例：乱数→接触", tr_mask, te_mask)

    print("\n--- 本題：接触の有無を、体の状態から当てられるか ---")
    a1, _ = fit_classify(P, Yb, "①固有感覚のみ→接触", tr_mask, te_mask)
    a2, _ = fit_classify(np.concatenate([P, V], 1), Yb, "②固有感覚＋前庭→接触", tr_mask, te_mask)
    a3, l3 = fit_classify(S, Yb, "③真の状態(オラクル)→接触", tr_mask, te_mask,
                          names=names, verbose=True)

    # ★①②が③（物理の真の状態＝オラクル）を上回ったら、それは**異常**である。
    # 接触は (qpos,qvel) から決定論的に決まるので、その関数である固有感覚が上回るはずがない。
    # 唯一ありうる説明＝**固有感覚が「答え」を直接含んでいる**：torque(関節にかかる力)と
    # actuation(筋の活性)は、接触力そのものの読み出しであり、「幾何から接触力を計算する」という
    # 難しい関数を**迂回**できる。＝トルクを抜けばオラクル並みに落ちるはず。これを実測する。
    if a1 > a3 + 0.02 or a2 > a3 + 0.02:
        lay, tot = proprio_layout()
        print(f"\n--- ★異常の切り分け：固有感覚がオラクルを上回った ---")
        print(f"接触は(qpos,qvel)から決定論的に決まる＝その下流である固有感覚が上回るのは異常。")
        print(f"仮説：固有感覚は torque(力)・actuation(筋活性) を含む＝**接触の答えを直接持っている**。")
        print(f"固有感覚{tot}次元の内訳（実測）: " +
              " / ".join(f"{k}[{a}:{b}]={b-a}次元" for k, (a, b) in lay.items()))
        if "torques" in lay and P.shape[1] == tot:
            ta, tb = lay["torques"]
            keep = np.r_[0:ta, tb:tot]
            fit_classify(P[:, keep], Yb, "④固有感覚 −トルク →接触", tr_mask, te_mask)
            fit_classify(P[:, ta:tb], Yb, "⑤トルクのみ →接触", tr_mask, te_mask)
            if "qpos" in lay:
                qa, qb = lay["qpos"]
                fit_classify(P[:, qa:qb], Yb, "⑥関節角度のみ →接触", tr_mask, te_mask)
        else:
            print(f"  （固有感覚の次元{P.shape[1]}が内訳の合計{tot}と食い違うので切り分け不能）")

    print("\n=== 判定 ===")
    print("同じ入力(真の状態)・同じデータ・同じ分割で、**予測対象の形式だけ**を変えた：")
    print(f"    力の回帰   （既知） 検証R2  -54.4%  ＝『平均を答える』より悪い")
    print(f"    接触の分類 （今回） 平均AUC {a3:.3f}   ＝0.5がでたらめ")
    if np.isnan(a3):
        print("\n→ 採点できる部位が無い＝この設定では判定不能。")
    elif a3 > 0.75:
        print("\n→ ★**分類なら学べる**。3日間の『触覚は学べない"
              "』は**測り方（形式）の問題だった**。")
        print("   ただし文献調査の警告どおり、これは形式の問題が解けただけで、")
        print("   **信号量の問題（床が82.5%・他者は3.9%）は別に残る**。")
    elif a3 > 0.6:
        print("\n→ 中間。分類のほうが回帰よりましだが決定的ではない。部位ごとの内訳を見る。")
    else:
        print("\n→ **分類でも学べない**＝形式の問題ではなかった。仮説は棄却。")
        print("   ＝接触の発生そのものが姿勢からカオス的、という既存の結論が強まる。")

    print("\n【この実験が言えないこと】")
    print("・人間側の文献は『分類形式に裏づけがある』だけで『力回帰が誤り』とは言っていない")
    print("  （証拠の非対称であり反証ではない）。よってこれは工学的な仮説検証である。")
    print("・他者が触覚の3.9%しかないという信号量の問題は、形式とは独立で未解決のまま。")


if __name__ == "__main__":
    main()
