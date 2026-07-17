"""
★アナログ検証：触れる瞬間の不連続を避ければ、力は予測できるのか。

【背景・2026-07-16】
MIMoの触覚は最初から**アナログ**（力の大きさ・ニュートン連続値）。実測で確認済み：
胸を押す深さを 0.18→0.08 と下げると触覚合計が 0→1.78→3.82→…→19.21 と11段階でなめらかに増える。
＝**0/1にしていたのは `d_touch_classify.py` で私が二値化していただけ**で、生信号は連続。

【ユーザーの仮説】
人間の触覚予測は「一瞬こう動いたら一瞬こう触れる」ではなく「**押し続けたら感じ続ける**
（強さは一定でないがなめらかに続く）」。＝不連続なのは**触れる瞬間（境界）だけ**で、
押し続けている間の力はなめらか＝予測できるはず。3日間の失敗（R²=−54%）は、
**全瞬間を混ぜて**測り、境界の不連続を食らっていたからではないか。

【測り方＝サンプルの選び方だけを変える】
同じ入力(真の状態/固有感覚)・同じデータ・同じ予測対象(アナログの力)で、対象にする瞬間だけ変える：
  ①全部の瞬間（従来＝−54%の再現）
  ②**その部位が触れ続けている瞬間だけ**（境界＝0→非0の飛びを除く）
部位ごとに「触れている瞬間」を取り出し、その中で力を回帰してR²を測り、①と比べる。

  ②が①より大きく跳ねる → 仮説が正しい（境界だけが問題・押し続けの感覚は予測可能）
  ②も低いまま         → 境界だけの問題ではない（押し続けの力も姿勢から読めない
                        ＝外から押される力＝自分の情報でないから、が濃厚）

【落とし穴の回避】
・②は「触れている瞬間だけ」＝力は必ず非0。だが**値は連続的に散らばっている**ので、
  「常に平均を答える」だけでは当たらない（R²の基準＝その区間内の力の分散）。二値化の罠とは別。
・部位ごとに標本数が違う。触れる瞬間が少なすぎる部位（<30）は除外して平均を水増ししない。
・入力に「1つ前の力」は入れない（それを入れると"さっきと同じ"で当たる別の罠＝持続予測）。
  ここで測るのは純粋に「姿勢→今の力」。時間文脈版は別途（本文の注記参照）。

使い方: python d_touch_analog.py [model_path] [n_dec]
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
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "taro_core"))
import paths
paths.setup_brain_path()
sys.path.insert(0, paths.MIMO_DIR)

import d_touch_predictability as dtp

THRESH = 1e-6
MIN_CONTACT = 30   # 触れている瞬間がこれ未満の部位は、標本不足として採点から除く


def fit_r2(X, y, tr, te, steps=3000):
    """Xから連続値yを回帰し、検証R²を返す（1部位ぶん・1次元出力）。

    R²＝『学習データの平均をいつも答える』と比べてどれだけ二乗誤差を減らせたか。
    0%=平均以上のことは何もできていない、100%=完璧。負=平均より悪い。
    """
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xtr = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32)
    Xte = torch.tensor((X[te] - mu) / sd, dtype=torch.float32)
    ym, ys = y[tr].mean(), y[tr].std() + 1e-6
    ytr = torch.tensor((y[tr] - ym) / ys, dtype=torch.float32).unsqueeze(1)
    net = nn.Sequential(nn.Linear(X.shape[1], 128), nn.SiLU(),
                        nn.LayerNorm(128), nn.Linear(128, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    n = len(Xtr)
    for _ in range(steps):
        idx = torch.randperm(n)[:256]
        loss = ((net(Xtr[idx]) - ytr[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = net(Xte).squeeze(1).numpy() * ys + ym
    base = ((y[te] - y[tr].mean()) ** 2).mean()
    err = ((pred - y[te]) ** 2).mean()
    return 1 - err / max(base, 1e-12)


def main():
    mp = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _HERE, os.pardir, "models", "supine_touch1_seed1.pt")
    n_dec = int(sys.argv[2]) if len(sys.argv) > 2 else 400

    print("=== ★アナログ検証：触れ続けている間の力は予測できるか ===")
    print(f"model={os.path.basename(mp)}  {n_dec}判断ぶん収集")
    print("既知（全瞬間・アナログの力の回帰）: R²=−54%（境界の不連続込み）\n")

    P, V, S, Y, TH, EP, names = dtp.collect(mp, n_dec)
    te_mask = EP >= (EP.max() * 0.8)
    tr_mask = ~te_mask
    inC = Y > THRESH   # 各(時刻,部位)で触れているか

    print(f"標本数={len(P)}（学習{int(tr_mask.sum())} / 検証{int(te_mask.sum())}・エピソード単位で分割）")
    print("入力＝真の状態(qpos+qvel 225次元)。予測＝各部位のアナログな力。\n")

    print(f"{'部位':22s} {'接触率':>6s} {'①全瞬間R²':>10s} {'②触れてる間だけR²':>16s}")
    rows = []
    for j in range(Y.shape[1]):
        # ②の標本：その部位が触れている瞬間（学習・検証それぞれ）
        tr_c = tr_mask & inC[:, j]
        te_c = te_mask & inC[:, j]
        if tr_c.sum() < MIN_CONTACT or te_c.sum() < MIN_CONTACT:
            continue
        r2_all = fit_r2(S, Y[:, j], tr_mask, te_mask)                 # ①全瞬間
        r2_con = fit_r2(S, Y[:, j], np.where(tr_c)[0], np.where(te_c)[0])  # ②触れてる間だけ
        rows.append((names[j], float(inC[:, j].mean()), r2_all, r2_con))

    rows.sort(key=lambda r: -r[3])
    for nm, rate, a, c in rows:
        print(f"{nm:22s} {rate*100:5.0f}% {a*100:9.1f}% {c*100:15.1f}%")

    allm = np.mean([r[2] for r in rows])
    conm = np.mean([r[3] for r in rows])
    print(f"\n{'平均':22s} {'':6s} {allm*100:9.1f}% {conm*100:15.1f}%")

    print("\n=== 判定 ===")
    print(f"①全瞬間（境界込み）        平均R² {allm*100:+.1f}%")
    print(f"②触れ続けている間だけ      平均R² {conm*100:+.1f}%")
    if conm > 0.3 and conm - allm > 0.3:
        print("\n→ ★**跳ねた**。触れる瞬間の不連続だけが問題で、押し続けている間の力は予測できる。")
        print("   ＝ユーザーの仮説が実証。触覚の内部モデルは『瞬間』でなく『押し続け』で作るべき。")
        print("   3日間の『触覚は学べない』は、境界を含めて測っていたための結論だった。")
    elif conm - allm > 0.15:
        print(f"\n→ 境界を除くと改善する（{allm*100:.0f}%→{conm*100:.0f}%）が、決定的ではない。")
        print("   境界は一因だが、押し続けの力にも姿勢から読めない成分が残る。")
    else:
        print(f"\n→ **境界を除いても跳ねない**（{allm*100:.0f}%→{conm*100:.0f}%）。")
        print("   ＝不連続だけの問題ではなかった。押し続けている間の力も姿勢から読めない")
        print("   ＝その力は『外から押される量』で、自分の姿勢からは決まらない（自分の情報でない）。")

    print("\n【この実験が測っていないこと】")
    print("・入力に『1つ前の力』を入れる時間文脈版は別（それは『さっきと同じ』で当たる持続予測の")
    print("  罠を切り分ける必要がある）。ここでは純粋に『姿勢→今の力』のみを測った。")


if __name__ == "__main__":
    main()
