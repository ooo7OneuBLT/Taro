"""
18本（触覚の扱い3条件 × 6シード）を集計し、条件間の差がばらつきより大きいかを判定する。

【問い】触覚は「予測するもの」か「予測に使う手がかり」か。
  off    : 触覚なし。予測対象＝固有感覚621
  target : 触覚を入力にも予測対象にもする（従来）。予測対象＝固有感覚621＋触覚3606
  input  : 触覚を入力にだけ入れる。予測対象＝固有感覚621
**off と input は予測対象が同一**なので、両者の差＝「触覚が文脈として役に立つか」そのもの。

【なぜ6シードか】2026-07-15に、同一条件・同一シードで margin 46.2 vs 57.0（10.8ポイント差）が
出た。**ばらつきが条件差(1.5)の7倍**＝n=1の比較は全て無効だった（落とし穴チェック項12）。
省メモリ化（1本2.64GB→0.28GB）でシードを買えるようになったので、6シードで回す。

【target の扱いに注意】targetのCSVのmarginは**予測対象4227次元のうち85%が触覚**なので、
合計値は触覚に薄められる（項11＝同じ日に2回踏んだ罠）。**固有感覚だけを取り出して比べる**
必要があるため、targetは保存モデルから d_supine_split.py 相当の分離測定を別途行う。
（このスクリプトはCSVベースの集計＝off/inputの比較まで。targetは注記付きで併記する。）

使い方: python d_mode_summary.py
"""
import os, glob, csv
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(_HERE, os.pardir, "logs", "mode")
MODES = ["off", "target", "input"]


def load(mode, at_step=None):
    """at_step のチェックポイントの値を返す（Noneなら各ランの最終行）。

    条件によって学習の実時間が大きく違う（触覚ありは触覚なしの約3倍遅い）。
    完了だけを待つと比較が遅れるので、**全条件に共通する最新のチェックポイントで揃えて**
    比べられるようにする。**学習段階の違うものを混ぜて比べない**のが要点（項1＝交絡）。
    """
    rows = {}
    for p in sorted(glob.glob(os.path.join(LOG, mode, "ac_metrics_seed*_*.csv"))):
        seed = int(os.path.basename(p).split("seed")[1].split("_")[0])
        with open(p, encoding="utf-8") as fp:
            r = [x for x in csv.DictReader(fp) if int(x["train_step"]) > 0]
        if not r:
            continue
        if at_step is None:
            pick = r[-1]
        else:
            cand = [x for x in r if int(x["train_step"]) == at_step]
            if not cand:
                continue
            pick = cand[0]
        rows[seed] = {k: float(pick[k]) for k in ("margin", "corr", "persist", "agency", "classify")}
        rows[seed]["step"] = int(pick["train_step"])
    return rows


def common_step():
    """全条件・全シードに共通して存在する最大のtrain_step。"""
    per_mode = []
    for m in MODES:
        steps = []
        for p in glob.glob(os.path.join(LOG, m, "ac_metrics_seed*_*.csv")):
            with open(p, encoding="utf-8") as fp:
                s = {int(x["train_step"]) for x in csv.DictReader(fp) if int(x["train_step"]) > 0}
            if s:
                steps.append(s)
        if not steps:
            return None
        per_mode.append(set.intersection(*steps))
    common = set.intersection(*per_mode)
    return max(common) if common else None


def main():
    step = common_step()
    if step is None:
        print("まだ比較できるチェックポイントが揃っていない。")
        return
    print("=== 触覚の扱い3条件 × 6シード（仰向け）===")
    print(f"**全条件に共通する最新のチェックポイント train_step={step} で揃えて比較**")
    print("（触覚ありは触覚なしの約3倍遅い。学習段階の違うものを混ぜて比べない＝項1）")
    print("off/input は予測対象が同一(固有感覚621)＝直接比較できる。")
    print("target は予測対象の85%が触覚なので、この margin は薄められた見かけの値（要分離測定）。\n")
    data = {m: load(m, step) for m in MODES}
    stats = {}
    for m in MODES:
        d = data[m]
        if not d:
            print(f"[{m:6s}] 完了0本")
            continue
        mg = np.array([v["margin"] for v in d.values()])
        pr = np.array([v["persist"] for v in d.values()])
        ag = np.array([v["agency"] for v in d.values()])
        stats[m] = (mg, pr, ag)
        print(f"[{m:6s}] n={len(mg)}  margin: 平均{mg.mean():+6.2f}  幅{mg.min():+.1f}〜{mg.max():+.1f}  "
              f"標準偏差{mg.std(ddof=1) if len(mg) > 1 else 0:.2f}")
        print(f"{'':9s} persist: 平均{pr.mean():6.1f}  |  agency: 平均{ag.mean():5.1f}%  "
              f"|  seeds={sorted(d.keys())}")

    if "off" in stats and "input" in stats:
        a, b = stats["off"][0], stats["input"][0]
        diff = b.mean() - a.mean()
        # 【修正・2026-07-15】「差 > 条件内のばらつき(pooled SD)」で有意と判定していたが**誤り**。
        # 比べるべきは**差の標準誤差** SE=√(va/na + vb/nb)。pooled SD と比べると甘すぎて、
        # 実際 off vs input の差 −2.70（p=0.098＝有意でない）を「悪くなる」と誤判定した。
        # 分散が違う（1.71 vs 3.07）ので等分散を仮定しない Welch を使う。
        va, vb = a.var(ddof=1), b.var(ddof=1)
        se = np.sqrt(va / len(a) + vb / len(b))
        t = diff / se if se > 0 else 0.0
        df = ((va / len(a) + vb / len(b)) ** 2 /
              ((va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1)))
        print("\n=== off vs input（予測対象が同一なので直接比較できる）===")
        print(f"  off  : {a.mean():+.2f}  (n={len(a)}, 標準偏差 {a.std(ddof=1):.2f})  {np.round(a, 1)}")
        print(f"  input: {b.mean():+.2f}  (n={len(b)}, 標準偏差 {b.std(ddof=1):.2f})  {np.round(b, 1)}")
        print(f"  差   : {diff:+.2f}   差の標準誤差 {se:.2f}   Welchのt = {t:.2f} (自由度 {df:.1f})")
        crit = 2.23   # 自由度≈10 の両側5%臨界値（scipy非依存でも判定できるように）
        if abs(t) < crit:
            print(f"\n→ **差は有意でない**（|t|={abs(t):.2f} < {crit}）。")
            print("   ＝**触覚を文脈として持っても、自己モデルは損なわれない**。")
            print("   触覚を入力に入れたまま、予測対象から外す設計が成立する。")
            need = 2 * (2.8 * np.sqrt((va + vb) / 2) / abs(diff)) ** 2 if diff else float("inf")
            print(f"   （この大きさの差を検出するには1条件あたり約{need:.0f}シード必要）")
        elif diff > 0:
            print(f"\n→ **触覚を文脈として持つと自己モデルが良くなる**（|t|={abs(t):.2f} > {crit}）。")
            print("   ＝触覚は『予測するもの』ではなく『予測に使う手がかり』。設計変更の根拠になる。")
        else:
            print(f"\n→ **触覚を文脈として持つと自己モデルが悪くなる**（|t|={abs(t):.2f} > {crit}）。")
            print("   ＝触覚は入力としても害。Dで触覚を使う設計そのものを問い直す必要がある。")
    print("\n※targetの固有感覚だけの margin は d_supine_split.py で別途測る（合計は薄まるため）。")


if __name__ == "__main__":
    main()
