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


def load(mode):
    rows = {}
    for p in sorted(glob.glob(os.path.join(LOG, mode, "ac_metrics_seed*_*.csv"))):
        seed = int(os.path.basename(p).split("seed")[1].split("_")[0])
        with open(p, encoding="utf-8") as fp:
            r = list(csv.DictReader(fp))
        if not r:
            continue
        last = r[-1]
        if int(last["train_step"]) < 3600:      # 未完了は除く（途中の値を混ぜない）
            continue
        rows[seed] = {k: float(last[k]) for k in ("margin", "corr", "persist", "agency", "classify")}
    return rows


def main():
    print("=== 触覚の扱い3条件 × 6シード（仰向け・3600step）===")
    print("off/input は予測対象が同一(固有感覚621)＝直接比較できる。")
    print("target は予測対象の85%が触覚なので、この margin は薄められた見かけの値（要分離測定）。\n")
    data = {m: load(m) for m in MODES}
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
        # ばらつきと差を比べる。差がばらつきに埋もれていれば「差がある」とは言わない（項12）。
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) if len(a) > 1 and len(b) > 1 else float("nan")
        print(f"\n=== off vs input（予測対象が同一なので直接比較できる）===")
        print(f"  off  : {a.mean():+.2f}  (n={len(a)}, 標準偏差 {a.std(ddof=1):.2f})")
        print(f"  input: {b.mean():+.2f}  (n={len(b)}, 標準偏差 {b.std(ddof=1):.2f})")
        print(f"  差   : {diff:+.2f}   条件内のばらつき(pooled SD): {pooled:.2f}")
        if abs(diff) < pooled:
            print(f"\n→ **差はばらつきに埋もれている**（差{abs(diff):.2f} < ばらつき{pooled:.2f}）。")
            print("   ＝触覚を文脈として持っても、自己モデルは良くも悪くもならない。")
            print("   触覚は『予測するもの』でも『役に立つ手がかり』でもない、が今のところの読み。")
        elif diff > 0:
            print(f"\n→ **触覚を文脈として持つと自己モデルが良くなる**（+{diff:.2f} > ばらつき{pooled:.2f}）。")
            print("   ＝触覚は『予測するもの』ではなく『予測に使う手がかり』。設計変更の根拠になる。")
        else:
            print(f"\n→ **触覚を文脈として持つと自己モデルが悪くなる**（{diff:.2f}）。")
            print("   ＝触覚は入力としても害。Dで触覚を使う設計そのものを問い直す必要がある。")
    print("\n※targetの固有感覚だけの margin は d_supine_split.py で別途測る（合計は薄まるため）。")


if __name__ == "__main__":
    main()
