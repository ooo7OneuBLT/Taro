# -*- coding: utf-8 -*-
"""2つの条件の学習ログ(CSV)を並べて比べる。

【なぜ要るか、2026-07-31】条件を変えて回したあと、「良くなったのか悪くなったのか」を
目分量で言わないため。最終行1点だけを見ると学習の揺れに騙されるので、
**後半の平均**と**シード間の幅**を並べて出す。

注意：ここが出すのは数字だけ。**数字で良く見えても行動が退化していることがある**
  （落とし穴：太郎の指標は「動かない」を高く評価しがち）。必ず Viewer で目視すること。

使い方:
    .venv/Scripts/python.exe run/tools/compare_runs.py 触覚なし=E/logs/a/*.csv 触覚あり=E/logs/b/*.csv
"""
import glob
import sys

sys.stdout.reconfigure(encoding="utf-8")

# 見る列と、どちらが良いか（+1＝大きいほど良い / -1＝小さいほど良い / 0＝良し悪しでない）
COLS = [
    ("classify", +1, "自分の体だと分かる率"),
    ("margin", +1, "自分と他人の見分けの余裕"),
    ("corr", +1, "予測と実際のズレの相関"),
    ("persist", -1, "動かなさ（100未満が正常）"),
    ("d_action2", 0, "動きの激しさ（0.5未満で固まり）"),
    ("act_abs", 0, "筋の出力の大きさ"),
    ("hand_in_view", +1, "手が視界に入った回数"),
    ("cereb_err", -1, "小脳の誤差"),
]


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        head = f.readline().strip().split(",")
        for line in f:
            v = line.strip().split(",")
            if len(v) != len(head):
                continue
            # 空欄は「その回は測っていない」なので、その列だけ落とす（行は使う）
            row = {}
            for k, x in zip(head, v):
                try:
                    row[k] = float(x)
                except ValueError:
                    pass
            rows.append(row)
    return rows


def summarize(paths, tail_frac=0.5):
    """各CSVの**後半**の平均を取り、シードごとの値を返す。"""
    per_seed = []
    for p in sorted(paths):
        rows = load(p)
        if not rows:
            continue
        cut = int(len(rows) * (1 - tail_frac))
        tail = rows[cut:]
        s = {}
        for c, _, _ in COLS:
            vals = [r[c] for r in tail if c in r]
            s[c] = sum(vals) / len(vals) if vals else float("nan")
        per_seed.append(s)
    return per_seed


groups = {}
for arg in sys.argv[1:]:
    name, _, pat = arg.partition("=")
    files = glob.glob(pat)
    if not files:
        print(f"注意 {name}: {pat} に合うファイルが無い")
        continue
    groups[name] = (files, summarize(files))

if len(groups) < 2:
    print("2つ以上の条件を指定する")
    sys.exit(1)

names = list(groups)
print("=" * 78)
print(" 条件の比較（各ランの**後半半分**の平均。シードごとに出す）")
print("=" * 78)
for n in names:
    files, per = groups[n]
    print(f"  {n}: {len(files)}ラン")
    for f in sorted(files):
        print(f"      {f}")

print()
hdr = f"{'指標':<26}" + "".join(f"{n:>22}" for n in names)
print(hdr)
print("-" * len(hdr))
for col, better, jp in COLS:
    line = f"{jp:<26}"
    means = {}
    for n in names:
        vals = [s[col] for s in groups[n][1]]
        m = sum(vals) / len(vals)
        means[n] = m
        spread = max(vals) - min(vals) if len(vals) > 1 else 0.0
        line += f"{m:>13.3f} (幅{spread:>5.2f})"
    print(line)
    if better != 0 and len(names) == 2:
        a, b = names
        d = means[b] - means[a]
        good = (d > 0) if better > 0 else (d < 0)
        base = abs(means[a]) if abs(means[a]) > 1e-9 else 1.0
        print(f"{'':<26}  → {b} は {a} より {d:+.3f}"
              f"（{d / base * 100:+.1f}%）{'良い' if good else '悪い'}")

print()
print("注意：シード間の幅より小さい差は、条件の差とは言えない")
print("注意：数字が良くても行動が退化していることがある。必ず Viewer で目視する")
