# -*- coding: utf-8 -*-
"""指標の「暴れ」を既存CSVだけで数える。学習は回さない（タダ）。

【なぜ要るか、2026-07-30】「条件Cは不安定」と書いていたが、既存のCSVを
数え直したら条件の問題ではなく **1本のラン（C_seed0）だけが壊れていた**。
しかも進行性で、d_action2（行動の変化量）が回を追うごとに小さくなり、
最後に margin=6.1 / persist=181.4 と崩壊した＝太郎が固まっていた。
⇒ 落とし穴チェックリスト 項83・項84／研究日誌 2026-07-30（続き5）

見るもの:
  ① 隣り合う測定の差の大きさ（＝暴れ）を条件ごとに比べる
  ② 暴れが時間とともに減るか（減らない＝収束していない）
  ③ 指標が何と連動しているか（d_action2 が小さい＝固まっている）
  ④ 終盤の平均±標準偏差（報告する数値のばらつき）

使い方:
    python run/tools/check_jitter.py                       # 既定 E/logs/selfmodel_v2
    python run/tools/check_jitter.py E/logs/selfmodel_v3   # フォルダを指定
    python run/tools/check_jitter.py E/logs/selfmodel_v2 E/logs/selfmodel_v3  # 並べて比べる
"""
import csv, glob, os, math, sys

sys.stdout.reconfigure(encoding="utf-8")
_R = r"C:\claude\AI\Taro"
_dirs = sys.argv[1:] or [os.path.join(_R, "E", "logs", "selfmodel_v2")]
FILES = []
for _d in _dirs:
    _d = _d if os.path.isabs(_d) else os.path.join(_R, _d)
    FILES += sorted(p for p in glob.glob(os.path.join(_d, "*.csv"))
                    if not p.endswith(".meta.json"))
KEYS = ["classify", "margin", "corr", "persist"]
CTX = ["age_months", "noise", "act_abs", "d_action2", "cereb_err", "life_min"]


def load(p):
    rows = []
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({k: float(v) for k, v in r.items() if v not in ("", "-", None)})
            except ValueError:
                pass
    return rows


def sd(xs):
    if len(xs) < 2: return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def corr(a, b):
    n = min(len(a), len(b))
    if n < 3: return 0.0
    a, b = a[:n], b[:n]
    ma, mb = sum(a) / n, sum(b) / n
    va = math.sqrt(sum((x - ma) ** 2 for x in a))
    vb = math.sqrt(sum((x - mb) ** 2 for x in b))
    if va == 0 or vb == 0: return 0.0
    return sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / (va * vb)


print("=" * 78)
print("① 条件ごとの「暴れ」＝隣り合う測定の差の大きさ（平均絶対変化）")
print("=" * 78)
print(f"{'ファイル':<26}", end="")
for k in KEYS: print(f"{k:>11}", end="")
print(f"{'測定数':>7}")

data = {}
for p in FILES:
    rows = load(p)
    data[p] = rows
    name = os.path.basename(p).replace(".csv", "")
    print(f"{name:<26}", end="")
    for k in KEYS:
        xs = [r[k] for r in rows if k in r]
        d = [abs(xs[i + 1] - xs[i]) for i in range(len(xs) - 1)]
        print(f"{(sum(d)/len(d) if d else 0):>11.2f}", end="")
    print(f"{len(rows):>7}")

print()
print("=" * 78)
print("② 暴れは時間とともに減るか（前半17点 vs 後半17点の平均絶対変化）")
print("=" * 78)
for p in FILES:
    rows = data[p]
    name = os.path.basename(p).replace(".csv", "")
    print(f"{name:<26}", end="")
    for k in KEYS:
        xs = [r[k] for r in rows if k in r]
        h = len(xs) // 2
        d1 = [abs(xs[i + 1] - xs[i]) for i in range(h - 1)]
        d2 = [abs(xs[i + 1] - xs[i]) for i in range(h, len(xs) - 1)]
        a = sum(d1) / len(d1) if d1 else 0
        b = sum(d2) / len(d2) if d2 else 0
        print(f"  {k}:{a:>6.1f}→{b:>6.1f}", end="")
    print()

print()
print("=" * 78)
print("③ 暴れが何と連動しているか（指標と文脈変数の相関）")
print("=" * 78)
for p in FILES:
    rows = data[p]
    name = os.path.basename(p).replace(".csv", "")
    print(f"\n--- {name}")
    for k in ("margin", "persist"):
        xs = [r[k] for r in rows if k in r]
        line = f"  {k:<9}"
        for c in CTX:
            ys = [r[c] for r in rows if k in r and c in r]
            line += f" {c}={corr(xs, ys):+.2f}"
        print(line)

print()
print("=" * 78)
print("④ 終盤8点の平均と標準偏差（＝報告した数値のばらつき）")
print("=" * 78)
print(f"{'ファイル':<26}", end="")
for k in KEYS: print(f"{k+'(平均±sd)':>20}", end="")
print()
for p in FILES:
    rows = data[p]
    name = os.path.basename(p).replace(".csv", "")
    print(f"{name:<26}", end="")
    for k in KEYS:
        xs = [r[k] for r in rows if k in r][-8:]
        m = sum(xs) / len(xs) if xs else 0
        print(f"{m:>13.1f}±{sd(xs):<6.1f}", end="")
    print()

print()
print("=" * 78)
print("⑤ 外れ値：終盤で平均から2sd以上ずれた測定はどこか")
print("=" * 78)
for p in FILES:
    rows = data[p]
    name = os.path.basename(p).replace(".csv", "")
    for k in KEYS:
        xs = [(r["step"], r[k]) for r in rows if k in r and "step" in r]
        vals = [v for _, v in xs]
        m = sum(vals) / len(vals); s = sd(vals)
        out = [(int(st), v) for st, v in xs if s > 0 and abs(v - m) > 2 * s]
        if out:
            print(f"{name:<26} {k:<9} 平均{m:6.1f} sd{s:5.1f} → 外れ {out}")
