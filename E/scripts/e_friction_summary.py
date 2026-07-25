"""床のころがり摩擦スイープ（学習済みモデル）の集計。

【なぜ・2026-07-25】太郎は新生児にできない寝返りをする。原因の候補として
「床のころがり摩擦が事実上ゼロ（friction[1]=0.005・condim=3＝すべりしか計算しない）」
を検証した。学習なし（e_friction_probe.py）では本物の振幅だとそもそも寝返らないと
分かったので、**学習済みモデル**で振り直したのがこのデータ。

判定は型B（→ doc/壁にぶつかったときの型.md）＝「効かないことを示す」。
効果量と**単調性**を見る。振り幅に対して単調でないならノイズ。

⚠️出力は必ずASCIIにする（cp932の端末で日本語が化ける＝チェックリスト項35）。
"""
import glob
import os
import re
import sys

import numpy as np

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_DIR = os.path.join(_HERE, os.pardir, "logs", "E", "frictionsweep")

# (ラベル, ファイル接頭辞, ころがり摩擦の値)
CONDS = [
    ("roll=0.005 (current)", "post_r0005", 0.005),
    ("roll=0.05           ", "post_r005", 0.05),
    ("roll=0.5            ", "post_r05", 0.5),
    ("roll=2.0            ", "post_r20", 2.0),
]

# 日本語に依存しない拾い方（化けても壊れないように数値の並びで特定する）
# ⚠️「(最大180.0)」は固定値ではなく**そのランでの実測最大値**（例: 169.3）。
#   ここを 180.0 決め打ちで書いて「データが欠けている」と誤判定した（2026-07-25）。
#   体幹の回転=90.7度(最大169.3)  うつ伏せ時間=64.8%
RE_ROT = re.compile(r"=([\d.]+)\D{1,4}\(\D{1,6}([\d.]+)\)\D{1,20}=([\d.]+)%")
RE_TRANS = re.compile(r"(\d+)\D{1,4}\(.*?(\d+)\D{1,4}/.*?(\d+)\D{1,4}\)")  # 計N回 (A回 / B回)
RE_JERK = re.compile(r"jerk=([\d.]+)")


def parse(path):
    t = open(path, encoding="utf-8", errors="replace").read()
    out = {}
    m = RE_ROT.search(t)
    if m:
        out["rot"] = float(m.group(1))
        out["rot_max"] = float(m.group(2))
        out["prone"] = float(m.group(3))
    m = RE_JERK.search(t)
    if m:
        out["jerk"] = float(m.group(1))
    # 遷移は「姿勢の切り替わり」の行だけを見る（->を含む行）
    for line in t.splitlines():
        if "->" in line and "tick=" in line:
            m = RE_TRANS.search(line)
            if m:
                out["trans"] = int(m.group(1))
                out["to_prone"] = int(m.group(2))
                out["to_supine"] = int(m.group(3))
            break
    return out


def main():
    print("Floor rolling-friction sweep on TRAINED models (K=10, 6000 steps)")
    print(f"{'condition':<22}{'n':>3}{'prone%':>9}{'trunkRot':>10}"
          f"{'trans':>8}{'toProne':>9}{'toSupine':>10}")
    print("-" * 72)
    table = {}
    for label, prefix, val in CONDS:
        files = sorted(glob.glob(os.path.join(_DIR, prefix + "_s*.txt")))
        rows = [parse(f) for f in files]
        rows = [r for r in rows if "prone" in r]
        if not rows:
            print(f"{label:<22}  (no data)")
            continue
        g = lambda k: np.array([r.get(k, np.nan) for r in rows], dtype=float)
        table[label] = dict(val=val, prone=g("prone"), rot=g("rot"),
                            rot_max=g("rot_max"), trans=g("trans"),
                            to_supine=g("to_supine"), to_prone=g("to_prone"))
        print(f"{label:<22}{len(rows):>3}{np.nanmean(g('prone')):>9.1f}"
              f"{np.nanmean(g('rot')):>10.1f}{np.nanmean(g('trans')):>8.1f}"
              f"{np.nanmean(g('to_prone')):>9.1f}{np.nanmean(g('to_supine')):>10.1f}")

    base_label = CONDS[0][0]
    if base_label not in table:
        return
    b = table[base_label]
    print("\n=== effect size vs current (diff / pooled SD) ===")
    print("  |d| < 0.2 = no difference.  ** check MONOTONICITY across the 3 rows **")
    for key, nm in [("prone", "prone%"), ("rot", "trunkRot"),
                    ("rot_max", "maxRot"), ("to_supine", "toSupine")]:
        print(f"\n  [{nm}]  current mean = {np.nanmean(b[key]):.2f}")
        ds = []
        for label, _, _ in CONDS[1:]:
            if label not in table:
                continue
            v = table[label][key]
            sd = ((np.nanstd(v) ** 2 + np.nanstd(b[key]) ** 2) / 2) ** 0.5
            d = (np.nanmean(v) - np.nanmean(b[key])) / sd if sd > 1e-9 else 0.0
            ds.append(d)
            j = ("none" if abs(d) < 0.2 else "small" if abs(d) < 0.5
                 else "medium" if abs(d) < 0.8 else "LARGE")
            print(f"    {label} mean={np.nanmean(v):7.2f}  "
                  f"diff={np.nanmean(v)-np.nanmean(b[key]):+7.2f}  d={d:+5.2f} ({j})")
        if len(ds) == 3:
            mono = (ds[0] <= ds[1] <= ds[2]) or (ds[0] >= ds[1] >= ds[2])
            print(f"    -> monotonic: {'YES' if mono else 'NO (= noise)'}"
                  f"   [{ds[0]:+.2f} -> {ds[1]:+.2f} -> {ds[2]:+.2f}]")


if __name__ == "__main__":
    main()
