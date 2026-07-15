"""target条件の全シードについて、固有感覚と触覚を分けて測る（合計は触覚85%に薄められるため）。

【なぜ必要か】2026-07-15に**同じ日に2回**踏んだ罠（落とし穴チェック項11）：
  D0     : 合計 persist 86% を見て「自己モデル未確立」と誤報告 → 分けると固有感覚 +39.8 で確立
  仰向け  : 合計 margin +15〜24 を見て「明確に劣化」と誤報告 → 分けると固有感覚 +32〜46 で同等
targetのCSVの margin は予測対象4227次元のうち3606(85%)が触覚なので、**そのまま off/input と
比べてはいけない**。固有感覚だけを取り出して初めて比較できる。

使い方: python d_mode_split_all.py
"""
import os, glob, subprocess, sys, re

_HERE = os.path.dirname(os.path.abspath(__file__))
models = sorted(glob.glob(os.path.join(_HERE, os.pardir, "models", "mode", "target_seed*.pt")))
if not models:
    print("targetのモデルがまだ無い（学習が完了していない）")
    sys.exit(0)
print(f"=== target {len(models)}本の分離測定（固有感覚 vs 触覚）===\n")
prop_margins = []
for mp in models:
    seed = re.search(r"seed(\d+)", os.path.basename(mp)).group(1)
    r = subprocess.run([sys.executable, os.path.join(_HERE, "d_supine_split.py"), mp, "50"],
                       capture_output=True, text=True, encoding="utf-8",
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    for line in (r.stdout or "").splitlines():
        if "固有感覚" in line and "persist=" in line:
            m = re.search(r"persist=\s*([\d.]+)%\s+margin=\s*([+\-\d.]+)%", line)
            if m:
                prop_margins.append(float(m.group(2)))
                print(f"seed{seed}: 固有感覚 persist={m.group(1)}%  margin={m.group(2)}%")
        elif "触覚" in line and "persist=" in line and "固有" not in line:
            m = re.search(r"persist=\s*([\d.]+)%\s+margin=\s*([+\-\d.]+)%", line)
            if m:
                print(f"{'':7s} 触覚     persist={m.group(1)}%  margin={m.group(2)}%")
if prop_margins:
    import numpy as np
    a = np.array(prop_margins)
    print(f"\ntarget の固有感覚 margin: 平均{a.mean():+.2f}  幅{a.min():+.1f}〜{a.max():+.1f}  "
          f"標準偏差{a.std(ddof=1) if len(a) > 1 else 0:.2f}  (n={len(a)})")
    print("→ この値を d_mode_summary.py の off / input と比べる（これで初めて3条件が同じ土俵）")
