# -*- coding: utf-8 -*-
"""★触覚センサーが「体を育てても同じ部位に属し続けるか」を確かめる。

【なぜ要るか、2026-07-31】太郎は触覚を諦めていた。理由は
「体を育てるとセンサー点が 0ヶ月1734点 → 4ヶ月4274点 に変わり、
  観測の次元がずれる」（落とし穴チェックリスト 項75）。
`run/config.py` も触覚ONで体を育てようとすると**エラーで止める**作りになっている。

⇒ ★Baby Sophia（arXiv:2511.09727）は触覚17,175次元を**解剖学的68部位に分けて平均**
  するだけで扱えるようにしている。オートエンコーダも次元圧縮も使っていない。
  ★部位の数は体が育っても変わらないはず。それを確かめる。

★確かめること
  ① 月齢を変えても「センサーを持つ部位（body）の集合」が同じか
  ② 部位ごとの点数は変わってよい（平均を取るので次元は部位の数で固定される）
  ③ 部位ごとに平均すると、月齢によらず同じ次元になるか

使い方:
    .venv/Scripts/python.exe run/tools/check_touch_growth.py
    .venv/Scripts/python.exe run/tools/check_touch_growth.py 0 2 4
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
_R = r"C:\claude\AI\Taro"
for p in ("E/scripts", "D/scripts", "taro_core/src/body", "taro_core/src/brain",
          "taro_core/src/senses", "taro_core/src/wrapper", "MIMo", ""):
    sys.path.insert(0, os.path.join(_R, p))
os.chdir(_R)

import numpy as np              # noqa: E402
import e_scene                  # noqa: E402

ages = [float(a) for a in sys.argv[1:]] or [0.0, 1.0, 2.0, 3.0, 4.0]

print("=" * 78)
print(" 触覚センサーは体を育てても同じ部位に属し続けるか")
print("=" * 78)

result = {}
for age in ages:
    sc = e_scene.load("新生児_仰向け_柵なし")
    sc["body"]["age_months"] = age
    sc["fingerprint"] = None
    env, _ = e_scene.build(sc, seed=0, verbose=False)
    u = env.unwrapped

    touch = getattr(u, "touch", None)
    if touch is None:
        print(f"⚠️月齢{age}: 触覚が有効になっていない。"
              "シーンかenvの設定で触覚をONにする必要がある")
        env.close()
        continue

    m = u.model
    # センサー点は geom ごとに持たれる。その geom がどの body に属するかを見る
    per_body = {}
    total = 0
    for geom_id, pts in touch.sensor_positions.items():
        n = int(pts.shape[0])
        total += n
        bid = int(m.geom_bodyid[geom_id])
        bname = m.body(bid).name
        per_body[bname] = per_body.get(bname, 0) + n
    result[age] = per_body
    print(f"\n--- 月齢 {age:.1f} ヶ月")
    print(f"  センサー点の合計 {total:,} 点")
    print(f"  ★センサーを持つ部位（body）の数 {len(per_body)}")
    env.close()

if len(result) >= 2:
    keys = sorted(result)
    base = set(result[keys[0]])
    print("\n" + "=" * 78)
    print(" ★★判定")
    print("=" * 78)
    same = all(set(result[a]) == base for a in keys)
    print(f"  部位の集合が全月齢で同じか  ★{'はい' if same else 'いいえ'}")
    for a in keys:
        s = set(result[a])
        print(f"    月齢{a:.1f}: {len(s)}部位  "
              f"点数{sum(result[a].values()):,}"
              + ("" if s == base else f"  ⚠️差分 {sorted(s ^ base)}"))
    if same:
        print(f"\n  ⇒ ★部位ごとに平均すれば、月齢によらず**{len(base)}次元**で固定できる")
        print("     ＝ 触覚を「体が育つ実験」で使えるようになる")
    else:
        print("\n  ⇒ ⚠️部位の集合が月齢で変わる。★別の方法が要る")

    # 部位ごとの点数の変化（上位10部位）
    print("\n" + "-" * 78)
    print(" 部位ごとの点数（★数は変わってよい。平均を取るので次元は部位数で固定）")
    print("-" * 78)
    top = sorted(base, key=lambda b: -result[keys[-1]].get(b, 0))[:10]
    print(f"{'部位':<24}" + "".join(f"{a:>10.1f}ヶ月" for a in keys))
    for b in top:
        print(f"{b:<24}" + "".join(f"{result[a].get(b, 0):>15,}" for a in keys))
