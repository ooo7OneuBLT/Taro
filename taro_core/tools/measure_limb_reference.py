"""四肢の筋力補正の「基準」を測る／保存値と照合する。

【この道具が要る理由、2026-07-29】四肢の筋力補正は
「18ヶ月児と同じ相対強度（筋力÷自重を支えるのに要る力）まで下げる」という作りで、
その基準を **体を作るたびに裏で18ヶ月の体を1体作って測って**いた。ところが
それが呼ばれるのは太郎の体を作っている最中で、新生児の体型補正（手0.70倍など）が
既に効いた状態だった。補正が裏の18ヶ月にも漏れ、基準が2.1倍ずれていた：

    何も作らずに測る               median = 64.00   ← ★正しい
    age=0.0 を作る途中で測る       median = 135.93  ← 汚染

その結果、**0〜3ヶ月の太郎は79個中31個しか筋力が下がっていなかった**。
＝この補正が直そうとした「新生児のほうが相対的に強い」という逆転が、
新生児でだけ直っていなかった（落とし穴チェックリスト 項76）。

→ 基準は測り直さず、`taro_core/src/body/limb_reference_18mo.json` に保存した値を使う。
   このスクリプトだけが、**他の体を1体も作っていない状態で**測って保存する。

⚠️体の定義（体型・質量・関節・アクチュエータ）を変えたら必ず走らせて照合すること。

使い方:
    # 保存値と実測を比べるだけ（変更しない）
    .venv/Scripts/python.exe taro_core/tools/measure_limb_reference.py

    # 測り直して保存する
    .venv/Scripts/python.exe taro_core/tools/measure_limb_reference.py --update
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
for _p in ("taro_core/src/body", "taro_core/src/wrapper", "taro_core/src/senses",
           "D/scripts", "E/scripts"):
    sys.path.insert(0, os.path.join(_ROOT, *_p.split("/")))

import numpy as np      # noqa: E402
import mimoEnv         # noqa: E402,F401  （環境の登録に要る）

TOL = 0.01              # 相対差の許容（1%）。これを超えたら「体の定義が変わった」


def main():
    update = "--update" in sys.argv
    # ★import はここまでで、体はまだ1体も作っていない。この順序が生命線。
    import infant_limbs as il

    print("=" * 70)
    print(f" 四肢の筋力補正の基準を測る（age={il.REFERENCE_AGE}ヶ月の素の体）")
    print("=" * 70)
    ratios = il.measure_reference_ratios(il.REFERENCE_AGE)
    vals = np.array(list(ratios.values()), dtype=float)
    print(f"  アクチュエータ {len(ratios)} 個  "
          f"median={np.median(vals):.2f}  min={vals.min():.2f}  max={vals.max():.2f}")

    path = il._REF_PATH
    old = None
    if os.path.exists(path):
        old = json.load(open(path, encoding="utf-8"))

    if old is not None:
        o = old["ratios"]
        miss = [k for k in ratios if k not in o] + [k for k in o if k not in ratios]
        diffs = [(k, float(o[k]), ratios[k],
                  abs(ratios[k] - float(o[k])) / max(abs(float(o[k])), 1e-9))
                 for k in ratios if k in o]
        bad = [d for d in diffs if d[3] > TOL]
        print(f"\n  保存値: median={np.median([float(v) for v in o.values()]):.2f} "
              f"n={len(o)}（{old.get('measured_note', '')}）")
        if miss:
            print(f"  ⚠️アクチュエータの顔ぶれが違う: {miss[:6]}{'...' if len(miss) > 6 else ''}")
        if bad:
            print(f"  ⚠️★{len(bad)} 個が {TOL*100:.0f}% を超えてずれている（体の定義が変わった？）")
            for k, a, b, r in sorted(bad, key=lambda d: -d[3])[:8]:
                print(f"      {k:<28} 保存={a:8.2f} → 実測={b:8.2f}  ({r*100:+.1f}%)")
        if not miss and not bad:
            print("  ✓ 保存値と実測は一致（体の定義は変わっていない）")

    if update:
        blob = {"age": float(il.REFERENCE_AGE),
                "ratios": {k: float(v) for k, v in ratios.items()},
                "measured_note": "他の体を1体も作らずに測定（汚染なし）",
                "median": float(np.median(vals)),
                "how": "taro_core/tools/measure_limb_reference.py --update"}
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(blob, fp, ensure_ascii=False, indent=1, sort_keys=True)
        print(f"\n  → 保存した: {os.path.relpath(path, _ROOT)}")
        print("  ⚠️基準が変わると**太郎の体が変わる**。既存の学習結果と混ぜないこと。")
    elif old is None:
        print("\n  ⚠️保存値がまだない。--update を付けて作ること")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
