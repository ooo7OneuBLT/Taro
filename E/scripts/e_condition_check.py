"""条件を振った実験で「条件が本当に効いていたか」をモデルの重みから検証する。

【なぜ必要か・2026-07-25】★同じシードで条件だけを変えて学習したのに、
**モデルの重みが完全一致していた**（＝条件が届いていなかった）という事故が起きた。

    distal（末端質量）  x1 = x2 = x6.7 が同一 ／ x3.4 だけ違う
    limb（四肢の筋力）  x0.25 = x0.50 = x2.00 = nofix が同一 ／ x1.00 だけ違う

原因：`e_growth_train.py` が身体設定の統一した入口（`body_kwargs_from_env`）を使わず、
個別に関数を呼んでいたため、あとから足された引数が学習ループに届いていなかった。
＝★**感度分析2件（四肢の筋力32倍／末端質量6.7倍）が丸ごと無効**になった。

【この検証が強い理由】
指標（margin・うつ伏せ%）の差を見るより先に、**そもそも別のモデルになっているか**を見る。
  同一 → 条件が届いていない。実験は無意味（指標を見る意味すらない）
  相違 → 条件は効いている。効果の大小はその先の話
★これは検証の落とし穴チェックリスト 項17（設定した値が効いているか）の最も強い形で、
「入力を揺らして出力が動くか」をモデルの重みで直接確認している。

使い方:
    # 2条件を比べる
    python E/scripts/e_condition_check.py limb_x0.25_seed0.pt limb_x2.00_seed0.pt

    # 接頭辞でまとめて総当たり（seed0 同士）
    python E/scripts/e_condition_check.py --sweep limb_x0.25 limb_x0.50 limb_x1.00 limb_x2.00
"""
import itertools
import os
import sys

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

import torch  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS = os.path.join(_HERE, os.pardir, "models", "growth_curriculum")


def load_brain(name):
    path = name if os.path.isabs(name) else os.path.join(_MODELS, name)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return blob["brain"], blob.get("config", {})


def max_diff(a, b):
    shared = [k for k in a if k in b]
    if not shared:
        return float("nan"), 0
    return max(float((a[k].float() - b[k].float()).abs().max()) for k in shared), len(shared)


def compare(n1, n2):
    a, ca = load_brain(n1)
    b, cb = load_brain(n2)
    d, n = max_diff(a, b)
    same = d < 1e-9
    verdict = "IDENTICAL -> condition had NO effect" if same else "different -> condition OK"
    print(f"  {n1:<28} vs {n2:<28} maxdiff={d:.3e} ({n} tensors)  {verdict}")
    if same:
        # 設定が同じなら当然一致するので、config の差も見せる（切り分けの助けになる）
        keys = sorted(set(ca) | set(cb))
        diffs = [f"{k}: {ca.get(k)} vs {cb.get(k)}" for k in keys if ca.get(k) != cb.get(k)]
        print(f"      config diff: {diffs if diffs else 'none (settings identical too)'}")
    return same


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1
    if args[0] == "--sweep":
        names = [a if a.endswith(".pt") else a + "_seed0.pt" for a in args[1:]]
        print("Condition check (pairwise). IDENTICAL means the condition never reached "
              "the training loop.\n")
        bad = 0
        for x, y in itertools.combinations(names, 2):
            try:
                if compare(x, y):
                    bad += 1
            except FileNotFoundError as e:
                print(f"  [missing] {e.filename}")
        print(f"\n  {bad} identical pair(s) found.", end=" ")
        print("-> ★that sweep is INVALID" if bad else "-> all conditions took effect")
        return 1 if bad else 0
    if len(args) != 2:
        print("give two model names, or --sweep with a list")
        return 1
    print()
    return 1 if compare(*args) else 0


if __name__ == "__main__":
    sys.exit(main())
