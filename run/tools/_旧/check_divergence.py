"""同じ設定を2回回して「どこで最初に食い違うか」を突き止める。

【なぜ要るか、2026-07-30】同じ設定・同じ乱数の種で学習を2回回すと結果が違う
（落とし穴チェックリスト 項79）。「結果が違う」しか分かっていないので推測で原因を
探すことになり、実際に3回外した（PYTHONHASHSEED／BLASのスレッド数／env.reset）。
⇒ **推測をやめて実測する**：毎ステップの内部の値の指紋を2回ぶん取り、
  最初に食い違ったステップと、そのステップで最初に食い違った量を出す。

食い違った量から原因の性質が決まる：
    obs_in.interoception → 内臓（泣く・寝る・うとうと）
    obs_in.eye_*         → 視覚のレンダリング
    obs_in.observation   → 物理（関節の状態）
    sv                   → 感覚をまとめる層（fusion）
    z                    → 脳の内部（予測符号化の乱数）
    mean / std           → 行動を作るところ
    a                    → 探索のゆらぎ
    W だけ違う          → 学習の計算（浮動小数の足し算の順序）

【使い方】
    .venv/Scripts/python.exe -m run.tools.check_divergence
    .venv/Scripts/python.exe -m run.tools.check_divergence --steps 300 --probe

  --probe を付けると自己モデルの測定（e_probes.evaluate）も入れる。
  測定を入れると一致しなくなる、というのがここまでに分かっていること。
"""
import argparse
import csv
import os
import subprocess
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

SCENE = "新生児_仰向け_柵あり"


def _spec(steps, ckpt, probe, out):
    plugins = {"trace": {"out": out}}
    if probe:
        plugins["self_model"] = True
    return {"name": "食い違いの追跡", "scene": SCENE,
            "taro": {"actuation": "muscle", "age_months": 0.0},
            "run": {"type": "train", "steps": steps, "seed": 0, "checkpoint": ckpt},
            "plugins": plugins}


def _read(path):
    with open(path, encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--ckpt", type=int, default=50)
    ap.add_argument("--probe", action="store_true",
                    help="自己モデルの測定も入れる（これを入れると一致しなくなる）")
    # 2回では足りない。ずれは**稀にしか起きない**（実測で3回中1回）。
    #   n=2 の一致を根拠に「直った」と判断して外したことが今日2回ある（項79）。
    ap.add_argument("--runs", type=int, default=5, help="何回回して比べるか（3以上）")
    a = ap.parse_args()

    import json
    tmp = os.path.join(_ROOT, "E", "logs", "trace")
    os.makedirs(tmp, exist_ok=True)
    paths = []
    for k in range(1, max(2, a.runs) + 1):
        out = f"E/logs/trace/trace_{k}.csv"
        sp = os.path.join(tmp, f"spec_{k}.json")
        with open(sp, "w", encoding="utf-8") as fp:
            json.dump(_spec(a.steps, a.ckpt, a.probe, out), fp, ensure_ascii=False,
                      indent=2)
        print(f"--- {k} 回目を実行（{a.steps}回学習 / "
              f"測定{'あり' if a.probe else 'なし'}）", flush=True)
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        r = subprocess.run([os.path.join(_ROOT, ".venv", "Scripts", "python.exe"),
                            "-u", "-m", "run.main", sp],
                           cwd=_ROOT, env=env, capture_output=True, text=True,
                           errors="replace")
        if r.returncode != 0:
            print("注意実行が失敗した")
            print(r.stdout[-2000:])
            print(r.stderr[-2000:])
            return 1
        paths.append(os.path.join(_ROOT, out.replace("/", os.sep)))

    runs = [_read(p) for p in paths]
    print("\n" + "=" * 78)
    print(f" 食い違いの追跡（{len(runs)}回 × {len(runs[0])} ステップ / "
          f"測定{'あり' if a.probe else 'なし'}）")
    print("=" * 78)
    if len({len(r) for r in runs}) != 1:
        print(f"注意行数が違う {[len(r) for r in runs]}＝比べられない")
        return 1
    if len(runs[0]) == 0:
        print("注意記録が空＝比べられない（trace プラグインが動いていない）")
        return 1

    cols = [c for c in runs[0][0] if c != "step"]
    # 1本目を基準に、他のどれかが食い違った最初のステップを探す
    A = runs[0]
    first_step, first_cols, which = None, [], None
    for idx in range(len(A)):
        for j, B in enumerate(runs[1:], start=2):
            diff = [c for c in cols if A[idx].get(c) != B[idx].get(c)]
            if diff:
                first_step, first_cols, which = A[idx]["step"], diff, j
                break
        if first_step is not None:
            break

    if first_step is None:
        print(f"  {len(runs)}回すべて、{len(A)} ステップ全部が完全一致")
        print("    ⇒ この条件では再現している")
        print("    注意ずれは稀にしか起きないので、回数を増やしてまだ探す価値がある")
        return 0

    print(f"  最初に食い違ったステップ : {first_step} / {len(A)}"
          f"（1本目 vs {which}本目）")
    print(f"  そのステップで食い違った量:")
    for c in first_cols:
        print(f"      {c:22s} {A[int(first_step)-1].get(c)} vs "
              f"{runs[which-1][int(first_step)-1].get(c)}")
    print()
    print("  一致していた量（同じステップ）:")
    ra, rb = A[int(first_step) - 1], runs[which - 1][int(first_step) - 1]
    ok = [c for c in cols if ra.get(c) == rb.get(c)]
    print("      " + ("  ".join(ok) if ok else "（なし）"))

    # 「どの量が最初に壊れたか」で原因の場所を言い当てる
    print("\n  " + "-" * 74)
    hints = [
        ("obs_in.interoception", "内臓（泣く・寝る・うとうと）。内受容感覚が違う"),
        ("obs_in.eye_left", "視覚のレンダリング"),
        ("obs_in.eye_right", "視覚のレンダリング"),
        ("obs_in.observation", "物理（関節の状態）または前のステップの行動"),
        ("obs_in.vestibular", "前庭感覚（体の傾き）"),
        ("sv", "感覚をまとめる層（fusion）"),
        ("z", "脳の内部（予測符号化の乱数など）"),
        ("mean", "行動を作るところ（運動野＋小脳）"),
        ("std", "ノルアドレナリン（探索の強さ）"),
        ("a", "探索のゆらぎ（explore）"),
        ("W", "学習の計算だけが違う＝浮動小数の足し算の順序"),
    ]
    said = False
    for key, msg in hints:
        if key in first_cols:
            print(f"  ⇒ {msg}")
            said = True
            break
    if not said:
        print(f"  ⇒ 上の表に無い量から壊れている: {first_cols}")
    print(f"\n  詳細は {os.path.relpath(paths[0], _ROOT)} と "
          f"{os.path.relpath(paths[1], _ROOT)} を比べる")
    return 0


if __name__ == "__main__":
    sys.exit(main())
