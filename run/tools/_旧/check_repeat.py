"""同じプロセスの中で学習を2回して、結果が一致するかを調べる。

【なぜ要るか、2026-07-30】同じ設定・同じ乱数の種で学習を2回回すと結果が一致しない
（落とし穴チェックリスト 項79）。ここまでに分かったこと：
    物理だけ                       → 一致
    学習だけ（測る道具を外す）      → 一致
    学習＋自己モデルの測定         → 一致しない
    env.reset() を6回繰り返す      → 一致（`run/tools/check_reset.py`）
    PYTHONHASHSEED / BLASのスレッド数を固定 → 一致しない

**次に切り分けたいこと**：一致しないのは「プロセスを分けたから」か、
「同じプロセスの中でも起きる」のか。
    同じプロセスで一致しない → 乱数や内部状態の**撒き直し漏れ**の可能性が高い
                              （＝1回目の学習が2回目に影響している）
    同じプロセスなら一致する → プロセス起動時に決まる何か（メモリ配置など）

【使い方】
    .venv/Scripts/python.exe -m run.tools.check_repeat
    .venv/Scripts/python.exe -m run.tools.check_repeat --steps 200

注意：1回あたり数分かかる（既定100回学習で約1分）。
"""
import argparse
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import warnings                                     # noqa: E402
warnings.filterwarnings("ignore")

SCENE = "新生児_仰向け_柵あり"


def one(steps, ckpt, n_eval, use_probe):
    """学習を1本回して、チェックポイントごとの数値を返す。"""
    from run.config import Config
    from run import trainer
    from run.plugins.common.self_model import SelfModel
    cfg = Config(taro={"actuation": "muscle", "age_months": 0.0},
                 run={"steps": steps, "seed": 0, "checkpoint": ckpt,
                      "n_eval": n_eval},
                 scene=SCENE)
    rows = []
    plugins = [SelfModel({})] if use_probe else []
    trainer.train(cfg, plugins=plugins, verbose=False,
                  log_row=lambda r: rows.append(dict(r)))
    return rows


def fmt(rows):
    return [f"{r['step']}:cl={r['classify']:.4f} mg={r['margin']:.4f} "
            f"pr={r['persist']:.4f}" for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--ckpt", type=int, default=50)
    ap.add_argument("--n-eval", type=int, default=80)
    a = ap.parse_args()

    print("=" * 78)
    print(f" 同じプロセスの中で学習を2回（{a.steps}回学習 / 測定 n_eval={a.n_eval}）")
    print("=" * 78)
    out = []
    for k in (1, 2):
        print(f"\n--- {k} 回目 ---", flush=True)
        out.append(fmt(one(a.steps, a.ckpt, a.n_eval, True)))
        for line in out[-1]:
            print("   ", line)

    print("\n" + "-" * 78)
    if out[0] == out[1]:
        print("  同じプロセスの中では**一致した**")
        print("    ⇒ 一致しない原因は「プロセスを分けたこと」に関わる")
        print("      （プロセス起動時に決まる何か。メモリ配置・ライブラリの初期化など）")
    else:
        print("  注意同じプロセスの中でも**一致しない**")
        print("    ⇒ 1回目の学習が2回目に影響している疑い＝**撒き直し漏れ**")
        print("      （どこかのグローバルな状態が持ち越されている）")
        for i, (x, y) in enumerate(zip(out[0], out[1])):
            if x != y:
                print(f"      {i}: {x}")
                print(f"         {y}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
