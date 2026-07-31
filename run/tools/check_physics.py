"""物理（MuJoCo）を「激しく動かしたとき」に決定的かを調べる。脳を通さない。

【なぜ要るか、2026-07-30】同じ設定・同じ乱数の種で学習を2回回すと結果が違う
（落とし穴チェックリスト 項79）。ここまでの切り分け：

    物理だけ（力を入れない）        → 一致
    学習だけ（測る道具を外す）        → 一致（4回すべて）
    学習＋自己モデルの測定           → step 1 で既に食い違う
       食い違うのは obs（関節・前庭・触覚・視覚）で、
       脳と感覚層の重みは一致。内受容（内臓）も一致

＝**脳は同じことをしているのに、体の状態だけが違う**。
測定（`e_probes.evaluate`）は学習と違って**ゆらぎ無しの決定的な行動**を使うので、
体が同じ方向へ押し続ける＝関節が可動域の端に当たり、接触が持続する。
⇒ そういう状況で MuJoCo が決定的かどうかを、**脳を通さずに**確かめる。

【使い方】2回実行して要約の行を比べる
    .venv/Scripts/python.exe -m run.tools.check_physics
    .venv/Scripts/python.exe -m run.tools.check_physics --mode hold --level 0.8

  --mode random  決まった乱数列の行動を流す（学習に近い）
  --mode hold    同じ行動を流し続ける（測定に近い。可動域の端に当たる）

注意：脳を一切通さないので、ここで食い違えば原因は**物理側**。
  一致すれば原因は**脳側（測定の中の計算）**。
"""
import argparse
import hashlib
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import warnings                                     # noqa: E402
warnings.filterwarnings("ignore")
import numpy as np                                  # noqa: E402
import torch                                        # noqa: E402

SCENE = "新生児_仰向け_柵あり"


def _h(a):
    return hashlib.sha1(np.ascontiguousarray(a, dtype=np.float64).tobytes()
                        ).hexdigest()[:10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("hold", "random"), default="hold")
    ap.add_argument("--level", type=float, default=0.8,
                    help="筋の活性化（0〜1）。大きいほど激しく動く")
    ap.add_argument("--ticks", type=int, default=1200, help="物理ステップ数")
    a = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    import random
    random.seed(0)
    from run.plugins.common import scene as scene_mod
    env, sc, hands = scene_mod.build(SCENE, taro={"actuation": "muscle",
                                                  "age_months": 0.0},
                                     seed=0, verbose=False, hybrid=True)
    u = env.unwrapped
    d = u.model, u.data
    m, d = u.model, u.data
    n = env.action_space.shape[0]
    obs, _ = env.reset(seed=0)

    # 行動の作り方。random は torch の乱数から作る＝2回の実行で必ず同じ列になる
    if a.mode == "random":
        acts = torch.rand(a.ticks // 10 + 1, n).numpy().astype(np.float32) * a.level
    else:
        acts = np.tile(np.full(n, a.level, dtype=np.float32),
                       (a.ticks // 10 + 1, 1))

    print("=" * 78)
    print(f" 物理は決定的か（mode={a.mode} level={a.level} ticks={a.ticks}）")
    print("=" * 78)
    marks = []
    ncon_max = 0
    for k in range(a.ticks):
        act = acts[k // 10]
        obs, _r, te, tr, _i = env.step(act)
        ncon_max = max(ncon_max, int(d.ncon))
        if (k + 1) % 200 == 0:
            marks.append((k + 1, _h(d.qpos), _h(d.qvel), int(d.ncon)))
            print(f"  {k+1:>5} tick  qpos={marks[-1][1]}  qvel={marks[-1][2]}"
                  f"  接触={marks[-1][3]}")
        if te or tr:
            obs, _ = env.reset()

    print("-" * 78)
    print(f"  接触の最大数 {ncon_max}（多いほど接触ソルバが働いている）")
    print(f"\n  この実行の要約（2回実行して**この行だけ**比べる）")
    print(f"    {_h([int(x[1], 16) for x in marks] + [int(x[2], 16) for x in marks])}")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
