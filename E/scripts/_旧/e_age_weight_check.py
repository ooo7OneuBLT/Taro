"""月齢ごとの体重が人間に合っているかだけを、汚染なしで測る。

【なぜ専用にしたか、2026-07-28】`mimoGrowth/growth.py:167` はグローバル辞書
`SCHEMA_V2` を破壊的に書き換えるため、**同じプロセスで環境を作り直すと体型補正が累積**する
（`infant_body._restore_growth_schema` の説明を参照）。1回の実行で「補正あり」と「補正なし」を
交互に測ると、後から作った方が汚れる。実際 `e_age_muscle_check.py` の初版では
素のage=0が11.892kg（人間3.3kgの3.6倍）というありえない値が出た。

⇒ ここでは **1プロセス1条件**にして、体重だけを測る。呼び出し側でループする。

使い方:
    E_AGE=4 E_SHAPE_ON=1 .venv/Scripts/python.exe E/scripts/e_age_weight_check.py
"""

# 注意：古い方式（2026-07-30 に整理）。新しい実験は `run/main.py` を通す。
#   【経緯】目標Eの実験スクリプトが118本あり、うち66本が**独立に環境を組み立てていた**。
#     そのため「学習は関節モード（90関節を独立に駆動＝逸脱リスト 逸脱5）、
#     測定とViewerは筋肉モード（拮抗筋2本/関節）」という**別の体で動く**事故が起きた
#     （ユーザーの目視「視線誘導反射の実験の時とは動きが全然違う」で発覚。
#      実測で動きが人間の新生児の約3.3倍速かった）。
#   【設計と移行計画】`E/docs/実行基盤_設計.md`
#   注意：このファイルは**記録として残す**（削除しない方針）。
#     中の測り方は再利用できるので、プラグインへ移すときの元にする。
import os
import sys
import warnings
import contextlib
import io

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
for _p in [os.path.join(_ROOT, "D", "scripts"), os.path.join(_ROOT, "MIMo"),
           os.path.join(_ROOT, "taro_core"),
           os.path.join(_ROOT, "taro_core", "src", "body"), _HERE]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main():
    from d_supine_env import SupineMimoEnv
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env

    age = float(os.environ.get("E_AGE", "0"))
    shape_on = os.environ.get("E_SHAPE_ON", "1") == "1"
    corrections = os.environ.get("E_CORR", "0") == "1"

    kw = body_kwargs_from_env(age, verbose=False) if shape_on else {}
    kw["body_corrections"] = corrections
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        env = SupineMimoEnv(vision_params=None, age=age,
                            actuation_model=MuscleModel, **kw)
        env.reset(seed=0)
    m = env.unwrapped.model
    skip = ("floor", "wall", "fence", "object", "toy", "target", "world")
    ids = [i for i in range(1, m.nbody)
           if not any(k in m.body(i).name for k in skip)]
    total = float(sum(m.body_mass[i] for i in ids))
    head = float(m.body_mass[int(m.body("head").id)])

    # 身長の代わりに頭頂〜足先の z 方向の広がり（仰向けなので x 方向）を測る
    import numpy as np
    d = env.unwrapped.data
    pts = [d.xipos[i] for i in ids]
    span = float(np.max([p[0] for p in pts]) - np.min([p[0] for p in pts]))

    print(f"AGE={age:g} SHAPE={'on' if shape_on else 'off'} "
          f"CORR={'on' if corrections else 'off'} "
          f"MASS={total:.3f} HEAD={head:.3f} SPAN={span*100:.1f}")
    env.close()


if __name__ == "__main__":
    main()
