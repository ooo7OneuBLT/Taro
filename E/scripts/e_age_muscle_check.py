"""★MIMoは月齢で「筋肉の力」を変えているのか？ を直接確かめる。

【疑い、2026-07-28】`e_age_body_audit.py` の実測で、首の筋力が
    0ヶ月 31.390 / 2ヶ月 31.390 / 3ヶ月 31.390 / 4ヶ月 31.390 / 6ヶ月 31.390
と**全月齢で完全に同一**だった。月齢で体は大きくなるのに筋力が1桁も動かないのは不自然。

【コードを読んで分かったこと】`MIMo/mimoGrowth/physics.py` の `calc_motor_gear` は
**`gear` しか計算していない**。一方 MuscleModel の筋力は
`actuator_user[:, 1] / [:, 2]`（FMAX）から読まれる（`muscle.py:248-251`）。
＝**筋肉モデルを使うと、月齢を変えても筋力は1ミリも変わらない**という仮説。

このスクリプトは全90関節×2方向の FMAX を月齢間で突き合わせて、仮説を確定させる。
ついでに体重（体型補正あり／なし）も月齢別に測り、体型補正が新生児用のまま
他の月齢に流用されていないかを見る。

使い方:
    .venv/Scripts/python.exe E/scripts/e_age_muscle_check.py
"""
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

import numpy as np  # noqa: E402

AGES = [float(x) for x in os.environ.get("E_AGES", "0,4,18").split(",")]

# 人間の体重の中央値 [kg]（WHO Child Growth Standards 2006・男児）
HUMAN_WEIGHT = {0: 3.3, 2: 5.6, 3: 6.4, 4: 7.0, 6: 7.9, 18: 10.9}


def probe(age, shape=True, corrections=True):
    """月齢 age の太郎を作り、FMAX・gear・体重を返す。"""
    from d_supine_env import SupineMimoEnv
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env

    if shape:
        kw = body_kwargs_from_env(age, verbose=False)
    else:
        # ⚠️★体型補正を使わない場合も、MIMoのスキーマを初期状態へ戻してから作る。
        #   `mimoGrowth/growth.py:167` はグローバル辞書を破壊的に書き換えるので、
        #   同じプロセスで補正ありの環境を作った後だと**汚れたスキーマが残る**。
        #   これを忘れて測ったとき、素のage=0が11.892kg（人間3.3kgの3.6倍）という
        #   ありえない値になった（2026-07-28）。→ 落とし穴チェックリスト参照。
        from infant_body import _restore_growth_schema
        _restore_growth_schema()
        kw = {}
    kw["body_corrections"] = corrections
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        env = SupineMimoEnv(vision_params=None, age=age,
                            actuation_model=MuscleModel, **kw)
        env.reset(seed=0)
    m = env.unwrapped.model
    am = getattr(env.unwrapped, "actuation_model", None)

    names = [m.actuator(i).name for i in range(m.nu)]
    gear = np.array([abs(float(m.actuator_gear[i, 0])) for i in range(m.nu)])
    user = np.array(m.actuator_user).copy()
    fmax = np.asarray(getattr(am, "fmax", np.array([])), dtype=float).copy()

    skip = ("floor", "wall", "fence", "object", "toy", "target", "world")
    ids = [i for i in range(1, m.nbody)
           if not any(k in m.body(i).name for k in skip)]
    total = float(sum(m.body_mass[i] for i in ids))
    env.close()
    return dict(age=age, names=names, gear=gear, user=user, fmax=fmax, total=total)


def main():
    print("=" * 76)
    print(" MIMoは月齢で筋力を変えているのか（2026-07-28）")
    print("=" * 76)

    runs = {}
    for age in AGES:
        print(f"  [{age:g}ヶ月をつくっています...]")
        runs[age] = probe(age, shape=True, corrections=False)   # ★補正なしの素の値

    a0 = runs[AGES[0]]
    print("\n" + "=" * 76)
    print("1. FMAX（筋肉モデルの筋力）が月齢で変わるか")
    print("   ※太郎の補正は切ってある＝MIMoの素の値どうしの比較")
    print("-" * 76)
    print(f"{'月齢':>6}{'FMAXの合計':>14}{'0ヶ月との差':>14}{'一致する数':>14}")
    for age in AGES:
        f = runs[age]["fmax"]
        if f.size == 0:
            print(f"{age:>6g}   FMAXが読めない（キャリブレーション未設定）")
            continue
        same = int(np.sum(np.isclose(f, a0["fmax"]))) if f.shape == a0["fmax"].shape else -1
        print(f"{age:>6g}{f.sum():>14.2f}{f.sum() - a0['fmax'].sum():>14.2f}"
              f"{same:>10}/{f.size}")

    print("\n2. gear（トルクモータモデルの筋力）が月齢で変わるか")
    print("-" * 76)
    print(f"{'月齢':>6}{'gearの合計':>14}{'0ヶ月比':>12}")
    for age in AGES:
        g = runs[age]["gear"]
        print(f"{age:>6g}{g.sum():>14.2f}{g.sum()/max(a0['gear'].sum(),1e-9):>12.3f}")

    print("\n3. actuator_user（FMAXの元データ）が月齢で変わるか")
    print("-" * 76)
    for age in AGES:
        u = runs[age]["user"]
        same = int(np.sum(np.isclose(u, a0["user"]))) if u.shape == a0["user"].shape else -1
        print(f"  {age:>4g}ヶ月: 合計={u.sum():>12.2f}  0ヶ月と一致 {same}/{u.size}")

    # --- 体重（体型補正あり／なし） ---
    print("\n" + "=" * 76)
    print("4. 体重は人間の月齢に合っているか")
    print("-" * 76)
    print(f"{'月齢':>6}{'補正あり':>12}{'補正なし':>12}{'人間':>10}"
          f"{'あり/人間':>12}{'なし/人間':>12}")
    for age in AGES:
        with_shape = runs[age]["total"]
        bare = probe(age, shape=False, corrections=False)["total"]
        h = HUMAN_WEIGHT.get(int(age))
        hs = f"{h:.1f}" if h else "?"
        r1 = f"{with_shape/h:.2f}" if h else "-"
        r2 = f"{bare/h:.2f}" if h else "-"
        print(f"{age:>6g}{with_shape:>12.3f}{bare:>12.3f}{hs:>10}{r1:>12}{r2:>12}")

    print("\n" + "=" * 76)
    print("読み方")
    print("-" * 76)
    print("  1で「一致する数」が全部なら → ★MIMoは筋肉モデルの筋力を月齢で変えていない")
    print("  2のgearだけが変わるなら     → 変えているのはトルクモータモデルの側だけ")
    print("  4で補正ありが人間より重いなら → 体型補正（新生児用）を他の月齢に流用している")


if __name__ == "__main__":
    main()
