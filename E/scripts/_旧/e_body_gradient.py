"""身体の「近位-遠位の勾配」を測る＝**発達の順序が自然に出る条件**を満たしているか。

【なぜ測るか、2026-07-25】
太郎は新生児にできない寝返り・寝返り返りをしてしまう。対処を探して文献調査を重ねたが、
「制約を足す」方向は**実装の標準手法が存在せず**、制約の強さも決められないと判明した
（詳細は `doc/やることリスト.md`「寝返り問題」節）。

そこで**第三の道**を先に確かめる。根拠は：

  **Stulp & Oudeyer (2018)** *Developmental Science*
  "Proximodistal Exploration in Motor Learning as an Emergent Property of Optimization"
  https://arxiv.org/abs/1712.05249
  ＝発達の順序（付け根から先へ、順に動かせるようになる）は、**制約を書かなくても
    最適化の副産物として自然に出る**。**ただし身体形態に強く依存する**：
      - 人間的な腕（近位ほど大きい筋・関節）→ 順序が出る
      - **等距離腕**（equidistant arm）→ 順序が出ない（最大固有値差 0.3未満で不明瞭）
      - **左右逆転**した腕（遠位が大きい）→ 順序が**壊れる**

⇒ **MIMoの身体がこの条件を満たしているか**を測る。
   満たしていなければ「制約を足す」のではなく「**身体を直す**」話になり、
   「どれくらい制約するか」という決められない問題が**消える**。

この測定は**値を1つも振らない**（学習もしない）。測るだけ。

使い方:
    python E/scripts/e_body_gradient.py           # 体型v3（既定）
    E_SHAPE=0 python E/scripts/e_body_gradient.py # 補正なし（素のmimoGrowth）と比較
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

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
for p in [os.path.join(_ROOT, "D", "scripts"), os.path.join(_ROOT, "MIMo"), _HERE]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

import numpy as np  # noqa: E402
import mujoco  # noqa: E402

# 人間の体節質量比（体重に対する%）。注意**成人**のデータ（Winter, Biomechanics and Motor
# Control of Human Movement の標準表）。新生児の体節質量比の実測は見つかっていないが、
# 「近位ほど重い」という**順序**は変わらないはずなので、勾配の向きの参照に使う。
HUMAN_ADULT_PCT = {
    "upper_arm": 2.8, "lower_arm": 1.6, "hand": 0.6,
    "upper_leg": 10.0, "lower_leg": 4.65, "foot": 1.45,
}

# 測る連鎖（近位 → 遠位）。MIMoのbody名。
CHAINS = {
    "右腕": ["right_upper_arm", "right_lower_arm", "right_hand"],
    "左腕": ["left_upper_arm", "left_lower_arm", "left_hand"],
    "右脚": ["right_upper_leg", "right_lower_leg", "right_foot"],
    "左脚": ["left_upper_leg", "left_lower_leg", "left_foot"],
}


def subtree_mass(model, bid):
    """そのbodyから先（子孫すべて）の総質量。＝その関節を動かすときに動かす重さ。"""
    total = 0.0
    for b in range(model.nbody):
        p = b
        while p != 0:
            if p == bid:
                total += float(model.body_mass[b])
                break
            p = int(model.body_parentid[p])
    return total


def main():
    from d_supine_env import SupineMimoEnv
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env

    print("身体の近位-遠位の勾配を測る（値は振らない。測るだけ）\n")
    env = SupineMimoEnv(actuation_model=MuscleModel, vision_params=None, age=0.0,
                        **body_kwargs_from_env(0.0, verbose=False))
    m, d = env.unwrapped.model, env.unwrapped.data
    d.qpos[:] = m.qpos0
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)

    # 太郎の身体だけの総質量（床・柵・おもちゃを除く）
    skip = ("floor", "wall", "fence", "object", "toy", "target", "world")
    ids = [i for i in range(1, m.nbody)
           if not any(k in m.body(i).name for k in skip)]
    total = float(sum(m.body_mass[i] for i in ids))
    print(f"全身の質量: {total:.3f} kg\n")

    ok_all = True
    for label, chain in CHAINS.items():
        print(f"=== {label} ===")
        print(f"  {'部位':<16}{'質量g':>9}{'体重%':>8}{'人間(成人)%':>12}"
              f"{'慣性(最大)':>12}{'先の総質量g':>12}")
        print("  " + "-" * 68)
        masses, inertias = [], []
        for name in chain:
            try:
                bid = int(m.body(name).id)
            except Exception:
                print(f"  {name:<16}  注意見つからない")
                continue
            mass = float(m.body_mass[bid])
            inert = float(np.max(m.body_inertia[bid]))
            sub = subtree_mass(m, bid)
            key = name.replace("right_", "").replace("left_", "")
            human = HUMAN_ADULT_PCT.get(key, float("nan"))
            masses.append(mass)
            inertias.append(inert)
            print(f"  {name:<16}{mass*1000:>9.1f}{mass/total*100:>8.2f}{human:>12.2f}"
                  f"{inert:>12.2e}{sub*1000:>12.1f}")

        # 判定：近位 → 遠位で単調に軽くなっているか
        mono_m = all(masses[i] > masses[i + 1] for i in range(len(masses) - 1))
        mono_i = all(inertias[i] > inertias[i + 1] for i in range(len(inertias) - 1))
        ratio = masses[0] / masses[-1] if masses and masses[-1] > 0 else float("nan")
        print(f"  -> 質量が単調に減る: {'[OK]' if mono_m else '[NG]'}   "
              f"慣性が単調に減る: {'[OK]' if mono_i else '[NG]'}   "
              f"最も近位/最も遠位 = {ratio:.1f}倍")
        if not (mono_m and mono_i):
            ok_all = False
        print()

    # 人間との比較（比の大きさ）
    print("=== 判定 ===")
    hu_arm = HUMAN_ADULT_PCT["upper_arm"] / HUMAN_ADULT_PCT["hand"]
    hu_leg = HUMAN_ADULT_PCT["upper_leg"] / HUMAN_ADULT_PCT["foot"]
    print(f"  人間(成人)の 近位/遠位 比: 腕 {hu_arm:.1f}倍 / 脚 {hu_leg:.1f}倍")
    if ok_all:
        print("  [OK] 4本すべてで質量・慣性が近位->遠位に単調減少している")
        print("       = Stulp & Oudeyer 2018 の『順序が自然に出る』条件を形の上では満たす")
        print("       -> 身体は正常。寝返り問題は身体形態では説明できない")
    else:
        print("  [NG] 単調減少していない連鎖がある")
        print("       = 『等距離腕』『左右逆転』に近い状態の可能性")
        print("       -> 身体を直す話になる（制約を足す必要がなくなるかもしれない）")
    print("\n  注意人間の値は**成人**の体節質量比（新生児の実測は見つかっていない）。")
    print("     順序の向きの参照にのみ使い、比の絶対値は当てにしない。")
    env.close()


if __name__ == "__main__":
    main()
