"""床の摩擦が「寝返り」に効くかを測る（学習なし・値を振るだけ）。

【なぜ測るか、2026-07-25】
太郎は新生児にできない寝返り・寝返り返りをする。原因を探して
「筋力ではない」（32倍振っても不変）「体の硬さでもない」（新生児の脊柱はむしろ過可動）
までは分かったが、**では何が原因かは特定できていない**。

重要な手がかり：**学習なし・色付きノイズだけでも寝返る**
（実測：K=100 でうつ伏せ36.2%、K=10 で65.0%）＝**学習の産物ではない**。
⇒ 身体・物理・環境のどこか。

そして床の設定を見ると：
    床の friction = [1.0, 0.005, 0.0001]   （すべり / ころがり / ねじれ）
    床の condim   = 3                       ＝ **すべり摩擦しか計算しない**
＝**転がることへの抵抗が事実上ゼロ**。

実際の新生児は**服＋寝具（布と布）**の上にいて、転がるとき布が引っかかる。
太郎は bare skin ＋ 平面床。この差が原因ではないか、というのがユーザーの指摘。

注意：**2026-07-23 に「床摩擦を人工的に上げる補正は入れない」と決めている**が、
それは**対処**への判断であって、**原因かどうかの検証**はしていない。
しかも当時の症状は「滑る（drift）」で、今回は「転がる（寝返り）」＝別の症状。
→ もし摩擦が原因なら、対処は「摩擦を上げる」（工学的対処）ではなく
  **服・寝具をモデル化する**（人間模倣として正当）になりうる。

【この測定の性格】型B（振って効くか効かないかを見る）。
→ `doc/壁にぶつかったときの型.md`

使い方:
    python E/scripts/e_friction_probe.py            # 4条件 × 10シード
    E_FRIC_SEEDS=3 python ...                       # シード数を変える
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
for p in [os.path.join(_ROOT, "D", "scripts"), os.path.join(_ROOT, "MIMo"),
          os.path.join(_ROOT, "taro_core"), _HERE,
          os.path.join(_ROOT, "taro_core", "src", "brain", "spinal_cord")]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

import numpy as np  # noqa: E402
import mujoco  # noqa: E402

N_STEP = int(os.environ.get("E_FRIC_STEPS", "6000"))
K = int(os.environ.get("E_K", "10"))
N_SEED = int(os.environ.get("E_FRIC_SEEDS", "10"))

# (ラベル, ころがり摩擦, condim)
# 注意：MuJoCoの接触の condim は「両geomの condim の**大きい方**」が使われる。
#   床が3でも、太郎側が6のgeom（手など）との接触では6になる。
#   ここでは**床側**を変える＝体のどの部位が触れても転がり抵抗が効くようにする。
CONDS = [
    ("現行 (roll=0.005, condim=3)", 0.005, 3),
    ("roll=0.05, condim=6",          0.05,  6),
    ("roll=0.5,  condim=6",          0.5,   6),
    ("roll=2.0,  condim=6",          2.0,   6),
]


def set_floor_friction(model, roll, condim):
    """床geomのころがり摩擦と condim を書き換える。すべり摩擦(1.0)は触らない。"""
    n = 0
    for gi in range(model.ngeom):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
        if nm == "floor" or model.body(model.geom_bodyid[gi]).name == "world":
            model.geom_friction[gi][1] = float(roll)
            model.geom_condim[gi] = int(condim)
            n += 1
    return n


def trunk_rotation_deg(data, R0):
    """体幹が初期姿勢(仰向け)からどれだけ回転したか。0度=仰向け、180度=うつぶせ。"""
    R = data.body("upper_body").xmat.reshape(3, 3)
    rel = R0.T @ R
    cos = (np.trace(rel) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def run_one(roll, condim, seed):
    from d_supine_env import SupineMimoEnv
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env
    from cpg import ColoredNoiseGenerator

    env = SupineMimoEnv(actuation_model=MuscleModel, vision_params=None, age=0.0,
                        **body_kwargs_from_env(0.0, verbose=False))
    m, d = env.unwrapped.model, env.unwrapped.data
    set_floor_friction(m, roll, condim)
    env.reset(seed=seed)
    R0 = d.body("upper_body").xmat.reshape(3, 3).copy()

    n_act = env.action_space.shape[0]
    gen = ColoredNoiseGenerator(n_act, seed=seed)
    rots = []
    a = np.full(n_act, 0.5)
    for tick in range(max(1, N_STEP // K)):
        # 学習なし＝色付きノイズだけで筋を動かす（探索の実効値 std≈0.174 に合わせる）
        a = np.clip(0.5 + 0.174 * gen.sample(0.7), 0.0, 1.0)
        for _ in range(K):
            env.step(a)
        rots.append(trunk_rotation_deg(d, R0))
    env.close()

    r = np.asarray(rots)
    prone = r > 90.0
    to_supine = int(np.sum(~prone[1:] & prone[:-1]))
    return dict(prone_pct=float(np.mean(prone)) * 100,
                mean_rot=float(np.mean(r)), max_rot=float(np.max(r)),
                back=to_supine)


def main():
    print(f"床の摩擦が寝返りに効くか（学習なし・色付きノイズのみ）"
          f"  K={K} step={N_STEP} seeds={N_SEED}\n")
    print(f"{'条件':<26}{'うつ伏せ%':>10}{'体幹回転':>10}{'最大':>8}{'寝返り返り':>11}")
    print("-" * 66)
    results = {}
    for label, roll, condim in CONDS:
        pr, mr, mx, bk = [], [], [], []
        for s in range(N_SEED):
            try:
                r = run_one(roll, condim, s)
            except Exception as e:
                print(f"  [ERR] {label} seed{s}: {e}")
                continue
            pr.append(r["prone_pct"]); mr.append(r["mean_rot"])
            mx.append(r["max_rot"]); bk.append(r["back"])
        if pr:
            results[label] = (pr, bk)
            print(f"{label:<26}{np.mean(pr):>10.1f}{np.mean(mr):>10.1f}"
                  f"{np.mean(mx):>8.1f}{np.mean(bk):>11.1f}", flush=True)

    # 効果量（現行との比較）
    base_label = CONDS[0][0]
    if base_label in results:
        b_pr, b_bk = results[base_label]
        print("\n=== 現行との効果量 ===")
        for label, _, _ in CONDS[1:]:
            if label not in results:
                continue
            v_pr, v_bk = results[label]
            for nm, b, v in [("うつ伏せ%", b_pr, v_pr), ("寝返り返り", b_bk, v_bk)]:
                sd = ((np.std(v) ** 2 + np.std(b) ** 2) / 2) ** 0.5
                dd = (np.mean(v) - np.mean(b)) / sd if sd > 1e-9 else 0.0
                j = ("ほぼ差なし" if abs(dd) < 0.2 else
                     "小" if abs(dd) < 0.5 else "中" if abs(dd) < 0.8 else "[大]")
                print(f"  {label:<24} {nm:<10} {np.mean(v)-np.mean(b):+7.1f}  "
                      f"効果量 {dd:+5.2f} ({j})")
    print("\n  [注] 学習なし＝色付きノイズだけ。学習の産物でないことは確認済み")
    print("       (K=100でうつ伏せ36.2%、K=10で65.0%＝学習なしでも寝返る)")


if __name__ == "__main__":
    main()
