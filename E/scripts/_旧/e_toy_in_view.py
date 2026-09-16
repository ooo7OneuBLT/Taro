"""おもちゃが視界に入っている時間の割合を測る。

【なぜ】ユーザーの目視：「眼球や首の角度が Reset するごとに変わるから、
視界に入らないときもある」。E1 では対象を見て手を伸ばすので、
**そもそも見えていない時間**がどれくらいあるかを知る必要がある。

【ユーザーの仮説】実際の人間なら、親が視界に入るようにおもちゃを動かす。
→ 割合が低ければ「親の介入」が必要だと数字で裏付く。
→ `doc/壁にぶつかったときの型.md` に既に
  「環境側に『親が戻す』介入を置く。実際の新生児がそうされている＝人間模倣として正当」
  と書かれている。

【測るもの】
  1. リセット直後の視線とおもちゃのなす角（シード間のばらつき）
  2. 自発運動を続けたときの、視界内にいる時間の割合
  視野の半角は 30度（fovy=60）。これを超えると視界の外。

使い方:
    .venv/Scripts/python.exe E/scripts/e_toy_in_view.py
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
import os, sys, json, warnings
warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
for p in [os.path.join(_ROOT, "D", "scripts"), os.path.join(_ROOT, "MIMo"),
          os.path.join(_ROOT, "taro_core"),
          os.path.join(_ROOT, "taro_core", "src", "brain", "spinal_cord"), _HERE]:
    if p not in sys.path:
        sys.path.insert(0, p)
try: sys.stdout.reconfigure(errors="replace")
except Exception: pass

import numpy as np

HALF_FOV = 30.0        # 視野角60度の半分
SEEDS = list(range(10))
SIM_SEC = 30.0
BETA = 0.7             # 自発運動の色付き度（現行の既定）
STD = 0.174


def gaze_angle(m, d, cam_id, toy_bid):
    """視線方向とおもちゃ方向のなす角[度]。"""
    cpos = d.cam_xpos[cam_id]
    fwd = -d.cam_xmat[cam_id].reshape(3, 3)[:, 2]     # カメラは -Z を見る
    v = d.xpos[toy_bid] - cpos
    n = np.linalg.norm(v)
    if n < 1e-9:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(fwd, v / n), -1, 1))))


def run(seed, babble=True, sec=SIM_SEC):
    from e_toy_env import ToySupineEnv, infant_vision_params
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env
    from cpg import ColoredNoiseGenerator

    kw = body_kwargs_from_env(0.0, verbose=False)
    kw["flexion"] = True
    env = ToySupineEnv(actuation_model=MuscleModel,
                       vision_params=infant_vision_params(),
                       age=0.0, toy=True, orient=False, **kw)
    m, d = env.unwrapped.model, env.unwrapped.data
    env.reset(seed=seed)

    cam_id = next(c for c in range(m.ncam) if "eye" in (m.camera(c).name or ""))
    toy_bid = m.body("test_object1").id
    n_act = env.action_space.shape[0]
    dt = float(m.opt.timestep) * int(env.unwrapped.frame_skip)

    a0 = gaze_angle(m, d, cam_id, toy_bid)     # リセット直後
    gen = ColoredNoiseGenerator(n_act, seed=seed)
    angles = []
    K = 10                                      # 0.1秒ホールド（現行の階層化に合わせる）
    for i in range(int(sec / dt)):
        if babble and i % K == 0:
            a = np.clip(0.5 + STD * gen.sample(BETA), 0.0, 1.0)
        elif not babble:
            a = np.zeros(n_act, dtype=np.float32)
        env.step(a)
        angles.append(gaze_angle(m, d, cam_id, toy_bid))
    env.close()
    angles = np.array(angles)
    return a0, angles


def main():
    print("=== おもちゃが視界に入っている割合 ===")
    print(f"  視野の半角 {HALF_FOV:.0f}度（fovy=60）。これを超えると視界の外")
    print(f"  {len(SEEDS)}シード × {SIM_SEC:.0f}秒（自発運動あり）\n")

    print(f"{'seed':>5}{'リセット直後':>14}{'平均':>10}{'最小':>8}{'最大':>8}"
          f"{'視界内の割合':>14}")
    print("-" * 62)
    all_in, all_a0 = [], []
    for s in SEEDS:
        try:
            a0, ang = run(s)
        except Exception as e:
            print(f"{s:>5}  [ERR] {type(e).__name__}: {e}")
            continue
        inside = float((ang < HALF_FOV).mean())
        all_in.append(inside)
        all_a0.append(a0)
        print(f"{s:>5}{a0:>14.1f}{ang.mean():>10.1f}{ang.min():>8.1f}"
              f"{ang.max():>8.1f}{inside*100:>13.0f}%")

    if all_in:
        a0 = np.array(all_a0)
        inn = np.array(all_in)
        print("-" * 62)
        print(f"{'平均':>5}{a0.mean():>14.1f}{'':>10}{'':>8}{'':>8}{inn.mean()*100:>13.0f}%")
        print()
        print(f"  リセット直後の角度のばらつき: {a0.mean():.1f} ± {a0.std():.1f}度"
              f"（{a0.min():.1f}〜{a0.max():.1f}）")
        print(f"  視界内にいる時間の割合: {inn.mean()*100:.0f}% "
              f"（{inn.min()*100:.0f}〜{inn.max()*100:.0f}%）")
        print()
        if inn.mean() < 0.5:
            print("  → 半分以上の時間、おもちゃが見えていない。")
            print("     『親が視界に戻す』介入が要る、というユーザーの仮説を支持する")
        else:
            print("  → 過半の時間は見えている。介入なしでも E1 は成立しうる")


if __name__ == "__main__":
    main()
