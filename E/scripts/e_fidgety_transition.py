"""★writhing→fidgety移行を、今の体（筋力バグ修正後）で測り直す。

【なぜ・2026-07-26】前回の目視確認（研究日誌 2026-07-23 続き16）は
**筋力の配線バグが未修正の体**（首24倍・四肢54倍が誤って強かった）で行った。
その後
  ① 階層化（K=100→10）
  ② MuscleModel への切り替え
  ③ ★筋力バグ修正
の3つの大改修が入っており、fidgety移行の実装が今の体で機能するかは未確認。

【何を測るか・writhing vs fidgety の判別】
  writhing (0〜2ヶ月)  ： 中〜大振幅、遅い
  fidgety  (3〜5ヶ月)  ： 小振幅、中等度の速さ、変動する加速度
  期待される移行方向：
    ROM ↓（振幅が小さくなる）
    speed ↑ or ~（速さは中庸〜やや上）
    ★β ↑（López 2026：8週 0.686 → 30週 0.877）
    cycle ↓（1往復が短くなる）

【スケジュールの中身】developmental_schedule.py（1.5〜5ヶ月で線形）
  w_mean : 0 → 0.5（皮質脊髄路の再導入）
  syn_w  : 0.6 → 0.85（多筋協調が構造化）

【学習しない】純粋な生成器の性格を測る（1シード、数分で終わる）。
学習後の方策はもっと収束するはずなので、これはあくまで
「素材（noise + synergy + w_mean）の月齢による違い」を見るテスト。

使い方：
    python E/scripts/e_fidgety_transition.py
"""

# ⚠️★古い方式（2026-07-30 に整理）。新しい実験は `run/main.py` を通す。
#   【経緯】目標Eの実験スクリプトが118本あり、うち66本が**独立に環境を組み立てていた**。
#     そのため「学習は関節モード（90関節を独立に駆動＝逸脱リスト 逸脱5）、
#     測定とViewerは筋肉モード（拮抗筋2本/関節）」という**別の体で動く**事故が起きた
#     （ユーザーの目視「視線誘導反射の実験の時とは動きが全然違う」で発覚。
#      実測で動きが人間の新生児の約3.3倍速かった）。
#   【設計と移行計画】`E/docs/実行基盤_設計.md`
#   ⚠️このファイルは**記録として残す**（削除しない方針）。
#     中の測り方は再利用できるので、プラグインへ移すときの元にする。
import os
import sys
import warnings

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
for p in [os.path.join(_ROOT, "D", "scripts"), os.path.join(_ROOT, "MIMo"),
          os.path.join(_ROOT, "taro_core"),
          os.path.join(_ROOT, "taro_core", "src", "brain"),
          os.path.join(_ROOT, "taro_core", "src", "brain", "spinal_cord"),
          _HERE]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

import numpy as np  # noqa: E402

AGES = [0.0, 1.5, 3.0, 5.0]   # [Tier1範囲] writhing / 移行開始 / 途中 / 完了
SEEDS = [0, 1, 2]
N_STEP = 6000                  # 60秒相当
K = 10                          # 0.1秒ホールド（現行の階層化に合わせる）
STD = 0.174                     # 現行既定
BETA = 0.7                      # 現行既定（8週相当。将来は月齢連動）

# 見る関節（e_movement_scale.py と同じ）
LEG_R = [72, 73, 75, 76]
LEG_L = [81, 82, 84, 85]
ARM_R = [14, 15, 17, 19]
ARM_L = [43, 44, 46, 48]
WATCH = ["right_shoulder_ad_ab", "right_elbow", "right_hip1", "right_knee"]


def autocorr_time(x, dt, max_lag=200):
    x = np.asarray(x, dtype=float) - np.mean(x)
    if np.std(x) < 1e-12:
        return float("nan")
    n = min(max_lag, len(x) // 2)
    ac = np.array([1.0] + [float(np.corrcoef(x[:-k], x[k:])[0, 1]) for k in range(1, n)])
    below = np.where(ac < 1.0 / np.e)[0]
    return float(below[0] * dt) if len(below) else float(n * dt)


def spectral_beta(x, dt):
    x = np.asarray(x, dtype=float) - np.mean(x)
    if np.std(x) < 1e-12:
        return float("nan")
    f = np.fft.rfftfreq(len(x), d=dt)
    p = np.abs(np.fft.rfft(x)) ** 2
    m = (f > 0) & (p > 0)
    if m.sum() < 10:
        return float("nan")
    fl, pl = f[m], p[m]
    keep = fl <= max(fl.min() * 30, np.percentile(fl, 20))
    if keep.sum() < 10:
        keep = np.ones_like(fl, dtype=bool)
    return float(-np.polyfit(np.log10(fl[keep]), np.log10(pl[keep]), 1)[0])


def cycle_time(x, dt):
    x = np.asarray(x, dtype=float)
    if np.std(x) < 1e-12:
        return float("nan")
    up = x > x.mean() + 0.3 * x.std()
    idx, i = [], 0
    while i < len(up):
        if up[i]:
            j = i
            while j < len(up) and up[j]:
                j += 1
            idx.append((i + j) // 2)
            i = j
        else:
            i += 1
    if len(idx) < 3:
        return float("nan")
    return float(np.mean(np.diff(idx)) * dt)


def jerk_rms(x, dt):
    """加加速度のRMS（fidgetyの"変動する加速度"の代理）。"""
    x = np.asarray(x, dtype=float)
    if len(x) < 4 or np.std(x) < 1e-12:
        return float("nan")
    j = np.diff(x, n=3) / (dt ** 3)
    return float(np.sqrt(np.mean(j ** 2)))


def run(age, seed):
    from d_supine_env import SupineMimoEnv
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env
    from cpg import CPG
    from developmental_schedule import schedule_w_mean, schedule_syn_w

    w_mean = schedule_w_mean(age)
    syn_w = schedule_syn_w(age)

    kw = body_kwargs_from_env(age, verbose=False)
    env = SupineMimoEnv(actuation_model=MuscleModel, vision_params=None, age=age, **kw)
    m, d = env.unwrapped.model, env.unwrapped.data
    env.reset(seed=seed)
    n_act = env.action_space.shape[0]
    n_joint = n_act // 2
    dt = float(m.opt.timestep) * int(env.unwrapped.frame_skip)

    gen = CPG(n_act, leg_r=LEG_R, leg_l=LEG_L, arm_r=ARM_R, arm_l=ARM_L,
              seed=seed, pair_offset=n_joint)
    dofs = [int(m.jnt_dofadr[m.joint("robot:" + nm).id]) for nm in WATCH]
    qadr = [int(m.jnt_qposadr[m.joint("robot:" + nm).id]) for nm in WATCH]

    # w_mean 相当の減衰は「学習後のmean出力」が無いここでは近似的に
    # ★s に (1 - w_mean) を掛けて探索を弱める（w_mean が上がると精密制御が主役に
    # なる方向を粗く再現）。学習ループそのものではないので厳密ではない。
    scale = 1.0 - 0.5 * w_mean  # w_mean=0→1.0、w_mean=0.5→0.75

    ang, vel = [], []
    for _ in range(max(1, N_STEP // K)):
        s = gen.sample(BETA, synergy=True, syn_w=syn_w)
        a = np.clip(0.5 + STD * scale * s, 0.0, 1.0)
        for _ in range(K):
            env.step(a)
            ang.append(np.degrees(d.qpos[qadr]).copy())
            vel.append(np.degrees(d.qvel[dofs]).copy())
    env.close()
    A, V = np.asarray(ang), np.asarray(vel)
    out = {}
    for i, nm in enumerate(WATCH):
        out[nm] = dict(rom=float(np.ptp(A[:, i])),
                       speed=float(np.mean(np.abs(V[:, i]))),
                       tau=autocorr_time(V[:, i], dt),
                       beta=spectral_beta(V[:, i], dt),
                       cyc=cycle_time(A[:, i], dt),
                       jerk=jerk_rms(A[:, i], dt))
    return out, w_mean, syn_w


def main():
    print("=== writhing→fidgety transition (post 筋力バグ修正) ===")
    print(f"  K={K} STD={STD} BETA={BETA} steps={N_STEP} (~{N_STEP*0.01:.0f}s)")
    print(f"  seeds={SEEDS}  ages(months)={AGES}")
    print("  writhing(0-2mo): LARGE amp, SLOW / fidgety(3-5mo): small amp, moderate")
    print("  expected: age↑ → ROM↓, beta↑, cycle↓, jerk↑\n")

    header = f"{'age':>4}{'w_m':>6}{'syn':>5}  {'joint':<24}"
    header += f"{'ROM':>8}{'speed':>8}{'tau':>7}{'beta':>7}{'cyc':>7}{'jerk':>10}"
    print(header)
    print("-" * len(header))

    # 集計用（joint別・age別のシード平均）
    agg = {nm: {age: {k: [] for k in ["rom", "speed", "beta", "cyc", "jerk"]} for age in AGES}
           for nm in WATCH}

    for age in AGES:
        for seed in SEEDS:
            try:
                r, w_m, syn = run(age, seed)
            except Exception as e:
                print(f"  age={age} seed={seed}  [ERR] {type(e).__name__}: {e}")
                continue
            for nm in WATCH:
                v = r[nm]
                print(f"{age:>4.1f}{w_m:>6.2f}{syn:>5.2f}  {nm:<24}"
                      f"{v['rom']:>8.1f}{v['speed']:>8.1f}{v['tau']:>7.3f}"
                      f"{v['beta']:>7.2f}{v['cyc']:>7.2f}{v['jerk']:>10.1f}",
                      flush=True)
                for k in agg[nm][age]:
                    agg[nm][age][k].append(v[k])
            print()

    # ★シード平均でのagesの推移をまとめて表示
    print("\n=== summary: seed-mean, per joint ===")
    for nm in WATCH:
        print(f"\n  {nm}")
        print(f"    {'age':>4}{'ROM':>10}{'speed':>10}{'beta':>10}{'cyc':>10}{'jerk':>12}")
        for age in AGES:
            row = agg[nm][age]
            def _mean(k):
                vs = [x for x in row[k] if not np.isnan(x)]
                return np.mean(vs) if vs else float("nan")
            print(f"    {age:>4.1f}{_mean('rom'):>10.1f}{_mean('speed'):>10.1f}"
                  f"{_mean('beta'):>10.2f}{_mean('cyc'):>10.2f}{_mean('jerk'):>12.1f}")

    print("\n  --- 判定の見方 ---")
    print("  age=0 → age=5 で：")
    print("    ROM が★減る、beta が★上がる（0.7→0.9方向）、cycle が★短くなる")
    print("    → fidgety方向。文献と整合。")
    print("  逆方向 or 変化なし：")
    print("    → 移行が今の体でも効いていない。実装の再検討が必要。")


if __name__ == "__main__":
    main()
