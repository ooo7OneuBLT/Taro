"""★太郎の動きは writhing なのか fidgety なのかを数値で判定する（学習なし）。

【なぜ・2026-07-25】ユーザーの目視：
> やっぱり運動が細かい。僕が見た自発運動の動画はもうちょっと**ゆっくり大きく**動いてた
> （腕を上下に動かす、足を伸ばしたり縮めたりする）

★Prechtl の General Movements の分類（⚠️定義は一次文献で要確認）：
    writhing movements（新生児〜約2ヶ月）… **中等度の振幅・遅い〜中等度の速さ**
    fidgety movements（約3〜5ヶ月）      … **小さい振幅・中等度の速さ・変動する加速度**
＝ユーザーの記述は writhing の定義と一致し、太郎は fidgety 寄りに見える
＝★月齢が先に進んだ動きをしている疑い。

【ユーザーの仮説】「0.1秒ごとに命令が変わるのが一番よくない気がする」
→ K（何physics-stepごとに指令を更新するか）を振って検証する。
   K=10 なら 0.1秒ごと、K=100 なら 1秒ホールド。

【もう一つの候補】90関節に独立なノイズが流れている（シナジーOFF）
→ シナジー ON/OFF でも比べる。「まとまれば大きくなる」かどうか。

【★第三の候補（未検証・重要）】國吉研は**振動子**で動かしている
    Kuniyoshi & Sangawa (2006) Biol Cybern 95:589-605
    ＝予め与えたのは①脊髄の伸張反射 ②各筋に個別接続された medulla の**BVP振動子**のみ
    → 「rolling over と crawling-like motion が創発した」
★振動子は周期的にゆっくり大きく動く。ランダムノイズでは反復的な動きが出にくい。
＝「腕を上下に動かす」という**反復**は、原理的にノイズでは作れない可能性がある。
⚠️このスクリプトでは振動子は試さない（未実装）。まず現状を測る。

【測る量】
  1. 自己相関の時定数 … 関節角速度が同じ向きを保つ時間（★動きの「ゆっくりさ」）
  2. スペクトル指数 β … ★López et al. 2026 の実測（8週 β=0.686 / 30週 0.877）と直接比較
  3. 振幅（ROM）と速さ（|qvel|）を**分けて**見る
     writhing = 振幅 大 / 速さ 小   ／   fidgety = 振幅 小 / 速さ 大
  4. ★「1往復の時間」… ピークからピークまでの間隔（腕を上下させる周期に相当）

使い方:
    python E/scripts/e_movement_scale.py
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
          os.path.join(_ROOT, "taro_core"), _HERE,
          os.path.join(_ROOT, "taro_core", "src", "brain", "spinal_cord")]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

import numpy as np  # noqa: E402

N_STEP = int(os.environ.get("E_MS_STEPS", "6000"))
STD = float(os.environ.get("E_MS_STD", "0.174"))
BETA = float(os.environ.get("E_BETA", "0.7"))
SYN_W = float(os.environ.get("E_SYN_W", "0.6"))
LIMB = float(os.environ.get("E_LIMB_SCALE", "16.0"))

LEG_R = [72, 73, 75, 76]
LEG_L = [81, 82, 84, 85]
ARM_R = [14, 15, 17, 19]
ARM_L = [43, 44, 46, 48]

# 見る関節（ユーザーが見た「腕を上下」「足を伸ばす/縮める」に対応する軸）
WATCH = ["right_shoulder_ad_ab", "right_elbow", "right_hip1", "right_knee"]


def autocorr_time(x, dt, max_lag=200):
    """自己相関が 1/e まで落ちるまでの時間（秒）＝動きが同じ向きを保つ時間。"""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    if np.std(x) < 1e-12:
        return float("nan")
    n = min(max_lag, len(x) // 2)
    ac = np.array([1.0] + [float(np.corrcoef(x[:-k], x[k:])[0, 1]) for k in range(1, n)])
    below = np.where(ac < 1.0 / np.e)[0]
    return float(below[0] * dt) if len(below) else float(n * dt)


def spectral_beta(x, dt):
    """パワースペクトルの傾き -β を推定（1/f^β）。★López 2026 と比較する量。"""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    if np.std(x) < 1e-12:
        return float("nan")
    f = np.fft.rfftfreq(len(x), d=dt)
    p = np.abs(np.fft.rfft(x)) ** 2
    m = (f > 0) & (p > 0)
    if m.sum() < 10:
        return float("nan")
    # 低周波側の1桁分だけで傾きを取る（高周波は離散化の影響を受ける）
    fl, pl = f[m], p[m]
    keep = fl <= max(fl.min() * 30, np.percentile(fl, 20))
    if keep.sum() < 10:
        keep = np.ones_like(fl, dtype=bool)
    return float(-np.polyfit(np.log10(fl[keep]), np.log10(pl[keep]), 1)[0])


def cycle_time(x, dt):
    """ピークからピークまでの平均間隔（秒）＝「1往復の時間」。"""
    x = np.asarray(x, dtype=float)
    if np.std(x) < 1e-12:
        return float("nan")
    # 平均より上に出た区間の中心をピークとみなす（ノイズに強い簡易版）
    up = x > x.mean() + 0.3 * x.std()
    idx = []
    i = 0
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


def run(K, synergy, seed=0):
    from d_supine_env import SupineMimoEnv
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env
    from cpg import CPG, ColoredNoiseGenerator

    kw = body_kwargs_from_env(0.0, verbose=False)
    kw["limb_scale"] = LIMB
    env = SupineMimoEnv(actuation_model=MuscleModel, vision_params=None, age=0.0, **kw)
    m, d = env.unwrapped.model, env.unwrapped.data
    env.reset(seed=seed)
    n_act = env.action_space.shape[0]
    n_joint = n_act // 2
    dt = float(m.opt.timestep) * int(env.unwrapped.frame_skip)

    gen = (CPG(n_act, leg_r=LEG_R, leg_l=LEG_L, arm_r=ARM_R, arm_l=ARM_L,
               seed=seed, pair_offset=n_joint) if synergy
           else ColoredNoiseGenerator(n_act, seed=seed))
    dofs = [int(m.jnt_dofadr[m.joint("robot:" + nm).id]) for nm in WATCH]
    qadr = [int(m.jnt_qposadr[m.joint("robot:" + nm).id]) for nm in WATCH]

    ang, vel = [], []
    for _ in range(max(1, N_STEP // K)):
        s = (gen.sample(BETA, synergy=True, syn_w=SYN_W) if synergy else gen.sample(BETA))
        a = np.clip(0.5 + STD * s, 0.0, 1.0)
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
                       cyc=cycle_time(A[:, i], dt))
    return out


def main():
    print("Is Taro's movement 'writhing' or 'fidgety'?  (no learning, colored noise only)")
    print(f"  limb_scale={LIMB}  std={STD}  beta_target={BETA}  steps={N_STEP}")
    print("  writhing (newborn-2mo) = LARGE amplitude, SLOW")
    print("  fidgety  (3-5mo)       = small amplitude, moderate speed")
    print("  ★reference: Lopez et al. 2026 measured spectral index beta = 0.686 at 8 weeks\n")

    conds = [("K=10  syn OFF", 10, False),
             ("K=10  syn ON ", 10, True),
             ("K=100 syn OFF", 100, False),
             ("K=100 syn ON ", 100, True)]
    print(f"{'condition':<16}{'joint':<24}{'ROM(deg)':>10}{'speed':>9}"
          f"{'tau(s)':>9}{'beta':>7}{'cycle(s)':>10}")
    print("-" * 86)
    for label, K, syn in conds:
        try:
            r = run(K, syn)
        except Exception as e:
            print(f"{label:<16}  [ERR] {type(e).__name__}: {e}")
            continue
        for nm in WATCH:
            v = r[nm]
            print(f"{label:<16}{nm:<24}{v['rom']:>10.1f}{v['speed']:>9.1f}"
                  f"{v['tau']:>9.3f}{v['beta']:>7.2f}{v['cyc']:>10.2f}", flush=True)
        print()

    print("  ROM   = range of motion (amplitude)")
    print("  speed = mean |angular velocity| (deg/s)")
    print("  tau   = autocorrelation time of velocity (how long it keeps the same direction)")
    print("  ★cycle = mean peak-to-peak interval ~ 'one back-and-forth' of the joint")
    print("\n  [!] If cycle is much shorter than a few seconds, the movement is")
    print("      too rapid for writhing regardless of amplitude.")


if __name__ == "__main__":
    main()
