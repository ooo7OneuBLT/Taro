"""関節角どうしの連動が「どこから来ているか」を分解する（学習なし）。

【なぜ・2026-07-25】文献調査の指摘：
  MIMoは物理シミュレータなので、**関節の連動は身体の力学から勝手に発生する**
  （Schneider, Zernicke, Ulrich, Jensen, Thelen 1990, J Motor Behavior：
   3ヶ月児の自発キッキングを逆動力学解析し、滑らかな軌道は「筋トルクが重力だけでなく、
   連結セグメントの運動が生む**運動依存トルク**にも対抗し、かつ補完することで」生じると示した）
  ⇒ `syn_w=0.6` で足しているシナジーの一部は、**受動力学と二重計上**かもしれない。

【何を測るか】3条件で「関節角どうしの相関」を測り、寄与を分解する。
  条件P（受動）  ：行動を一切出さない（筋活性化=0）＝重力と接触だけ
                   → ここで既に出る相関が「身体の力学に由来する分」
  条件I（独立）  ：独立ノイズ（シナジーOFF）
                   → 受動 + 独立な指令 で、どれだけ相関が増えるか
  条件S（相関）  ：色付きノイズ + シナジーON（syn_w=0.6）
                   → シナジーが**上乗せしている分**

人間の実測（Thelen 1985 / Piek & Gasson 1997 の関節間相互相関の実数値）は
  有料誌のため未取得。だから「人間と比べて多いか」はまだ言えない。
  ここで分かるのは**太郎の内部での寄与の分解**だけ。

注意：出力は必ずASCII（cp932で日本語が化ける＝チェックリスト項35）。
注意：条件名を出力パスに入れる（項21）。

使い方:
    python E/scripts/e_joint_corr.py           # 3条件 x 3シード
    E_CORR_SEEDS=5 python E/scripts/e_joint_corr.py
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

N_STEP = int(os.environ.get("E_CORR_STEPS", "6000"))
K = int(os.environ.get("E_K", "10"))
N_SEED = int(os.environ.get("E_CORR_SEEDS", "3"))
STD = float(os.environ.get("E_CORR_STD", "0.174"))   # 学習初期の実効値
BETA = float(os.environ.get("E_BETA", "0.7"))
SYN_W = float(os.environ.get("E_SYN_W", "0.6"))

# 90-actuator index（シナジーの対象。e_growth_train.py と同じ）
LEG_R = [72, 73, 75, 76]
LEG_L = [81, 82, 84, 85]
ARM_R = [14, 15, 17, 19]
ARM_L = [43, 44, 46, 48]

# 相関を見たい関節（名前で引く。actuator index とは別物なので注意）
LEG_JOINTS = ["right_hip1", "right_hip2", "right_knee", "right_foot1"]
TRUNK_JOINTS = ["hip_bend1", "hip_lean1", "hip_rot1", "chest_lean", "chest_rot"]


def joint_dofs(model, names):
    out = []
    for nm in names:
        try:
            j = model.joint("robot:" + nm)
        except Exception:
            continue
        out.append((nm, int(model.jnt_dofadr[j.id])))
    return out


def mean_abs_offdiag(qpos_series, idx):
    """指定した関節の角度時系列から、対どうしの相関の絶対値の平均を返す。"""
    if len(idx) < 2:
        return float("nan")
    x = qpos_series[:, idx]
    # 分散がほぼ0の列があると相関が nan になるので落とす
    keep = [i for i in range(x.shape[1]) if np.std(x[:, i]) > 1e-9]
    if len(keep) < 2:
        return float("nan")
    c = np.corrcoef(x[:, keep].T)
    n = c.shape[0]
    vals = [abs(c[i, j]) for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(vals))


def run_one(cond, seed):
    from d_supine_env import SupineMimoEnv
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env
    from cpg import CPG, ColoredNoiseGenerator

    env = SupineMimoEnv(actuation_model=MuscleModel, vision_params=None, age=0.0,
                        **body_kwargs_from_env(0.0, verbose=False))
    m, d = env.unwrapped.model, env.unwrapped.data
    env.reset(seed=seed)
    n_act = env.action_space.shape[0]
    n_joint = n_act // 2

    gen = None
    if cond == "I":
        gen = ColoredNoiseGenerator(n_act, seed=seed)
    elif cond == "S":
        # 筋肉モードなので pair_offset=n_joint（拮抗筋ペアに符号反転で適用）
        gen = CPG(n_act, leg_r=LEG_R, leg_l=LEG_L, arm_r=ARM_R, arm_l=ARM_L,
                  seed=seed, pair_offset=n_joint)

    leg = joint_dofs(m, LEG_JOINTS)
    trunk = joint_dofs(m, TRUNK_JOINTS)
    all_dofs = [dof for _, dof in leg + trunk]
    leg_idx = list(range(len(leg)))
    trunk_idx = list(range(len(leg), len(leg) + len(trunk)))

    series = []
    for _ in range(max(1, N_STEP // K)):
        if cond == "P":
            a = np.zeros(n_act)                       # 完全に受動（筋活性化=0）
        elif cond == "I":
            a = np.clip(0.5 + STD * gen.sample(BETA), 0.0, 1.0)
        else:
            a = np.clip(0.5 + STD * gen.sample(BETA, synergy=True, syn_w=SYN_W), 0.0, 1.0)
        for _ in range(K):
            env.step(a)
            series.append(d.qpos[all_dofs].copy())
    env.close()

    s = np.asarray(series)
    return dict(leg=mean_abs_offdiag(s, leg_idx),
                trunk=mean_abs_offdiag(s, trunk_idx),
                leg_range=float(np.mean(np.ptp(s[:, leg_idx], axis=0))),
                trunk_range=float(np.mean(np.ptp(s[:, trunk_idx], axis=0))))


CONDS = [("P", "passive (action=0)"),
         ("I", "independent noise"),
         ("S", f"synergy ON (w={SYN_W})")]


def main():
    print(f"Where does joint coupling come from?  K={K} step={N_STEP} seeds={N_SEED}")
    print("  mean |correlation| between joint ANGLES (not the noise)")
    print(f"  legs  = {LEG_JOINTS}")
    print(f"  trunk = {TRUNK_JOINTS}\n")
    print(f"{'condition':<24}{'legs':>9}{'trunk':>9}{'legROM':>10}{'trunkROM':>10}")
    print("-" * 62)
    res = {}
    for c, label in CONDS:
        rows = []
        for s in range(N_SEED):
            try:
                rows.append(run_one(c, s))
            except Exception as e:
                print(f"  [ERR] {label} seed{s}: {type(e).__name__}: {e}")
        if not rows:
            continue
        g = lambda k: float(np.nanmean([r[k] for r in rows]))
        res[c] = rows
        print(f"{label:<24}{g('leg'):>9.3f}{g('trunk'):>9.3f}"
              f"{np.degrees(g('leg_range')):>9.1f}d{np.degrees(g('trunk_range')):>9.1f}d",
              flush=True)

    if "P" in res and "S" in res:
        gp = lambda c, k: float(np.nanmean([r[k] for r in res[c]]))
        print("\n=== decomposition ===")
        for nm in ["leg", "trunk"]:
            p, i, s = gp("P", nm), gp("I", nm), gp("S", nm)
            print(f"  {nm:<6} passive={p:.3f}  independent={i:.3f}  synergy={s:.3f}")
            print(f"         -> body dynamics contributes {p:.3f}, "
                  f"synergy adds {s - i:+.3f} on top of independent")
        print("\n  [note] If 'passive' is already high, part of syn_w is DOUBLE-COUNTING")
        print("         the coupling that the physics produces by itself.")
        print("  [note] Human reference values (Thelen 1985 / Piek 1997 cross-correlation)")
        print("         are NOT obtained yet (paywalled), so we cannot say")
        print("         whether these numbers are human-like.")


if __name__ == "__main__":
    main()
