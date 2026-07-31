"""生理的屈曲（physiological flexion）を入れるための実測（学習なし・設計の前段）。

【なぜ・2026-07-25】新生児は**放っておいても股・膝・肘が曲がっている**（屈曲拘縮）。
一次文献の実測値：

| 関節 | 伸展の限界 | 出典 |
|---|---|---|
| 股関節 | **-32度** | Ishida et al. 1997, Rev Bras Ortop 32(1):37-45（n=80、満期産・生後0-4日、受動ROM、OA） |
| 膝 | **-21度** | Broughton, Wright & Menelaus 1993, J Pediatr Orthop 13(2):263-4（n=57、縦断） |
| 肘 | **-14度** | Watanabe et al. 1979（n=62。Norkin & White 教科書 Table 16-2 経由＝☆二次） |

太郎の実測（settle後）：股 -6.5度 / 膝 -22.7度 / 肘 -11.7度
＝**膝はほぼ合っているが、股関節が25.5度も伸びすぎている**。

注意：【2026-07-25 重要な訂正】当初「MIMoは既に jnt_stiffness=1.72 のバネを持っている」と
報告したが、それは **SpringDamperModel（筋肉なしの既定モデル）** の値だった。
**学習に使う MuscleModel では jnt_stiffness = 0** で、代わりに筋の受動力
`fp(lce)`（lce>1＝筋が伸ばされたときだけ出る抵抗、fpmax=1.3）で代替している。
＝actuation_model を指定せずに測っていた＝**学習と違う体を測っていた**（今日4回目）。

【実装方法の判断（重要）】
生理的屈曲は**受動的な組織特性**（屈筋トーン・関節包の張力）であって、
**筋の能動的な姿勢保持ではない**。
注意：後者は 2026-07-25 に「仰臥位では姿勢筋の需要が最小」「固める方向は人間でも病的サイン
（cramped-synchronised＝CPの早期マーカー）」という2つの[Tier1]で否定され、実装が丸ごと消えた。
→ だから **springref（バネの中立位置）＋ stiffness（バネの強さ）** で表現する。
   MIMoは既に 93/96 関節に stiffness を持ち、springref も負（屈曲側）に設定してある。
   つまり「無い機構を足す」のではなく「**既にある機構の値を人間に合わせる**」作業。

【何を測るか】stiffness の文献値（N·m/rad）は存在しない（今回の調査でも出なかった）。
だから「**目標角度に落ち着く stiffness を逆算する**」＝実測から決める [Tier2]。
同時に 注意**stiffness を上げると関節が固くなって自発運動が潰れる**ので、
**ノイズを入れたときの可動範囲（ROM）も一緒に測る**＝トレードオフを見る。
（これを測らないと「屈曲は合ったが動かない体」を作ってしまう＝チェックリスト項10の型）

使い方:
    python E/scripts/e_flexion_probe.py
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

# (関節名, 目標角度[deg], 出典の強さ)
TARGETS = [
    ("hip1", -32.0, "Ishida1997 n=80 [primary,OA]"),
    ("knee", -21.0, "Broughton1993 n=57 [primary,abstract]"),
    ("elbow", -14.0, "Watanabe1979 n=62 [secondary]"),
]
SETTLE = int(os.environ.get("E_FLEX_SETTLE", "600"))   # 平衡に達するまで
ROM_STEP = int(os.environ.get("E_FLEX_ROM", "2000"))   # ROM測定のstep数
STD = float(os.environ.get("E_CORR_STD", "0.174"))
BETA = float(os.environ.get("E_BETA", "0.7"))


def joints_of(model, base):
    """left_/right_ を両方拾う（体幹側の同名関節と混ざらないよう接頭辞で限定）。"""
    out = []
    for side in ("right_", "left_"):
        try:
            j = model.joint("robot:" + side + base)
        except Exception:
            continue
        out.append((side + base, j.id, int(model.jnt_qposadr[j.id])))
    return out


def measure(spring_scale, stiff_scale, with_noise):
    """springref を目標角へ、stiffness を stiff_scale 倍にして平衡位置とROMを測る。

    spring_scale=None なら springref を触らない（現状のまま）。
    """
    from d_supine_env import SupineMimoEnv
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env
    from cpg import ColoredNoiseGenerator

    env = SupineMimoEnv(actuation_model=MuscleModel, vision_params=None, age=0.0,
                        **body_kwargs_from_env(0.0, verbose=False))
    m, d = env.unwrapped.model, env.unwrapped.data

    tracked = []
    for base, target, _ in TARGETS:
        for nm, jid, qadr in joints_of(m, base):
            if spring_scale is not None:
                # springref を目標角に置く（＝バネの中立位置を人間の実測に合わせる）
                m.qpos_spring[qadr] = np.radians(target)
            # stiffness は MuscleModel では 0 なので「倍率」では動かない。絶対値で置く。
            m.jnt_stiffness[jid] = stiff_scale
            tracked.append((nm, target, qadr))

    env.reset(seed=0)
    for _ in range(SETTLE):
        mujoco.mj_step(m, d)
    eq = {nm: float(np.degrees(d.qpos[qadr])) for nm, _, qadr in tracked}

    rom = {}
    if with_noise:
        n_act = env.action_space.shape[0]
        gen = ColoredNoiseGenerator(n_act, seed=0)
        hist = {nm: [] for nm, _, _ in tracked}
        for _ in range(ROM_STEP // 10):
            a = np.clip(0.5 + STD * gen.sample(BETA), 0.0, 1.0)
            for _ in range(10):
                env.step(a)
                for nm, _, qadr in tracked:
                    hist[nm].append(float(np.degrees(d.qpos[qadr])))
        rom = {nm: float(np.ptp(v)) for nm, v in hist.items()}
    env.close()
    return eq, rom


def main():
    print("Physiological flexion: how much stiffness reaches the literature angle?")
    print(f"  settle={SETTLE} steps, ROM measured over {ROM_STEP} steps with colored noise\n")
    for base, target, src in TARGETS:
        print(f"  target {base:<6} = {target:+6.1f} deg   [{src}]")
    print()

    # MuscleModel は jnt_stiffness=0 なので、倍率でなく【絶対値】を振る。
    #   参考：SpringDamperModel の既定は hip1=1.72 / knee=1.63 / elbow=0.13
    rows = [("current (stiffness=0)", None, 0.0)]
    for s in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]:
        rows.append((f"springref->target, stiff={s:g}", 1.0, s))

    print(f"{'condition':<34}{'hip1':>16}{'knee':>16}{'elbow':>16}")
    print(f"{'':<34}{'eq / ROM':>16}{'eq / ROM':>16}{'eq / ROM':>16}")
    print("-" * 80)
    for label, sp, st in rows:
        try:
            eq, rom = measure(sp, st, with_noise=True)
        except Exception as e:
            print(f"{label:<34}  [ERR] {type(e).__name__}: {e}")
            continue
        cells = []
        for base, target, _ in TARGETS:
            # 左右の平均
            vs = [v for k, v in eq.items() if k.endswith(base)]
            rs = [v for k, v in rom.items() if k.endswith(base)]
            e_ = float(np.mean(vs)) if vs else float("nan")
            r_ = float(np.mean(rs)) if rs else float("nan")
            mark = "*" if abs(e_ - target) <= 3.0 else " "     # 目標±3度で達成とみなす
            cells.append(f"{e_:+6.1f}{mark}/{r_:5.1f}")
        print(f"{label:<34}{cells[0]:>16}{cells[1]:>16}{cells[2]:>16}", flush=True)

    print("\n  eq = equilibrium angle after settle (deg)   ROM = range of motion with noise (deg)")
    print("  * = within 3 deg of the literature target")
    print("\n  [!] Watch the ROM column. If ROM collapses while eq reaches the target,")
    print("      we would be building a body that is correctly flexed but cannot move")
    print("      -- that is the 'cramped-synchronised' direction (abnormal in humans).")


if __name__ == "__main__":
    main()
