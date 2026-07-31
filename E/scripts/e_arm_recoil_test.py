"""屈筋トーン（バネ）の強さを、肘の arm recoil（腕の反跳）で較正する。

【なぜこのテストか】屈筋トーンのパラメータのうち、**バネの強さだけは文献の実測値で
決められる**。他（股・膝の目標角）は安静時の一次文献が存在せず目視に頼るしかないが、
肘の反跳は測定値がある：

    Farmania et al. (2017) J Neurosci Rural Pract 8(3) [PMC5602260]
    満期産 n=74 の arm recoil = **105.3 ± 14.2 度**
    （肘を伸展させて離すと、この角度まで戻る）

【手順】
  1. 肘を伸ばした状態にする（他の関節はそのまま）
  2. 離す（脱力＝行動ゼロ）
  3. 何度で止まるか測る
  4. 105.3度に近くなるバネの強さを探す

注意：これは「バネが伸張反射と同じ機構か」を検証するものではない（機構は別物＝近似）。
  「反跳の結果が実測に合うか」だけを見る。

使い方:
    .venv/Scripts/python.exe E/scripts/e_arm_recoil_test.py
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
import os, sys, warnings
warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
for p in [os.path.join(_ROOT, "D", "scripts"), os.path.join(_ROOT, "MIMo"),
          os.path.join(_ROOT, "taro_core"),
          os.path.join(_ROOT, "taro_core", "src", "body"), _HERE]:
    if p not in sys.path:
        sys.path.insert(0, p)
try: sys.stdout.reconfigure(errors="replace")
except Exception: pass

import numpy as np

TARGET_RECOIL = 105.3      # Farmania 2017（満期産 n=74）
TARGET_SD = 14.2
STIFFNESS_LEVELS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
SETTLE_SEC = 3.0           # 離してから測るまでの時間


def run(stiffness):
    from d_supine_env import SupineMimoEnv
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env
    from infant_body import apply_flexor_tone

    kw = body_kwargs_from_env(0.0, verbose=False)
    kw["flexion"] = True
    env = SupineMimoEnv(actuation_model=MuscleModel, vision_params=None,
                        age=0.0, **kw)
    m, d = env.unwrapped.model, env.unwrapped.data
    # 強さを上書きして適用し直す
    apply_flexor_tone(m, 0.0, stiffness=stiffness, verbose=False, data=d)
    env.reset(seed=0)

    qadr = {}
    for side in ("right_", "left_"):
        j = m.joint("robot:" + side + "elbow")
        qadr[side] = int(m.jnt_qposadr[j.id])

    n_act = env.action_space.shape[0]
    zero = np.zeros(n_act, dtype=np.float32)
    dt = float(m.opt.timestep) * int(env.unwrapped.frame_skip)

    # ① 肘を伸ばす（可動域の伸展側いっぱい）
    for side, qa in qadr.items():
        jid = m.joint("robot:" + side + "elbow").id
        d.qpos[qa] = float(m.jnt_range[jid, 1])       # 伸展側の限界
    ext = float(np.degrees(d.qpos[qadr["right_"]]))

    # ② 離す（脱力のまま経過させる）
    trace = []
    for _ in range(int(SETTLE_SEC / dt)):
        env.step(zero)
        trace.append(float(np.degrees(d.qpos[qadr["right_"]])))
    env.close()

    trace = np.array(trace)
    final = float(np.mean(trace[-20:]))
    # 反跳角＝伸展位から何度戻ったか（絶対値で「曲がった量」）
    recoil = abs(final)
    return dict(ext=ext, final=final, recoil=recoil,
                peak=float(np.abs(trace).max()), trace=trace)


def main():
    print("=== 肘の arm recoil で屈筋トーンのバネを較正する ===")
    print(f"  目標: {TARGET_RECOIL} ± {TARGET_SD} 度")
    print(f"        Farmania et al. 2017 [PMC5602260] 満期産 n=74")
    print(f"  手順: 肘を伸展位にして離し、{SETTLE_SEC}秒後の角度を測る\n")

    print(f"{'stiffness':>10}{'伸展位':>10}{'落ち着いた角度':>16}"
          f"{'反跳量':>10}{'目標との差':>12}  判定")
    print("-" * 76)
    best, best_err = None, 1e9
    for k in STIFFNESS_LEVELS:
        try:
            r = run(k)
        except Exception as e:
            print(f"{k:>10.3f}  [ERR] {type(e).__name__}: {e}")
            continue
        err = abs(r["recoil"] - TARGET_RECOIL)
        inside = err <= TARGET_SD
        if err < best_err:
            best, best_err = k, err
        print(f"{k:>10.3f}{r['ext']:>10.1f}{r['final']:>16.1f}"
              f"{r['recoil']:>10.1f}{r['recoil']-TARGET_RECOIL:>+12.1f}"
              f"  {'○ 範囲内' if inside else '× 範囲外'}")

    print(f"\n  最も近い強さ: stiffness = {best}（差 {best_err:.1f}度）")
    if best_err <= TARGET_SD:
        print(f"  → 実測の 1SD（{TARGET_SD}度）以内。この値を採用できる")
    else:
        print(f"  注意 どの水準も 1SD 以内に入らない。刻みを細かくするか、"
              f"バネ以外の要因（重力・他の関節の影響）を疑う")


if __name__ == "__main__":
    main()
