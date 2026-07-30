"""首を回している力の正体を、MuJoCo の力の内訳から特定する。

【前提＝e_reset_settle_probe.py で判明したこと】
  リセット直後、首（head_swivel）が 0.5秒で 65度回る。
  ・屈筋トーンを切っても回る（52.98度）
  ・★**重力を切っても回る**（47.78度 → 0.5秒で 64.98度でぴたりと止まる）
  → 重力でもバネでもない。**何かが能動的に押している**。

【MuJoCo の力の内訳】関節にかかる力は次の4つに分解できる。
  qfrc_bias        重力・コリオリ・遠心力
  qfrc_passive     バネ・減衰・**筋の受動張力**（MuscleModel は activation=0 でも
                   筋長に応じた受動力を出す＝人間の筋にもある性質）
  qfrc_actuator    アクチュエータが出している力（action=0 なら 0 のはず）
  qfrc_constraint  接触・可動域の制限・等式拘束からの反力

どれが大きいかで犯人が決まる。
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
import os, sys, warnings
warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
for p in [os.path.join(_ROOT, "D", "scripts"), os.path.join(_ROOT, "MIMo"),
          os.path.join(_ROOT, "taro_core"),
          os.path.join(_ROOT, "taro_core", "src", "body"),
          _HERE]:
    if p not in sys.path:
        sys.path.insert(0, p)
try: sys.stdout.reconfigure(errors="replace")
except Exception: pass

import numpy as np

SAMPLE_AT = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]


def main():
    from e_toy_env import ToySupineEnv, infant_vision_params
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env

    kw = body_kwargs_from_env(0.0, verbose=False)
    kw["flexion"] = True
    env = ToySupineEnv(actuation_model=MuscleModel,
                       vision_params=infant_vision_params(),
                       age=0.0, toy=True, vor=True, orient=False, **kw)
    m, d = env.unwrapped.model, env.unwrapped.data
    env.reset(seed=0)
    dt = float(m.opt.timestep) * int(env.unwrapped.frame_skip)
    n_act = env.action_space.shape[0]

    # 首の関節
    neck = []
    for j in range(m.njnt):
        nm = m.joint(j).name
        if any(k in nm for k in ("head", "neck")):
            neck.append(dict(jid=j, name=nm.replace("robot:", ""),
                             qadr=int(m.jnt_qposadr[j]),
                             dof=int(m.jnt_dofadr[j]),
                             lo=float(np.degrees(m.jnt_range[j, 0])),
                             hi=float(np.degrees(m.jnt_range[j, 1]))))

    print("=== 首の関節の可動域と初期角度 ===")
    for u in neck:
        print(f"  {u['name']:<16} 可動域 {u['lo']:>7.1f} 〜 {u['hi']:>7.1f} 度"
              f"   初期 {np.degrees(d.qpos[u['qadr']]):>7.2f} 度"
              f"   stiffness={float(m.jnt_stiffness[u['jid']]):.4f}"
              f"   springref={np.degrees(m.qpos_spring[u['qadr']]):.2f}")

    # 首を動かすアクチュエータ
    print("\n=== 首を動かすアクチュエータ ===")
    for i in range(m.nu):
        jid = int(m.actuator_trnid[i, 0])
        if jid >= 0 and any(u["jid"] == jid for u in neck):
            print(f"  {m.actuator(i).name:<22} gear={float(m.actuator_gear[i,0]):.4f}"
                  f"  ctrlrange={m.actuator_ctrlrange[i]}")

    print("\n=== 首にかかる力の内訳 [Nm]（action=0＝完全脱力）===")
    print("  bias=重力等 / passive=バネ・筋の受動張力 / actuator=筋の能動力 / constraint=接触・可動域")

    a = np.zeros(n_act, dtype=np.float32)
    picks = {round(t / dt): t for t in SAMPLE_AT}
    rows = {}

    def snap():
        return {u["name"]: dict(
            ang=float(np.degrees(d.qpos[u["qadr"]])),
            bias=float(d.qfrc_bias[u["dof"]]),
            passive=float(d.qfrc_passive[u["dof"]]),
            actuator=float(d.qfrc_actuator[u["dof"]]),
            constraint=float(d.qfrc_constraint[u["dof"]]),
        ) for u in neck}

    rows[0.0] = snap()
    for step in range(1, int(1.0 / dt) + 1):
        env.step(a)
        if step in picks:
            rows[picks[step]] = snap()

    for u in neck:
        nm = u["name"]
        print(f"\n--- {nm} ---")
        print(f"{'t[s]':>6}{'角度':>10}{'bias':>12}{'passive':>12}"
              f"{'actuator':>12}{'constraint':>13}")
        for t in SAMPLE_AT:
            r = rows[t][nm]
            print(f"{t:>6.2f}{r['ang']:>10.2f}{r['bias']:>12.5f}{r['passive']:>12.5f}"
                  f"{r['actuator']:>12.5f}{r['constraint']:>13.5f}")

    # 筋の活性化が本当に0か（action=0 でも target_activity が 0 とは限らない）
    am = getattr(env.unwrapped, "actuation_model", None)
    if am is not None and hasattr(am, "activity"):
        act = np.asarray(am.activity, dtype=float)
        print(f"\n=== 筋の活性化（action=0 のあと1秒）===")
        print(f"  最小 {act.min():.4f}  平均 {act.mean():.4f}  最大 {act.max():.4f}")
        print("  ★0でないなら、脱力の指令が筋に届いていない")

    print("\n=== 読み方 ===")
    print("  passive が大きい   → 筋の受動張力（初期姿勢が筋の中立から外れている）")
    print("  actuator が大きい  → action=0 なのに筋が力を出している＝配線の誤り")
    print("  constraint が大きい→ 可動域や接触に食い込んでいて弾かれている")
    env.close()


if __name__ == "__main__":
    main()
