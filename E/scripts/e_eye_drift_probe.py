"""眼球だけが下に落ちる問題の原因を切り分ける。

【目視で分かったこと（2026-07-26、Viewer）】
  ・頭はまったく動かない。手で動かしても戻らない＝首は問題ない。
  ・眼球だけが下に引っ張られる。手で上に向けてもすぐ下に戻る。
  → 直前まで疑っていた「屈筋トーンのバネ→頭が揺れる→VORが飽和」という筋は**外れ**。
     頭が揺れていないのだから、頭の角速度を打ち消すVORの本体は何もしていない。

【切り分けたい候補】
  ①重力：眼球ボディに質量があり、重力トルクで下へ落ちる。
     VORは「速度をゼロにする」制御なので、定常的な力に対して**位置を保持できない**。
     人間には神経積分器（velocity-to-position integrator, 前庭核・NPH）があり
     これが眼の位置を保持している。太郎にはそれが無い。
  ②VORの出力そのものが下向きに偏っている（三半規管や耳石器のベースラインのずれ）。
  ③関節のバネ/減衰の設定（qpos_spring が下向きにある等）。

【測り方】
  action=0（完全脱力）で 10 秒。以下の4条件で眼球関節の角度を追う。
      A: VOR OFF・重力あり
      B: VOR ON ・重力あり
      C: VOR OFF・重力なし
      D: VOR ON ・重力なし
  ・Aで落ちる → ①か③（身体側の問題。VORは無関係）
  ・Aで落ちずBで落ちる → ②（VORの出力が偏っている）
  ・Cで落ちない → ①（重力が原因）
  ・Cでも落ちる → ③（バネ等の設定）
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
          os.path.join(_ROOT, "taro_core", "src", "body"),
          _HERE]:
    if p not in sys.path:
        sys.path.insert(0, p)
try: sys.stdout.reconfigure(errors="replace")
except Exception: pass

import numpy as np

SEC = 10.0
SAMPLE_AT = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]


def eye_joints(m):
    out = []
    for j in range(m.njnt):
        name = m.joint(j).name
        if "eye" in name:
            out.append(dict(jid=j, name=name,
                            qadr=int(m.jnt_qposadr[j]),
                            dofadr=int(m.jnt_dofadr[j]),
                            lo=float(np.degrees(m.jnt_range[j, 0])),
                            hi=float(np.degrees(m.jnt_range[j, 1]))))
    return out


def static_report(m, d, joints):
    """身体側の設定（質量・バネ・減衰・重力トルク）を出す。"""
    print("=== 眼球まわりの身体設定 ===")
    print(f"{'関節':<26}{'可動域(deg)':>16}{'stiffness':>11}{'springref':>11}"
          f"{'damping':>10}{'armature':>10}")
    print("-" * 84)
    for u in joints:
        j = u["jid"]
        print(f"{u['name']:<26}{u['lo']:>7.1f}〜{u['hi']:>6.1f}"
              f"{float(m.jnt_stiffness[j]):>11.4f}"
              f"{float(np.degrees(m.qpos_spring[u['qadr']])):>11.2f}"
              f"{float(m.dof_damping[u['dofadr']]):>10.4f}"
              f"{float(m.dof_armature[u['dofadr']]):>10.5f}")

    bids = sorted({int(m.jnt_bodyid[u["jid"]]) for u in joints})
    print("\n  眼球ボディの質量:")
    for b in bids:
        print(f"    {m.body(b).name:<20} mass={float(m.body_mass[b]):.6f} kg")

    # qfrc_bias ＝ 重力・コリオリ・遠心力をまとめた項（＝外部から関節にかかる力）
    print("\n  qfrc_bias（重力などが関節にかける力 [Nm]。正負が向き）:")
    for u in joints:
        print(f"    {u['name']:<26}{float(d.qfrc_bias[u['dofadr']]):>12.6f}")


def run_case(env, joints, vor_on, gravity_on, dt, n_act):
    m, d = env.unwrapped.model, env.unwrapped.data
    saved_g = np.array(m.opt.gravity, dtype=float).copy()
    env.reset(seed=0)
    if not gravity_on:
        m.opt.gravity[:] = 0.0
    else:
        m.opt.gravity[:] = saved_g

    vor = env.unwrapped._vor
    env.unwrapped._vor = vor if vor_on else None
    if vor_on and vor is not None:
        vor.reset()

    a = np.zeros(n_act, dtype=np.float32)
    n = int(SEC / dt)
    picks = {round(t / dt): t for t in SAMPLE_AT}
    trace = {}
    trace[0.0] = {u["name"]: float(np.degrees(d.qpos[u["qadr"]])) for u in joints}
    for step in range(1, n + 1):
        env.step(a)
        if step in picks:
            trace[picks[step]] = {u["name"]: float(np.degrees(d.qpos[u["qadr"]]))
                                  for u in joints}
    m.opt.gravity[:] = saved_g
    env.unwrapped._vor = vor
    return trace


def main():
    from e_toy_env import ToySupineEnv, infant_vision_params
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env

    age = 0.0
    kw = body_kwargs_from_env(age, verbose=False)
    # VORは作らせておいて、条件ごとに付け外しする（環境の作り直しを避ける＝メモリ対策）
    env = ToySupineEnv(actuation_model=MuscleModel,
                       vision_params=infant_vision_params(),
                       age=age, vor=True, orient=False, **kw)
    m, d = env.unwrapped.model, env.unwrapped.data
    env.reset(seed=0)
    dt = float(m.opt.timestep) * int(env.unwrapped.frame_skip)
    n_act = env.action_space.shape[0]
    joints = eye_joints(m)

    static_report(m, d, joints)

    cases = [("A  VOR OFF / 重力あり", False, True),
             ("B  VOR ON  / 重力あり", True, True),
             ("C  VOR OFF / 重力なし", False, False),
             ("D  VOR ON  / 重力なし", True, False)]

    results = {}
    for label, vor_on, grav_on in cases:
        results[label] = run_case(env, joints, vor_on, grav_on, dt, n_act)

    print(f"\n\n=== 眼球の角度の推移（action=0＝完全脱力、単位 deg）===")
    for u in joints:
        name = u["name"]
        print(f"\n--- {name}（可動域 {u['lo']:.0f}〜{u['hi']:.0f} deg）---")
        print(f"{'条件':<24}" + "".join(f"{t:>10.1f}s" for t in SAMPLE_AT))
        print("-" * (24 + 11 * len(SAMPLE_AT)))
        for label, _, _ in cases:
            row = "".join(f"{results[label][t][name]:>11.2f}" for t in SAMPLE_AT)
            print(f"{label:<24}{row}")

    print("\n=== 読み方 ===")
    print("  Aで下がる          → 身体側の問題（重力かバネ）。VORは無関係。")
    print("  Aは平ら・Bで下がる → VORの出力が下向きに偏っている。")
    print("  Cで平らになる      → 原因は重力（＝眼の位置を保つ機構が無い）。")
    print("  Cでも下がる        → 原因はバネ等の設定。")
    env.close()


if __name__ == "__main__":
    main()
