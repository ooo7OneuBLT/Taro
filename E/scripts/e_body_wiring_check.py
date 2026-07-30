"""【身体の配線チェック】身体設定が"設定したつもり"で終わっていないかを機械的に検証する。

【なぜ作るか＝同じ失敗を1日に3件した（2026-07-25）】
  ① `distal_mass`（末端の質量）… 学習ループに届いておらず、感度分析4条件のモデルの
     重みが**完全一致**していた（＝10シード×4条件が無意味）
  ② `limb_scale`（四肢の筋力）… 同じく届いていなかった。★さらに届くように直しても
     **MuscleModel は毎ステップ `actuator_gear` を上書きする**（muscle.py:331）ため、
     gear を書き換える実装では**筋力が一切変わっていなかった**
     ＝「新生児は四肢が弱い」「首がすわらない」が**学習では一度も効いていなかった**
     （実際の強さは記録の 24倍（首）／54倍（四肢））
  ③ `flexion`（生理的屈曲）… Viewerに届かず、ユーザーの目視
     「ひじは曲がってるけど**ひざはほとんど曲がってない**」で発覚

★共通点：**ログには出るのに実効がない**。だから「ログを見た」では検証にならない。

【この検査の考え方】
設定を**2つの値で作って物理量を比べる**。変わらなければ配線が死んでいる。
＝チェックリスト項17（入力を揺らして出力が動くか）を身体設定に対して機械化したもの。
⚠️学習を伴う実験では `E/scripts/e_condition_check.py`（モデルの重みを比較）と**対**で使う。
  こちらは「物理が変わるか」、あちらは「学習が変わるか」。

使い方:
    python E/scripts/e_body_wiring_check.py
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
          os.path.join(_ROOT, "taro_core", "src", "body")]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

import numpy as np  # noqa: E402
import mujoco  # noqa: E402

_RESULTS = []


def check(name, ok, detail=""):
    _RESULTS.append((name, bool(ok)))
    print(f"  [{'OK ' if ok else 'NG!'}] {name}" + (f"   {detail}" if detail else ""))


def build(**over):
    """身体設定を上書きして環境を作る（筋肉モデル＝学習と同じ条件）。"""
    from d_supine_env import SupineMimoEnv
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env
    kw = body_kwargs_from_env(0.0, verbose=False)
    kw.update(over)
    env = SupineMimoEnv(actuation_model=MuscleModel, vision_params=None, age=0.0, **kw)
    return env


def probe(env):
    """物理量を取り出す（配線が効いていれば、設定を変えるとどれかが動く）。"""
    from infant_body import actuator_strength
    m, d = env.unwrapped.model, env.unwrapped.data
    am = getattr(env.unwrapped, "actuation_model", None)
    env.reset(seed=0)
    for _ in range(600):
        mujoco.mj_step(m, d)          # 平衡まで落ち着かせる（筋活性化ゼロ）
    q = lambda nm: float(np.degrees(d.qpos[m.jnt_qposadr[m.joint("robot:" + nm).id]]))
    s = lambda nm: actuator_strength(m, mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_ACTUATOR, nm), am)
    hand = [b for b in range(m.nbody) if "right_hand" in (m.body(b).name or "")]
    return dict(
        hip=q("right_hip1"), knee=q("right_knee"), elbow=q("right_elbow"),
        leg_strength=s("act:right_hip_flex"), neck_strength=s("act:head_tilt"),
        hand_mass=float(sum(m.body_mass[b] for b in hand)),
        total_mass=float(m.body_mass.sum()),
        head_size=float(np.max(m.geom("head").size)),
        stiff_hip=float(m.jnt_stiffness[m.joint("robot:right_hip1").id]),
    )


def main():
    print("=== 身体の配線チェック（設定を振って物理量が動くか）===")
    print("  ⚠️ログが出るだけでは検証にならない。MuscleModel は gear を毎ステップ上書きする\n")

    base = probe(build())
    print("  基準（現在の既定設定）:")
    for k, v in base.items():
        print(f"      {k:<14} {v:.4f}")
    print()

    # (表示名, 上書きする設定, 動くべき物理量, 期待する最小の変化)
    cases = [
        ("四肢の筋力 limb_scale", {"limb_scale": 4.0}, "leg_strength", 1e-6),
        ("四肢の補正 limb_fix", {"limb_fix": False}, "leg_strength", 1e-6),
        ("末端の質量 distal_mass", {"distal_mass": 5.0}, "hand_mass", 1e-6),
        ("生理的屈曲 flexion（膝）", {"flexion": True}, "knee", 1.0),
        ("生理的屈曲 flexion（股）", {"flexion": True}, "hip", 1.0),
        ("屈曲の強さ flexion_stiffness", {"flexion": True, "flexion_stiffness": 16.0},
         "stiff_hip", 1e-6),
        ("頭の楕円化 head_elongation", {"head_elongation": 1.5}, "head_size", 1e-6),
    ]
    for label, over, key, eps in cases:
        try:
            v = probe(build(**over))
        except Exception as e:
            check(label, False, f"例外 {type(e).__name__}: {e}")
            continue
        diff = abs(v[key] - base[key])
        check(label, diff > eps, f"{key}: {base[key]:.4f} -> {v[key]:.4f} (差 {diff:.4f})")

    # ★首の補正が「効いた結果」を持続するか（gear上書き問題の直接検査）
    print("\n  --- ★gearの上書き検査（MuscleModelの罠） ---")
    env = build()
    m, d = env.unwrapped.model, env.unwrapped.data
    am = getattr(env.unwrapped, "actuation_model", None)
    from infant_body import actuator_strength, scale_actuator_strength
    aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "act:right_hip_flex")
    before = actuator_strength(m, aid, am)
    scale_actuator_strength(m, aid, 0.1, am)
    just_after = actuator_strength(m, aid, am)
    env.reset(seed=0)
    for _ in range(50):
        env.step(np.full(env.action_space.shape[0], 0.5))   # 実際にstepを回す
    after_steps = actuator_strength(m, aid, am)
    check("筋力の変更がstep後も残る", abs(after_steps - just_after) < 1e-6 * max(1.0, abs(just_after)),
          f"{before:.3f} -> 書換直後 {just_after:.3f} -> 50step後 {after_steps:.3f}")
    env.close()

    ng = [n for n, ok in _RESULTS if not ok]
    print(f"\n  結果: {len(_RESULTS) - len(ng)}/{len(_RESULTS)} 通過")
    if ng:
        print("  ★NG:", " / ".join(ng))
        print("  → 設定が物理に届いていない。呼び出し側が body_kwargs_from_env を"
              "使っているか、MuscleModel が上書きしていないかを確認する")
    return 1 if ng else 0


if __name__ == "__main__":
    sys.exit(main())
