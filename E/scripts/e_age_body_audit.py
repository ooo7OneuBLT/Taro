"""月齢を変えたときに太郎の身体（質量・筋力・バネ）がどう変わるかを実測する。

【なぜ要るか、2026-07-28】首座り（運動発達ロードマップ段階2）に着手するにあたり
体年齢を4ヶ月に上げる方針になった。ところが太郎の身体補正には**月齢の期限**があり、
    首の筋力を弱める      … 4ヶ月で解除（infant_neck.HEAD_CONTROL_AGE）
    頭の質量を25%にする   … 4ヶ月で解除（infant_body.HEAD_MASS_UNTIL_MO）
    首のバネ（筋緊張）    … 3ヶ月で解除（infant_body.TONE_UNTIL_MO）
    生理的屈曲・屈筋トーン… 3ヶ月で解除（同上）
    四肢の筋力を弱める    … 18ヶ月まで継続（infant_limbs.REFERENCE_AGE）
判定は `age >= 期限` なので **4.0 を指定すると首関連が一斉に落ちる**。
＝「4ヶ月の太郎」は補正の谷間に落ちる可能性がある。それを数値で確かめる。

【何を測るか】月齢ごとに環境を1つ作り、
  1. 全体重・頭の質量・頭が占める割合
  2. 首の持ち上げ能力比 = 首の筋力 ÷ 頭の重力モーメント（<1 なら頭を持ち上げられない）
  3. 首のバネの強さと目標角
  4. 代表的な四肢の「筋力 ÷ 自重」比
  5. どの補正が効いていて、どれが解除されたか

⚠️筋力は `actuator_gear` ではなく `actuation_model.fmax` から読む。MuscleModel は
  毎ステップ gear を上書きするので、gear を読むと「今出しているトルク」を読んでしまう
  （チェックリスト項43）。読み取りは infant_body.actuator_strength に一本化してある。

使い方:
    .venv/Scripts/python.exe E/scripts/e_age_body_audit.py
    E_AGES=0,3,4 .venv/Scripts/python.exe E/scripts/e_age_body_audit.py
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
import contextlib
import io

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
for _p in [os.path.join(_ROOT, "D", "scripts"), os.path.join(_ROOT, "MIMo"),
           os.path.join(_ROOT, "taro_core"),
           os.path.join(_ROOT, "taro_core", "src", "body"), _HERE]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

import numpy as np  # noqa: E402

AGES = [float(x) for x in os.environ.get("E_AGES", "0,2,3,4,6").split(",")]
G = 9.81

# 代表として見る関節（左右対称なので左側だけ）
LIMB_KEYS = ["act:left_shoulder_horizontal", "act:left_elbow",
             "act:left_hip1", "act:left_knee"]


def _descendants(model, bid):
    ids = []
    for b in range(model.nbody):
        p = b
        while p != 0:
            if p == bid:
                ids.append(b)
                break
            p = int(model.body_parentid[p])
    return ids


def measure(age):
    """月齢 age の太郎を1体つくって測る。"""
    from d_supine_env import SupineMimoEnv
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env
    from infant_body import actuator_strength
    from infant_neck import lift_ratio, head_gravity_torque

    kw = body_kwargs_from_env(age, verbose=False)
    kw["flexion"] = True          # 屈曲・屈筋トーンも込みで見る（月齢で自動解除される）
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        env = SupineMimoEnv(vision_params=None, age=age,
                            actuation_model=MuscleModel, **kw)
        env.reset(seed=0)
    log = buf.getvalue()
    m, d = env.unwrapped.model, env.unwrapped.data
    am = getattr(env.unwrapped, "actuation_model", None)

    # --- 質量 ---
    skip = ("floor", "wall", "fence", "object", "toy", "target", "world")
    ids = [i for i in range(1, m.nbody)
           if not any(k in m.body(i).name for k in skip)]
    total = float(sum(m.body_mass[i] for i in ids))
    head_ids = _descendants(m, int(m.body("head").id))
    head_mass = float(sum(m.body_mass[i] for i in head_ids))

    # --- 首 ---
    ratio, neck_strength, head_torque = lift_ratio(m, d, am)
    _, _, arm = head_gravity_torque(m, d)

    # 首のバネ
    neck_spring = {}
    for j in range(m.njnt):
        short = m.joint(j).name.split(":")[-1]
        if short.startswith("head"):
            adr = int(m.jnt_qposadr[j])
            dof = int(m.jnt_dofadr[j])
            neck_spring[short] = dict(
                k=float(m.jnt_stiffness[j]),
                target=float(np.degrees(m.qpos_spring[adr])),
                damping=float(m.dof_damping[dof]),
                angle=float(np.degrees(d.qpos[adr])),
            )

    # --- 四肢 ---
    limbs = {}
    for i in range(m.nu):
        name = m.actuator(i).name
        if name not in LIMB_KEYS:
            continue
        jid = int(m.actuator_trnid[i, 0])
        if jid < 0:
            continue
        dids = _descendants(m, int(m.jnt_bodyid[jid]))
        mass = float(sum(m.body_mass[k] for k in dids))
        com = sum(m.body_mass[k] * d.xipos[k] for k in dids) / max(mass, 1e-12)
        a = float(np.linalg.norm((com - d.xanchor[jid])[:2]))
        tau = mass * G * a
        s = actuator_strength(m, i, am)
        limbs[name] = dict(strength=s, tau=tau, ratio=s / max(tau, 1e-9))

    env.close()
    return dict(age=age, total=total, head_mass=head_mass,
                head_frac=head_mass / max(total, 1e-12),
                neck_ratio=ratio, neck_strength=neck_strength,
                head_torque=head_torque, head_arm=arm,
                neck_spring=neck_spring, limbs=limbs, log=log)


def main():
    print("=" * 78)
    print(" 月齢ごとの身体の実測（首座りに向けた事前確認・2026-07-28）")
    print("=" * 78)
    rows = []
    for age in AGES:
        print(f"\n[{age:g}ヶ月をつくって測っています…]")
        rows.append(measure(age))

    # --- 1. 質量 ---
    print("\n" + "=" * 78)
    print("1. 質量")
    print("-" * 78)
    print(f"{'月齢':>6}{'全体重[kg]':>12}{'頭[kg]':>10}{'頭の割合':>10}   人間の目安")
    human_w = {0: "3.5kg", 2: "5.6kg", 3: "6.4kg", 4: "7.0kg", 6: "7.9kg"}
    for r in rows:
        print(f"{r['age']:>6g}{r['total']:>12.3f}{r['head_mass']:>10.3f}"
              f"{r['head_frac']*100:>9.1f}%   "
              f"{human_w.get(int(r['age']), '?')}（WHO中央値・男児）")

    # --- 2. 首 ---
    print("\n" + "=" * 78)
    print("2. 首（頭を持ち上げられるか）")
    print("-" * 78)
    print("   持ち上げ能力比 = 首の筋力 ÷ 頭の重力モーメント")
    print("   1.0 未満 = 頭を持ち上げられない（新生児の head lag）")
    print()
    print(f"{'月齢':>6}{'首の筋力':>12}{'頭の負荷':>12}{'比':>9}{'腕[cm]':>9}   判定")
    for r in rows:
        v = "持ち上げられない" if r["neck_ratio"] < 1.0 else "持ち上げられる"
        print(f"{r['age']:>6g}{r['neck_strength']:>12.3f}{r['head_torque']:>12.4f}"
              f"{r['neck_ratio']:>9.2f}{r['head_arm']*100:>9.2f}   {v}")

    # --- 3. 首のバネ ---
    print("\n" + "=" * 78)
    print("3. 首のバネ（筋緊張）— これが無いと重力で頭が倒れ続ける")
    print("-" * 78)
    keys = sorted({k for r in rows for k in r["neck_spring"]})
    for key in keys:
        print(f"\n  --- {key} ---")
        print(f"{'月齢':>6}{'バネの強さ':>12}{'目標角[度]':>12}{'減衰':>10}{'初期角[度]':>12}")
        for r in rows:
            s = r["neck_spring"].get(key)
            if s is None:
                continue
            mark = "  ← バネなし" if s["k"] <= 0 else ""
            print(f"{r['age']:>6g}{s['k']:>12.3f}{s['target']:>12.1f}"
                  f"{s['damping']:>10.4f}{s['angle']:>12.2f}{mark}")

    # --- 4. 四肢 ---
    print("\n" + "=" * 78)
    print("4. 四肢の「筋力 ÷ 自重」比")
    print("-" * 78)
    names = sorted({k for r in rows for k in r["limbs"]})
    print(f"{'関節':<28}" + "".join(f"{r['age']:>9g}mo" for r in rows))
    for nm in names:
        line = f"{nm.replace('act:', ''):<28}"
        for r in rows:
            v = r["limbs"].get(nm)
            line += f"{v['ratio']:>11.2f}" if v else f"{'-':>11}"
        print(line)

    # --- 5. どの補正が効いたか ---
    print("\n" + "=" * 78)
    print("5. 月齢ごとに、どの補正が効いたか（環境構築時のログ）")
    print("-" * 78)
    for r in rows:
        print(f"\n--- {r['age']:g}ヶ月 ---")
        for line in r["log"].splitlines():
            if any(t in line for t in ("[head]", "[neck", "[limbs]", "[flexion",
                                       "[tone]", "[distal]", "[body]")):
                print("  " + line.strip())

    print("\n" + "=" * 78)
    print("読み方")
    print("-" * 78)
    print("  ・首の比が急に跳ね上がる月齢 = 補正が解除された地点")
    print("  ・首のバネが 0 になる月齢 = 頭を支えるものが無くなる地点")
    print("  ・体重が人間の目安から大きく外れていたら、体型補正が新生児用のまま")
    print("    （NEWBORN_SHAPE_DEFAULTS は 0ヶ月に合わせて決めた係数）")


if __name__ == "__main__":
    main()
