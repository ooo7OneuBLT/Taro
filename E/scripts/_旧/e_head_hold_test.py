"""親が頭を支えたとき、本当に頭が固定されるかを測る。

【なぜ要るか、2026-07-28】視線誘導反射の実験が成立していなかった原因は、太郎の首が
倒れて対象を視野に捉え続けられないことだった。人間の乳児実験（Hunter & Richards 2003）
では**実験者が頭を支えている**ので、太郎も同じ条件にする。

その「支え」が効いているかを、反射を回す前に単体で確かめる。
＝落とし穴チェックリスト項62「設定した値が本当に体に届いているか確認する」。

【測ること】
  1. 支えなし   … 何秒でどれだけ頭が倒れるか（＝これまでの状態）
  2. B-1 firm   … ほぼ完全固定。ずれが何度に収まるか
  3. B-2 soft   … 柔らかく支える。ずれが何度になるか
  いずれも**完全脱力（action=0）**で測る。太郎が自分で首を動かす分は含めない。

使い方:
    .venv/Scripts/python.exe E/scripts/e_head_hold_test.py
    E_SECONDS=30 .venv/Scripts/python.exe E/scripts/e_head_hold_test.py
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

import numpy as np  # noqa: E402

AGE = float(os.environ.get("E_AGE", "0"))
SECONDS = float(os.environ.get("E_SECONDS", "20"))
SAMPLE_AT = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0]


def run(mode):
    """mode: 'none'（支えなし）/ 'firm'（B-1）/ 'soft'（B-2）"""
    from d_supine_env import SupineMimoEnv
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env
    from e_head_hold import CaregiverHands, HOLD_STIFFNESS_FIRM, HOLD_STIFFNESS_SOFT

    kw = body_kwargs_from_env(AGE, verbose=False)
    kw["flexion"] = True
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        env = SupineMimoEnv(vision_params=None, age=AGE,
                            actuation_model=MuscleModel, **kw)
        env.reset(seed=0)
    m, d = env.unwrapped.model, env.unwrapped.data
    dt = float(m.opt.timestep) * int(env.unwrapped.frame_skip)
    n_act = env.action_space.shape[0]

    hands = CaregiverHands(m, d)
    if mode == "firm":
        hands.hold(stiffness=HOLD_STIFFNESS_FIRM)
    elif mode == "soft":
        hands.hold(stiffness=HOLD_STIFFNESS_SOFT)

    start = hands.head_angles()
    a = np.zeros(n_act, dtype=np.float32)
    picks = {round(t / dt): t for t in SAMPLE_AT if t <= SECONDS}
    rows = {0.0: dict(ang=dict(start), off=hands.offsets())}
    blew_up = False
    for step in range(1, int(SECONDS / dt) + 1):
        env.step(a)
        if not np.all(np.isfinite(d.qpos)):
            blew_up = True
            break
        if step in picks:
            rows[picks[step]] = dict(ang=hands.head_angles(), off=hands.offsets())
    env.close()
    return dict(mode=mode, start=start, rows=rows, dt=dt, blew_up=blew_up)


def main():
    print("=" * 76)
    print(f" 親が頭を支えたとき本当に固定されるか（{AGE:g}ヶ月・完全脱力・{SECONDS:g}秒）")
    print("=" * 76)
    outs = []
    for mode in ("none", "firm", "soft"):
        label = {"none": "支えなし（これまでの状態）",
                 "firm": "B-1 ほぼ完全固定", "soft": "B-2 柔らかく支える"}[mode]
        print(f"  [{label}...]")
        outs.append(run(mode))

    for o in outs:
        label = {"none": "支えなし", "firm": "B-1 firm", "soft": "B-2 soft"}[o["mode"]]
        print("\n" + "=" * 76)
        print(f"{label}   （制御周期 {o['dt']*1000:.1f}ms）")
        if o["blew_up"]:
            print("  注意発散した（qpos が有限でなくなった）＝バネが強すぎる")
        print("-" * 76)
        keys = list(o["start"].keys())
        print(f"{'t[s]':>6}" + "".join(f"{k:>18}" for k in keys))
        for t in sorted(o["rows"]):
            r = o["rows"][t]
            line = f"{t:>6.1f}"
            for k in keys:
                line += f"{r['ang'][k]:>11.2f}度"
            print(line)
        # 目標からのずれ（支えているときだけ意味がある）
        if o["mode"] != "none":
            last = o["rows"][max(o["rows"])]
            worst = max(abs(v) for v in last["off"].values())
            print(f"  → 目標角からのずれ（最後）: " +
                  " ".join(f"{k}{v:+.2f}度" for k, v in last["off"].items()))
            print(f"  → 最大のずれ {worst:.2f}度")

    # まとめ
    print("\n" + "=" * 76)
    print("まとめ：頭がどれだけ動いたか（開始からの変化量の最大）")
    print("-" * 76)
    print(f"{'条件':<14}" + "".join(f"{k:>18}" for k in outs[0]["start"].keys()))
    for o in outs:
        label = {"none": "支えなし", "firm": "B-1 firm", "soft": "B-2 soft"}[o["mode"]]
        last = o["rows"][max(o["rows"])]
        line = f"{label:<14}"
        for k in o["start"]:
            line += f"{last['ang'][k] - o['start'][k]:>+11.2f}度"
        print(line)

    print("\n" + "=" * 76)
    print("読み方")
    print("-" * 76)
    print("  ・支えなしで大きく動き、firm でほぼ動かなければ、支えは効いている")
    print("  ・firm でも動くなら、バネが弱いか、別の力（筋の受動張力など）が勝っている")
    print("  ・発散したらバネが強すぎる。制御周期に対して硬すぎると陽的積分が破綻する")


if __name__ == "__main__":
    main()
