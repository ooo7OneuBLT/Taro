"""リクライニングで体がどう動くかを時系列で見る（姿勢が保てない原因の切り分け）。

【症状、2026-07-28】背もたれを70度で作ったのに、4秒後の体幹の傾きが **-40.4度**
（マイナス＝逆向き）になった。45度でも 7.8度 しか起きず、1.56cm ずり落ちている。
＝「置いた瞬間の姿勢」と「落ち着いた姿勢」が違う。どこで崩れるかを見る。

【見ること】
  ・初期姿勢（reset直後）は設定どおりか  → 違えば置き方（_apply_recline）の問題
  ・時間とともにどう変わるか              → 崩れるなら支えの問題
  ・背もたれ／座面と接触しているか        → していなければ空中や別の場所にいる

使い方:
    E_RECLINE=70 .venv/Scripts/python.exe E/scripts/e_recline_debug.py
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

AGE = float(os.environ.get("E_AGE", "4.0"))
REC = float(os.environ.get("E_RECLINE", "70"))
SECONDS = float(os.environ.get("E_SECONDS", "5.0"))
SAMPLE = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0]


def main():
    from e_toy_env import ToySupineEnv, infant_vision_params, SEAT_HALF_LEN, SEAT_HALF_WID
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env

    kw = body_kwargs_from_env(AGE, verbose=False)
    kw["flexion"] = True
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        env = ToySupineEnv(actuation_model=MuscleModel,
                           vision_params=infant_vision_params(acuity_age=AGE),
                           age=AGE, toy=True, vor=True, orient=False,
                           recline_deg=REC, **kw)
        env.reset(seed=0)
    u = env.unwrapped
    m, d = u.model, u.data
    dt = float(m.opt.timestep) * int(u.frame_skip)
    a = np.zeros(env.action_space.shape[0], dtype=np.float32)

    print("=" * 78)
    print(f" リクライニング {REC:g}度 で体がどう動くか（体年齢 {AGE:g}ヶ月）")
    print("=" * 78)
    # 背もたれ・座面の位置を出す
    for nm in ("recline_back", "recline_seat"):
        try:
            gid = int(m.geom(nm).id)
            print(f"  {nm:<16} 位置 {np.array(m.geom_pos[gid]).round(3)}"
                  f"  大きさ {np.array(m.geom_size[gid]).round(3)}")
        except Exception:
            print(f"  {nm:<16} 存在しない")

    def bpos(n):
        return np.array(d.xpos[int(m.body(n).id)], dtype=float)

    def snap():
        head, hip = bpos("head"), bpos("hip")
        v = head - hip
        trunk = float(np.degrees(np.arctan2(v[2], np.linalg.norm(v[:2]))))
        # 接触の内訳
        touch = {}
        for c in range(d.ncon):
            con = d.contact[c]
            n1 = m.geom(con.geom1).name or "?"
            n2 = m.geom(con.geom2).name or "?"
            for nm in ("recline_back", "recline_seat", "floor"):
                if nm in n1 or nm in n2:
                    touch[nm] = touch.get(nm, 0) + 1
        return dict(trunk=trunk, hip=hip.copy(), head=head.copy(), touch=touch,
                    ncon=int(d.ncon))

    picks = {round(t / dt): t for t in SAMPLE if t <= SECONDS}
    rows = {0.0: snap()}
    for step in range(1, int(SECONDS / dt) + 1):
        env.step(a)
        if step in picks:
            rows[picks[step]] = snap()

    print("\n" + "=" * 78)
    print(" 時系列")
    print("=" * 78)
    print(f"{'t[s]':>6}{'体幹の傾き':>12}{'骨盤の位置(x,z)':>22}{'頭の位置(x,z)':>22}"
          f"{'接触':>8}")
    for t in sorted(rows):
        r = rows[t]
        tc = " ".join(f"{k.replace('recline_','')}×{v}" for k, v in r["touch"].items())
        print(f"{t:>6.1f}{r['trunk']:>11.1f}度"
              f"({r['hip'][0]:>+7.3f},{r['hip'][2]:>+7.3f})"
              f"      ({r['head'][0]:>+7.3f},{r['head'][2]:>+7.3f})      "
              f"{tc if tc else '(なし)'}")

    print("\n" + "=" * 78)
    print(" 読み方")
    print("=" * 78)
    print(f"  ・t=0 の体幹の傾きが {REC:g}度 に近くなければ、**置き方**が違っている")
    print("  ・時間とともに崩れるなら、**支えが足りない**（背もたれから落ちている）")
    print("  ・背もたれと接触していなければ、そもそも板に乗っていない")
    env.close()


if __name__ == "__main__":
    main()
