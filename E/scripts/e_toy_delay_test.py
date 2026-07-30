"""おもちゃの登場を遅らせる案（ユーザー提案 2026-07-26）の効果を測る。

【案】最初はおもちゃを遠くに置き、しばらくしてから規定の位置へ運んでくる。

【何が直るはずか】
  ・リセット直後の太郎は落ち着いておらず、規定位置に置いたおもちゃが顔に
    15.6mm めり込んで拘束反力 1961 Nm を生み、0.4秒で首を64度回していた
  → 落ち着いてから置けば、めり込まないはず

【★確かめる懸念】私は「1秒待っても同じ位置に出せばやはりめり込む」と予想した。
  太郎の顔はそこにあるままだから。実際どうなるかを測る。
  なお実装では**到着時点の視線の正面**を行き先にしている（待つ間に首が回るため）。

【測るもの】
  1. おもちゃと太郎のめり込み（最大の貫入 mm）
  2. 視線とおもちゃのなす角（視野の半角30度以内なら見えている）
  3. 首の角速度（弾かれていないか）
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

SEC = 6.0
SAMPLE_AT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
HALF_FOV = 30.0


def measure(delay, approach):
    """delay 秒待って approach 秒かけて運ぶ設定で1エピソード測る。"""
    os.environ["E_TOY_DELAY"] = str(delay)
    os.environ["E_TOY_APPROACH"] = str(approach)
    # 定数はモジュール読み込み時に決まるので、読み直す
    for mod in ("e_toy_env",):
        if mod in sys.modules:
            del sys.modules[mod]
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
    toy_bid = int(m.body("test_object1").id)
    head_bid = int(m.body("head").id)
    cam_id = next(c for c in range(m.ncam) if "eye_left" in (m.camera(c).name or ""))

    def snap():
        # おもちゃが何かにめり込んでいないか（床は除く。★柵＝world は含める）
        # ⚠️2026-07-26：以前は「到着した後」しか集計しておらず、**運搬中に柵へ
        #   引っかかるのを見落としていた**。ユーザーの目視で発覚（落とし穴 項1・項6）。
        worst = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist >= -1e-5:
                continue
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if toy_bid not in (b1, b2):
                continue
            other = b2 if b1 == toy_bid else b1
            if m.body(other).name == "floor":
                continue
            worst = max(worst, -float(c.dist) * 1000)
        cpos = d.cam_xpos[cam_id]
        fwd = -d.cam_xmat[cam_id].reshape(3, 3)[:, 2]
        v = d.xpos[toy_bid] - cpos
        n = float(np.linalg.norm(v))
        gz = float(np.degrees(np.arccos(np.clip(np.dot(fwd, v / n), -1, 1)))) if n > 1e-9 else float("nan")
        w = float(np.linalg.norm(d.cvel[head_bid][:3]))
        return worst, gz, w

    a = np.zeros(n_act, dtype=np.float32)
    picks = {round(t / dt): t for t in SAMPLE_AT}
    rows = {0.0: snap()}
    trace = [snap()]
    for step in range(1, int(SEC / dt) + 1):
        env.step(a)
        s = snap()
        trace.append(s)
        if step in picks:
            rows[picks[step]] = s
    env.close()
    arr = np.array(trace)
    return rows, arr


def main():
    cases = [("遅延なし（従来）", 0.0, 0.0),
             ("1.0秒待って0.5秒で運ぶ（提案）", 1.0, 0.5),
             ("2.5秒待って0.5秒で運ぶ", 2.5, 0.5)]

    print("=== おもちゃの登場を遅らせる案の効果 ===")
    print(f"  action=0（完全脱力）・{SEC:.0f}秒・視野の半角 {HALF_FOV:.0f}度\n")

    summary = {}
    for label, delay, approach in cases:
        rows, arr = measure(delay, approach)
        print(f"--- {label} ---")
        print(f"{'t[s]':>6}{'めり込み[mm]':>15}{'視線のズレ[度]':>17}{'頭の角速度':>13}")
        for t in SAMPLE_AT:
            pen, gz, w = rows[t]
            mark = "" if gz < HALF_FOV else "  視界の外"
            pm = "  ★" if pen > 1.0 else ""
            print(f"{t:>6.1f}{pen:>15.2f}{pm}{gz:>15.1f}{w:>13.3f}{mark}")
        # ★運搬中と到着後を分けて出す（運搬中を落とすと柵への引っかかりを見逃す）
        i_arrive = int((delay + approach) / 0.01) if delay > 0 else 0
        during = arr[:i_arrive] if i_arrive > 0 else arr[:1]
        after = arr[i_arrive:]
        seen = float((after[:, 1] < HALF_FOV).mean()) * 100
        print(f"  ★運搬中：めり込み最大 {during[:,0].max():.2f} mm"
              f"（柵に引っかかっていないか）")
        print(f"  到着後  ：めり込み最大 {after[:,0].max():.2f} mm ／ "
              f"視界に入っていた時間 {seen:.1f}% ／ "
              f"頭の角速度 平均 {after[:,2].mean():.3f}\n")
        summary[label] = (during[:, 0].max(), after[:, 0].max(), seen,
                          after[:, 2].mean(), arr[:, 2].max())

    print("=== まとめ ===")
    print(f"{'条件':<34}{'運搬中':>10}{'到着後':>10}{'視界内':>9}"
          f"{'頭ω平均':>10}{'頭ω最大':>10}")
    for k, (pd, pen, seen, wm, wmax) in summary.items():
        print(f"{k:<34}{pd:>8.2f}mm{pen:>8.2f}mm{seen:>8.1f}%{wm:>10.3f}{wmax:>10.3f}")
    print("\n  めり込みが 0 になれば、ユーザーの案で解決したことになる")
    print("  視界内が上がれば、視線誘導反射のテストが成立する状態になった")


if __name__ == "__main__":
    main()
