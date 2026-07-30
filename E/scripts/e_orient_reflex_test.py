"""★視線誘導反射のテスト（本番）。おもちゃを視野の中心に持ってこられるか。

【なぜ今なのか】これまで環境そのものが壊れていて成立しなかった。
  ・反射が筋肉モデルに対応せず、眼も首も片方向にしか動けなかった（項50）
  ・おもちゃが顔にめり込み、リセットのたびに太郎を弾いていた
  ・眼球が31度ずれて始まり、視線が足の方を向いていた
  ・測定器が遮蔽を見ておらず、柵の向こうでも「視界内」と数えていた（項51）
全部直したので、ここで初めて意味のある数字が出る。

【何を測るか】反射の目的は「動くものを**中心視野に持ってくる**」こと。
だから指標は**画像の中でおもちゃがどこにいるか**にする。

    ずれ = sqrt(cx^2 + cy^2)      cx, cy は画像中心を0・端を1とした重心

    ずれが**減れば**寄せられている。増えれば逆効果。

【条件】おもちゃを視野のいろいろな位置に置いて、それぞれ反射ON/OFFで比べる。
  ⚠️自発運動は流さない（親がおもちゃを見せるのは、赤ちゃんがバタついている
    最中ではない。ユーザーの指摘 2026-07-26）。

【★測定器】`e_visibility.py` の3つの判定を全部出す。
  角度／光線（遮蔽）／画像の画素。食い違いがあればそれ自体が情報。
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
import mujoco
import e_visibility as VIS

SEC = 5.0                # おもちゃが置かれてから何秒見るか
SHAKE_HZ = 2.5           # 「小さく激しく」
SHAKE_AMP = 0.015        # 1.5cm

# おもちゃを置く位置＝規定位置からのずらし[m]。視野の端に置いて、寄せられるかを見る。
OFFSETS = {
    "正面      ": (0.0, 0.000, 0.000),
    "右へ 3cm  ": (0.0, +0.030, 0.000),
    "左へ 3cm  ": (0.0, -0.030, 0.000),
    "上へ 3cm  ": (0.0, 0.000, +0.030),
    "下へ 3cm  ": (0.0, 0.000, -0.030),
}


def make_env():
    """★環境は1つだけ作って使い回す。条件ごとに作り直すとメモリと OpenGL が枯れる
    （実際に `WARNING: OpenGL error 0x505` が出た）。"""
    from e_toy_env import ToySupineEnv, infant_vision_params
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env

    kw = body_kwargs_from_env(0.0, verbose=False)
    kw["flexion"] = True
    return ToySupineEnv(actuation_model=MuscleModel,
                        vision_params=infant_vision_params(),
                        age=0.0, toy=True, vor=True, orient=True, **kw)


def run(env, offset, reflex_on):
    u = env.unwrapped
    m, d = u.model, u.data
    env.reset(seed=0)
    dt = float(m.opt.timestep) * int(u.frame_skip)
    n_act = env.action_space.shape[0]
    a = np.zeros(n_act, dtype=np.float32)
    toy_bid = int(m.body("test_object1").id)
    toy_jid = next(j for j in range(m.njnt)
                   if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                   and m.body(m.jnt_bodyid[j]).name == "test_object1")
    toy_qadr = int(m.jnt_qposadr[toy_jid])
    toy_dof = int(m.jnt_dofadr[toy_jid])

    rf = u._orienting
    if not reflex_on:
        u._orienting = None
    elif rf is not None:
        rf.reset()

    # おもちゃが設置されるまで進める（登場の遅延ぶん）
    import e_toy_env as TE
    settle = TE.TOY_APPEAR_DELAY + TE.TOY_APPROACH_SEC + 0.2
    for _ in range(int(settle / dt)):
        env.step(a)
    # ★おもちゃの位置は `_rest_pos` を書き換えて動かす。
    #   TOY_MODE="hold" は毎ステップ `_rest_pos` に位置を書き戻すので、
    #   `data.qpos` を直接書いても**上書きされて効かない**（実際に踏んだ：
    #   置き場所を変えても全条件が同じ数値になった。落とし穴 項17）。
    base = (np.array(u._rest_pos, dtype=float) if getattr(u, "_rest_pos", None) is not None
            else d.qpos[toy_qadr:toy_qadr + 3].copy()) + np.array(offset)

    rows = []
    t = 0.0
    for _ in range(int(SEC / dt)):
        # 環境側から小さく激しく揺らす（親が振って注意を引く）
        off = SHAKE_AMP * np.sin(2 * np.pi * SHAKE_HZ * t)
        u._rest_pos = base + np.array([0.0, off, 0.0])
        d.qpos[toy_qadr:toy_qadr + 3] = u._rest_pos
        d.qvel[toy_dof:toy_dof + 6] = 0.0
        env.step(a)
        t += dt
        imgs = u.get_vision_obs()
        img = imgs.get("eye_left") if isinstance(imgs, dict) else None
        r = VIS.report(m, d, toy_bid, img)
        dev = (np.hypot(r["cx"], r["cy"]) if r["pix_seen"] else np.nan)
        rows.append((t, r["angle"], float(r["in_fov"]), float(r["ray_ok"]),
                     float(bool(r["pix_seen"])), r["n_pixels"], dev,
                     r["cx"], r["cy"]))
    n_sacc = getattr(rf, "n_saccades", 0) if reflex_on and rf is not None else 0
    u._orienting = rf
    return np.array(rows, dtype=float), n_sacc


def main():
    os.environ.setdefault("E_ORIENT_V", "2")
    os.environ.setdefault("E_TOY_MODE", "hold")
    print("=== 視線誘導反射：おもちゃを視野の中心に持ってこられるか ===")
    print(f"  {SEC:.0f}秒・自発運動なし・おもちゃを {SHAKE_HZ}Hz で {SHAKE_AMP*100:.1f}cm 揺らす")
    print("  指標『中心からのずれ』＝画像の中でおもちゃがどこにいるか")
    print("    0.0 = ど真ん中 ／ 1.0 = 画面の端。**減れば寄せられている**\n")

    print(f"{'置き場所':<12}{'反射':<6}{'見えた割合':>11}{'画素数':>9}"
          f"{'中心からのずれ':>16}{'変化':>10}{'サッケード':>11}")
    print("-" * 78)
    env = make_env()
    summary = {}
    for label, off in OFFSETS.items():
        res = {}
        for on in (False, True):
            arr, n_sacc = run(env, off, on)
            seen = float(np.nanmean(arr[:, 4])) * 100
            pix = float(np.nanmean(arr[:, 5]))
            dev = float(np.nanmean(arr[:, 6]))
            dev0 = float(arr[0, 6]) if not np.isnan(arr[0, 6]) else np.nan
            devN = float(np.nanmean(arr[-20:, 6]))
            res[on] = (seen, pix, dev, dev0, devN, n_sacc)
        for on in (False, True):
            seen, pix, dev, dev0, devN, n_sacc = res[on]
            delta = devN - dev0
            print(f"{label:<12}{'ON ' if on else 'OFF':<6}{seen:>10.1f}%{pix:>9.0f}"
                  f"{dev:>16.3f}{delta:>+10.3f}{n_sacc:>11d}")
        d_off = res[False][2]
        d_on = res[True][2]
        summary[label] = (d_off, d_on)
        verdict = ("★寄せている" if d_on < d_off - 0.02
                   else "★遠ざけている" if d_on > d_off + 0.02
                   else "変わらない")
        print(f"{'':<12}{'':6}→ 反射OFF {d_off:.3f} vs ON {d_on:.3f}   {verdict}\n")

    env.close()
    print("=== まとめ ===")
    ok = sum(1 for a, b in summary.values() if b < a - 0.02)
    ng = sum(1 for a, b in summary.values() if b > a + 0.02)
    print(f"  寄せた {ok}/{len(summary)}  ／  遠ざけた {ng}/{len(summary)}")

    # ★テスト自体の健全性：置き場所を変えたのに結果が同じなら、条件が効いていない
    #   （落とし穴 項17・項47。実際に踏んだ：hold モードが位置を上書きしていた）
    offs = [v[0] for v in summary.values()]
    if max(offs) - min(offs) < 1e-6:
        print("\n  ★★このテストは無効です。置き場所を変えても反射OFFの結果が")
        print("     まったく同じ＝**おもちゃの位置が変わっていない**。")
        print("     条件が環境に届いているかを先に確かめること。")
    print("\n  ★『変化』は最初と最後20tickの差。負なら中心に寄っている。")
    print("  ★サッケードが0発なら、動き検出の閾値に届いていない（別の原因）。")


if __name__ == "__main__":
    main()
