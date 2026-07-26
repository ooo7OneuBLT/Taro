"""視線誘導反射の配線チェック（環境テストの前に必ず通す）。

【なぜ作るか・2026-07-26】反射のロジックは人工画像テストで通ったが、
環境に繋いだとき「実装済みなのに機能しない」が繰り返し起きている：
  ・`vision_params=None` にしていて視覚が動かず、反射に入力が来ていなかった（今日）
  ・`E_ORIENT` が e_toy_env にしかなく、学習ループには繋がっていない
  ・MuscleModel が actuator_gear を毎ステップ上書きし、筋力補正が効いていなかった
  ・生理的屈曲が既定OFFで、渡し忘れて効いていなかった（今日）
★共通点：**ログには出るのに実効がない**。だから「ログを見た」では検証にならない。

【この検査の考え方】反射の入口から出口まで、各段で「値が変わること」を確認する。
  入口： 視覚パイプラインが画像を返すか
  中間： update() が呼ばれて h_dir/v_dir が更新されるか
  出口： apply() の結果が action に反映され、実際に関節が動くか

使い方:
    .venv/Scripts/python.exe E/scripts/e_orient_wiring_check.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
for p in [os.path.join(_ROOT, "D", "scripts"), os.path.join(_ROOT, "MIMo"),
          os.path.join(_ROOT, "taro_core"), _HERE]:
    if p not in sys.path:
        sys.path.insert(0, p)
try: sys.stdout.reconfigure(errors="replace")
except Exception: pass

import numpy as np

_RESULTS = []


def check(name, ok, detail=""):
    _RESULTS.append((name, bool(ok)))
    print(f"  [{'OK ' if ok else 'NG!'}] {name}" + (f"   {detail}" if detail else ""),
          flush=True)


def main():
    print("=== 視線誘導反射の配線チェック ===")
    print("  ロジックは人工画像テストで確認済み。ここでは【環境に繋がっているか】を見る\n")

    from e_toy_env import ToySupineEnv, infant_vision_params
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env

    # ---------- 入口：環境の生成 ----------
    print("--- 入口：環境と視覚 ---")
    kw = body_kwargs_from_env(0.0, verbose=False)
    kw["flexion"] = True
    vp = infant_vision_params()
    check("vision_params が None でない", vp is not None,
          f"カメラ {list(vp.keys()) if vp else '—'}")

    env = ToySupineEnv(actuation_model=MuscleModel, vision_params=vp,
                       age=0.0, toy=True, orient=True, **kw)
    m, d = env.unwrapped.model, env.unwrapped.data
    env.reset(seed=0)
    n_act = env.action_space.shape[0]

    check("おもちゃが存在する", env.unwrapped._toy,
          f"test_object1 の質量 {float(m.body_mass[m.body('test_object1').id])*1000:.0f} g")

    imgs = env.unwrapped.get_vision_obs()
    ok_img = isinstance(imgs, dict) and len(imgs) > 0
    check("視覚が画像を返す", ok_img,
          f"{[f'{k}:{v.shape}' for k, v in imgs.items()] if ok_img else '—'}")

    if ok_img:
        var = max(float(np.var(v)) for v in imgs.values())
        check("画像が真っ黒でない（分散がある）", var > 1e-4, f"分散 {var:.4f}")

    # ---------- 中間：反射のインスタンスと更新 ----------
    print("\n--- 中間：反射の生成と方向の更新 ---")
    reflex = env.unwrapped._orienting
    check("反射のインスタンスが生成されている", reflex is not None,
          type(reflex).__name__ if reflex else "None（orient=False？）")
    if reflex is None:
        env.close()
        return 1

    check("首のアクチュエータを見つけている", len(reflex.neck_idx) > 0,
          f"{list(reflex.neck_idx.keys())}")
    check("目のアクチュエータを見つけている",
          len(reflex.eye_idx["h"]) > 0 and len(reflex.eye_idx["v"]) > 0,
          f"h={len(reflex.eye_idx['h'])} v={len(reflex.eye_idx['v'])}")

    # おもちゃを揺らしながら回して、方向が更新されるか
    toy_bid = m.body("test_object1").id
    toy_jid = m.body_jntadr[toy_bid]
    toy_qadr = int(m.jnt_qposadr[toy_jid])
    toy_dof = int(m.jnt_dofadr[toy_jid])
    center = d.qpos[toy_qadr:toy_qadr + 3].copy()
    dt = float(m.opt.timestep) * int(env.unwrapped.frame_skip)

    zero = np.zeros(n_act, dtype=np.float32)
    dirs = []
    for k in range(200):
        off = 0.015 * np.sin(2 * np.pi * 2.5 * k * dt)
        d.qpos[toy_qadr:toy_qadr + 3] = center + np.array([0.0, off, 0.0])
        d.qvel[toy_dof:toy_dof + 6] = 0.0
        env.step(zero)
        dirs.append((getattr(reflex, "h_dir", 0.0), getattr(reflex, "v_dir", 0.0)))
    dirs = np.array(dirs)
    nz = float((np.abs(dirs).sum(axis=1) > 1e-9).mean())
    check("step() を経由して方向が更新される", nz > 0.5,
          f"非ゼロだった割合 {nz*100:.0f}%  |h|平均 {np.abs(dirs[:,0]).mean():.3f}")

    # ---------- 出口：action への反映 ----------
    print("\n--- 出口：action への反映と関節の動き ---")
    # 反射だけを取り出す（同じ状態で apply の前後を比べる）
    a_before = np.zeros(n_act, dtype=np.float32)
    fired = False
    diff_max = 0.0
    for _ in range(80):        # サッケードの潜時（0.2秒＝20step）を跨ぐまで回す
        a_after = reflex.apply(a_before.copy())
        dmax = float(np.abs(np.asarray(a_after) - a_before).max())
        if dmax > 1e-9:
            fired = True
            diff_max = max(diff_max, dmax)
    n_sacc = getattr(reflex, "n_saccades", None)   # v2 のみ持つ
    check("apply() が action を書き換える", fired,
          f"最大の変化量 {diff_max:.4f}" +
          (f"  サッケード発火 {n_sacc} 回" if n_sacc is not None else "  （v1は間欠出力なし）"))

    # ⚠️MIMoの環境は1つで約1.7GB使う。比較の前に必ず閉じる
    #   （3つ同時に作って "Could not allocate memory" になった。2026-07-26）
    env.close()
    del env, m, d, reflex

    # 実際に環境の中で関節が動くか（脱力 vs 反射あり）
    def run(orient_on, steps=400):
        e = ToySupineEnv(actuation_model=MuscleModel, vision_params=vp,
                         age=0.0, toy=True, orient=orient_on, **kw)
        mm, dd = e.unwrapped.model, e.unwrapped.data
        e.reset(seed=0)
        qa = int(mm.jnt_qposadr[mm.body_jntadr[mm.body("test_object1").id]])
        qd = int(mm.jnt_dofadr[mm.body_jntadr[mm.body("test_object1").id]])
        c = dd.qpos[qa:qa+3].copy()
        eh = [int(mm.jnt_qposadr[j]) for j in range(mm.njnt)
              if "eye" in (mm.joint(j).name or "") and "horizontal" in (mm.joint(j).name or "")]
        hs = int(mm.jnt_qposadr[mm.joint("robot:head_swivel").id])
        tr_e, tr_h = [], []
        for k in range(steps):
            off = 0.015 * np.sin(2*np.pi*2.5*k*dt)
            dd.qpos[qa:qa+3] = c + np.array([0.0, off, 0.0])
            dd.qvel[qd:qd+6] = 0.0
            e.step(np.zeros(e.action_space.shape[0], dtype=np.float32))
            tr_e.append(float(np.degrees(np.mean([dd.qpos[i] for i in eh]))))
            tr_h.append(float(np.degrees(dd.qpos[hs])))
        e.close()
        return np.array(tr_e), np.array(tr_h)

    eye_off, neck_off = run(False)
    eye_on, neck_on = run(True)
    d_eye = float(np.abs(eye_on - eye_off).max())
    d_neck = float(np.abs(neck_on - neck_off).max())
    check("反射ON/OFFで目の動きが変わる", d_eye > 0.1,
          f"最大差 {d_eye:.3f}度  (OFF幅 {np.ptp(eye_off):.2f} / ON幅 {np.ptp(eye_on):.2f})")
    check("反射ON/OFFで首の動きが変わる", d_neck > 0.1,
          f"最大差 {d_neck:.3f}度  (OFF幅 {np.ptp(neck_off):.2f} / ON幅 {np.ptp(neck_on):.2f})")

    ng = [n for n, ok in _RESULTS if not ok]
    print(f"\n  結果: {len(_RESULTS) - len(ng)}/{len(_RESULTS)} 通過")
    if ng:
        print("  ★NG:", " / ".join(ng))
        print("  → 反射のロジックではなく【配線】を疑う。"
              "vision_params・orient フラグ・update の呼び出し位置を確認する")
    return 1 if ng else 0


if __name__ == "__main__":
    sys.exit(main())
