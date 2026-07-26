"""サッケードの間隔を変えると暴走が収まるか。あわせて「いつ反応しているか」を分ける。

【測ること2つ】

★測定1：反応はサッケード中に起きているのか、それ以外の時間に起きているのか
  これまで「自己運動への反応 0.26」とだけ測っていたが、
    ・サッケード中の一時的な流れ         → 撃つ間隔を空ければ収まる
    ・首や眼がゆっくり動き続ける流れ     → 間隔では解決しない
  のどちらかで対処が変わる。分けていなかった（調査で指摘された測定の穴）。

★測定2：サッケードの間隔（SACCADE_LATENCY）を振る
  太郎は 200ms。新生児の実測で対応するのは**サッケード間の間隔 500〜900ms**
  （Aslin & Salapatek 1975。※「潜時 800〜1480ms」は「対象が出てから最初の1発まで」で
    別の量。混同しないこと ← 落とし穴 項48「文献の値が何の値かを区別する」）

【条件】おもちゃを揺らす／止める × 間隔 0.2/0.5/0.9/1.3秒
  「止める」条件で撃つ数が減れば、間隔で暴走が収まったことになる。
"""
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

SEC = 8.0
SHAKE_HZ = 2.5
SHAKE_AMP = 0.015
LATENCIES = [0.2, 0.5, 0.9, 1.3]


def run(env, latency, shake):
    import e_orienting_v2 as OR
    OR.SACCADE_LATENCY = float(latency)

    u = env.unwrapped
    m, d = u.model, u.data
    env.reset(seed=0)
    dt = float(m.opt.timestep) * int(u.frame_skip)
    a = np.zeros(env.action_space.shape[0], dtype=np.float32)
    toy_bid = int(m.body("test_object1").id)
    toy_jid = next(j for j in range(m.njnt)
                   if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                   and m.body(m.jnt_bodyid[j]).name == "test_object1")
    toy_qadr = int(m.jnt_qposadr[toy_jid])
    toy_dof = int(m.jnt_dofadr[toy_jid])
    cam_id = int(m.camera("eye_left").id)

    rf = u._orienting
    rf.reset()

    import e_toy_env as TE
    for _ in range(int((TE.TOY_APPEAR_DELAY + TE.TOY_APPROACH_SEC + 0.2) / dt)):
        env.step(a)
    base = np.array(u._rest_pos, dtype=float)

    # ★サッケード中かどうかで分けて集計する
    during = {"s": [], "gaze": []}     # サッケードを撃っている最中
    between = {"s": [], "gaze": []}    # それ以外（撃っていない時間）
    devs, seen = [], []
    prev_fwd = [None]
    t = 0.0
    for _ in range(int(SEC / dt)):
        off = SHAKE_AMP * np.sin(2 * np.pi * SHAKE_HZ * t) if shake else 0.0
        u._rest_pos = base + np.array([0.0, off, 0.0])
        d.qpos[toy_qadr:toy_qadr + 3] = u._rest_pos
        d.qvel[toy_dof:toy_dof + 6] = 0.0
        # 撃っている最中か（apply の内部状態）
        in_sacc = float(getattr(rf, "_sacc_remaining", 0.0)) > 0.0
        env.step(a)
        t += dt
        fwd = -np.array(d.cam_xmat[cam_id], dtype=float).reshape(3, 3)[:, 2]
        g = 0.0
        if prev_fwd[0] is not None:
            c = float(np.clip(np.dot(fwd, prev_fwd[0]), -1, 1))
            g = float(np.degrees(np.arccos(c))) / dt
        prev_fwd[0] = fwd
        bucket = during if in_sacc else between
        bucket["s"].append(float(rf.strength))
        bucket["gaze"].append(g)
        imgs = u.get_vision_obs()
        img = imgs.get("eye_left") if isinstance(imgs, dict) else None
        if img is not None:
            v = VIS.visible_in_image(img)
            seen.append(1.0 if v["seen"] else 0.0)
            if v["seen"]:
                devs.append(float(np.hypot(v["cx"], v["cy"])))

    def mn(x):
        return float(np.mean(x)) if x else float("nan")

    return dict(n_sacc=rf.n_saccades,
                frac_in=len(during["s"]) / max(1, len(during["s"]) + len(between["s"])),
                s_in=mn(during["s"]), s_out=mn(between["s"]),
                g_in=mn(during["gaze"]), g_out=mn(between["gaze"]),
                seen=mn(seen) * 100, dev=mn(devs))


def main():
    os.environ.setdefault("E_ORIENT_V", "2")
    os.environ.setdefault("E_TOY_MODE", "hold")
    from e_toy_env import ToySupineEnv, infant_vision_params
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env

    kw = body_kwargs_from_env(0.0, verbose=False)
    kw["flexion"] = True
    env = ToySupineEnv(actuation_model=MuscleModel,
                       vision_params=infant_vision_params(),
                       age=0.0, toy=True, vor=True, orient=True, **kw)

    print("=== 測定1：反応はサッケード中に起きているのか ===")
    print(f"  {SEC:.0f}秒・自発運動なし。反応の強さを『撃っている最中』と『それ以外』で分ける\n")
    print(f"{'間隔[s]':>8}{'おもちゃ':<10}{'撃った数':>9}{'撃ってる割合':>13}"
          f"{'強さ:撃中':>11}{'強さ:撃外':>11}{'視線:撃中':>11}{'視線:撃外':>11}")
    print("-" * 88)
    res = {}
    for lat in LATENCIES:
        for shake in (True, False):
            r = run(env, lat, shake)
            res[(lat, shake)] = r
            print(f"{lat:>8.1f}{'揺らす' if shake else '止める':<10}"
                  f"{r['n_sacc']:>9d}{r['frac_in']*100:>12.0f}%"
                  f"{r['s_in']:>11.4f}{r['s_out']:>11.4f}"
                  f"{r['g_in']:>9.1f}°/s{r['g_out']:>9.1f}°/s")

    print("\n=== 測定2：間隔を変えると暴走は収まるか ===")
    print(f"{'間隔[s]':>8}{'撃った数(揺)':>13}{'撃った数(止)':>13}"
          f"{'見えた割合':>12}{'中心からのずれ':>16}")
    print("-" * 64)
    for lat in LATENCIES:
        a_ = res[(lat, True)]
        b_ = res[(lat, False)]
        print(f"{lat:>8.1f}{a_['n_sacc']:>13d}{b_['n_sacc']:>13d}"
              f"{a_['seen']:>11.1f}%{a_['dev']:>16.3f}")

    env.close()
    print("\n=== 読み方 ===")
    print("  ★強さ:撃中 >> 強さ:撃外 → サッケード中の流れが主因。間隔で収まる見込み")
    print("  ★強さ:撃中 ≈ 強さ:撃外 → 常時流れている。間隔では解決しない")
    print("  『止める』条件の撃った数が減れば、間隔が効いている")
    print("  ★『見えた割合』が上がり『中心からのずれ』が下がれば、反射が機能し始めている")


if __name__ == "__main__":
    main()
