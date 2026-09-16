# -*- coding: utf-8 -*-
"""眼球の「指令1あたりに出る力」を直接測る（走行とは別の、机上のプローブ）。

【なぜ要るか】2026-09-11、保持中の加速度を指令で回帰して「指令1.0あたり574度/秒²」
と出したが、これは誤り。振動する系では加速度と位置のずれが常に比例する（a=-ω²x）ため、
位置のずれに比例するP項と共線になり、回帰は ω²/ゲイン を返すだけだった
（P項だけで測ると19093度/秒²＝飽和時の実測5382を超える、という物理的にありえない値が出た）。
詳細：F/logs/_机上/保持の残り_原因の切り分け_2026-09-11.md 項目19〜21

【やり方】ほかの制御を全部止めて、眼球の水平アクチュエータに一定の指令を入れ、
そのときの角加速度を測る。指令を変えて繰り返す。力が指令に比例するか、
どこで飽和するかが直接見える。

【先行する道具・2026-09-11に気づいた】`E/scripts/e_eye_power_probe.py` が
ほぼ同じことを測る（最大指令で何度動くか・可動域・VOR ON/OFF・月齢スケーリング）。
書く前に探すべきだった（TaroMap が「もう試したか」を調べる最初の場所）。
違いは、こちらが**指令を段階的に変えて比例するか／どこで飽和するか**を見る点だけ。
先に e_eye_power_probe.py を読むこと。

使い方: python F/scripts/f105_eye_force_probe.py
出力  : F/logs/_机上/眼球の力_実測_YYYY-MM-DD.txt と同名 .json
"""
import os, sys, io, json, datetime
sys.path.insert(0, os.path.abspath("taro_core/src"))
sys.path.insert(0, os.path.abspath("."))
import numpy as np

CMDS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
SETTLE_S = 0.20      # 指令を入れてから測り始めるまで（筋の立ち上がりを待つ）
MEASURE_S = 0.08     # 加速度を測る区間


def build():
    """太郎の身体だけを持つ最小の環境を作る（シーンの物や親は要らない）。"""
    from run.scene_tools import scene_io
    spec = {"world": {"toys": [], "backdrop": None, "eye_centering": False,
                       "orienting_hold": False},
            "taro": {"actuation": "muscle", "age_months": 12.0,
                     "orienting_reflex": False, "hearing": False}}
    return scene_io.build(spec)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        env = build()
    except Exception as e:
        print("環境を作れなかった: %r" % (e,))
        print("→ 走行と同じ経路（run/main.py）で作る必要があるかもしれない。")
        return 1
    u = env.unwrapped
    dt = float(u.model.opt.timestep) * int(u.frame_skip)
    # 眼球の水平アクチュエータと関節を探す
    import mujoco
    ai = mujoco.mj_name2id(u.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act:left_eye_horizontal")
    ji = mujoco.mj_name2id(u.model, mujoco.mjtObj.mjOBJ_JOINT, "robot:left_eye_horizontal")
    qadr = int(u.model.jnt_qposadr[ji]); dadr = int(u.model.jnt_dofadr[ji])
    print("1ステップ = %.4f 秒 ／ アクチュエータ %d ／ 関節 qpos %d" % (dt, ai, qadr))
    rows = []
    for c in CMDS:
        env.reset()
        act = np.zeros(u.model.nu)
        n_settle = max(int(SETTLE_S/dt), 1); n_meas = max(int(MEASURE_S/dt), 3)
        ang = []
        for k in range(n_settle + n_meas):
            act[:] = 0.0; act[ai] = c
            u.data.ctrl[:] = act
            mujoco.mj_step(u.model, u.data, nstep=int(u.frame_skip))
            if k >= n_settle:
                ang.append(np.degrees(float(u.data.qpos[qadr])))
        ang = np.array(ang)
        # 二次のあてはめ x = a/2 t² + v t + x0 → 加速度 a
        tt = np.arange(len(ang))*dt
        a2 = np.polyfit(tt, ang, 2)[0]*2.0
        rows.append({"指令": c, "角加速度[度/秒²]": float(a2),
                     "指令1.0換算": float(a2/c)})
        print("  指令 %.2f → 角加速度 %8.0f 度/秒²   （指令1.0換算 %8.0f）" % (c, a2, a2/c))
    d = datetime.date.today().isoformat()
    os.makedirs("F/logs/_机上", exist_ok=True)
    io.open("F/logs/_机上/眼球の力_実測_%s.json" % d, "w", encoding="utf-8").write(
        json.dumps(rows, ensure_ascii=False, indent=2))
    lin = [r["指令1.0換算"] for r in rows]
    print("\n指令1.0換算のばらつき: 最小 %.0f 最大 %.0f （比 %.2f）"
          % (min(lin), max(lin), max(lin)/max(min(lin), 1e-9)))
    print("比が1に近ければ力は指令に比例。大きく外れるなら飽和か非線形。")
    print("保存: F/logs/_机上/眼球の力_実測_%s.json" % d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
