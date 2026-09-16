# -*- coding: utf-8 -*-
"""小さいサッケードに**直接命令を出して**、命令と実際の動きを突き合わせる。

【なぜ・2026-09-11】ユーザーの提案。
「1度未満の命令の向きの一致率が0%」という症状を、走行（run/main.py）の中で
調べていたが、**600歩で9〜10発しか出ない**ので統計として弱く、原因も絞れなかった。
今日4つの修正（保持の無条件化／制動の5段階／共収縮をなくす／サッケード中の
VOR抑制）を試して、1度未満だけは**全部 0% のまま動かなかった**。

地図が自然に小さい命令を出すのを待つ必要はない。**こちらから命令を出せばよい。**
命令の大きさを 0.1〜5度まで振り、1条件あたり何十発でも取れる。

【先行する道具】`E/scripts/e_saccade_size_probe.py`（2026-07-26）が
「サッケード1発で視線が何度動くか」を測るが、**おもちゃを置いて反射が自然に
撃つのを待つ**方式で、命令の大きさを指定できない。こちらは `set_map_target` で
直接命令を出す。環境の組み立ては `run/plugins/common/scene.build`（run/main.py と
同じ入口）を使う ── 昔、独立に組み立てて「学習は関節モード・測定は筋肉モード」と
いう別の体で動く事故が起きたため（`E/docs/実行基盤_設計.md`）。

【測ること】1発ごとに
  ・命令の大きさ[度]（こちらが指定した値）
  ・実際に動いた距離[度]（命令の向きへの射影）
  ・向きが合ったか
  ・撃つ直前の眼球の速度[度/秒]
  ・撃つ直前の頭の角速度[度/秒]
  ・サッケード中のVORの指令

使い方:
    python F/scripts/f106_small_saccade_probe.py
    python F/scripts/f106_small_saccade_probe.py --reps 30 --vor-suppress 1
"""
import argparse
import io
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
for p in [_ROOT, os.path.join(_ROOT, "run"), os.path.join(_ROOT, "run", "scene_tools"),
          os.path.join(_ROOT, "MIMo"), os.path.join(_ROOT, "taro_core"),
          os.path.join(_ROOT, "taro_core", "src"),
          os.path.join(_ROOT, "E", "scripts")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

SCENE = ("座位_12ヶ月_F2-49_実物8択_r1_個体1_消失発話_気づき待ち_3語黙る"
         "_2026-09-06_背景あり_2026-09-10")
AMPS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.5]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="小さいサッケードに直接命令を出して測る")
    ap.add_argument("--reps", type=int, default=20, help="1条件あたりの発数（向きごと）")
    ap.add_argument("--seed", type=int, default=91)
    ap.add_argument("--motion", type=float, default=0.0,
                    help="体を動かす雑音の大きさ。0なら静止（本番は頭が31度/秒で動く）")
    ap.add_argument("--out", default="F/logs/_机上/小さいサッケード_直接命令_2026-09-11.json")
    a = ap.parse_args()

    from run.plugins.common import scene as scene_mod
    env, sc, _hands = scene_mod.build(
        SCENE, taro={"actuation": "muscle", "age_months": 12.0,
                     "orienting_reflex": True}, seed=a.seed)
    u = env.unwrapped
    orient = getattr(u, "_orienting", None)
    if orient is None:
        print("定位反射が付いていない。シーンの orienting_version を確認すること。")
        return 1
    half = orient._smap.half_fov if getattr(orient, "_smap", None) is not None else 30.0
    dt = float(u.model.opt.timestep) * int(u.frame_skip)
    print("視野の半分 %.1f 度 ／ 1ステップ %.3f 秒" % (half, dt))
    print("VOR抑制: %s" % os.environ.get("E_VOR_SUPPRESS", "(既定)"))

    act = np.zeros(u.action_space.shape, dtype=np.float32)
    rng = np.random.default_rng(a.seed)

    def _step():
        """1ステップ進める。--motion があれば体を揺らして頭を動かす。

        本番（run/main.py）では方策の自発運動で頭が中央値31.4度/秒で動いており、
        止まっているのは4%しかない（F2-122pre の実測）。静止した状態で測ると
        別の条件を測ることになるので、揺らせるようにした。
        """
        if a.motion > 0.0:
            act[:] = np.clip(rng.normal(0.0, a.motion, act.shape), 0.0, 1.0)
        env.step(act)

    rows = []
    # 命令の大きさ × 向き（左右上下）を順に出す
    for amp in AMPS:
        for sgn_h, sgn_v in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            for rep in range(a.reps):
                # 撃つ前に少し進めて、眼球の状態をばらけさせる
                for _ in range(12):
                    _step()
                h0 = orient._version_h_deg()
                v0 = orient._angle_deg(orient.eye_qadr["v"])
                vh0 = orient._version_h_vel_deg()
                hw = orient._head_omega_deg()
                # SACCADE_FRAC が掛かるので、狙った振幅になるよう割り戻す
                import brain.midbrain.orienting as O
                need = amp / max(O.SACCADE_FRAC * half, 1e-9)
                dh = need * sgn_h
                dv = need * sgn_v
                if abs(dh) > 1.0 or abs(dv) > 1.0:
                    continue          # 視野の外＝出せない命令
                # 【2026-09-11・直し】`_sacc_remaining > 0` で撃ったかを見ていたが、
                #   **0.5度未満の命令は同じステップ内で終わる**ので一度も見えず、
                #   1発も記録できなかった（それ自体が症状）。
                #   撃った記録（sacc_log）が増えたかで数える。
                n_before = len(orient.sacc_log)
                orient.set_map_target(dh, dv, latency_s=0.0)
                vor_cmds = []
                for k in range(12):
                    _step()
                    if orient._sacc_remaining > 0.0:
                        vor_cmds.append(abs(orient._incoming_eye_h(act)))
                    elif len(orient.sacc_log) > n_before:
                        break
                fired = len(orient.sacc_log) > n_before
                rec = orient.sacc_log[-1] if fired else None
                h1 = orient._version_h_deg()
                v1 = orient._angle_deg(orient.eye_qadr["v"])
                cx = (orient._tgt.get("eye_h", h0) - h0)
                cy = (orient._tgt.get("eye_v", v0) - v0)
                cmd = float(np.hypot(cx, cy))
                if cmd < 1e-9 or not fired:
                    continue
                # サッケードが終わった時点の角度（記録があればそれを使う）
                if rec is not None:
                    h1 = float(rec.get("h1", h1))
                    v1 = float(rec.get("v1", v1))
                    dur = float(rec.get("t_end", 0.0) - rec.get("t", 0.0))
                else:
                    dur = 0.0
                moved = float(((h1 - h0) * cx + (v1 - v0) * cy) / cmd)
                rows.append({
                    "狙った振幅": amp, "実際の命令": cmd, "長さ[ms]": 1000.0 * dur,
                    "動いた距離": moved, "向きが合った": bool(moved > 0),
                    # 【2026-09-11】「反対に動いた」と「1ミリも動かなかった」は
                    #   別物。向きの判定を内積>0 と書いていたため、不動（内積0）を
                    #   「反対向き」と数えていた。分けて記録する。
                    "まったく動かず": bool(abs(h1 - h0) < 1e-9 and abs(v1 - v0) < 1e-9),
                    "撃つ前の眼球速度": float(vh0),
                    "撃つ前の頭の角速度": float(hw[0]),
                    "サッケード中のVOR": float(np.median(vor_cmds)) if vor_cmds else 0.0,
                })
        print("  振幅 %.1f度 まで完了（%d発）" % (amp, len(rows)))

    os.makedirs(os.path.dirname(os.path.join(_ROOT, a.out)), exist_ok=True)
    io.open(os.path.join(_ROOT, a.out), "w", encoding="utf-8").write(
        json.dumps(rows, ensure_ascii=False, indent=1))
    report(rows)
    print("\n保存: %s" % a.out)
    return 0


def report(rows):
    if not rows:
        print("1発も取れなかった。")
        return
    amp = np.array([r["実際の命令"] for r in rows])
    mov = np.array([r["動いた距離"] for r in rows])
    ok = np.array([1.0 if r["向きが合った"] else 0.0 for r in rows])
    v0 = np.array([abs(r["撃つ前の眼球速度"]) for r in rows])
    vor = np.array([r["サッケード中のVOR"] for r in rows])
    zero = np.array([1.0 if r.get("まったく動かず") else 0.0 for r in rows])
    print("\n【結果】%d発\n" % len(rows))
    dur = np.array([r.get("長さ[ms]", 0.0) for r in rows])
    print("  命令[度]        発数  動かなかった  向きが合った  長さ  動いた距離")
    print("  " + "-" * 74)
    for lo, hi in [(0, 0.15), (0.15, 0.25), (0.25, 0.35), (0.35, 0.45),
                   (0.45, 0.6), (0.6, 0.85), (0.85, 1.3), (1.3, 2.0), (2.0, 99)]:
        m = (amp >= lo) & (amp < hi)
        if m.sum() < 3:
            continue
        print("  %4.1f〜%4.1f      %4d     %5.0f%%       %5.0f%%   %5.0fms   %+7.2f度"
              % (lo, min(hi, 99), m.sum(), 100 * zero[m].mean(), 100 * ok[m].mean(),
                 np.median(dur[m]), np.median(mov[m])))
    print("\n  全体 %d発 ／ 向きが合った %.0f%%" % (len(rows), 100 * ok.mean()))
    # 勢いとの比で並べ直す
    ratio = v0 * 0.05 / np.maximum(amp, 1e-9)
    print("\n  ── 勢い÷命令 で並べ直すと ──")
    for lo, hi in [(0, 0.3), (0.3, 0.7), (0.7, 1.0), (1.0, 2.0), (2.0, 1e9)]:
        m = (ratio >= lo) & (ratio < hi)
        if m.sum() < 3:
            continue
        print("   %4.1f〜%4.1f : %4d発 → 向きが合った %3.0f%%（命令の中央値 %.2f度）"
              % (lo, min(hi, 9), m.sum(), 100 * ok[m].mean(), np.median(amp[m])))


if __name__ == "__main__":
    sys.exit(main())
