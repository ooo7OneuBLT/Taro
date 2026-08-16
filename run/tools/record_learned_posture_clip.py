# -*- coding: utf-8 -*-
"""本日学習した座位保持モデル（座位_6ヶ月・muscle・reward=posture_height）を、
学習時とまったく同じ条件で動かし、30秒・等倍速の第三者視点動画を作る。

指示は本セッションで直接受けた（管理用ファイル名は無し）。実験ファイル
`E/experiments/座位保持_6ヶ月_2026-08-16.json` と同じ taro 設定を使う：

    scene                    座位_6ヶ月_土台_2026-08-16
    actuation                muscle
    reward                   posture_height
    posture_reflex           false
    righting_reflex          true
    righting_reflex_gain     0.02
    posture_fall_deg         20.0
    posture_fall_delay_sec   1.0
    model                    E/logs/座位保持_6ヶ月/model_seed0.pt

【なぜ Taro 構築のあとに reset_to_scene するか（重要）】
`run/tools/record_posture_reset_clip.py`（本日別件で作られたもの）と同じ問題を
踏む：`Taro.__init__` は内部で `env.reset()` を呼ぶため、シーンJSONの座位姿勢
（scene_mod.build 時点で一度適用したもの）が既定の仰向け姿勢へ戻ってしまう。
そこで Taro を構築した「あと」に `e_scene.reset_to_scene()` を呼び直し、
座位へ戻してから録画を始める。

【_posture_seated_qpos の想定外は既に解消済み（確認した）】
`record_posture_reset_clip.py` のdocstringにある想定外（座り直しの復帰先が
reset_model() 直後の仰向け姿勢のまま固定されてしまう問題）は、本日
`E/scripts/e_toy_env.py` の step() 側で修正済み（`_posture_baseline_pending`
フラグにより、最初の step() 呼び出し時点＝reset_to_scene() 適用後の姿勢で
`_posture_seated_qpos` を確定し直す。同ファイル1190〜1240行）。実測でも
座り直しの復帰先が座位（体幹66度付近）になっていることを確認した
（下記「診断」ログ参照）。よってこのスクリプトでは手動での上書きは行っていない。

【行動の作り方（run/viewer.py policy_fn と同じ・探索なし）】
`run/viewer.py` 135〜169行目の非探索側の経路と同じ：
  fusion.encode → infer_latent → act_mean → clamp(-1,1)
Goal Babbling・見比べモードは使わない（このスクリプトでは常に固定の1脳）。
探索の揺らぎは無し（view_explore=false 相当、`a = clamp(mean, -1, 1)`）。
学習時は探索ノイズが乗った行動で集められた経験から学習しているが、
評価用の録画としては決定的行動（探索なし）を選んだ（ノウハウ.md
「決定的行動（act_mean）」の項と同じ考え方＝探索込みだと動きの質が変わる）。
厳密に学習時と同一ではない点は報告に明記する。

【触ってよいファイル】このファイルと、出力先のmp4のみ。他は一切変更しない。
"""
import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import cv2                                              # noqa: E402
import mujoco                                            # noqa: E402
import numpy as np                                       # noqa: E402
import torch                                              # noqa: E402

from run.config import Config                            # noqa: E402
from run.taro_setup import Taro, rescale_action           # noqa: E402
from run.plugins.common import scene as scene_mod         # noqa: E402

SCENE = "座位_6ヶ月_土台_2026-08-16"
MODEL_PATH = os.path.join(_ROOT, "E", "logs", "座位保持_6ヶ月", "model_seed0.pt")
OUT_PATH = os.path.join(_ROOT, "E", "docs", "figures",
                         "座位保持_6ヶ月_学習済みモデル_2026-08-16.mp4")

DT = 0.01                    # run/trainer.py と同じ（MuJoCoの1物理ステップの秒数）
K = 10                        # 1判断あたりの物理ステップ数（cfg.K既定値、実験と同じ）
RENDER_EVERY = 4              # 4物理stepに1回描画 → fps=25
FPS = (1.0 / DT) / RENDER_EVERY
SEC_TOTAL = 30.0
TOTAL_TICKS = int(SEC_TOTAL / DT)          # 3000
SEED = 0

H, W = 480, 640               # 横長（体全体が横から見えるように）


def main():
    print("=" * 78)
    print(" 座位保持_6ヶ月・学習済みモデルの録画（学習時と同一条件・30秒・等倍速）")
    print("=" * 78)

    assert os.path.isfile(MODEL_PATH), f"モデルが見つからない: {MODEL_PATH}"
    model_size = os.path.getsize(MODEL_PATH)
    print(f"[診断] モデルファイル: {MODEL_PATH}（{model_size}バイト）", flush=True)

    spec = {
        "name": "座位保持_6ヶ月_録画",
        "scene": SCENE,
        "taro": {
            "actuation": "muscle",
            "reward": "posture_height",
            "posture_reflex": False,
            "righting_reflex": True,
            "righting_reflex_gain": 0.02,
            "posture_fall_deg": 20.0,
            "posture_fall_delay_sec": 1.0,
            "model": MODEL_PATH,
        },
        "run": {"seed": SEED, "K": K},
    }
    cfg = Config.from_spec(spec)

    taro_spec = dict(cfg._taro)
    env, sc, hands = scene_mod.build(
        cfg.scene, taro=taro_spec, seed=cfg.seed, verbose=True, hybrid=True)

    # ---- Taro構築（学習済み重みを読み込む。env.reset()が内部で呼ばれ、
    #   シーンの座位姿勢がいったん失われる＝下でreset_to_sceneして戻す） --------
    taro = Taro(cfg, env, seed=cfg.seed, verbose=True)
    u = env.unwrapped
    assert getattr(u, "taro", None) is taro, \
        "[想定外] env.unwrapped.taro が配線されていない（righting_reflex=True なら " \
        "_setup_righting_damper が配線するはず）"
    print(f"[診断] env.taro配線OK: postural_gate={taro.postural_gate is not None} "
          f"righting_damper={taro.righting_damper is not None} "
          f"posture_fall_deg={cfg.posture_fall_deg} "
          f"posture_fall_delay_sec={cfg.posture_fall_delay_sec}", flush=True)

    # ---- シーンの姿勢（座位）へ戻す ------------------------------------------
    import e_scene
    e_scene.reset_to_scene(env, sc, hands=hands, seed=cfg.seed)
    tilt_after_reset = u._posture_trunk_tilt_deg()
    head_z_after_reset = float(u.data.body("head").xpos[2])
    print(f"[診断] reset_to_scene直後の体幹傾き={tilt_after_reset:.1f}度 "
          f"頭の高さ={head_z_after_reset * 100:.1f}cm", flush=True)

    # ---- render_modeを構築後に付け足す。0コスト診断で必ず確認する ------------
    u.render_mode = "rgb_array"
    test_frame = env.render()
    assert test_frame is not None, \
        "render_modeの後付けが効いていない。実装を止めて報告すること"
    print(f"[診断] render_modeの後付けOK：フレーム形状={test_frame.shape}", flush=True)

    # ---- 倒れ→座り直しの回数を数える（_check_posture_fallをラップして数える）---
    #   e_toy_env.pyは変更しない。インスタンス属性の上書きだけ（触ってよい範囲）。
    orig_check = u._check_posture_fall
    reset_events = []

    def counting_check(taro_arg):
        # 【なぜqposの前後比較にしたか】_check_posture_fallは「傾きが閾値を
        #   割っている間」毎tick呼ばれるが、実際にqposを書き戻す（＝座り直しが
        #   発動する）のはposture_fall_delay_sec待った1回だけ（同関数docstring
        #   参照）。tilt<fall_degを毎回数えると「発動回数」ではなく「割っていた
        #   tick数」になってしまう（最初の実装でこれを取り違え、1073回という
        #   明らかに過大な値が出たため実測で気づいて直した）。
        #   実際に書き戻ったかは、qposそのものの変化で判定する。
        qpos_before = u.data.qpos.copy()
        tilt_before = u._posture_trunk_tilt_deg()
        result = orig_check(taro_arg)
        jumped = float(np.abs(u.data.qpos - qpos_before).max()) > 1e-6
        if jumped:
            tilt_after = u._posture_trunk_tilt_deg()
            reset_events.append({"tick": tick_counter[0], "tilt_before_reset": tilt_before,
                                  "tilt_after_reset": tilt_after})
        return result

    u._check_posture_fall = counting_check

    # ---- カメラ（第三者・横からの構図。骨盤を中心に追従） ---------------------
    m, d = u.model, u.data
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), W)
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), H)
    ren = mujoco.Renderer(m, height=H, width=W)

    def make_side_camera():
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(m, cam)
        hip_xpos = d.body("hip").xpos
        cam.lookat[:] = [hip_xpos[0] + 0.03, hip_xpos[1], 0.09]
        cam.distance = 0.55
        cam.elevation = -8.0
        cam.azimuth = 90.0        # 真横（world +Y方向から見る。体はworld +X向き）
        return cam

    # ---- 行動：学習済みの脳（run/viewer.py policy_fn 135〜169行と同じ、探索なし）--
    #   探索の揺らぎは使わない（view_explore=false相当）＝ a = clamp(mean, -1, 1)
    n_act = env.action_space.shape[0]
    hidden = taro.brain.init_motor_hidden()
    prev_a = torch.zeros(n_act)

    def decide_action(obs, prev_a, hidden):
        sv = taro.fusion.encode(obs)
        cf = taro.target_fusion.encode(obs).detach()
        z, _kl, _rc, hn = taro.infer_latent(sv, prev_a, cf, hidden)
        z = z.detach()
        mean = taro.act_mean(z)
        a = torch.clamp(mean, -1.0, 1.0).detach()
        return a, hn.detach()

    # 【なぜ、実測で判明・2026-08-16】reset_to_scene()内のenv.reset(seed=seed)は
    #   まだシーンの姿勢を書き戻す前のobsを返す（apply_state()等はreset()呼び出し
    #   の後に実行される）。そのため、そのobsを使うと古い姿勢の値のまま脳へ渡って
    #   しまう。姿勢を全部適用し終えたこの時点で、生の観測(u._get_obs())へ
    #   HybridEnvの内受容(_augment_obs)を足し直して最新のobsを作る。
    obs = env._augment_obs(u._get_obs())

    frames = []
    tilt_log = []
    d_action2_log = []
    tick_counter = [0]

    t0 = time.time()
    for i in range(TOTAL_TICKS):
        tick_counter[0] = i
        if i % K == 0:
            a, hidden = decide_action(obs, prev_a, hidden)
            d_action2_log.append(float(((a - prev_a) ** 2).mean()))
            prev_a = a
            a_env = taro.brain.to_env_action(a) if hasattr(taro.brain, "to_env_action") else a
            ctrl = rescale_action(a_env, env.action_space)
        obs, r, te, tr, info = env.step(ctrl)
        if i % RENDER_EVERY == 0:
            cam = make_side_camera()
            ren.update_scene(d, camera=cam)
            frames.append(ren.render().copy())
        if i % 100 == 0:      # 1.0秒おきに記録（診断ログ・報告用）
            tilt_log.append((i * DT, u._posture_trunk_tilt_deg(),
                              float(u.data.body("head").xpos[2])))
        if te or tr:
            print(f"[診断] tick={i}でエピソード終端（term={te} trunc={tr}）。"
                  "録画は打ち切らずゼロ行動で継続するのは不適切なので、"
                  "そのまま最後まで回す想定だったが終端が起きた。報告に明記する。",
                  flush=True)

    elapsed = time.time() - t0
    print(f"[診断] {TOTAL_TICKS}物理tick（{SEC_TOTAL:.0f}秒ぶん）を{elapsed:.1f}秒で実行", flush=True)

    env.close()

    # ---- 動画として書き出す ---------------------------------------------------
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    h, w, _c = frames[0].shape
    vw = cv2.VideoWriter(OUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for fr in frames:
        vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"\nRECORDED {OUT_PATH} ({len(frames)}フレーム, {FPS:.0f}fps, "
          f"約{len(frames) / FPS:.1f}秒)", flush=True)

    # ---- 報告用の実測 ----------------------------------------------------------
    print(f"\n[結果] 開始時の体幹傾き: {tilt_after_reset:.1f}度 "
          f"頭の高さ: {head_z_after_reset * 100:.1f}cm", flush=True)
    print(f"[結果] 座り直し（座位への復帰）が発動した回数: {len(reset_events)}回", flush=True)
    for ev in reset_events:
        print(f"    tick={ev['tick']:5d} sim={ev['tick'] * DT:5.2f}秒"
              f"  発動前={ev['tilt_before_reset']:.1f}度 -> 発動後={ev['tilt_after_reset']:.1f}度",
              flush=True)
    print("\n[参考] 1.0秒おきの体幹傾き・頭の高さ：", flush=True)
    for t_sec, tilt, head_z in tilt_log:
        print(f"    {t_sec:5.2f}秒  体幹={tilt:6.1f}度  頭={head_z * 100:5.1f}cm", flush=True)
    d2 = np.asarray(d_action2_log, dtype=np.float64)
    print(f"\n[参考] d_action2（行動の変化量^2の平均）: "
          f"mean={d2.mean():.4f} max={d2.max():.4f} n={len(d2)}", flush=True)


if __name__ == "__main__":
    main()
