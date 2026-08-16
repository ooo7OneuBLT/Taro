# -*- coding: utf-8 -*-
"""発表用デモ映像：座位（シーン「座位_床_柵なし」）で「座る→前へ倒れる→座位へ
戻される」が繰り返される様子をmp4にする。

指示は本セッションで直接受けた（管理用ファイル名は無し）。ユーザーが事前に実測した
前提（2026-08-16）：

    再読込直後   体幹 39.7度   頭 15.40cm   ← 座っている
         1.0秒   体幹 33.6度   頭 13.04cm
         2.0秒   体幹 16.3度   頭  6.96cm
         3.0秒   体幹 13.6度   頭  5.85cm   ← 前へ倒れきった（骨盤は動かず上体だけ折れる）

【なぜ env.taro を手で配線するか（重要）】
`E/scripts/e_toy_env.py` の `_check_posture_fall`（倒れ判定→座り直し）は
`self.taro` が無いと1度も呼ばれない。`env.taro` は `run/taro_setup.py` の
`_setup_postural_gate`/`_setup_righting_damper` が **posture_reflex か
righting_reflex のどちらかが True のときだけ** 配線する（既定は両方 False）。
＝ run/main.py の train と同じ配線に乗せるには、この2つのどちらかを True に
する必要がある（依頼文の「太郎の脳を通す経路で走らせること」の具体的な意味）。
このスクリプトは実際の座位保持の本番実験（`E/experiments/座位保持_学習.json`）
と同じ組み合わせ（posture_reflex=true・righting_reflex=true）で `Taro` を構築する。

【なぜ行動を常にゼロにするか（ユーザーの実測との整合性）】
depend（ユーザー）が示した実測タイムライン（上記）は「脱力（action=0）」で得た
ものであり、このスクリプトが再現すべき挙動もそれ。posture_gate は行動配列への
**マスク**（傾きと逆側の筋を通すだけ）なので、行動そのものがゼロなら
posture_reflex=True でも False でも trunk_tilt への影響はゼロ（マスクしても
ゼロはゼロ）。righting_damper だけは行動に依存しない独自のバイアスを
head_tilt の1アクチュエータへ加える（前庭覚由来）ので、行動ゼロでも
頭の揺れをわずかに抑える方向に働く。＝「体幹の崩れ方は depend の実測どおり、
頭だけ本番の反射がわずかに効く」という一番忠実な組み合わせ。
未学習の脳（act_mean）を通す方式も検討したが、初期化直後の重みで生じる
小さいが非ゼロの出力が depend の実測条件（action=0）と厳密には一致しなくなる
ため採用しなかった（判断。違うなら直す）。

【_posture_seated_qpos の想定外（重要・作業記録に明記した内容と同じ）】
`E/scripts/e_toy_env.py` の `reset_model()` は、`super().reset_model()`
（仰向け＋jitter＋settle の**既定姿勢**）の直後に
`self._posture_seated_qpos = self.data.qpos.copy()` を記録する。
一方、シーンの本当の姿勢（このシーンでは座位）は `reset_to_scene()` が
`env.reset()` の**あとに** `apply_state()` で書き込む。
＝ そのまま使うと、座り直しの復帰先が「座位」ではなく「仰向け＋jitterの
既定姿勢」になってしまう（実際に最初の実装でこの通りに再現し、体幹角度が
既定姿勢の値へジャンプすることを確認した＝下記「検証1」）。
run/main.py の train も reset_to_scene を呼ばない素の env.reset() のままで
このシーンを使うので、**本番の学習でも同じ想定外が起きる**可能性が高い
（このシーンは2026-08-16作成の新しいシーンで、まだ学習に使われていない）。
このスクリプトでは `reset_to_scene()` の直後に
`u._posture_seated_qpos = u.data.qpos.copy()` を明示的に上書きして正しい
座位を復帰先にした（触ってよいファイルの一覧に e_toy_env.py は無いので、
実行時に属性を上書きするだけに留める）。この想定外は作業記録で上に上げる。

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

from run.config import Config                            # noqa: E402
from run.taro_setup import Taro                          # noqa: E402
from run.plugins.common import scene as scene_mod         # noqa: E402

SCENE = "座位_床_柵なし"
OUT_PATH_SIDE = os.path.join(_ROOT, "E", "docs", "figures",
                             "posture_reset_side_2026-08-16.mp4")

DT = 0.01                    # run/trainer.py と同じ（MuJoCoの1物理ステップの秒数）
K = 10                        # 1判断あたりの物理ステップ数（cfg.K既定値）
RENDER_EVERY = 4              # 4物理stepに1回描画 → fps=25
FPS = (1.0 / DT) / RENDER_EVERY
SEC_TOTAL = 15.0
TOTAL_TICKS = int(SEC_TOTAL / DT)          # 1500
SEED = 0

H, W = 480, 640               # 横長（体全体が横から見えるように）


def main():
    print("=" * 78)
    print(" 座位保持の座り直しデモ映像（座位_床_柵なし・15秒・第三者視点）")
    print("=" * 78)

    spec = {
        "name": "posture_reset_demo",
        "scene": SCENE,
        "taro": {
            "actuation": "muscle",
            "posture_reflex": True,
            "righting_reflex": True,
            "righting_reflex_gain": 0.02,
            "posture_fall_deg": 20.0,     # 依頼文の指定値（ユーザーが実測から決定）
            "reward": "posture_height",
        },
        "run": {"seed": SEED, "K": K},
    }
    cfg = Config.from_spec(spec)

    taro_spec = dict(cfg._taro)
    env, sc, hands = scene_mod.build(
        cfg.scene, taro=taro_spec, seed=cfg.seed, verbose=True, hybrid=True)

    # ---- Taro構築（posture_reflex/righting_reflex=Trueなので、taro_setup.py内で
    #   env.unwrapped.taro が自動配線される＝run/main.pyのtrain/viewと同じ配線）----
    taro = Taro(cfg, env, seed=cfg.seed, verbose=True)
    u = env.unwrapped
    assert getattr(u, "taro", None) is taro, \
        "[想定外] env.unwrapped.taro が配線されていない（posture_reflex/righting_reflex " \
        "のどちらかがTrueなら _setup_postural_gate/_setup_righting_damper が配線するはず）"
    print(f"[診断] env.taro配線OK: postural_gate={taro.postural_gate is not None} "
          f"righting_damper={taro.righting_damper is not None} "
          f"posture_fall_deg={cfg.posture_fall_deg}", flush=True)

    # ---- シーンの姿勢（座位）へ戻す ------------------------------------------
    import e_scene
    e_scene.reset_to_scene(env, sc, hands=hands, seed=cfg.seed)

    # ---- 想定外の手当て：座り直しの復帰先を「座位」に上書きする ---------------
    #   （docstring【_posture_seated_qpos の想定外】参照）
    tilt_before_fix = u._posture_trunk_tilt_deg()
    u._posture_seated_qpos = u.data.qpos.copy()
    u._posture_head_height_ref = float(u.data.body("head").xpos[2])
    print(f"[診断] reset_to_scene直後の体幹傾き={tilt_before_fix:.1f}度 "
          f"（座り直しの復帰先をこの姿勢に上書きした）", flush=True)

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
        fall_deg = float(getattr(taro_arg.cfg, "posture_fall_deg", 0.0) or 0.0)
        tilt_now = u._posture_trunk_tilt_deg()
        will_reset = fall_deg > 0.0 and tilt_now < fall_deg
        if will_reset:
            reset_events.append({"tick": tick_counter[0], "tilt_before_reset": tilt_now})
        return orig_check(taro_arg)

    u._check_posture_fall = counting_check

    # ---- カメラ（第三者・横からの構図。骨盤を中心に追従） ---------------------
    m, d = u.model, u.data
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), W)
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), H)
    ren = mujoco.Renderer(m, height=H, width=W)

    def make_side_camera():
        # 【カメラの調整、2026-08-16】最初 distance=1.1・lookat=hip xpos で撮ったところ、
        #   体が画面の中で小さく写り、姿勢が判別しにくかった（実測で確認：0コスト診断
        #   として複数アングルを別撮りし、az=0(正面)では明らかに体育座りに見えるが、
        #   同じ姿勢をdistance=1.1で撮ると小さくて分かりにくかった）。distanceを
        #   0.55へ縮め、lookatを胴体中心（hipとheadの中間程度の高さ）に変えた。
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(m, cam)
        hip_xpos = d.body("hip").xpos
        cam.lookat[:] = [hip_xpos[0] + 0.03, hip_xpos[1], 0.09]
        cam.distance = 0.55
        cam.elevation = -8.0
        cam.azimuth = 90.0        # 真横（world +Y方向から見る。体はworld +X向き）
        return cam

    # ---- 行動ループ：常にゼロ行動（脱力）。docstring参照 -----------------------
    zero_ctrl = np.zeros(env.action_space.shape[0], dtype=np.float32)
    if float(env.action_space.low[0]) >= 0.0:
        zero_ctrl[:] = 0.0    # 筋肉モードは[0,1]＝0が「力を入れない」

    frames_side = []
    tilt_log = []
    tick_counter = [0]

    t0 = time.time()
    for i in range(TOTAL_TICKS):
        tick_counter[0] = i
        env.step(zero_ctrl)
        if i % RENDER_EVERY == 0:
            cam = make_side_camera()
            ren.update_scene(d, camera=cam)
            frames_side.append(ren.render().copy())
        if i % 25 == 0:      # 0.25秒おきに記録（診断ログ用）
            tilt_log.append((i * DT, u._posture_trunk_tilt_deg()))

    elapsed = time.time() - t0
    print(f"[診断] {TOTAL_TICKS}物理tick（{SEC_TOTAL:.0f}秒ぶん）を{elapsed:.1f}秒で実行", flush=True)

    env.close()

    # ---- 動画として書き出す ---------------------------------------------------
    os.makedirs(os.path.dirname(OUT_PATH_SIDE), exist_ok=True)
    h, w, _c = frames_side[0].shape
    vw = cv2.VideoWriter(OUT_PATH_SIDE, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for fr in frames_side:
        vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"\nRECORDED {OUT_PATH_SIDE} ({len(frames_side)}フレーム, {FPS:.0f}fps, "
          f"約{len(frames_side) / FPS:.1f}秒)", flush=True)

    # ---- 報告用の実測 ----------------------------------------------------------
    print(f"\n[結果] 座り直し（座位への復帰）が発動した回数: {len(reset_events)}回", flush=True)
    for ev in reset_events:
        print(f"    tick={ev['tick']:5d} sim={ev['tick'] * DT:5.2f}秒"
              f"  発動時の体幹傾き={ev['tilt_before_reset']:.1f}度", flush=True)
    print("\n[参考] 0.25秒おきの体幹傾き（度）：", flush=True)
    for t_sec, tilt in tilt_log[:20]:
        print(f"    {t_sec:5.2f}秒  {tilt:6.1f}度", flush=True)
    if len(tilt_log) > 20:
        print(f"    ...(以下略、全{len(tilt_log)}点)", flush=True)


if __name__ == "__main__":
    main()
