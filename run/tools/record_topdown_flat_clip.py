# -*- coding: utf-8 -*-
"""発表用デモ映像：自発運動（もがき運動）の区切りの細かさを、真上からの
   見下ろし構図で見せるmp4を作る（2026-08-07長井研発表・付録スライド用）。

仕様：作業記録（非公開）
（ファイル名は目安。実際の指示は同日のチャット指示）

【背景・このスクリプトが何の主張を裏付けるためのものか】
スライドの主張は「太郎の動きの区切り間隔は74〜80ミリ秒、人間は190ミリ秒＝
約2.4倍細かい」。この数値は **2026-07-31 に既に実測・確定済み**
（`E/docs/研究日誌.md` 同日の記録、`run/tools/check_movement_units.py` で測定）。
このスクリプトは**新しい測定をしない**。確定済みの数値がどんな動きから
出てきたのかを、実際の映像として見せるだけ。

【なぜこのモデルを使うか（依頼時の案から変更した理由）】
依頼書の初期案は `E/models/growth_curriculum/stage0_age0_MUSCLE_8000_seed0.pt`
だったが、これは「目標C自己モデルのmargin+58.8」という**別の実績**に対応する
モデルで、74〜80msの実測とは無関係と判明したため使わない。
74〜80msの実測に実際に使われたのは次の4本（2026-07-31・18000ステップ・
0→4ヶ月成長・筋肉モード）：
    E/logs/selfmodel_v3/model_柵なし_線形_seed0.pt        （触覚なし seed0）
    E/logs/selfmodel_v3/model_柵なし_線形_seed1.pt        （触覚なし seed1）
    E/logs/selfmodel_v3_touch/model_柵なし_線形_触覚_seed0.pt  （触覚あり seed0）
    E/logs/selfmodel_v3_touch/model_柵なし_線形_触覚_seed1.pt  （触覚あり seed1）
日誌の実測値は「触覚なし12.7個/秒・78.5ms／触覚あり12.5個/秒・80.0ms」で、
4本とも12.5〜13.5個/秒（間隔74〜80ms）の範囲に収まる。このうち
**触覚なし・seed0**（一番シンプルで、78.5msが引用範囲74〜80msにちょうど
収まる）を採用する。

【なぜこのシーンを使うか】
上記4本の学習に使われたシーン `新生児_仰向け_柵なし`
（`run/scene_tools/scene_io.py` の `load()` が読む名前）が、
world.fence=false / world.toy.enabled=false / world.plain=true と、
既に「おもちゃなし・柵なし・平坦な床・仰向け」の要件を満たしている。
新しいシーンを作る必要はない。

【探索ノイズ std=0.174 を使う理由（決定的行動を使わない理由）】
74〜80msの実測は「探索ノイズ込みの自発運動（もがき運動）」を見た結果であって、
決定的行動（`act_mean` のみ）ではない。`check_movement_units.py` の実測ループが
`t.brain.explore(mean, torch.full_like(mean, 0.174))` を使っているのと同じ理由で、
このデモでも同じ探索ノイズを使う。決定的行動だけを使うと、震動のない
（実測と異なる）滑らかな動きを撮ってしまい、デモの趣旨（震動を見せる）を外す。

【行動ループ・環境構築の手順】
`run/tools/check_movement_units.py` 18〜26行目（importパス）・107〜163行目
（環境構築・行動ループ）と同じ手順を踏襲する（真似るのではなく、この手順
そのものが74〜80msという数値を生んだ手順だから）。

【カメラの描画方式】
`E/scripts/e_body_shot.py` 46〜93行目の方式（`mujoco.Renderer` を直接使い、
`mujoco.MjvCamera` で真上アングルを手動指定する）を参考にする。ただし
e_body_shot.py は体だけを見せるため床(world)・柵(fence)のgeomを透明にしているが、
このシーンは元々 fence=false・toy無効なので、床を消す処理は行わない
（平らな床の上に寝ていることを見せたいので、むしろ床は見せる）。

【録画の長さ・fps】
`run/tools/record_self_touch_clip.py` と同じ形式：DT=0.01秒/物理step、
RENDER_EVERY=4（4stepに1回描画）→fps=25。5秒間＝物理step500回分
（decision 50回ぶん、1decisionあたり10 step＝K=10）。

【触ってよいファイル】このファイルと、出力先の1つのmp4ファイルのみ。
他のファイル（check_movement_units.py・e_body_shot.py・record_self_touch_clip.py
含む）は読むだけで、一切変更していない。

【検証】学習・勾配更新は一切行わない（optimizer関連のコードは無し）。
モデルを読み込んで推論（ロールアウト）するだけ。
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
_R = r"C:\claude\AI\Taro"
for p in ("run/scene_tools", "D/scripts", "taro_core/src/body", "taro_core/src/brain",
          "taro_core/src/senses", "taro_core/src/wrapper", "MIMo", ""):
    sys.path.insert(0, os.path.join(_R, p))
os.chdir(_R)

import cv2                                              # noqa: E402
import mujoco                                            # noqa: E402
import numpy as np                                       # noqa: E402
import torch                                              # noqa: E402

import scene_io                                            # noqa: E402
from hybrid_env import HybridEnv                          # noqa: E402
from run.config import Config, touch_setting_of           # noqa: E402
from run.taro_setup import Taro, rescale_action            # noqa: E402
from run.plugins.common.movement_units import MovementUnits  # noqa: E402

MODEL = os.path.join(_R, "E", "logs", "selfmodel_v3", "model_柵なし_線形_seed0.pt")
OUT_PATH = os.path.join(_R, "E", "docs", "figures",
                         "spontaneous_movement_topdown_flat_2026-08-03.mp4")

DT = 0.01                 # 1物理stepの秒数（run/trainer.pyと同じ）
RENDER_EVERY = 4           # 4物理stepに1回描画 → fps=25
FPS = (1.0 / DT) / RENDER_EVERY
SEC = 5.0
TOTAL_STEPS = int(SEC / DT)         # 500
K = 10                              # 10物理stepに1回、行動を選び直す
EXPLORE_STD = 0.174                 # check_movement_units.py と同じ探索ノイズ
SEED = 0

H, W = 480, 480            # 真上から見るので正方形にする


def main():
    print("=" * 78)
    print(" 自発運動の区切り（74〜80ms）デモ映像を、真上からの見下ろし構図で撮る")
    print("=" * 78)

    # ---- ① シーン・環境（check_movement_units.py 119〜124行目と同じ） --------
    sc = scene_io.load("新生児_仰向け_柵なし")
    sc["body"]["age_months"] = 4.0     # 成長しきった状態（4ヶ月）＝実測と同じ
    sc["fingerprint"] = None
    env0, _ = scene_io.build(sc, seed=SEED, verbose=False)
    env = HybridEnv(env0)
    env.reset(seed=SEED)

    # ---- ② 脳（check_movement_units.py 126〜130行目と同じ）-------------------
    spec = {"actuation": "muscle", "age_months": 4.0, "model": MODEL}
    spec.update(touch_setting_of(MODEL))
    cfg = Config(spec, {"seed": SEED, "K": K}, scene=sc["name"], name="topdown-demo")
    t = Taro(cfg, env, seed=SEED, verbose=False)
    st = t.init_state(t.first_obs)

    print(f"[設定] シーン={sc['name']}  age_months={sc['body']['age_months']}"
          f"  actuation=muscle  model={os.path.relpath(MODEL, _R)}"
          f"  K={cfg.K}  seed={SEED}  探索std={EXPLORE_STD}"
          f"  touch={spec.get('touch')}", flush=True)

    # ---- ③ 診断用プラグイン（movement_units・0コスト診断として参考値を出す）----
    class _Ctx:
        pass
    ctx = _Ctx()
    ctx.env, ctx.model, ctx.data = env, env0.unwrapped.model, env0.unwrapped.data
    mu = MovementUnits({})
    mu.setup(ctx)
    print(f"[参考] movement_units: 1ステップ={mu.dt * 1000:.1f}ms "
          f"最短間隔={mu.min_gap_steps}ステップ", flush=True)

    # ---- ④ カメラ・描画器の準備（e_body_shot.py 76〜89行目を踏襲）------------
    m, d = ctx.model, ctx.data
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), W)
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), H)
    ren = mujoco.Renderer(m, height=H, width=W)

    def make_camera():
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(m, cam)
        cam.lookat[:] = d.body("upper_body").xpos
        cam.distance = 0.95
        cam.elevation = -89.0      # 真上
        cam.azimuth = 90.0
        return cam

    # ---- ⑤ 行動ループ（check_movement_units.py 141〜163行目と同じ手順）------
    #   注意：決定的行動(act_mean)ではなく explore(std=0.174) を使う。
    #     実測(74〜80ms)は探索ノイズ込みの自発運動を見た結果であって、
    #     決定的行動だけでは震動のない別の動きになってしまう（このタスクの核心）。
    act = np.zeros(env.action_space.shape[0], dtype=np.float32)
    obs = None
    frames = []
    act_abs_per_decision = []
    for i in range(TOTAL_STEPS):
        if i % K == 0:
            if obs is not None:
                st["obs"] = obs
            sv = t.fusion.encode(st["obs"])
            cf = t.target_fusion.encode(st["obs"]).detach()
            z, _k, _r, hn = t.infer_latent(sv, st["prev_a"], cf, st["hidden"])
            mean = t.act_mean(z.detach())
            a, _lp = t.brain.explore(mean, torch.full_like(mean, EXPLORE_STD))
            a = a.detach()
            act_abs_per_decision.append(float(a.abs().mean().item()))
            st["hidden"], st["prev_a"] = hn.detach(), a
            act = rescale_action(t.brain.to_env_action(a),
                                 env.action_space).astype(np.float32)
        obs = env.step(act)[0]
        mu.on_step(ctx)

        if i % RENDER_EVERY == 0:
            cam = make_camera()
            ren.update_scene(d, camera=cam)
            frames.append(ren.render().copy())

    env0.close()

    # ---- ⑥ 動画として書き出す（record_self_touch_clip.py 170〜178行目相当）--
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    h, w, _c = frames[0].shape
    vw = cv2.VideoWriter(OUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for fr in frames:
        vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"\nRECORDED {OUT_PATH} ({len(frames)}フレーム, {FPS:.0f}fps, "
          f"約{len(frames) / FPS:.1f}秒)", flush=True)

    # ---- ⑦ 0コスト診断：方策が凍結（振幅ゼロ）していないか ---------------------
    aa = np.asarray(act_abs_per_decision)
    print(f"[診断] 各decisionの行動の絶対値平均: "
          f"mean={aa.mean():.4f} min={aa.min():.4f} max={aa.max():.4f}"
          f"（全てゼロ近辺なら方策が凍結している可能性）", flush=True)

    # ---- ⑧ 参考値：この約500stepの軌道でmovement_unitsを見る -----------------
    #   注意：日誌の実測は3000stepで、ここは500stepしかない。ばらつきが大きく、
    #     これは「近い値が出るかの参考確認」であって新しい統計的主張ではない。
    rep = mu.report(ctx)
    print("\n[参考・500stepぶんのmovement_units（日誌の3000step実測との桁比較用）]")
    for k, v in rep.items():
        print(f"    {k:22s} {v}")
    print("\n注意：500stepは日誌の実測(3000step)より短くばらつきが大きい。"
          "これは参考値であり、新しい統計的主張として扱わないこと。")


if __name__ == "__main__":
    main()
