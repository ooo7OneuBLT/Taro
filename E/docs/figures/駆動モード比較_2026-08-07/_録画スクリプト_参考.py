# -*- coding: utf-8 -*-
"""駆動モード比較（white / colored / colored+synergy）のmp4を3本作る。

依頼：太郎の自発運動の「駆動モード」を見比べて、どれが一番人間の赤ちゃんっぽいか
目視で判断したい（測定への依頼、2026-08-07）。

【配線状況の確認結果】
`E/scripts/e_viewer.py`（"編集ウィンドウ付き" Viewer、`run.type=edit` の実体）は、
脳を読み込む `_load_brain()` の中で
    taro_spec = {"actuation": ..., "age_months": ..., "model": p}
を**その場で作り直しており**、`os.environ.get("E_NOISE")` を一切参照していない。
`run/config.py` の `Config(dict, ...)` 直接コンストラクタは環境変数を読まない設計
（環境変数を読むのは `Config.from_env()` だけ）なので、
**`E_NOISE=colored` を環境変数で立てても、編集ウィンドウ経由では
ColoredNoiseGenerator は有効化されない（配線が無い）**。これが今回確認できた
「配線未完了」の実体。

一方、`run/taro_setup.py` の `Taro.__init__`（153〜187行目付近、`cfg.noise` を見て
`brain.enable_spinal_babble(...)` を呼ぶ）は、実験ファイル（`taro.noise` キー）
経由なら正しく配線されている。この経路は `run/tools/record_topdown_flat_clip.py`
（2026-08-07既存、太郎の自発運動デモ映像を作った実績スクリプト）と全く同じ手順
（`Config(spec, ...)` → `Taro(cfg, env, ...)` → 決定ごとに `t.brain.explore(mean, std)`
を呼び cv2.VideoWriter で書き出す）で動くことが分かっている。

**「編集ウィンドウ」自体の tkinter コード（e_viewer.py）は一切書き換えていない**
（測定の禁止2）。このスクリプトは `record_topdown_flat_clip.py` と同じ低レベル
API（Config/Taro を直接組み立てる）を使った**新規の使い捨てスクリプト**
（scratchpad配下）で、リポジトリのファイルは1つも変更していない。

【速度（倍速表示バグ）の検証】
過去に見つかった「Viewerの倍速表示が嘘だった」バグ（`doc/検証の落とし穴チェックリスト.md`
項17-表内#4：「sleepを減らすだけの実装で、計算時間が律速だった」＝壁時計sleepの
累積誤差が原因）は、**壁時計のsleepに一切依存しないこの方式では構造的に起こらない**。
DT(1物理stepの秒数)×TOTAL_STEPS(物理step数)＝シミュレーション上の経過秒数を、
RENDER_EVERY・FPSで割った動画の再生時間と**設計上一致させている**
（DT=0.01, RENDER_EVERY=4 → FPS=25。SEC秒分のstepを撮ると動画もSEC秒になる）。
書き出し後に実フレーム数・動画秒数が期待値と一致するかもコードで検証する
（run_viewer.py側の現行実装＝目標時刻積み上げ方式・実効速度オーバーレイ表示、も
別途確認済みで、旧来の「sleepを減らすだけ」の実装からは既に修正されている
＝ライブGUIで見る場合もE_REALTIME=1なら等倍速で問題ない。ただし本スクリプトは
GUIの画面録画を伴わないため、この構造的な検証で十分と判断した）。

【触ってよいファイル】このファイルと、出力先の3本のmp4のみ。
`run/tools/record_topdown_flat_clip.py`・`run/config.py`・`run/taro_setup.py`・
`E/scripts/e_viewer.py` 含め、他のファイルは読むだけで一切変更していない。

【検証】学習・勾配更新は一切行わない（optimizer関連のコードは無し）。
モデルを読み込んで推論（ロールアウト）するだけ。3条件とも同じ乱数シード(0)で
env.reset するので、開始姿勢は3条件で同一になる（比較の公平性）。
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

import e_scene                                            # noqa: E402
from hybrid_env import HybridEnv                          # noqa: E402
from run.config import Config, touch_setting_of           # noqa: E402
from run.taro_setup import Taro, rescale_action            # noqa: E402

MODEL = os.path.join(_R, "E", "logs", "selfmodel_v3", "model_柵なし_線形_seed0.pt")
# 注意：元は一時作業フォルダ（scratchpad）の絶対パスを直書きしていたが、
#   ユーザー名を含むためこの参考コピーでは伏せた（このリポジトリは公開）。
#   実行するときは書き出し先を自分の環境に合わせて指定すること。
OUT_DIR = os.environ.get("E_RECORD_OUT_DIR",
                         os.path.join(_R, "E", "docs", "figures", "noise_compare"))
os.makedirs(OUT_DIR, exist_ok=True)

SCENE_NAME = "新生児_仰向け_柵なし"   # fit-check: シーン一覧.md参照。
                                    # おもちゃなし・柵なし・自発運動の観察に使われてきたシーン。
                                    # 今回の目的（駆動モードの目視比較）と矛盾する記述なし。
AGE_MONTHS = 4.0     # 成長しきった状態（record_topdown_flat_clip.pyと同じ、比較の基準を合わせる）
DT = 0.01            # 1物理stepの秒数（run/trainer.pyと同じ。record_topdown_flat_clip.py踏襲）
RENDER_EVERY = 4      # 4物理stepに1回描画 → fps=25
FPS = (1.0 / DT) / RENDER_EVERY
SEC = 15.0           # 比較のため、5秒より長めに取ってリズム性の違いを見せる
TOTAL_STEPS = int(SEC / DT)          # 1500
K = 10               # 10物理stepに1回、行動を選び直す（record_topdown_flat_clip.pyと同じ）
EXPLORE_STD = 0.174   # 学習初期の実効値（check_movement_units.py等と同じ探索std）
SEED = 0
H, W = 480, 480

# 条件：3条件とも model・scene・age・K・seed・std・長さは完全に同一。
#   変えるのは taro.noise / taro.synergy / taro.beta のみ（単一変数比較）。
CONDITIONS = [
    ("A_white",           {}),                                         # 既定＝白色ガウス
    ("B_colored",         {"noise": "colored", "beta": 0.8, "synergy": False}),
    ("C_colored_synergy", {"noise": "colored", "beta": 0.8, "synergy": True}),
]


def record_condition(tag, extra_spec):
    print("=" * 78)
    print(f" 条件 {tag}  追加設定={extra_spec}")
    print("=" * 78)

    sc = e_scene.load(SCENE_NAME)
    sc["body"]["age_months"] = AGE_MONTHS
    sc["fingerprint"] = None
    env0, _ = e_scene.build(sc, seed=SEED, verbose=False)
    env = HybridEnv(env0)
    env.reset(seed=SEED)

    spec = {"actuation": "muscle", "age_months": AGE_MONTHS, "model": MODEL}
    spec.update(touch_setting_of(MODEL))
    spec.update(extra_spec)
    cfg = Config(spec, {"seed": SEED, "K": K}, scene=sc["name"], name=f"noise-compare-{tag}")
    t = Taro(cfg, env, seed=SEED, verbose=True)
    st = t.init_state(t.first_obs)

    print(f"[設定] noise={cfg.noise} synergy={cfg.synergy} beta={cfg.beta} "
          f"K={cfg.K} seed={SEED} 探索std={EXPLORE_STD} touch={spec.get('touch')}",
          flush=True)

    m, d = env0.unwrapped.model, env0.unwrapped.data
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), W)
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), H)
    ren = mujoco.Renderer(m, height=H, width=W)

    def make_camera():
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(m, cam)
        cam.lookat[:] = d.body("upper_body").xpos
        cam.distance = 1.05
        cam.elevation = -40.0     # 斜め上から全身が見える角度（真上90度と真横0度の中間）
        cam.azimuth = 100.0
        return cam

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

        if i % RENDER_EVERY == 0:
            cam = make_camera()
            ren.update_scene(d, camera=cam)
            frames.append(ren.render().copy())

    env0.close()

    out_path = os.path.join(OUT_DIR, f"駆動モード比較_{tag}.mp4")
    h, w, _c = frames[0].shape
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for fr in frames:
        vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    vw.release()

    expect_frames = TOTAL_STEPS // RENDER_EVERY
    dur = len(frames) / FPS
    ok = (len(frames) == expect_frames) and (abs(dur - SEC) < 1e-6)
    print(f"\nRECORDED {out_path} ({len(frames)}フレーム, {FPS:.0f}fps, "
          f"約{dur:.2f}秒)", flush=True)
    print(f"[速度検証] 期待フレーム数={expect_frames} 実フレーム数={len(frames)} "
          f"期待秒数={SEC:.2f} 動画秒数={dur:.2f} → "
          f"{'OK：構造上、等倍速と一致' if ok else '不一致（要調査）'}", flush=True)

    aa = np.asarray(act_abs_per_decision)
    print(f"[診断] 各decisionの行動の絶対値平均: mean={aa.mean():.4f} "
          f"min={aa.min():.4f} max={aa.max():.4f}"
          f"（全てゼロ近辺なら方策が凍結している可能性）", flush=True)
    return out_path, ok


def main():
    print("=" * 78)
    print(" 駆動モード比較（white / colored / colored+synergy）3本撮り")
    print("=" * 78)
    results = []
    for tag, extra in CONDITIONS:
        results.append(record_condition(tag, extra))
    print("\n完了：")
    all_ok = True
    for p, ok in results:
        print(f"  {p}  速度検証={'OK' if ok else 'NG'}")
        all_ok = all_ok and ok
    print(f"\n全条件の速度検証: {'OK（全て等倍速の構造）' if all_ok else '一部NG'}")


if __name__ == "__main__":
    main()
