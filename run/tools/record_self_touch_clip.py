"""発表用デモ映像：学習済み太郎が自己接触（頭に手が触れる）する瞬間のmp4を作る。

仕様：作業記録（非公開）

【何をするか】
`E/logs/contact_reward_compare/seed4_model.pt`（学習済みモデル）を、対応する
`seed4.meta.json` の設定で読み込み、決定的な行動（`taro.act_mean`、探索なし）で
進めながら低頻度でローリングバッファに描画し続け、`taro.double_touch.detect()`が
頭への自己接触（ダブルタッチ）を検出した瞬間の前後を切り出してmp4に書き出す。

【なぜ常時描画するか】
接触の**3秒前**からの動画が要る。事前に描画していない過去の状態は、後から
描画し直すことができない（物理シミュレーションはリプレイ可能な描画バッファを
持たない）。そのため「ヒットするまで高速に進める」ことと「3秒前から撮る」ことを
両立させるには、常に低頻度で描画してローリングバッファ（`collections.deque`、
上限長）に貯め続けるしかない。

【触ってよいファイル】このファイルのみ。他は一切変更しない
（`run/`の共通ファイルは並行して他の実装作業が触っているため）。

【render_modeを構築後に付け足す件】
`gymnasium/envs/mujoco/mujoco_env.py`のMujocoEnv.render()は
`self.mujoco_renderer.render(self.render_mode)`を呼ぶだけで、
`MujocoRenderer.render(render_mode)`は呼ばれた瞬間の引数から遅延生成する
（`mujoco_rendering.py` 745行、`_get_viewer`774行）。コンストラクタに渡す必要はない。
`HybridEnv`（`taro_core/src/wrapper/hybrid_env.py`）は`gymnasium.Wrapper`の
サブクラスなので、`.render()`と`.unwrapped`はラップされたままでも正しく動く
（Wrapperの既定実装が委譲する）。
"""
import collections
import json
import os
import random
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cv2                                              # noqa: E402
import numpy as np                                      # noqa: E402
import torch                                             # noqa: E402

from run.config import Config                            # noqa: E402
from run.taro_setup import Taro, rescale_action, to_tensor  # noqa: E402
from run.plugins.common import scene as scene_mod         # noqa: E402

META_PATH = os.path.join(_ROOT, "E", "logs", "contact_reward_compare", "seed4.meta.json")
MODEL_PATH = os.path.join(_ROOT, "E", "logs", "contact_reward_compare", "seed4_model.pt")
OUT_PATH = os.path.join(_ROOT, "E", "docs", "figures", "self_touch_clip_2026-08-03.mp4")

DT = 0.01                    # run/trainer.py と同じ（MuJoCoの1物理ステップの秒数）
RENDER_EVERY = 4              # 物理ステップ何回に1回描画するか
FPS = (1.0 / DT) / RENDER_EVERY     # 等速再生になるfps（=25）
SEC_BEFORE = 3.0
SEC_AFTER = 2.0
MAX_DECISIONS = 8000          # まずこの値で試す。見つからなければ延長を判断して報告する


def main():
    with open(META_PATH, encoding="utf-8") as f:
        spec = json.load(f)
    cfg = Config.from_spec(spec)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    taro_spec = dict(cfg._taro)
    if cfg.age_months is not None:
        taro_spec["age_months"] = float(cfg.age_months)
    env, scene_dict, hands = scene_mod.build(
        cfg.scene, taro=taro_spec, seed=cfg.seed, verbose=True, hybrid=True)

    taro = Taro(cfg, env, seed=cfg.seed, verbose=True)
    # 注意：meta.jsonのtaro欄に"model"キーが無いので、Taro.__init__は自動で
    #   読み込まない。ここで明示的に読み込む（二重読込にはならない）。
    taro._load(MODEL_PATH)

    # ---- render_modeを構築後に付け足す。0コスト診断で必ず確認する ------------
    env.unwrapped.render_mode = "rgb_array"
    obs, _ = env.reset(seed=cfg.seed)
    test_frame = env.render()
    assert test_frame is not None, \
        "render_modeの後付けが効いていない。実装を止めて報告すること"
    print(f"[診断] render_modeの後付けOK：フレーム形状={test_frame.shape}"
          f" dtype={test_frame.dtype} std={float(np.std(test_frame)):.3f}", flush=True)

    # ---- ローリングバッファの用意 --------------------------------------------
    maxlen = int(SEC_BEFORE * FPS) + 2
    buf = collections.deque(maxlen=maxlen)

    toucher_name = f"{cfg.reach_arm_side}_palm"
    hidden = taro.brain.init_motor_hidden()
    prev_a = torch.zeros(taro.n_act)
    tick = 0
    hit_tick = None
    hit_info = None
    after_frames = []
    after_needed = int(SEC_AFTER * FPS)
    act_abs_accum = []

    t0 = time.time()
    n_decisions_done = 0
    for i in range(MAX_DECISIONS):
        n_decisions_done = i + 1
        sv = taro.fusion.encode(obs)
        cf = taro.target_fusion.encode(obs).detach()
        z, _kl, _rc, hn = taro.infer_latent(sv, prev_a, cf, hidden)
        z = z.detach()
        a = torch.clamp(taro.act_mean(z), -1.0, 1.0).detach()
        act_abs_accum.append(float(a.abs().mean().item()))
        a_env = taro.brain.to_env_action(a)
        ctrl = rescale_action(a_env, env.action_space)
        done = False
        te = tr = False
        for k in range(cfg.K):
            obs, r, te, tr, info = env.step(ctrl)
            tick += 1
            if tick % RENDER_EVERY == 0:
                fr = env.render()
                if fr is not None:
                    if hit_tick is None:
                        buf.append(fr)
                    else:
                        after_frames.append(fr)
            if hit_tick is None:
                h, toucher_p, touched_p = taro.double_touch.detect(
                    taro.target_fusion.touch, to_tensor(obs["touch"]),
                    toucher_name=toucher_name)
                if h:
                    hit_tick = tick
                    hit_info = {"tick": tick, "decision": i,
                                "toucher_presence": toucher_p, "touched_presence": touched_p}
                    print(f"[hit] tick={tick} decision={i} "
                          f"toucher={toucher_p:.3f} head={touched_p:.3f}", flush=True)
            if hit_tick is not None and len(after_frames) >= after_needed:
                done = True
                break
            if te or tr:
                break
        hidden = hn.detach()
        prev_a = a
        if done:
            break
        if te or tr:
            obs, _ = env.reset()
            hidden = taro.brain.init_motor_hidden()
            prev_a = torch.zeros(taro.n_act)

    elapsed = time.time() - t0
    print(f"探索にかかった時間: {elapsed:.1f}秒（{n_decisions_done}判断ぶん、"
          f"1判断あたり{elapsed / max(n_decisions_done, 1):.4f}秒）", flush=True)

    if hit_tick is None:
        act_arr = np.array(act_abs_accum) if act_abs_accum else np.zeros(1)
        print("[想定外] MAX_DECISIONS以内に自己接触（頭へのダブルタッチ）が"
              "見つからなかった。", flush=True)
        print(f"  探索した判断数: {n_decisions_done} / {MAX_DECISIONS}"
              f"（物理tick換算 約{n_decisions_done * cfg.K}）", flush=True)
        print(f"  実測にかかった時間: {elapsed:.1f}秒", flush=True)
        print(f"  決定的行動(act_mean)の平均絶対値: "
              f"mean={float(act_arr.mean()):.6f} max={float(act_arr.max()):.6f}"
              f"（ほぼ0なら方策が凍結・静止している可能性）", flush=True)
        env.close()
        return

    frames = list(buf) + after_frames
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    h, w, _c = frames[0].shape
    vw = cv2.VideoWriter(OUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for fr in frames:
        vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"RECORDED {OUT_PATH} ({len(frames)}フレーム, {FPS:.0f}fps, "
          f"約{len(frames) / FPS:.1f}秒)", flush=True)
    print(f"[hit_info] {hit_info}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
