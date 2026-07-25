"""共通の運動Viewer（測定器＝太郎を外から見る観測装置、目標横断で共有）。

【なぜ・2026-07-24】従来Viewerは D/scripts/d_c5_motor_quality.py の run_view にしかなく、
目標E側で「筋肉モデル+58.8を目視したい」となったときに、私(claude)が e_growth_train.py の中に
省略版のViewerを別途書いてしまった。ユーザー指摘：「目標が変わるごとにViewerを書いてるの?」
→ Viewerは太郎そのものでも環境でもなく、**測定器(観測装置)**。目標C/D/Eどれからも呼ばれるべき
共通ツール。ここに一本化する。[[feedback-core-target-neutral-naming]] の精神。

【呼び出し側が用意するもの】
- env：mujoco env (env.unwrapped.model/data/frame_skip が読める形)
- brain：init_motor_hidden() を持つ。to_env_action(action) メソッドがあれば拮抗筋等の写像も対応
- policy_fn(obs, prev_a, hidden, *, recompute, frac) -> (action, new_hidden)
  boundary(recompute=True) 時のみ policy 再計算、frac=[0,1) は連続制御時の内挿位置
- rescale_action(action, action_space) -> ctrl ndarray
- 好み：reflex_fns=[fn(action) -> action] のリストで反射を差し込める

【設定(環境変数で読む共通ルール、呼び出し側は E_* で上書き可)】
- E_REALTIME(0/1)：等倍速 / 最速。実行中キー操作(. , 0 M)で変更可
- E4_CONTINUOUS(0/1)：1秒ホールド / 連続制御
- E_CTRL_M(int)：連続制御時の内訳(K=100の何tickごとに更新するか、既定100=1秒ホールド相当)

【共通機能】
- 画面右上に速度オーバーレイ(要求x1.0 / 実効x0.9 等)
- キー操作 (. > で2倍速く、, < で2倍遅く、0で等倍、M で最速)
- 60Hzで描画同期(sync)
- 実時間追従(sleepの累積誤差を避けるための目標時刻積上げ)
- terminate/truncate時に自動reset

d_c5_motor_quality.py の run_view から抽出、機能は保つ。
"""
import os
import time

import mujoco
import mujoco.viewer as _mjv
import numpy as np
import torch


def _make_key_callback(speed_ref):
    def _cb(keycode):
        try:
            ch = chr(keycode)
        except ValueError:
            return
        if ch in ".>":
            speed_ref[0] = min(speed_ref[0] * 2 if speed_ref[0] > 0 else 64.0, 64.0)
        elif ch in ",<":
            speed_ref[0] = max(speed_ref[0] / 2 if speed_ref[0] > 0 else 1.0, 0.0625)
        elif ch == "0":
            speed_ref[0] = 1.0
        elif ch in "mM":
            speed_ref[0] = 0.0
        else:
            return
        print(f"  [speed] x{speed_ref[0]:.4g}" if speed_ref[0] > 0
              else "  [speed] MAX (no wait)", flush=True)
    return _cb


def _show_speed_overlay(viewer, speed_ref, eff):
    """右上に速度オーバーレイ表示。d_c5 の実装から流用（3D空間でなくオーバーレイ）。"""
    req = (f"x{speed_ref[0]:.4g}" if speed_ref[0] > 0 else "MAX")
    txt = f"speed {req} (real x{eff:.1f})"
    if getattr(viewer, "_last_speed_txt", None) == txt:
        return
    viewer._last_speed_txt = txt
    try:
        fig = mujoco.MjvFigure()
        mujoco.mjv_defaultFigure(fig)
        fig.title = txt
        fig.flg_legend = 0
        fig.flg_ticklabel[:] = [0, 0]
        fig.figurergba[:] = [0.0, 0.0, 0.0, 0.4]
        vp = viewer.viewport
        w = max(int(vp.width * 0.22), 180)
        h = 46
        rect = mujoco.MjrRect(int(vp.width - w - 10), int(vp.height - h - 10), w, h)
        viewer.set_figures([(rect, fig)])
    except Exception as e:
        print(f"  [{txt}] (overlay unavailable: {type(e).__name__})", flush=True)


def run_viewer(env, brain, policy_fn, rescale_action, *,
               K=100, n_act=None, reflex_fns=(), banner=""):
    """共通Viewer本体。

    Args:
        env: HybridEnv 等。env.unwrapped.model/data/frame_skip を使う。
        brain: init_motor_hidden() を持つ。to_env_action(action) があれば写像に使う。
        policy_fn: callable(obs, prev_a, hidden, *, recompute, frac) -> (action, new_hidden)
        rescale_action: callable(action, action_space) -> np.ndarray
        K: 学習時の1判断あたりの物理ステップ数(既定100)
        n_act: 行動次元(reset時にprev_aを作るため)
        reflex_fns: [callable(action) -> action] のリスト。把握反射・ATNR等を差し込む
        banner: 起動時に表示するメモ(条件表示など)

    環境変数：E_REALTIME, E4_CONTINUOUS, E_CTRL_M
    """
    realtime = os.environ.get("E_REALTIME", "0") == "1"
    continuous = os.environ.get("E4_CONTINUOUS", "0") == "1"
    ctrl_m = int(os.environ.get("E_CTRL_M", str(K)))

    m, d = env.unwrapped.model, env.unwrapped.data
    dt_env = m.opt.timestep * env.unwrapped.frame_skip
    if n_act is None:
        n_act = env.action_space.shape[0]

    speed = [1.0 if realtime else 0.0]
    key_cb = _make_key_callback(speed)

    _ctrl_desc = f"連続制御(ctrl_m={ctrl_m})" if continuous else f"1秒ホールド(K={K})"
    print(f"\n[Viewer] {banner}  制御刻み={_ctrl_desc}  再生={'等倍' if realtime else '最速'}",
          flush=True)
    print("  キー操作: . = 速く / , = 遅く / 0 = 等倍 / M = 最速（待たない）", flush=True)

    obs, _ = env.reset(seed=0)
    hidden = brain.init_motor_hidden()
    prev_a = torch.zeros(n_act)
    ctrl = None
    SYNC_DT = 1.0 / 60.0

    with _mjv.launch_passive(m, d, key_callback=key_cb) as viewer:
        t_wall = time.perf_counter()
        t_draw = 0.0
        eff_t0, eff_sim, eff = time.perf_counter(), 0.0, 0.0
        while viewer.is_running():
            for k in range(K):
                boundary = (k % ctrl_m == 0)
                if continuous:
                    frac = (k % ctrl_m) / ctrl_m
                    a, hidden = policy_fn(obs, prev_a, hidden, recompute=boundary, frac=frac)
                    for fn in reflex_fns:
                        a = fn(a)
                    a_env = brain.to_env_action(a) if hasattr(brain, "to_env_action") else a
                    ctrl = rescale_action(a_env, env.action_space)
                    if boundary:
                        prev_a = a
                elif boundary:
                    a, hidden = policy_fn(obs, prev_a, hidden, recompute=True, frac=0.0)
                    for fn in reflex_fns:
                        a = fn(a)
                    a_env = brain.to_env_action(a) if hasattr(brain, "to_env_action") else a
                    ctrl = rescale_action(a_env, env.action_space)
                    prev_a = a
                obs, r, te, tr, info = env.step(ctrl)
                now = time.perf_counter()
                eff_sim += dt_env
                if now - eff_t0 >= 0.5:
                    eff = eff_sim / (now - eff_t0)
                    eff_t0, eff_sim = now, 0.0
                if now - t_draw >= SYNC_DT:
                    _show_speed_overlay(viewer, speed, eff)
                    viewer.sync()
                    t_draw = now
                if speed[0] > 0:
                    t_wall += dt_env / speed[0]
                    lag = t_wall - time.perf_counter()
                    if lag > 0:
                        time.sleep(lag)
                    elif lag < -0.25:
                        t_wall = time.perf_counter()
                else:
                    t_wall = now
                if te or tr:
                    obs, _ = env.reset()
                    hidden = brain.init_motor_hidden()
                    prev_a = torch.zeros(n_act)
                    break
