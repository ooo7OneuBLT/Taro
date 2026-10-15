"""core_v2 の脳を、旧太郎の環境（体・部屋・親）と一緒に回す薄いループ（段階2・工程2）。

`run/trainer.py` は旧太郎（Taro）と密結合なので使わない。手本は `run/main.py` の
measure（脳なしで環境を回す）。

1回（脳の tick）＝環境の K=10歩（DT=0.01秒）＝0.1秒（10Hz）。体は動かさない（行動はゼロ。
筋肉モードでは 0＝力を入れない）。毎回の流れ：
  1. 環境を K 歩進める。親の発話は1歩にしか出ないので、K歩ぶんの info を全部見て集める
  2. 左目の画像を eyein へ、発話の文字列を earin へ入れて brain.tick()
  3. eyeout が方向を返したら、環境の中の目を動かす仕組みへ set_map_target(h, v)
     （同じ回に画像を見て決めた指令は、次の K 歩に効く）

乱数は4系統（torch・numpy・random・環境の np_random）を試行の頭で撒く（trainer.py:127-150 と同じ考え）。
"""

from __future__ import annotations

import os
import random
import sys
from typing import Any, Callable

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
for _p in (REPO, os.path.join(REPO, "run"), os.path.join(REPO, "run", "world"),
           os.path.join(REPO, "taro_core", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

K = 10                 # 脳1回あたりの環境の歩数（trainer.py:49）
DT = 0.01              # 環境1歩の秒数（run/config.py:424-427）
OBJECTS = {"ball": "test_object1", "cup": "test_object2", "distractor": "test_object3"}


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_env(scene: str, word: str | None, seed: int = 0, word_t: float = 4.0):
    """段階2の場面を作る。word を渡すと、word_t 秒に親がそれを言う（場面の既定を上書き）。"""
    from run.world import scene as scene_mod
    world = {}
    if word is not None:
        world["word_test"] = {"schedule": [{"t": float(word_t), "word": word}]}
    env, sc, hands = scene_mod.build(scene, taro={"orienting_reflex": True, "vision": True},
                                     seed=seed, world=world or None)
    u = env.unwrapped
    if getattr(u, "_orienting", None) is None:
        raise RuntimeError("環境に目を動かす仕組み（_orienting）が無い。taro.orienting_reflex=True が要る")
    if getattr(u, "taro", None) is not None:
        raise RuntimeError("env.taro が設定されている。core_v2 は旧太郎を環境に載せない（parent.py:901）")
    return env


def close_env(env) -> None:
    from run.trainer import close_env as _close
    _close(env)


def zero_action(env) -> np.ndarray:
    return np.zeros(env.action_space.shape[0], dtype=np.float32)


def gaze_angles(env) -> dict[str, float]:
    u = env.unwrapped
    return {k: u._gaze_angle_to(body) for k, body in OBJECTS.items()}


def frame_at(env) -> np.ndarray:
    """今の左目のカメラの向き（列＝右・上・後ろ）。試行の頭で控え、視線の軌跡の基準にする。"""
    u = env.unwrapped
    cid = int(u.model.camera("eye_left").id)
    return np.array(u.data.cam_xmat[cid], dtype=float).reshape(3, 3).copy()


def directions(env, R0: np.ndarray) -> dict[str, float]:
    """視線と3つの物の方向を、基準の向き R0 の右・上の角度（度）で表す（絵の軌跡用）。"""
    u = env.unwrapped
    cid = int(u.model.camera("eye_left").id)
    eye = np.array(u.data.cam_xpos[cid], dtype=float)
    vecs = {"gaze": u._gaze_dir()}
    for k, body in OBJECTS.items():
        v = np.array(u.data.body(body).xpos, dtype=float) - eye
        vecs[k] = v / np.linalg.norm(v)
    out = {}
    for k, v in vecs.items():
        out[f"x_{k}"] = float(np.degrees(np.arcsin(np.clip(v @ R0[:, 0], -1, 1))))
        out[f"y_{k}"] = float(np.degrees(np.arcsin(np.clip(v @ R0[:, 1], -1, 1))))
    return out


def path_record(brain) -> dict[str, Any]:
    """通り道の矢印（wiring.PATH_FORMATS）の今の中身を、CSV に書ける数へ要約する。
    brain.buffer は「この回に各係が出した物」＝「次の回に各係へ届く物」。
    荷札＝長さ（0なら空）、地図＝最大値と最大のマス、数1つ＝値、筋への指令＝値、
    物ごとの組＝個数、文字＝そのまま。係の中の状態も少し足す。"""
    from .formats import Fmt, GRID
    from .wiring import PATH_FORMATS
    rec: dict[str, Any] = {}
    for (src, dst), (fmt, _size) in PATH_FORMATS.items():
        v = brain.buffer[(src, dst)]
        k = f"{src}>{dst}"
        if fmt is Fmt.TAG:
            rec[k] = round(float(torch.linalg.vector_norm(v)), 3)
        elif fmt is Fmt.MAP:
            mx = float(v.max())
            r, c = divmod(int(torch.argmax(v)), GRID)
            rec[k] = round(mx, 3)
            rec[k + "@"] = f"{r},{c}" if mx > 0 else ""
        elif fmt is Fmt.AMOUNT:
            rec[k] = round(float(v), 3)
        elif fmt is Fmt.MOTOR:
            rec[k] = "" if float(torch.linalg.vector_norm(v)) == 0 else f"{float(v[0]):.3f},{float(v[1]):.3f}"
        elif fmt is Fmt.TAG_AND_MAP:
            rec[k] = len(v)
        else:
            rec[k] = v
    goal = brain.boxes["goal"]
    rec["goal.age"] = goal.age if goal.held is not None else ""
    rec["goal.cleared_by"] = goal.cleared_by
    rec["orienteye.reason"] = brain.boxes["orienteye"].last_reason
    return rec


def run_trial(brain, env, *, seed: int, duration_s: float = 10.0,
              on_tick: Callable[[dict[str, Any]], None] | None = None,
              manual_eye: Callable[[int], tuple[float, float] | None] | None = None,
              use_brain: bool = True) -> list[dict[str, Any]]:
    """1試行を回し、脳1回ごとの記録（辞書）の列を返す。

    manual_eye : 確かめ用。tick 番号を受けて方向を返すと、脳の代わりにそれを目へ入れる
    use_brain  : False なら脳を回さない（環境だけ・確かめ用）
    """
    seed_all(seed)
    obs, _ = env.reset(seed=seed)
    u = env.unwrapped
    brain.reset()
    R0 = frame_at(env)
    act = zero_action(env)
    rows = []
    n_ticks = int(round(duration_s / (K * DT)))
    for i in range(n_ticks):
        heard = []
        for _ in range(K):
            obs, _r, term, trunc, info = env.step(act)
            utt = info.get("parent_utterance")
            if utt and utt.get("text"):
                heard.append(str(utt["text"]))
            if term or trunc:
                raise RuntimeError(f"試行の途中で環境が終わった（t={u.data.time:.2f}）")
        text = " ".join(heard)
        cmd = None
        to_env = {}
        if use_brain:
            to_env = brain.tick({"eyein": obs["eye_left"], "earin": text})
            cmd = to_env.get("eyeout")
        if manual_eye is not None:
            cmd = manual_eye(i)
        if cmd is not None:
            u._orienting.set_map_target(float(cmd[0]), float(cmd[1]))
        row = {"tick": i, "t": float(u.data.time), "heard": text,
               "cmd_h": None if cmd is None else float(cmd[0]),
               "cmd_v": None if cmd is None else float(cmd[1])}
        row.update({f"ang_{k}": v for k, v in gaze_angles(env).items()})
        row.update(directions(env, R0))
        if on_tick is not None:
            on_tick(row)
        rows.append(row)
    return rows
