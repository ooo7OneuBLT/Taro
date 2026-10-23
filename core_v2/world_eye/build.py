"""run_v2.build_env と同じ引数で、目だけの場面を作る。

場面は MIMo と同じ run/scenes/<名前>.json を読む（物の形・色・置き方・親・語の予定表・段階6の2つずつの予定表）。
体（body）の欄は、画像の大きさ（vision_px）だけを使う。目だけの場面にしかない設定は world.eye_world に書く：
  {"table": false, "parent_figure": false, "objects": [...足す物...], "script": [...決まった時刻の見せる・隠す・言う...]}

記録：前後の画像と events.csv を log_dir（既定 G/logs/目だけの場面/<場面>_<日時>/）の下に、試行（reset）ごとのフォルダで残す。
環境変数 TARO_EYE_LOG で置き場を変えられる。TARO_EYE_LOG=off なら残さない（速さを測るときだけ）。
"""

from __future__ import annotations

import datetime
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
for _p in (REPO, os.path.join(REPO, "run", "scene_tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def resolve_scene(scene: str, word: str | None = None, word_t: float = 4.0, silent: bool = False,
                  world_extra: dict | None = None) -> dict:
    """run_v2.build_env と同じ重ね方（語を渡すと word_t 秒に言う／silent なら何も言わない／world_extra を重ねる）。"""
    import scene_io
    sc = scene_io.load(scene)
    world = {}
    if word is not None:
        world["word_test"] = {"schedule": [{"t": float(word_t), "word": word}]}
    elif silent:
        world["word_test"] = {"schedule": []}
    if world_extra:
        world = scene_io._merge(world, world_extra)
    if world:
        sc["world"] = scene_io._merge(sc["world"], world)
    sc.setdefault("name", os.path.basename(str(scene)))
    return sc


def default_log_dir(scene: str) -> str | None:
    v = os.environ.get("TARO_EYE_LOG")
    if v and v.lower() == "off":
        return None
    root = v or os.path.join(REPO, "G", "logs", "目だけの場面")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return os.path.join(root, f"{os.path.basename(str(scene))}_{stamp}")


def build_env(scene: str, word: str | None = None, seed: int = 0, word_t: float = 4.0,
              silent: bool = False, world_extra: dict | None = None, log_dir: str | None = "auto",
              save_images: bool = True):
    from .env import EyeWorldEnv
    sc = resolve_scene(scene, word, word_t, silent, world_extra)
    if log_dir == "auto":
        log_dir = default_log_dir(scene)
    return EyeWorldEnv(sc, seed=seed, log_dir=log_dir, save_images=save_images)


def close_env(env) -> None:
    env.close()
