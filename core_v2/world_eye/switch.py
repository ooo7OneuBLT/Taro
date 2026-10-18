"""道具に --world eye|mimo を足すための切り替え。

道具（core_v2/tools/*・G/scripts/*）は run_v2.build_env / run_v2.close_env で環境を作る。select("eye") を呼ぶと、
その2つをこのプロセスの中だけ目だけの場面のものに差し替える（run_v2.py 自体は書き換えない）。差し替えたことは
1行表示する。select("mimo") は何もしない（今までどおり）。

道具の側は2行だけ：
    from core_v2.world_eye.switch import pop_world_flag, select
    select(pop_world_flag(argv))          # argv から --world を取り除いて、その世界を選ぶ
"""

from __future__ import annotations

WORLDS = ("mimo", "eye")


def pop_world_flag(argv: list, default: str = "mimo") -> str:
    """argv（list、その場で書き換える）から `--world X` か `--world=X` を取り除き、X を返す。"""
    world = default
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--world" and i + 1 < len(argv):
            world = argv[i + 1]
            del argv[i:i + 2]
            continue
        if a.startswith("--world="):
            world = a.split("=", 1)[1]
            del argv[i]
            continue
        i += 1
    if world not in WORLDS:
        raise ValueError(f"--world は {WORLDS} のどれか（{world}）")
    return world


def select(world: str) -> str:
    from .. import run_v2
    if world == "mimo":
        return world
    from . import build
    run_v2.build_env = build.build_env
    run_v2.close_env = build.close_env
    run_v2.WORLD = "eye"
    print("[世界] 目だけの場面（core_v2/world_eye）で回す。前後の画像は G/logs/目だけの場面/ の下", flush=True)
    return world
