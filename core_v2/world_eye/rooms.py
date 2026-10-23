"""段階9：部屋3つ（食事・お風呂・遊び）。仕様書 §2「部屋を3つにする（今のまわりの市松模様の部屋の代わり）。
床・壁・家具（机・湯船・棚）が部屋ごとに違う」。

scene.build_xml の cfg に "room": "食事"|"お風呂"|"遊び" を書くと、市松模様の壁と無地の床の代わりにこの部屋を作る。
寸法は scene.py の壁と同じ（太郎のまわり 1.2m に4枚、高さ2m）。家具は物の置き場（目から 25〜45cm 先、目より少し下）の
向こうか下に置き、物と重ならないようにする。模様は MuJoCo の組み込みの模様（格子・点）だけ［Tier3・形と色はモデル上の選択］。
"""

from __future__ import annotations

from .scene import EYE_MID

ROOMS = ("食事", "お風呂", "遊び")

# 部屋ごとの見た目：床・壁の模様（builtin, rgb1, rgb2, 繰り返し）と家具
_SPEC = {
    "食事": {"floor": ("checker", (0.55, 0.38, 0.22), (0.5, 0.34, 0.2), (8, 40)),
             "wall": ("checker", (0.86, 0.82, 0.72), (0.78, 0.74, 0.64), (8, 6))},
    "お風呂": {"floor": ("checker", (0.85, 0.9, 0.95), (0.6, 0.75, 0.9), (40, 40)),
              "wall": ("checker", (0.8, 0.9, 0.95), (0.72, 0.84, 0.92), (24, 16))},
    "遊び": {"floor": ("checker", (0.4, 0.62, 0.38), (0.37, 0.58, 0.35), (60, 60)),
            "wall": ("dots", (0.96, 0.92, 0.75), (0.95, 0.65, 0.4), (6, 5))},
}


def _f(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in v)


def _tex(name: str, spec) -> list[str]:
    kind, rgb1, rgb2, rep = spec
    if kind == "checker":
        t = (f'<texture name="{name}_tex" type="2d" builtin="checker" width="128" height="128" '
             f'rgb1="{_f(rgb1)}" rgb2="{_f(rgb2)}"/>')
    else:
        t = (f'<texture name="{name}_tex" type="2d" builtin="flat" mark="random" random="0.02" width="256" '
             f'height="256" rgb1="{_f(rgb1)}" markrgb="{_f(rgb2)}"/>')
    return [t, f'<material name="{name}" texture="{name}_tex" texuniform="true" texrepeat="{rep[0]:g} {rep[1]:g}"/>']


def _box(name, pos, size, rgba) -> str:
    return (f'<geom name="{name}" type="box" pos="{_f(pos)}" size="{_f(size)}" rgba="{_f(rgba)}" '
            f'contype="0" conaffinity="0"/>')


def room_parts(room: str, dist: float = 1.2, height: float = 2.0) -> tuple[list[str], list[str]]:
    """(asset の XML の列, worldbody の XML の列)。床（floor_room）と壁4枚と家具。"""
    if room not in _SPEC:
        raise ValueError(f"知らない部屋: {room}（{ROOMS}）")
    sp = _SPEC[room]
    assets = _tex("room_floor", sp["floor"]) + _tex("room_wall", sp["wall"])
    world = [f'<geom name="floor_room" type="plane" size="5 5 0.1" material="room_floor" contype="0" conaffinity="0"/>']
    half_w, half_h = dist * 1.6, height / 2.0
    cz = -0.0 + half_h
    r2 = 0.70710678
    for nm, pos, quat in (("wall_nx", [-dist, 0, cz], [r2, 0, r2, 0]), ("wall_px", [dist, 0, cz], [r2, 0, -r2, 0]),
                          ("wall_ny", [0, -dist, cz], [r2, -r2, 0, 0]), ("wall_py", [0, dist, cz], [r2, r2, 0, 0])):
        world.append(f'<geom name="{nm}" type="plane" size="{half_w:.6g} {half_h:.6g} 0.1" pos="{_f(pos)}" '
                     f'quat="{_f(quat)}" material="room_wall" contype="0" conaffinity="0"/>')
    cx = float(EYE_MID[0])
    if room == "食事":       # 食卓：天板（物の置き場の下 0.2m）と脚
        top = 0.2
        tx = cx - 0.45
        world.append(_box("table_top", (tx, 0, top - 0.015), (0.35, 0.6, 0.015), (0.6, 0.4, 0.22, 1)))
        for sx in (-0.3, 0.3):
            for sy in (-0.55, 0.55):
                world.append(_box(f"table_leg_{sx}_{sy}", (tx + sx, sy, (top - 0.03) / 2), (0.02, 0.02, (top - 0.03) / 2),
                                  (0.5, 0.33, 0.18, 1)))
    elif room == "お風呂":   # 湯船：白い箱と青い湯の面（右の向こう。物の置き場 y≈0 と重ならない）
        bx, by = -dist + 0.3, 0.6
        world.append(_box("tub", (bx, by, 0.25), (0.25, 0.4, 0.25), (0.95, 0.95, 0.95, 1)))
        world.append(_box("tub_water", (bx, by, 0.505), (0.21, 0.36, 0.005), (0.35, 0.6, 0.9, 1)))
    else:                    # 棚：茶色の枠と板3枚（左の向こう）
        sx0, sy0 = -dist + 0.2, -0.6
        world.append(_box("shelf_back", (sx0 - 0.15, sy0, 0.5), (0.02, 0.4, 0.5), (0.55, 0.38, 0.22, 1)))
        for k, z in enumerate((0.02, 0.35, 0.68, 0.98)):
            world.append(_box(f"shelf_board_{k}", (sx0, sy0, z), (0.15, 0.4, 0.015), (0.62, 0.44, 0.26, 1)))
        for sy in (sy0 - 0.4, sy0 + 0.4):
            world.append(_box(f"shelf_side_{sy:.2f}", (sx0, sy, 0.5), (0.15, 0.015, 0.5), (0.55, 0.38, 0.22, 1)))
    return assets, world
