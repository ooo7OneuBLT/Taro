"""物の形の一覧（目だけの場面）。

どの物も「体（body）1つ＋形（geom）いくつか」。体には自由な関節（freejoint）を付けるが、この世界は物理を
積分しない（重力で落ちない）。物の場所と向きは、場面・親・道具が書いた所にそのまま留まる
（MIMo の場面でも物は親が手に持つ扱い（hold）で、毎歩、場所を書いていた。ここでは書いたときだけ動く）。

物は衝突しない（contype=0）。見えるだけ。隠すときは形の組（geom group）を描かない組へ移す（env.py の hide）。
透明度を下げるのではないので、影も落とさない。

形の寸法は、MIMo の段階2〜6の場面の値をそのまま使う（半径 3cm の球・円柱、2.5cm の箱）。
くつ・おさら・おわん・かばん・がおお・ばす・でんしゃ・とけい は、MIMo では型（メッシュ）の物だったが、型の
ファイルはこのリポジトリに無いので、単純な形を組み合わせて作る［Tier3・形はモデル上の選択］。大きさは
30cm 先で 6〜10cm（段階2〜6の物と同じくらいの見かけの大きさ）にそろえた。
"""

from __future__ import annotations

import numpy as np

# 段階2の妨害物が点滅するときの色（run/world/toy_env.py の TOY_RGBA_ON と同じ）
FLASH_RGBA = (1.0, 1.0, 0.45, 1.0)
# MIMo の円柱の半分の長さ＝半径×この比（toy_env.TOY_CYLINDER_HALF_LEN_RATIO）
CYL_HALF_LEN_RATIO = 1.0
# MIMo の楕円体の伸び（toy_env.TOY_ELLIPSOID_ELONGATION）
ELLIPSOID_ELONGATION = 1.4

# 物の種類 → 既定の色（場面の JSON に色が書いてあればそちらを使う）
DEFAULT_RGBA = {
    "cup": (0.95, 0.95, 0.95, 1.0),
    "shoe": (0.85, 0.2, 0.2, 1.0),
    "plate": (0.92, 0.9, 0.82, 1.0),
    "bowl": (0.95, 0.6, 0.2, 1.0),
    "bag": (0.55, 0.35, 0.2, 1.0),
    "lion": (0.95, 0.75, 0.3, 1.0),
    "bus": (0.95, 0.8, 0.1, 1.0),
    "train": (0.2, 0.55, 0.3, 1.0),
    "clock": (0.95, 0.95, 0.9, 1.0),
}
DARK = "0.12 0.12 0.14 1"


def _f(v) -> str:
    return " ".join(f"{float(x):.5g}" for x in v)


def _rgba(rgba) -> str:
    return _f(rgba if len(rgba) == 4 else list(rgba) + [1.0])


def geoms_for(kind: str, radius: float, rgba) -> list[str]:
    """物の種類から形（geom）の XML を作る。group は env.py が付け直すので、ここでは書かない。
    体の向き：+x が太郎の方（太郎は -x の向きを見る）、+z が上。"""
    r = float(radius)
    c = _rgba(rgba)
    g = 'contype="0" conaffinity="0"'
    if kind == "sphere":
        return [f'<geom type="sphere" size="{r:.5g}" rgba="{c}" {g}/>']
    if kind == "cylinder":
        return [f'<geom type="cylinder" size="{r:.5g} {r * CYL_HALF_LEN_RATIO:.5g}" rgba="{c}" {g}/>']
    if kind == "box":
        return [f'<geom type="box" size="{r:.5g} {r:.5g} {r:.5g}" rgba="{c}" {g}/>']
    if kind == "ellipsoid":
        return [f'<geom type="ellipsoid" size="{r:.5g} {r:.5g} {r * ELLIPSOID_ELONGATION:.5g}" rgba="{c}" {g}/>']
    if kind == "cup":     # 円柱＋取っ手（右側）
        return [f'<geom type="cylinder" size="{r:.5g} {r * 1.15:.5g}" rgba="{c}" {g}/>',
                f'<geom type="capsule" fromto="0 {-r * 0.95:.5g} {r * 0.6:.5g} 0 {-r * 1.55:.5g} 0" '
                f'size="{r * 0.18:.5g}" rgba="{c}" {g}/>',
                f'<geom type="capsule" fromto="0 {-r * 1.55:.5g} 0 0 {-r * 0.95:.5g} {-r * 0.6:.5g}" '
                f'size="{r * 0.18:.5g}" rgba="{c}" {g}/>']
    if kind == "shoe":    # 底＋甲＋つま先。横向き（つま先が太郎から見て右）
        return [f'<geom type="box" pos="0 0 {-r * 0.55:.5g}" size="{r * 0.75:.5g} {r * 1.7:.5g} {r * 0.15:.5g}" '
                f'rgba="0.95 0.95 0.92 1" {g}/>',
                f'<geom type="box" pos="0 {r * 0.35:.5g} {-r * 0.05:.5g}" size="{r * 0.7:.5g} {r * 1.0:.5g} '
                f'{r * 0.4:.5g}" rgba="{c}" {g}/>',
                f'<geom type="ellipsoid" pos="0 {-r * 0.9:.5g} {-r * 0.2:.5g}" size="{r * 0.7:.5g} {r * 0.8:.5g} '
                f'{r * 0.3:.5g}" rgba="{c}" {g}/>',
                f'<geom type="cylinder" pos="0 {r * 0.6:.5g} {r * 0.4:.5g}" size="{r * 0.55:.5g} {r * 0.08:.5g}" '
                f'rgba="{DARK}" {g}/>']
    if kind == "plate":   # 平たい円板（面が太郎の方を向くよう少し立てる）
        return [f'<geom type="cylinder" euler="0 60 0" size="{r * 1.6:.5g} {r * 0.12:.5g}" rgba="{c}" {g}/>',
                f'<geom type="cylinder" euler="0 60 0" pos="{r * 0.1:.5g} 0 {r * 0.06:.5g}" '
                f'size="{r * 1.1:.5g} {r * 0.08:.5g}" rgba="0.85 0.82 0.75 1" {g}/>']
    if kind == "bowl":    # 平たい楕円体＋ふち
        return [f'<geom type="ellipsoid" size="{r * 1.4:.5g} {r * 1.4:.5g} {r * 0.75:.5g}" rgba="{c}" {g}/>',
                f'<geom type="cylinder" pos="0 0 {r * 0.45:.5g}" size="{r * 1.25:.5g} {r * 0.06:.5g}" '
                f'rgba="0.98 0.95 0.9 1" {g}/>']
    if kind == "bag":     # 箱＋持ち手
        return [f'<geom type="box" size="{r * 0.5:.5g} {r * 1.2:.5g} {r * 0.95:.5g}" rgba="{c}" {g}/>',
                f'<geom type="capsule" fromto="0 {-r * 0.6:.5g} {r * 0.9:.5g} 0 {-r * 0.35:.5g} {r * 1.6:.5g}" '
                f'size="{r * 0.1:.5g}" rgba="{DARK}" {g}/>',
                f'<geom type="capsule" fromto="0 {r * 0.6:.5g} {r * 0.9:.5g} 0 {r * 0.35:.5g} {r * 1.6:.5g}" '
                f'size="{r * 0.1:.5g}" rgba="{DARK}" {g}/>',
                f'<geom type="capsule" fromto="0 {-r * 0.35:.5g} {r * 1.6:.5g} 0 {r * 0.35:.5g} {r * 1.6:.5g}" '
                f'size="{r * 0.1:.5g}" rgba="{DARK}" {g}/>']
    if kind == "lion":    # 胴＋たてがみ＋顔
        return [f'<geom type="ellipsoid" pos="{-r * 0.3:.5g} {r * 0.6:.5g} {-r * 0.35:.5g}" '
                f'size="{r * 0.6:.5g} {r * 1.1:.5g} {r * 0.6:.5g}" rgba="{c}" {g}/>',
                f'<geom type="sphere" pos="0 {-r * 0.6:.5g} {r * 0.3:.5g}" size="{r * 0.95:.5g}" '
                f'rgba="0.6 0.35 0.12 1" {g}/>',
                f'<geom type="sphere" pos="{r * 0.35:.5g} {-r * 0.6:.5g} {r * 0.3:.5g}" size="{r * 0.62:.5g}" '
                f'rgba="{c}" {g}/>',
                f'<geom type="sphere" pos="{r * 0.92:.5g} {-r * 0.6:.5g} {r * 0.22:.5g}" size="{r * 0.12:.5g}" '
                f'rgba="{DARK}" {g}/>']
    if kind in ("bus", "train"):   # 車体＋窓＋車輪。横向き（前が太郎から見て左）
        win = "0.75 0.88 0.98 1"
        out = [f'<geom type="box" size="{r * 0.55:.5g} {r * 1.7:.5g} {r * 0.75:.5g}" rgba="{c}" {g}/>']
        for k in range(4):
            y = -r * 1.2 + k * r * 0.8
            out.append(f'<geom type="box" pos="{r * 0.56:.5g} {y:.5g} {r * 0.25:.5g}" '
                       f'size="{r * 0.01:.5g} {r * 0.28:.5g} {r * 0.25:.5g}" rgba="{win}" {g}/>')
        for y in (-r * 1.1, r * 1.1):
            out.append(f'<geom type="cylinder" euler="90 0 0" pos="{r * 0.45:.5g} {y:.5g} {-r * 0.78:.5g}" '
                       f'size="{r * 0.3:.5g} {r * 0.12:.5g}" rgba="{DARK}" {g}/>')
        if kind == "train":
            out.append(f'<geom type="cylinder" pos="0 {-r * 1.35:.5g} {r * 1.0:.5g}" size="{r * 0.15:.5g} '
                       f'{r * 0.3:.5g}" rgba="{DARK}" {g}/>')
            out.append(f'<geom type="box" pos="0 {r * 1.3:.5g} {r * 0.95:.5g}" size="{r * 0.5:.5g} '
                       f'{r * 0.4:.5g} {r * 0.25:.5g}" rgba="{c}" {g}/>')
        return out
    if kind == "clock":   # 文字盤（面が太郎の方）＋ふち＋針2本
        return [f'<geom type="cylinder" euler="0 90 0" size="{r * 1.2:.5g} {r * 0.15:.5g}" rgba="0.3 0.3 0.75 1" {g}/>',
                f'<geom type="cylinder" euler="0 90 0" pos="{r * 0.05:.5g} 0 0" size="{r * 1.0:.5g} {r * 0.13:.5g}" '
                f'rgba="{c}" {g}/>',
                f'<geom type="box" pos="{r * 0.2:.5g} 0 {r * 0.3:.5g}" size="{r * 0.02:.5g} {r * 0.05:.5g} '
                f'{r * 0.4:.5g}" rgba="{DARK}" {g}/>',
                f'<geom type="box" pos="{r * 0.2:.5g} {-r * 0.25:.5g} 0" size="{r * 0.02:.5g} {r * 0.3:.5g} '
                f'{r * 0.04:.5g}" rgba="{DARK}" {g}/>']
    raise ValueError(f"知らない物の種類: {kind}")


# 親の名づけの的（ParentLabeling の utterances のキー）→ この世界の物。MIMo の10択の場面（toy5〜11）と同じ並び。
# toy1・toy2・toy3 は場面の JSON の toy・toy2・toy3 から作る（下の build で別扱い）
PRESENT_CATALOGUE = {
    "toy5": ("present_cup", "cup", 0.03),
    "toy6": ("present_plate", "plate", 0.03),
    "toy7": ("present_bowl", "bowl", 0.03),
    "toy8": ("present_bag", "bag", 0.03),
    "toy9": ("present_lion", "lion", 0.03),
    "toy10": ("present_bus", "bus", 0.03),
    "toy11": ("present_ball", "sphere", 0.03),
    "toy12": ("present_shoe", "shoe", 0.03),
    "toy13": ("present_train", "train", 0.03),
    "toy14": ("present_clock", "clock", 0.03),
}
PRESENT_RGBA = {"present_ball": (0.9, 0.3, 0.3, 1.0)}


def catalogue_rgba(body: str, kind: str):
    return PRESENT_RGBA.get(body, DEFAULT_RGBA.get(kind, (0.8, 0.8, 0.8, 1.0)))


def body_xml(name: str, kind: str, radius: float, rgba, pos=(0.0, 0.0, -5.0)) -> str:
    """物の体の XML（自由な関節つき）。最初の場所は env が reset で書くので、ここでは仮の場所。"""
    inner = "\n      ".join(geoms_for(kind, radius, rgba))
    return (f'    <body name="{name}" pos="{_f(pos)}">\n      <freejoint name="{name}_free"/>\n'
            f'      {inner}\n    </body>')


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v
