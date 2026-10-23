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


# ================================================================ 段階9：名詞を厚くする（20語の自作の形）
# 仕様書 `2026-10-18_段階9_仕様書_名詞を厚く.md` §2。全部の語を単純な形（球・円柱・楕円体・箱・カプセル）の組み合わせで
# 作る。種類ごとに「形の作り方」を1つ書き、個体ごとの違い（色・大きさ・模様・部品の長さ）は instances.py が決めて渡す。
# 上の geoms_for（段階2〜8の物）は変えない（前の段階の見え方を変えないため）。
# look  : 主な色の書き方（'rgba="…"' か、模様つきなら 'material="…"'）
# L     : 部品の長さの倍率 {名前: 0.8〜1.2}（無ければ1）
# 体の向きは geoms_for と同じ（+x が太郎の方、+y が太郎から見て右、+z が上）
WHITE = "0.95 0.95 0.92 1"
NOUN_ACCENT = {   # 種類ごとの決まった脇の色（目・車輪・へたなど。個体で変えない）
    "apple_stem": "0.4 0.25 0.1 1", "leaf": "0.2 0.6 0.2 1", "beak": "0.98 0.55 0.1 1",
    "mane": "0.55 0.3 0.1 1", "ear": "0.4 0.25 0.15 1", "window": "0.75 0.88 0.98 1", "sole": WHITE, "banana_tip": "0.35 0.25 0.1 1",
}


def _g(gtype: str, size, look: str, pos=None, euler=None, fromto=None) -> str:
    a = [f'type="{gtype}"']
    if fromto is not None:
        a.append(f'fromto="{_f(fromto)}"')
    if pos is not None:
        a.append(f'pos="{_f(pos)}"')
    if euler is not None:
        a.append(f'euler="{_f(euler)}"')
    a.append(f'size="{_f(size)}"')
    a.append(look)
    return f'<geom {" ".join(a)} contype="0" conaffinity="0"/>'


def _rg(c: str) -> str:
    return f'rgba="{c}"'


def noun_geoms(kind: str, r: float, look: str, L: dict | None = None) -> list[str]:
    """段階9の20種類の形。r＝基準の大きさ（m、個体の大きさの倍率を掛けたもの）。"""
    L = L or {}
    l = lambda k: float(L.get(k, 1.0))  # noqa: E731
    d = _rg(DARK)
    if kind == "cup":            # 円柱＋取っ手
        h = r * 1.15 * l("body")
        hw = r * 0.6 * l("handle")
        return [_g("cylinder", (r, h), look),
                _g("capsule", (r * 0.18,), look, fromto=(0, r * 0.95, hw, 0, r * 1.55, 0)),
                _g("capsule", (r * 0.18,), look, fromto=(0, r * 1.55, 0, 0, r * 0.95, -hw)),
                _g("cylinder", (r * 0.85, r * 0.02), d, pos=(0, 0, h))]
    if kind == "spoon":          # 丸いさじ＋柄（斜めに立てて太郎の方へ面を向ける）
        hl = r * 1.6 * l("handle")
        return [_g("ellipsoid", (r * 0.25, r * 0.55, r * 0.75), look, pos=(0, 0, hl * 0.6), euler=(0, 30, 0)),
                _g("capsule", (r * 0.12,), look, fromto=(0, 0, hl * 0.6 - r * 0.6, 0, 0, -hl))]
    if kind == "plate":          # 平たい円板＋内側の段
        rr = r * 1.6 * l("rim")
        return [_g("cylinder", (rr, r * 0.12), look, euler=(0, 60, 0)),
                _g("cylinder", (rr * 0.68, r * 0.08), _rg(WHITE), pos=(r * 0.1, 0, r * 0.06), euler=(0, 60, 0))]
    if kind == "bread":          # 山形の食パン：箱＋上が丸い円柱
        w, hh = r * 1.0 * l("width"), r * 0.7 * l("height")
        return [_g("box", (r * 0.45, w, hh), look),
                _g("cylinder", (w * 0.75, r * 0.45), look, pos=(0, -w * 0.35, hh), euler=(0, 90, 0)),
                _g("cylinder", (w * 0.75, r * 0.45), look, pos=(0, w * 0.35, hh), euler=(0, 90, 0)),
                _g("box", (r * 0.02, w * 0.85, hh * 0.85), _rg(WHITE), pos=(r * 0.45, 0, -hh * 0.05))]
    if kind == "banana":         # 弧に並べた5つのカプセル＋両端の黒い先
        ln = r * 0.55 * l("length")
        pts = [(0.0, (k - 2.5) * ln * 1.25, 0.1 * (k - 2.5) ** 2 * r) for k in range(6)]
        out = [_g("capsule", (r * 0.32,), look, fromto=(*pts[k], *pts[k + 1])) for k in range(5)]
        out += [_g("sphere", (r * 0.14,), _rg(NOUN_ACCENT["banana_tip"]), pos=p) for p in (pts[0], pts[-1])]
        return out
    if kind == "apple":          # 球＋へた＋葉
        st = r * 0.35 * l("stem")
        return [_g("sphere", (r,), look),
                _g("capsule", (r * 0.06,), _rg(NOUN_ACCENT["apple_stem"]), fromto=(0, 0, r * 0.85, 0, r * 0.05, r + st)),
                _g("ellipsoid", (r * 0.08, r * 0.3, r * 0.12), _rg(NOUN_ACCENT["leaf"]), pos=(0, r * 0.3, r + st * 0.6),
                   euler=(25, 0, 0))]
    if kind == "duck":           # 胴の楕円体＋頭＋くちばし＋目（左向き）
        nk = r * 0.9 * l("neck")
        return [_g("ellipsoid", (r * 0.8, r * 1.2, r * 0.75), look),
                _g("ellipsoid", (r * 0.45, r * 0.5, r * 0.25), look, pos=(0, r * 1.05, r * 0.35), euler=(-30, 0, 0)),
                _g("sphere", (r * 0.55,), look, pos=(0, -r * 0.65, nk)),
                _g("ellipsoid", (r * 0.22, r * 0.38, r * 0.12), _rg(NOUN_ACCENT["beak"]), pos=(0, -r * 1.25, nk - r * 0.08)),
                _g("sphere", (r * 0.09,), d, pos=(r * 0.42, -r * 0.85, nk + r * 0.15))]
    if kind == "fish":           # 細長い楕円体＋尾びれ＋背びれ＋目（左向き）
        bl = r * 1.4 * l("body")
        tl = r * 0.6 * l("tail")
        return [_g("ellipsoid", (r * 0.35, bl, r * 0.6), look),
                _g("box", (r * 0.06, tl * 0.5, tl), look, pos=(0, bl + tl * 0.3, 0), euler=(45, 0, 0)),
                _g("ellipsoid", (r * 0.06, r * 0.5, r * 0.3), look, pos=(0, 0, r * 0.6)),
                _g("sphere", (r * 0.12,), d, pos=(r * 0.25, -bl * 0.65, r * 0.15))]
    if kind == "towel":          # たたんだタオル：重ねた平たい箱2枚＋ふちの線
        w = r * 1.5 * l("width")
        return [_g("box", (r * 1.0, w, r * 0.22), look, pos=(0, 0, -r * 0.2), euler=(0, 30, 0)),
                _g("box", (r * 1.0, w * 0.98, r * 0.22), look, pos=(r * 0.05, 0, r * 0.25), euler=(0, 30, 0)),
                _g("box", (r * 1.02, w * 0.99, r * 0.04), _rg(WHITE), pos=(r * 0.05, 0, r * 0.25), euler=(0, 30, 0))]
    if kind == "ball":           # 球＋白い帯
        rr = r * 1.1 * l("size")
        return [_g("sphere", (rr,), look),
                _g("cylinder", (rr * 1.01, rr * 0.12), _rg(WHITE), euler=(0, 30, 0))]
    if kind in ("dog", "cat", "lion", "bear"):
        return _animal(kind, r, look, l, d)
    if kind in ("car", "bus", "train"):
        return _vehicle(kind, r, look, l, d)
    if kind == "shoe":           # 底＋甲＋つま先＋はき口
        ln = l("length")
        return [_g("box", (r * 0.75, r * 1.7 * ln, r * 0.15), _rg(NOUN_ACCENT["sole"]), pos=(0, 0, -r * 0.55)),
                _g("box", (r * 0.7, r * 1.0 * ln, r * 0.4), look, pos=(0, r * 0.35 * ln, -r * 0.05)),
                _g("ellipsoid", (r * 0.7, r * 0.8 * ln, r * 0.3), look, pos=(0, -r * 0.9 * ln, -r * 0.2)),
                _g("cylinder", (r * 0.55, r * 0.08), d, pos=(0, r * 0.6 * ln, r * 0.4))]
    if kind == "hat":            # 丸い山＋広いつば＋帯
        br = r * 1.7 * l("brim")
        return [_g("ellipsoid", (r * 0.95, r * 0.95, r * 0.85 * l("crown")), look, pos=(0, 0, 0)),
                _g("cylinder", (br, r * 0.06), look, pos=(0, 0, -r * 0.1)),
                _g("cylinder", (r * 0.97, r * 0.15), d, pos=(0, 0, r * 0.12))]
    if kind == "blocks":         # 積んだ積み木：箱2つ＋上に三角の代わりの円柱の半分
        s = r * 0.75 * l("block")
        return [_g("box", (s, s, s), look, pos=(0, -s * 1.05, -s)),
                _g("box", (s, s, s), look, pos=(0, s * 1.05, -s)),
                _g("box", (s, s * 2.0, s * 0.6), look, pos=(0, 0, s * 0.62)),
                _g("cylinder", (s * 0.8, s * 0.95), _rg(WHITE), pos=(0, 0, s * 1.3), euler=(0, 90, 0))]
    raise ValueError(f"段階9に無い物の種類: {kind}")


def _animal(kind, r, look, l, d):
    """四つ足の動物（犬・猫・ライオン）と座ったくま。どれも左向き（頭が -y）。"""
    if kind == "bear":           # 座ったくま：胴＋頭＋耳＋鼻先＋手足
        hd = r * 0.65
        return [_g("ellipsoid", (r * 0.75, r * 0.8, r * 0.95), look),
                _g("sphere", (hd,), look, pos=(0, 0, r * 1.35)),
                _g("sphere", (r * 0.24,), look, pos=(0, -hd * 0.75, r * 1.35 + hd * 0.75)),
                _g("sphere", (r * 0.24,), look, pos=(0, hd * 0.75, r * 1.35 + hd * 0.75)),
                _g("ellipsoid", (r * 0.25, r * 0.28, r * 0.2), _rg(WHITE), pos=(hd * 0.85, 0, r * 1.25)),
                _g("sphere", (r * 0.08,), d, pos=(hd * 1.05, 0, r * 1.32)),
                _g("sphere", (r * 0.07,), d, pos=(hd * 0.8, -r * 0.25, r * 1.5)),
                _g("sphere", (r * 0.07,), d, pos=(hd * 0.8, r * 0.25, r * 1.5)),
                _g("capsule", (r * 0.22,), look, fromto=(r * 0.3, -r * 0.7, r * 0.3, r * 0.5, -r * 0.8, -r * 0.3)),
                _g("capsule", (r * 0.22,), look, fromto=(r * 0.3, r * 0.7, r * 0.3, r * 0.5, r * 0.8, -r * 0.3)),
                _g("capsule", (r * 0.26,), look, fromto=(r * 0.3, -r * 0.45, -r * 0.8, r * 0.95 * l("leg"), -r * 0.5, -r * 0.9)),
                _g("capsule", (r * 0.26,), look, fromto=(r * 0.3, r * 0.45, -r * 0.8, r * 0.95 * l("leg"), r * 0.5, -r * 0.9))]
    body_l = r * {"dog": 1.15, "cat": 1.2, "lion": 1.3}[kind] * l("body")
    leg = r * {"dog": 0.85, "cat": 0.75, "lion": 0.9}[kind] * l("leg")
    thick = {"dog": 0.55, "cat": 0.42, "lion": 0.65}[kind]
    hd = r * {"dog": 0.5, "cat": 0.45, "lion": 0.55}[kind]
    hy, hz = -body_l - hd * 0.4, r * 0.55
    out = [_g("ellipsoid", (r * thick, body_l, r * thick * 1.05), look),
           _g("sphere", (hd,), look, pos=(0, hy, hz))]
    for sy in (-body_l * 0.65, body_l * 0.65):
        for sx in (-r * thick * 0.5, r * thick * 0.5):
            out.append(_g("capsule", (r * 0.14 * (1.3 if kind == "lion" else 1.0),), look,
                          fromto=(sx, sy, -r * 0.2, sx, sy, -r * 0.2 - leg)))
    tail = r * {"dog": 0.7, "cat": 1.3, "lion": 1.0}[kind] * l("tail")
    if kind == "cat":    # 長いしっぽを上へ
        out.append(_g("capsule", (r * 0.08,), look, fromto=(0, body_l * 0.9, r * 0.2, 0, body_l * 1.05, r * 0.2 + tail)))
    else:
        out.append(_g("capsule", (r * 0.1,), look, fromto=(0, body_l * 0.9, r * 0.2, 0, body_l + tail * 0.7, r * 0.2 + tail * 0.6)))
    eye_x = hd * 0.8
    out += [_g("sphere", (r * 0.07,), d, pos=(eye_x, hy - hd * 0.3, hz + hd * 0.25)),
            _g("sphere", (r * 0.07,), d, pos=(eye_x, hy + hd * 0.05, hz + hd * 0.25))]
    if kind == "dog":    # 突き出た鼻先＋垂れた耳
        out += [_g("ellipsoid", (hd * 0.45, hd * 0.6, hd * 0.4), look, pos=(0, hy - hd * 0.85, hz - hd * 0.25)),
                _g("sphere", (r * 0.1,), d, pos=(0, hy - hd * 1.4, hz - hd * 0.15)),
                _g("ellipsoid", (r * 0.12, r * 0.18, hd * 0.75 * l("ear")), _rg(NOUN_ACCENT["ear"]),
                   pos=(r * 0.35, hy + hd * 0.2, hz - hd * 0.1)),
                _g("ellipsoid", (r * 0.12, r * 0.18, hd * 0.75 * l("ear")), _rg(NOUN_ACCENT["ear"]),
                   pos=(-r * 0.35, hy + hd * 0.2, hz - hd * 0.1))]
    elif kind == "cat":  # とがった耳（立てた細い箱を45度回す）＋ひげ
        ear = hd * 0.45 * l("ear")
        out += [_g("box", (r * 0.04, ear * 0.7, ear * 0.7), look, pos=(0, hy - hd * 0.5, hz + hd * 0.85), euler=(45, 0, 0)),
                _g("box", (r * 0.04, ear * 0.7, ear * 0.7), look, pos=(0, hy + hd * 0.5, hz + hd * 0.85), euler=(45, 0, 0)),
                _g("capsule", (r * 0.015,), d, fromto=(hd * 0.9, hy - hd * 1.4, hz - hd * 0.1, hd * 0.9, hy - hd * 0.4, hz - hd * 0.2)),
                _g("sphere", (r * 0.07,), _rg("0.95 0.6 0.65 1"), pos=(hd * 0.95, hy - hd * 0.15, hz - hd * 0.1))]
    else:                # ライオン：頭のまわりのたてがみ
        mane = hd * 1.55 * l("mane")
        out.insert(1, _g("sphere", (mane,), _rg(NOUN_ACCENT["mane"]), pos=(-r * 0.1, hy + hd * 0.15, hz)))
    return out


def _vehicle(kind, r, look, l, d):
    """車・バス・電車：車体＋窓＋車輪。横向き（前が太郎から見て左）。"""
    ln = l("length")
    win = _rg(NOUN_ACCENT["window"])
    out = []
    if kind == "car":            # 低い車体＋小さい客室
        out.append(_g("box", (r * 0.6, r * 1.5 * ln, r * 0.38), look))
        out.append(_g("box", (r * 0.55, r * 0.85 * ln, r * 0.32 * l("cabin")), look, pos=(0, r * 0.15, r * 0.65)))
        for y in (-r * 0.35 * ln, r * 0.6 * ln):
            out.append(_g("box", (r * 0.01, r * 0.3 * ln, r * 0.22 * l("cabin")), win, pos=(r * 0.56, y, r * 0.66)))
        wy, wr, wz = (-r * 0.95 * ln, r * 0.95 * ln), r * 0.32, -r * 0.4
    else:
        hb = r * 0.75 * (1.0 if kind == "bus" else 0.85)
        out.append(_g("box", (r * 0.55, r * 1.7 * ln, hb), look))
        n = 4 if kind == "bus" else 3
        for k in range(n):
            y = -r * 1.2 * ln + k * (r * 2.4 * ln / (n - 1))
            out.append(_g("box", (r * 0.01, r * 0.28, r * 0.25 * l("window")), win, pos=(r * 0.56, y, r * 0.25)))
        if kind == "train":      # 煙突＋運転室＋前の丸い鼻
            out.append(_g("cylinder", (r * 0.15, r * 0.3 * l("chimney")), d, pos=(0, -r * 1.35 * ln, r * 1.0)))
            out.append(_g("box", (r * 0.5, r * 0.4, r * 0.25), look, pos=(0, r * 1.3 * ln, r * 0.95)))
            out.append(_g("cylinder", (r * 0.5, r * 0.15), d, pos=(0, -r * 1.75 * ln, 0), euler=(90, 0, 0)))
        wy, wr, wz = ((-r * 1.1 * ln, r * 1.1 * ln) if kind == "bus"
                      else (-r * 1.2 * ln, 0.0, r * 1.2 * ln)), r * 0.3, -hb - r * 0.03
    for y in wy:
        out.append(_g("cylinder", (wr, r * 0.12), d, pos=(r * 0.45, y, wz), euler=(90, 0, 0)))
    return out
