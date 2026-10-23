"""段階9：本物の物（3Dで撮った・作った家庭用品）で20語×4個体をそろえる（ユーザー決定 2026-10-18「自作の物は見た目が
分かりにくすぎる」「足りないのは別のデータで補って」）。

* 選んだ一覧は `core_v2/data/段階9_本物の物.json`（語・個体番号・出どころ・ID・ライセンス・作者・大きさ・色）。コミットする
* 物のファイルは G/assets の下（コミットしない）：
    元：G/assets/gso/models/<名前>/（Google Scanned Objects の MuJoCo 版、kevinzakka/mujoco_scanned_objects、CC-BY 4.0）
        G/assets/objaverse/glbs/<uid>.glb（Objaverse、Sketchfab の CC0・CC-BY・CC-BY-SA だけ）
    直したもの：G/assets/real/<種類>_<個体>/p<i>.obj（＋t<i>.png）… `python -m core_v2.tools.make_real_objects` で作る
* 直し方：上の軸を z にそろえ（Objaverse は y が上）、外枠の真ん中を原点へ寄せ、いちばん長い辺を一覧の size_m にする。
  部品（材質ごとの形）は1つずつ別の mesh にする（MuJoCo は1つの mesh に模様1枚）
* MuJoCo の注意：薄い面だけの形は「体積が小さすぎる」で読めないので mesh は inertia="shell"、形の重さは使わず体に
  決まった重さ（inertial）を書く。物はぶつからない（contype=0）＝自作の物と同じく親の手と目の前に置くだけ
* 色の名前（試験 B2 の「同じ色の別の種類」）は、模様の画像と材質の色の、面の広さで重みをつけた平均にいちばん近い色
  （自作の8色＋白・黒・灰）。make_real_objects が決めて一覧に書く
"""

from __future__ import annotations

import json
import os

from .objects import _f

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
LIST_PATH = os.path.join(REPO, "core_v2", "data", "段階9_本物の物.json")
ASSET_DIR = os.path.join(REPO, "G", "assets")
REAL_DIR = os.path.join(ASSET_DIR, "real")

COLOR_NAMES = {
    "赤": (0.85, 0.15, 0.15), "青": (0.15, 0.3, 0.85), "緑": (0.15, 0.62, 0.25), "黄": (0.95, 0.8, 0.1),
    "橙": (0.95, 0.5, 0.1), "紫": (0.55, 0.25, 0.75), "桃": (0.95, 0.55, 0.7), "茶": (0.5, 0.3, 0.15),
    "白": (0.92, 0.92, 0.9), "黒": (0.1, 0.1, 0.1), "灰": (0.5, 0.5, 0.5),
}

_LIST: dict | None = None


def load_list(path: str = LIST_PATH) -> dict:
    """一覧の json（{"items": [{word, k, ...}, ...], ...}）。"""
    global _LIST
    if _LIST is None or path != LIST_PATH:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if path != LIST_PATH:
            return data
        _LIST = data
    return _LIST


def item(word: str, k: int) -> dict:
    for it in load_list()["items"]:
        if it["word"] == word and int(it["k"]) == int(k):
            return it
    raise KeyError(f"本物の物の一覧に {word} の個体{k} が無い（{LIST_PATH}）")


def obj_dir(kind: str, k: int) -> str:
    return os.path.join(REAL_DIR, f"{kind}_{k}")


def nearest_color(rgb) -> str:
    return min(COLOR_NAMES, key=lambda n: sum((a - b) ** 2 for a, b in zip(COLOR_NAMES[n], rgb)))


def color_name(px, w=None) -> str:
    """表面の色の見本（[n, 3]、0〜1）から色の名前。色みのある見本が3割以上なら、その色相の平均で決める
    （平均の色そのものは白い縞や影で灰に寄るので使わない）。色みが少なければ明るさで白・灰・黒。"""
    import colorsys

    import numpy as np
    px = np.asarray(px, float).reshape(-1, 3)
    w = np.ones(len(px)) if w is None else np.asarray(w, float).reshape(-1)
    hsv = np.array([colorsys.rgb_to_hsv(*p) for p in px])
    chrom = (hsv[:, 1] > 0.35) & (hsv[:, 2] > 0.2)
    if w[chrom].sum() >= 0.3 * w.sum():
        hh = hsv[chrom, 0] * 2 * np.pi
        ww = w[chrom]
        h = (np.degrees(np.arctan2((ww * np.sin(hh)).sum(), (ww * np.cos(hh)).sum())) + 360) % 360
        s = float((ww * hsv[chrom, 1]).sum() / ww.sum())
        v = float((ww * hsv[chrom, 2]).sum() / ww.sum())
        if h < 15 or h >= 340:
            return "茶" if v < 0.4 else ("桃" if s < 0.5 and v > 0.7 else "赤")
        if h < 42:
            return "茶" if v < 0.6 else "橙"
        if h < 70:
            return "黄"
        if h < 170:
            return "緑"
        if h < 260:
            return "青"
        if h < 300:
            return "紫"
        return "桃"
    v = float((w * hsv[:, 2]).sum() / w.sum())
    return "白" if v > 0.7 else ("黒" if v < 0.3 else "灰")


def parts_of(kind: str, k: int) -> list[dict]:
    """直したファイルの部品の一覧（make_real_objects が書く parts.json）。無ければ作り方を書いて止める。"""
    p = os.path.join(obj_dir(kind, k), "parts.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"本物の物のファイルが無い：{p}。PC で `python -m core_v2.tools.make_real_objects` を回す"
                                "（元の物は G/assets/gso・G/assets/objaverse。取り方は make_real_objects の説明）")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def assets_xml(inst: dict) -> list[str]:
    d = obj_dir(inst["kind"], inst["k"]).replace("\\", "/")
    nm = inst["name"]
    out = []
    for i, prt in enumerate(parts_of(inst["kind"], inst["k"])):
        out.append(f'<mesh name="{nm}_m{i}" file="{d}/{prt["mesh"]}" inertia="shell"/>')
        rgba = _f(list(prt["rgba"][:3]) + [1.0])
        if prt.get("tex"):
            out.append(f'<texture name="{nm}_t{i}" type="2d" file="{d}/{prt["tex"]}"/>')
            out.append(f'<material name="{nm}_mt{i}" texture="{nm}_t{i}" rgba="{rgba}"/>')
        else:
            out.append(f'<material name="{nm}_mt{i}" rgba="{rgba}"/>')
    return out


def body_xml(inst: dict, name: str | None = None, pos=(0.0, 0.0, -5.0)) -> str:
    name = name or inst["name"]
    nm = inst["name"]
    geoms = [f'<geom type="mesh" mesh="{nm}_m{i}" material="{nm}_mt{i}" contype="0" conaffinity="0" mass="0"/>'
             for i in range(len(parts_of(inst["kind"], inst["k"])))]
    inner = "\n      ".join(['<inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>'] + geoms)
    return (f'    <body name="{name}" pos="{_f(pos)}">\n      <freejoint name="{name}_free"/>\n'
            f'      {inner}\n    </body>')


# ---------------------------------------------------------------- 直し方（PC。trimesh を使う）
def convert(src: str, ident: str, size_m: float, out_dir: str, tilt_deg: float = 0.0) -> dict:
    """元の物（src="gso"|"objaverse"、ident＝名前か uid）を部品ごとの obj＋png に直して out_dir へ。
    返り値 {"parts": [...], "rgb": 平均の色, "color": 色の名前, "orig_size_m": 元のいちばん長い辺}。"""
    import math

    import numpy as np
    import trimesh
    from PIL import Image

    if src == "gso":
        mdir = os.path.join(ASSET_DIR, "gso", "models", ident)
        m = trimesh.load(os.path.join(mdir, "model.obj"), process=False, force="mesh")
        img = Image.open(os.path.join(mdir, "texture.png"))
        uv = getattr(m.visual, "uv", None)
        meshes = [(m, uv, img, (1.0, 1.0, 1.0))]
        rot = np.eye(4)
    elif src == "objaverse":
        sc = trimesh.load(os.path.join(ASSET_DIR, "objaverse", "glbs", f"{ident}.glb"), force="scene")
        meshes = []
        for g in sc.dump():
            if not isinstance(g, trimesh.Trimesh) or not len(g.faces):
                continue
            vis, uv, img, rgb = g.visual, None, None, (0.7, 0.7, 0.7)
            if isinstance(vis, trimesh.visual.TextureVisuals):
                mat = vis.material
                img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
                if vis.uv is not None and len(vis.uv) == len(g.vertices):
                    uv = vis.uv
                col = getattr(mat, "baseColorFactor", None)
                if col is None:
                    col = getattr(mat, "main_color", None)
                if col is not None:
                    c = np.asarray(col, float)
                    rgb = tuple((c / 255.0 if c.max() > 1 else c)[:3])
            else:
                try:
                    rgb = tuple(np.asarray(vis.main_color, float)[:3] / 255.0)
                except Exception:  # noqa: BLE001
                    pass
            meshes.append((g, uv, img, rgb))
        rot = trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0])   # y が上 → z が上
    else:
        raise ValueError(src)
    if not meshes:
        raise ValueError(f"{src} {ident}: 形が無い")
    for g, *_ in meshes:
        g.apply_transform(rot)
    if tilt_deg:
        # 平たい物（すぷうん）：いちばん長い向きを目から見て横（y）に、いちばん薄い向き（面の向き）を上から tilt_deg 倒して
        # 目の方へ向ける。35cm・下20度の目からは、寝かせたままだと縁しか見えないため
        allv = np.vstack([g.vertices for g, *_ in meshes])
        c = allv - allv.mean(0)
        _, _, vt = np.linalg.svd(c[:: max(1, len(c) // 5000)], full_matrices=False)
        long_ax, thin_ax = vt[0], vt[2]
        t = math.radians(tilt_deg)
        target_thin = np.array([math.sin(t), 0.0, math.cos(t)])   # 上（z）から、目のいる側（+x。目は -x を向く）へ倒す
        src_b = np.stack([long_ax, np.cross(thin_ax, long_ax), thin_ax])
        dst_b = np.stack([np.array([0.0, 1.0, 0.0]), np.cross(target_thin, [0.0, 1.0, 0.0]), target_thin])
        R = np.eye(4)
        R[:3, :3] = dst_b.T @ src_b
        for g, *_ in meshes:
            g.apply_transform(R)
    allv = np.vstack([g.vertices for g, *_ in meshes])
    lo, hi = allv.min(0), allv.max(0)
    span = float((hi - lo).max())
    s, mid = size_m / span, (lo + hi) / 2
    os.makedirs(out_dir, exist_ok=True)
    parts, acc, wsum, samples, sw = [], np.zeros(3), 0.0, [], []
    for i, (g, uv, img, rgb) in enumerate(meshes[:60]):
        v = (np.asarray(g.vertices) - mid) * s
        tex = uv is not None and img is not None
        with open(os.path.join(out_dir, f"p{i}.obj"), "w") as f:
            f.writelines(f"v {x[0]:.6f} {x[1]:.6f} {x[2]:.6f}\n" for x in v)
            if tex:
                f.writelines(f"vt {t[0]:.6f} {t[1]:.6f}\n" for t in uv)
                f.writelines(f"f {a}/{a} {b}/{b} {c}/{c}\n" for a, b, c in np.asarray(g.faces) + 1)
            else:
                f.writelines(f"f {a} {b} {c}\n" for a, b, c in np.asarray(g.faces) + 1)
        prt = {"mesh": f"p{i}.obj", "tex": None, "rgba": [round(float(x), 4) for x in rgb] + [1.0]}
        area = float(g.area) * s * s
        if tex:
            im = img.convert("RGB")
            im.save(os.path.join(out_dir, f"t{i}.png"))
            prt["tex"] = f"t{i}.png"
            arr = np.asarray(im, float) / 255.0
            h, w = arr.shape[:2]
            t = np.clip(np.asarray(uv)[np.asarray(g.faces).ravel()], 0, 1)
            px = arr[((1 - t[:, 1]) * (h - 1)).astype(int), (t[:, 0] * (w - 1)).astype(int)]
            px = px * np.asarray(rgb)
            col = px.mean(0)
            pick = px[:: max(1, len(px) // 400)]
        else:
            col = np.asarray(rgb)
            pick = col[None, :]
        samples.append(pick)
        sw.append(np.full(len(pick), area / len(pick)))
        acc += area * col
        wsum += area
        parts.append(prt)
    with open(os.path.join(out_dir, "parts.json"), "w", encoding="utf-8") as f:
        json.dump(parts, f, ensure_ascii=False, indent=1)
    mean = acc / max(wsum, 1e-12)
    return {"parts": parts, "rgb": [round(float(x), 3) for x in mean], "color": color_name(np.vstack(samples), np.concatenate(sw)),
            "orig_size_m": round(span, 4)}
